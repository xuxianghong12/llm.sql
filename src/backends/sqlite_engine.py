"""
SQLite graph executor — runs an ``ExportedProgram`` using pure SQL.

Workflow
--------
1. ``compile(ep)``   — walks the FX graph once and builds a flat list of
                       ``_Instr`` objects (the *execution table*), same as
                       the numpy engine.
2. ``run(inputs)``   — runs a *preflight check* that ensures every op has
                       an sqlite implementation, then executes the table
                       using a SQLite connection with llm_ops.so loaded.

All tensor data flows as BLOB bytes through the SQLite connection.
The C extension (llm_ops.so) provides accelerated SQL functions for
tensor operations (SIMD + BLAS).
"""
import os
import sqlite3
import sys
import time as _time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.fx as fx
from torch.export.graph_signature import InputKind

from src.backends.sqlite_ops import (
    SqliteOpRegistry,
    tensor_to_blob,
    blob_to_numpy,
    fused_silu_mul,
    fused_rope,
    fused_rmsnorm,
    _PARAM_TABLE,
    _LAZY_SHAPE_OPS,
)


# ---------------------------------------------------------------------------
# Compile-time optimisation constants
# ---------------------------------------------------------------------------

# Sentinel for dict.get() to distinguish None from missing keys.
_SENTINEL = object()

# Ops that return their first positional argument unchanged (no-ops).
_NOOP_TARGETS = {
    torch.ops.aten._to_copy.default,
    torch.ops.aten.to.dtype,
    torch.ops.aten.to.dtype_layout,
    torch.ops.aten.lift_fresh_copy.default,
    torch.ops.aten.detach_.default,
    torch.ops.aten.contiguous.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.copy.default,
    torch.ops.aten.alias.default,
    torch.ops.aten._assert_tensor_metadata.default,
}


def _fuse_silu_mul_instrs(instrs: List["_Instr"]) -> None:
    """Fuse consecutive ``silu → mul`` pairs into ``llm_silu_mul_nd``.

    Detects the MLP gate pattern ``silu(gate) * up`` and replaces the
    ``mul`` instruction with a fused call, eliminating one SQL round-trip
    per MLP block per layer.
    """
    name_to_instr = {i.out_name: i for i in instrs}
    for instr in instrs:
        if (instr.op != "call_function"
                or instr.target is not torch.ops.aten.mul.Tensor
                or len(instr.arg_names) < 2):
            continue
        for pos in range(2):
            arg = instr.arg_names[pos]
            if not isinstance(arg, str):
                continue
            src = name_to_instr.get(arg)
            if (src is not None
                    and src.op == "call_function"
                    and src.target is torch.ops.aten.silu.default):
                gate_arg = src.arg_names[0]
                up_arg = instr.arg_names[1 - pos]
                instr.arg_names = [gate_arg, up_arg]
                instr.resolved_fn = fused_silu_mul
                # Turn silu node into alias so the name is still resolvable
                src.op = "alias"
                src.target = src.arg_names[0]
                src.resolved_fn = None
                break


def _fuse_rope_instrs(instrs: List["_Instr"]) -> None:
    """Fuse the 7-op HuggingFace RoPE pattern into a single ``llm_rope_nd``.

    Detects: ``add(x*cos, cat([-x[half:], x[:half]])*sin)``
    Replaces the ``add`` instruction with a fused call and disables the
    intermediate instructions that have no external uses.
    """
    name_to_instr = {i.out_name: i for i in instrs}
    # Pre-build reference count for each name (how many instrs use it)
    ref_counts: Dict[str, int] = {}
    for instr in instrs:
        for arg in instr.arg_names:
            if isinstance(arg, str):
                ref_counts[arg] = ref_counts.get(arg, 0) + 1
            elif isinstance(arg, list):
                for item in arg:
                    if isinstance(item, str):
                        ref_counts[item] = ref_counts.get(item, 0) + 1

    for instr in instrs:
        if (instr.op != "call_function"
                or instr.target is not torch.ops.aten.add.Tensor
                or len(instr.arg_names) < 2):
            continue
        mul_x_name, mul_sin_name = instr.arg_names[0], instr.arg_names[1]
        if not (isinstance(mul_x_name, str) and isinstance(mul_sin_name, str)):
            continue
        mul_x_i = name_to_instr.get(mul_x_name)
        mul_sin_i = name_to_instr.get(mul_sin_name)
        if (mul_x_i is None or mul_sin_i is None
                or mul_x_i.op != "call_function"
                or mul_sin_i.op != "call_function"
                or mul_x_i.target is not torch.ops.aten.mul.Tensor
                or mul_sin_i.target is not torch.ops.aten.mul.Tensor
                or len(mul_x_i.arg_names) < 2
                or len(mul_sin_i.arg_names) < 2):
            continue
        # mul_x: [x_name, cos_name]
        x_name, cos_name = mul_x_i.arg_names[0], mul_x_i.arg_names[1]
        if not (isinstance(x_name, str) and isinstance(cos_name, str)):
            continue
        # mul_sin: [cat_name, sin_name] (or reversed)
        cat_name, sin_name = None, None
        for pos in range(2):
            cand = mul_sin_i.arg_names[pos]
            if not isinstance(cand, str):
                continue
            cand_i = name_to_instr.get(cand)
            if (cand_i is not None and cand_i.op == "call_function"
                    and cand_i.target is torch.ops.aten.cat.default):
                cat_name = cand
                sin_name = mul_sin_i.arg_names[1 - pos]
                break
        if cat_name is None or not isinstance(sin_name, str):
            continue
        cat_i = name_to_instr[cat_name]
        cat_args = cat_i.arg_names
        if (len(cat_args) < 2 or not isinstance(cat_args[0], list)
                or len(cat_args[0]) != 2 or cat_args[1] != -1):
            continue
        neg_name, slice_a_name = cat_args[0][0], cat_args[0][1]
        if not (isinstance(neg_name, str) and isinstance(slice_a_name, str)):
            continue
        neg_i = name_to_instr.get(neg_name)
        sa_i = name_to_instr.get(slice_a_name)
        if (neg_i is None or sa_i is None
                or neg_i.op != "call_function"
                or neg_i.target is not torch.ops.aten.neg.default
                or not neg_i.arg_names):
            continue
        slice_b_name = neg_i.arg_names[0]
        if not isinstance(slice_b_name, str):
            continue
        sb_i = name_to_instr.get(slice_b_name)
        if (sa_i.op != "call_function"
                or sb_i is None or sb_i.op != "call_function"
                or sa_i.target is not torch.ops.aten.slice.Tensor
                or sb_i.target is not torch.ops.aten.slice.Tensor
                or not sa_i.arg_names or not sb_i.arg_names
                or sa_i.arg_names[0] != x_name
                or sb_i.arg_names[0] != x_name):
            continue
        # Pattern confirmed.  Fuse into llm_rope_nd.
        instr.arg_names = [x_name, cos_name, sin_name]
        instr.resolved_fn = fused_rope
        # Disable intermediates that are only used within this pattern
        skip_candidates = {mul_x_name, slice_a_name, slice_b_name,
                           neg_name, cat_name, mul_sin_name}
        # An intermediate is safe to disable only if it has exactly 1 reference
        # (the instruction that consumes it within the pattern).
        for skip_name in skip_candidates:
            if ref_counts.get(skip_name, 0) == 1:
                si = name_to_instr.get(skip_name)
                if si is not None and si.op == "call_function":
                    si.op = "alias"
                    si.target = si.arg_names[0] if si.arg_names else None
                    si.resolved_fn = None


def _fuse_rmsnorm_instrs(instrs: List["_Instr"]) -> None:
    """Fuse the 6-op Qwen2 RMSNorm pattern into a single ``llm_rmsnorm_nd``.

    Detects the pattern::

        pow(x, 2) → mean(dim=-1, keepdim=True) → add(eps) → rsqrt
                  → mul(x, rsqrt) → [to.dtype alias] → mul(norm, weight)

    and replaces the final ``mul(norm, weight)`` with
    ``llm_rmsnorm_nd(x, weight, eps)``, disabling the intermediate
    instructions.

    For Qwen2.5-0.5B (24 layers, 49 RMSNorm instances) this saves
    **245 SQL calls per decode step** (5 ops → 1 per instance).
    """
    name_to_instr = {i.out_name: i for i in instrs}
    # Pre-build reference count for each name.
    ref_counts: Dict[str, int] = {}
    for instr in instrs:
        for arg in instr.arg_names:
            if isinstance(arg, str):
                ref_counts[arg] = ref_counts.get(arg, 0) + 1
            elif isinstance(arg, list):
                for item in arg:
                    if isinstance(item, str):
                        ref_counts[item] = ref_counts.get(item, 0) + 1
        for v in instr.kwarg_names.values():
            if isinstance(v, str):
                ref_counts[v] = ref_counts.get(v, 0) + 1

    for instr in instrs:
        # The anchor is the final mul(norm_result, weight).
        # We look backward: pow → mean → add(eps) → rsqrt → mul(x, rsqrt)
        # then through alias (to.dtype) to the final mul(alias, weight).
        if (instr.op != "call_function"
                or instr.target is not torch.ops.aten.mul.Tensor
                or len(instr.arg_names) < 2):
            continue
        # Find which arg traces back to a rsqrt-based norm chain
        for pos in range(2):
            norm_arg = instr.arg_names[pos]
            weight_arg = instr.arg_names[1 - pos]
            if not isinstance(norm_arg, str):
                continue
            # norm_arg might be the direct mul(x, rsqrt) output,
            # or an alias (to.dtype) wrapping it.
            norm_i = name_to_instr.get(norm_arg)
            if norm_i is None:
                continue
            actual_mul_norm_i = None
            actual_mul_norm_name = None
            if (norm_i.op == "alias" and isinstance(norm_i.target, str)):
                # Follow alias chain
                alias_src = name_to_instr.get(norm_i.target)
                if alias_src is not None:
                    actual_mul_norm_i = alias_src
                    actual_mul_norm_name = norm_i.target
            elif (norm_i.op == "call_function"
                  and norm_i.target is torch.ops.aten.mul.Tensor):
                actual_mul_norm_i = norm_i
                actual_mul_norm_name = norm_arg
            if actual_mul_norm_i is None:
                continue
            if (actual_mul_norm_i.op != "call_function"
                    or actual_mul_norm_i.target is not torch.ops.aten.mul.Tensor
                    or len(actual_mul_norm_i.arg_names) < 2):
                continue
            # mul(x, rsqrt): one arg should be x, other should be rsqrt output
            rsqrt_name = None
            x_name = None
            for mpos in range(2):
                cand = actual_mul_norm_i.arg_names[mpos]
                if not isinstance(cand, str):
                    continue
                cand_i = name_to_instr.get(cand)
                if (cand_i is not None and cand_i.op == "call_function"
                        and cand_i.target is torch.ops.aten.rsqrt.default):
                    rsqrt_name = cand
                    x_name = actual_mul_norm_i.arg_names[1 - mpos]
                    break
            if rsqrt_name is None or not isinstance(x_name, str):
                continue
            rsqrt_i = name_to_instr[rsqrt_name]
            if not rsqrt_i.arg_names:
                continue
            # rsqrt(add_result)
            add_name = rsqrt_i.arg_names[0]
            if not isinstance(add_name, str):
                continue
            add_i = name_to_instr.get(add_name)
            if (add_i is None or add_i.op != "call_function"
                    or add_i.target is not torch.ops.aten.add.Tensor
                    or len(add_i.arg_names) < 2):
                continue
            # add(mean_result, eps)
            mean_name = add_i.arg_names[0]
            eps_val = add_i.arg_names[1]
            if not isinstance(mean_name, str):
                continue
            mean_i = name_to_instr.get(mean_name)
            if (mean_i is None or mean_i.op != "call_function"
                    or mean_i.target is not torch.ops.aten.mean.dim):
                continue
            # mean(pow_result, [-1], keepdim=True)
            if not mean_i.arg_names:
                continue
            pow_name = mean_i.arg_names[0]
            if not isinstance(pow_name, str):
                continue
            pow_i = name_to_instr.get(pow_name)
            if (pow_i is None or pow_i.op != "call_function"
                    or pow_i.target is not torch.ops.aten.pow.Tensor_Scalar
                    or len(pow_i.arg_names) < 2):
                continue
            # pow(x_source, 2) — x_source should resolve to x_name
            # (possibly through alias chains like to.dtype)
            pow_x = pow_i.arg_names[0]
            pow_exp = pow_i.arg_names[1]
            if pow_exp != 2:
                continue
            # Resolve pow_x through alias chains
            resolved_pow_x = pow_x
            while isinstance(resolved_pow_x, str):
                aliased = name_to_instr.get(resolved_pow_x)
                if aliased and aliased.op == "alias" and isinstance(aliased.target, str):
                    resolved_pow_x = aliased.target
                else:
                    break
            # Similarly resolve x_name through alias chains
            resolved_x = x_name
            while isinstance(resolved_x, str):
                aliased = name_to_instr.get(resolved_x)
                if aliased and aliased.op == "alias" and isinstance(aliased.target, str):
                    resolved_x = aliased.target
                else:
                    break
            if resolved_pow_x != resolved_x:
                continue
            # Pattern confirmed!
            # Fuse: instr becomes llm_rmsnorm_nd(x, weight, eps)
            instr.arg_names = [x_name, weight_arg, eps_val]
            instr.resolved_fn = fused_rmsnorm
            # Disable intermediate nodes that are only consumed within
            # the pattern (safe to alias out).
            skip_candidates = {pow_name, mean_name, add_name, rsqrt_name,
                               actual_mul_norm_name}
            for skip_name in skip_candidates:
                if ref_counts.get(skip_name, 0) == 1:
                    si = name_to_instr.get(skip_name)
                    if si is not None and si.op == "call_function":
                        si.op = "alias"
                        si.target = si.arg_names[0] if si.arg_names else None
                        si.resolved_fn = None
            break  # Found the pattern for this mul, move to next instr


def _fuse_inline_shape_ops(instrs: List["_Instr"]) -> None:
    """Replace single-use shape ops with lazy SQL-expression wrappers.

    Shape operations (unsqueeze, transpose, view, reshape, expand) that
    are consumed by exactly one downstream instruction are converted to
    return ``SqlExpr`` objects instead of executing immediately.  The
    consumer's SQL call then inlines the shape op's SQL fragment,
    turning two round-trips into one nested SQL expression.

    This pass should run **after** all other fusion passes (silu_mul,
    rope, rmsnorm) so that those patterns are already resolved and their
    aliased intermediates are excluded from the reference count.
    """
    name_to_instr = {i.out_name: i for i in instrs}

    # Build reference counts (only for active instructions).
    ref_counts: Dict[str, int] = {}
    for instr in instrs:
        if instr.op == "alias":
            # Aliases still consume their source
            if isinstance(instr.target, str):
                ref_counts[instr.target] = ref_counts.get(instr.target, 0) + 1
            continue
        for arg in instr.arg_names:
            if isinstance(arg, str):
                ref_counts[arg] = ref_counts.get(arg, 0) + 1
            elif isinstance(arg, list):
                for item in arg:
                    if isinstance(item, str):
                        ref_counts[item] = ref_counts.get(item, 0) + 1
        for v in instr.kwarg_names.values():
            if isinstance(v, str):
                ref_counts[v] = ref_counts.get(v, 0) + 1
        for v in instr.output_args:
            if isinstance(v, str):
                ref_counts[v] = ref_counts.get(v, 0) + 1

    for instr in instrs:
        if instr.op != "call_function":
            continue
        # Skip already-fused ops
        if instr.resolved_fn is not None:
            continue
        lazy_fn = _LAZY_SHAPE_OPS.get(instr.target)
        if lazy_fn is None:
            continue
        # Only inline if the output is consumed exactly once
        if ref_counts.get(instr.out_name, 0) != 1:
            continue
        instr.resolved_fn = lazy_fn


def _eliminate_aliases(instrs: List["_Instr"]) -> List["_Instr"]:
    """Resolve alias chains at compile time and remove alias instructions.

    For every ``alias`` instruction ``a = alias(src)``, rewrites all
    references to ``a`` in downstream instructions to point directly at
    ``src`` (following chains recursively).  Then removes all alias
    instructions from the list.

    This eliminates per-alias dict-get + dict-set overhead at runtime,
    reducing the instruction count by up to 35%.
    """
    # Build alias map: out_name → ultimate source (follow chains)
    alias_map: Dict[str, Any] = {}
    for instr in instrs:
        if instr.op == "alias" and isinstance(instr.target, str):
            # Follow chains: if target is also an alias, resolve recursively
            src = instr.target
            while src in alias_map:
                src = alias_map[src]
            alias_map[instr.out_name] = src

    if not alias_map:
        return instrs

    def _rewrite(arg):
        """Replace aliased names with their ultimate source."""
        if isinstance(arg, str):
            return alias_map.get(arg, arg)
        if isinstance(arg, list):
            return [_rewrite(a) for a in arg]
        if isinstance(arg, tuple):
            return tuple(_rewrite(a) for a in arg)
        return arg

    # Rewrite all arg_names/kwarg_names/output_args to bypass aliases
    for instr in instrs:
        instr.arg_names = [_rewrite(a) for a in instr.arg_names]
        if instr.kwarg_names:
            instr.kwarg_names = {k: _rewrite(v)
                                  for k, v in instr.kwarg_names.items()}
        if instr.output_args:
            instr.output_args = [_rewrite(a) for a in instr.output_args]

    # Remove alias instructions (they are now dead code)
    return [i for i in instrs if i.op != "alias"]


# ---------------------------------------------------------------------------
# Instruction (one row in the execution table)
# ---------------------------------------------------------------------------

class _Instr:
    """One instruction in the flattened execution table.

    Fields
    ------
    out_name     : name of the output tensor/value stored in the env
    op           : string tag ('placeholder', 'call_function', etc.)
    target       : the operator callable (or a str for placeholders/get_attr)
    arg_names    : positional argument names/values
    kwarg_names  : keyword arguments
    is_output    : True if this is the final graph output
    output_args  : arg_names for output node (the value(s) to return)
    subgraph_instr: reserved for nested submodules
    """
    __slots__ = ("out_name", "op", "target", "arg_names",
                 "kwarg_names", "is_output", "output_args",
                 "subgraph_instr", "resolved_fn")

    def __init__(self, out_name, op, target, arg_names, kwarg_names,
                 is_output=False, output_args=None, subgraph_instr=None):
        self.out_name       = out_name
        self.op             = op
        self.target         = target
        self.arg_names      = arg_names
        self.kwarg_names    = kwarg_names
        self.is_output      = is_output
        self.output_args    = output_args or []
        self.subgraph_instr = subgraph_instr
        self.resolved_fn    = None


# ---------------------------------------------------------------------------
# FX-node serialisation helpers
# ---------------------------------------------------------------------------

def _node_arg_to_serialisable(arg, node_map):
    """
    Convert an FX node argument to either:
      - the node's output name (str)          if it is a graph node
      - a Python constant (int / float / …)   otherwise
    """
    if isinstance(arg, fx.Node):
        return arg.name
    if isinstance(arg, (list, tuple)):
        cls = type(arg)
        return cls(_node_arg_to_serialisable(a, node_map) for a in arg)
    return arg


def _build_subgraph_instrs(subgm) -> List[_Instr]:
    """Recursively compile a sub-GraphModule into _Instr list."""
    node_map = {n.name: n for n in subgm.graph.nodes}
    instrs: List[_Instr] = []
    for node in subgm.graph.nodes:
        if node.op == "placeholder":
            instrs.append(_Instr(node.name, "placeholder", node.name, [], {}))
        elif node.op == "output":
            flat_args = node.args[0]
            if isinstance(flat_args, (list, tuple)):
                out_args = [_node_arg_to_serialisable(a, node_map)
                            for a in flat_args]
            else:
                out_args = [_node_arg_to_serialisable(flat_args, node_map)]
            instrs.append(_Instr(node.name, "output", None, [], {},
                                 is_output=True, output_args=out_args))
        elif node.op == "call_function":
            args   = [_node_arg_to_serialisable(a, node_map)
                      for a in node.args]
            kwargs = {k: _node_arg_to_serialisable(v, node_map)
                      for k, v in node.kwargs.items()}
            instrs.append(_Instr(node.name, "call_function",
                                 node.target, args, kwargs))
        elif node.op == "get_attr":
            instrs.append(_Instr(node.name, "get_attr", node.target, [], {}))
    return instrs


# ---------------------------------------------------------------------------
# Extension loader
# ---------------------------------------------------------------------------

def _find_extension(hint: Optional[str] = None) -> str:
    """Locate llm_ops.so relative to this source file or the repo root."""
    if hint and os.path.exists(hint):
        return os.path.abspath(hint)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, '../../sqlite-llm/llm_ops.so'),
        os.path.join(here, '../../../sqlite-llm/llm_ops.so'),
        'sqlite-llm/llm_ops.so',
        'llm_ops.so',
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        'Cannot find llm_ops.so.  Build it with: cd sqlite-llm && make'
    )


def create_connection(
    extension_path: Optional[str] = None,
    db_path: str = ':memory:',
) -> sqlite3.Connection:
    """Create a SQLite connection with the llm_ops extension loaded.

    Applies performance PRAGMAs suitable for read-heavy inference:

    * ``journal_mode=MEMORY``      – journal in RAM, no disk I/O
    * ``synchronous=OFF``          – skip fsync (safe: DB is read-only at inference)
    * ``locking_mode=EXCLUSIVE``   – avoid per-statement lock acquisition
      (only for in-memory DBs; file DBs keep NORMAL to allow concurrent access)
    * ``temp_store=MEMORY``        – keep temp results in RAM
    * ``cache_size=-524288``       – 512 MiB page cache
    * ``mmap_size``                – memory-map the DB file (file DBs only)
    """
    ext_path = _find_extension(extension_path)
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    try:
        conn.load_extension(ext_path, entrypoint='sqlite3_llm_ops_init')
    except TypeError:
        conn.load_extension(ext_path)

    # ── Performance PRAGMAs for inference ──
    # journal_mode may fail if another connection holds the database lock
    # (e.g. tests with shared file DBs); ignore errors.
    try:
        conn.execute("PRAGMA journal_mode=MEMORY")
    except sqlite3.OperationalError:
        pass
    conn.execute("PRAGMA synchronous=OFF")
    # EXCLUSIVE locking avoids per-statement lock acquisition/release
    # overhead.  Safe for in-memory DBs (single-connection) and standalone
    # engine (dedicated file DB).  File DBs that may be shared by multiple
    # connections keep the default NORMAL mode.
    if db_path == ':memory:':
        conn.execute("PRAGMA locking_mode=EXCLUSIVE")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-524288")  # 512 MiB
    if db_path != ':memory:':
        db_size = os.path.getsize(db_path) if os.path.isfile(db_path) else 0
        mmap_sz = max(db_size * 2, 256 * 1024 * 1024)  # at least 256 MiB
        conn.execute(f"PRAGMA mmap_size={mmap_sz}")

    return conn


# ---------------------------------------------------------------------------
# Parameter persistence — DB table helpers
# ---------------------------------------------------------------------------

_CREATE_PARAM_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_PARAM_TABLE} (
    name       TEXT    PRIMARY KEY,
    data       BLOB    NOT NULL,
    ndim       INTEGER NOT NULL,
    shape      TEXT    NOT NULL,
    dtype      TEXT    NOT NULL DEFAULT 'float32',
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
)
"""


def save_params_to_db(
    conn: sqlite3.Connection,
    param_data: Dict[str, bytes],
) -> None:
    """Persist parameter BLOBs into the ``model_params`` table.

    Creates the table if it does not exist, then inserts or replaces
    each entry from *param_data*.  Metadata columns (ndim, shape) are
    derived from the BLOB header so they can be queried cheaply later.
    """
    import json, struct as _st

    conn.execute(_CREATE_PARAM_TABLE_SQL)
    _MAGIC = _st.unpack('<i', _st.pack('<I', 0x80000000))[0]

    rows: List[Tuple] = []
    for name, blob in param_data.items():
        first_int = _st.unpack_from('<i', blob, 0)[0]
        if first_int == _MAGIC:
            ndim = _st.unpack_from('<i', blob, 4)[0]
            shape = list(_st.unpack_from(f'<{ndim}i', blob, 8))
        else:
            # Legacy 1-D
            ndim = 1
            shape = [first_int]
        rows.append((name, blob, ndim, json.dumps(shape), 'float32'))

    conn.executemany(
        f"INSERT OR REPLACE INTO {_PARAM_TABLE}"
        f" (name, data, ndim, shape, dtype) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def load_param_names_from_db(conn: sqlite3.Connection) -> List[str]:
    """Return the list of parameter names stored in ``model_params``."""
    try:
        rows = conn.execute(
            f"SELECT name FROM {_PARAM_TABLE} ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]
    except sqlite3.OperationalError:
        return []


def preload_params_from_db(conn: sqlite3.Connection) -> Dict[str, bytes]:
    """Eagerly load **all** parameter BLOBs from ``model_params`` into memory.

    This eliminates per-op sub-SELECT overhead during inference.  Instead of
    each SQL call containing ``(SELECT data FROM model_params WHERE name=?)``,
    the BLOB is passed directly as a bind-parameter — saving ~10 µs per
    parameter reference.

    For a typical Qwen2 model this converts ~40 sub-SELECTs per forward pass
    into direct BLOB binds, saving ~0.4 ms per step (20–30 % of the tiny
    model's forward time).  The memory cost is negligible because the BLOBs
    are already resident in SQLite's page cache.
    """
    try:
        rows = conn.execute(
            f"SELECT name, data FROM {_PARAM_TABLE}"
        ).fetchall()
        return {name: data for name, data in rows}
    except sqlite3.OperationalError:
        return {}


# ---------------------------------------------------------------------------
# Main graph executor
# ---------------------------------------------------------------------------

class SqliteGraphExecutor:
    """
    Executes an ``ExportedProgram`` using SQLite with the llm_ops extension.

    Usage::

        executor = SqliteGraphExecutor()
        executor.compile(exported_program)
        logits_tuple = executor.run(input_ids_np)

    When *db_path* points to a file (anything other than ``':memory:'``),
    model parameters are persisted to a ``model_params`` table inside that
    database.  During ``run()`` the executor passes parameter **names**
    (strings) instead of raw BLOBs; the helper ``_sql1`` in sqlite_ops
    transparently rewrites the SQL so that each parameter is fetched via
    an inline sub-SELECT — keeping parameter data inside SQLite and out
    of Python memory.
    """

    def __init__(self,
                 extension_path: Optional[str] = None,
                 db_path: str = ':memory:') -> None:
        self._instrs:     List[_Instr]          = []
        self._param_data: Dict[str, Any]         = {}   # str param names (file mode) or bytes BLOBs (memory mode)
        self._submodules: Dict[str, Any]         = {}
        self._ep:         Optional[Any]          = None
        self._db_path:    str                    = db_path
        self._conn:       sqlite3.Connection     = create_connection(
            extension_path, db_path)

    @property
    def db_path(self) -> str:
        """Path of the underlying SQLite database (``':memory:'`` or a file)."""
        return self._db_path

    @property
    def connection(self) -> sqlite3.Connection:
        """The underlying SQLite connection."""
        return self._conn

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile(self, ep: torch.export.ExportedProgram) -> None:
        """
        Walk the exported graph and build the execution table.

        Parameters are extracted from the exported program's state_dict /
        buffers.  When the executor uses a file-based database the BLOBs
        are persisted to the ``model_params`` table **and** eagerly loaded
        back into Python memory so that inference uses direct BLOB binds
        (avoiding per-op sub-SELECT overhead).
        """
        self._ep = ep
        gm = ep.graph_module

        # ── collect model parameter / buffer arrays via graph_signature ──
        all_tensors: Dict[str, torch.Tensor] = {}
        all_tensors.update(ep.state_dict)
        all_tensors.update({k: v for k, v in ep.named_buffers()})

        # Collect parameter node names for the graph signature.
        param_node_names: List[str] = []
        for spec in ep.graph_signature.input_specs:
            if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER):
                param_node_names.append(spec.arg.name)

        if self._db_path != ':memory:':
            # File DB mode — persist params then preload into memory.
            existing = set(load_param_names_from_db(self._conn))
            needed = set(param_node_names)
            if not (needed and needed.issubset(existing)):
                # First time or schema mismatch: serialize and write.
                raw_blobs: Dict[str, bytes] = {}
                for spec in ep.graph_signature.input_specs:
                    if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER):
                        node_name = spec.arg.name
                        tensor = all_tensors.get(spec.target)
                        if tensor is not None:
                            raw_blobs[node_name] = tensor_to_blob(
                                tensor.detach().float().numpy()
                            )
                save_params_to_db(self._conn, raw_blobs)
            # Eagerly load all BLOBs into Python memory — eliminates the
            # per-op ``(SELECT data FROM model_params WHERE name=?)``
            # sub-SELECT overhead (~10 µs per reference).
            self._param_data = preload_params_from_db(self._conn)
        else:
            # In-memory mode — keep BLOBs in Python dict (original path).
            raw_blobs = {}
            for spec in ep.graph_signature.input_specs:
                if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER):
                    node_name = spec.arg.name
                    tensor = all_tensors.get(spec.target)
                    if tensor is not None:
                        raw_blobs[node_name] = tensor_to_blob(
                            tensor.detach().float().numpy()
                        )
            self._param_data = raw_blobs

        # ── collect sub-modules ──────────────────────────────────────────
        for attr_name, submod in gm.named_children():
            self._submodules[attr_name] = submod

        # ── build instruction list ───────────────────────────────────────
        node_map = {n.name: n for n in ep.graph.nodes}
        instrs: List[_Instr] = []

        for node in ep.graph.nodes:
            if node.op == "placeholder":
                instrs.append(_Instr(node.name, "placeholder",
                                     node.name, [], {}))

            elif node.op == "get_attr":
                instrs.append(_Instr(node.name, "get_attr",
                                     node.target, [], {}))

            elif node.op == "call_function":
                args   = [_node_arg_to_serialisable(a, node_map)
                          for a in node.args]
                kwargs = {k: _node_arg_to_serialisable(v, node_map)
                          for k, v in node.kwargs.items()}

                from torch._higher_order_ops.wrap import wrap_with_set_grad_enabled
                if node.target is wrap_with_set_grad_enabled:
                    instrs.append(_Instr(node.name, "subgraph_call",
                                         node.target, args, kwargs))
                else:
                    instrs.append(_Instr(node.name, "call_function",
                                         node.target, args, kwargs))

            elif node.op == "output":
                flat = node.args[0]
                if isinstance(flat, (list, tuple)):
                    out_args = [_node_arg_to_serialisable(a, node_map)
                                for a in flat]
                else:
                    out_args = [_node_arg_to_serialisable(flat, node_map)]
                instrs.append(_Instr(node.name, "output", None, [], {},
                                     is_output=True, output_args=out_args))

        # ── Compile-time optimisation passes ──
        optimised: List[_Instr] = []
        for instr in instrs:
            if (instr.op == "call_function"
                    and instr.target in _NOOP_TARGETS):
                src = instr.arg_names[0] if instr.arg_names else None
                optimised.append(_Instr(instr.out_name, "alias",
                                        src, [], {}))
            else:
                optimised.append(instr)

        # ── Fuse silu → mul into llm_silu_mul_nd (1 round-trip per MLP) ──
        _fuse_silu_mul_instrs(optimised)

        # ── Fuse 7-op RoPE pattern into llm_rope_nd ──
        _fuse_rope_instrs(optimised)

        # ── Fuse 6-op RMSNorm pattern into llm_rmsnorm_nd ──
        _fuse_rmsnorm_instrs(optimised)

        # ── Inline single-use shape ops as SQL expressions ──
        _fuse_inline_shape_ops(optimised)

        # ── Eliminate alias instructions by rewiring references ──
        optimised = _eliminate_aliases(optimised)

        for instr in optimised:
            if instr.op == "call_function" and instr.resolved_fn is None:
                instr.resolved_fn = SqliteOpRegistry.get(instr.target)

        self._instrs = optimised

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, input_ids: np.ndarray,
            verbose: bool = False,
            profile: bool = False) -> Tuple:
        """
        Execute the compiled graph with *input_ids*.

        Returns a tuple mirroring the model's flat output, e.g. ``(logits,)``
        when use_cache=False.  Outputs are numpy arrays.

        When *profile* is ``True``, per-op timing data is stored in
        ``self.last_profile`` (a dict mapping op category to total µs).
        """
        self._preflight_check()

        env: Dict[str, Any] = {}
        env.update(self._param_data)

        for attr_name, submod in self._submodules.items():
            env[attr_name] = submod

        # Local variable bindings for hot-path speed
        conn = self._conn
        _resolve = self._resolve

        output_value = None
        op_timings: Optional[Dict[str, float]] = {} if profile else None
        sql_call_count = 0

        for instr in self._instrs:
            op = instr.op
            if op == "call_function":
                # ── Hot path: most instructions land here ──
                fn = instr.resolved_fn
                arg_names = instr.arg_names

                # Inline argument resolution for speed
                if instr.kwarg_names:
                    args = [_resolve(a, env) for a in arg_names]
                    kwargs = {k: _resolve(v, env)
                              for k, v in instr.kwarg_names.items()}
                else:
                    # Fast path: no kwargs (most ops)
                    # Inline _resolve for simple cases
                    args = []
                    for a in arg_names:
                        if isinstance(a, str):
                            args.append(env[a])
                        elif isinstance(a, list):
                            args.append(_resolve(a, env))
                        else:
                            args.append(a)
                    kwargs = {}

                if profile:
                    t0 = _time.perf_counter()

                result = fn(conn, *args, **kwargs)

                if profile:
                    elapsed_us = (_time.perf_counter() - t0) * 1e6
                    fn_name = getattr(fn, '__name__', '')
                    if fn_name.startswith('lazy_'):
                        pass
                    elif fn_name in ('fused_rmsnorm', 'fused_silu_mul', 'fused_rope'):
                        tgt = fn_name
                        op_timings[tgt] = op_timings.get(tgt, 0.0) + elapsed_us
                        sql_call_count += 1
                    else:
                        tgt = str(instr.target)
                        if 'aten.' in tgt:
                            tgt = tgt.split('aten.')[1].split('.')[0]
                        elif hasattr(instr.target, '__name__'):
                            tgt = instr.target.__name__
                        op_timings[tgt] = op_timings.get(tgt, 0.0) + elapsed_us
                        sql_call_count += 1

                env[instr.out_name] = result
                if verbose:
                    _log_value(instr.out_name, result)

            elif op == "placeholder":
                if instr.target not in env:
                    ids = np.asarray(input_ids, dtype=np.int64)
                    env[instr.out_name] = tensor_to_blob(
                        ids.astype(np.float32))

            elif op == "output":
                vals = []
                for a in instr.output_args:
                    v = _resolve(a, env)
                    if isinstance(v, bytes):
                        v = blob_to_numpy(v)
                    vals.append(v)
                output_value = tuple(vals)

            elif op == "subgraph_call":
                result = self._exec_subgraph_call(instr, env)
                env[instr.out_name] = result
                if verbose:
                    _log_value(instr.out_name, result)

            elif op == "get_attr":
                env[instr.out_name] = self._submodules.get(instr.target)

        if output_value is None:
            raise RuntimeError("Graph had no output node")

        if profile:
            self.last_profile = {
                "op_timings_us": op_timings,
                "sql_call_count": sql_call_count,
            }

        return output_value

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _preflight_check(self) -> None:
        """
        Scan every compiled instruction for ``call_function`` ops that have
        no sqlite implementation.  Raises a single comprehensive
        ``NotImplementedError`` listing *all* missing ops at once.
        """
        missing: Dict[str, str] = {}

        def _scan(instrs: List[_Instr], label: str) -> None:
            for instr in instrs:
                if instr.op == "call_function":
                    key = repr(instr.target)
                    if key not in missing and SqliteOpRegistry.get(instr.target) is None:
                        missing[key] = f"{label}:{instr.out_name}"

        _scan(self._instrs, "graph")

        for attr_name, submod in self._submodules.items():
            try:
                sub_instrs = _build_subgraph_instrs(submod)
                _scan(sub_instrs, f"subgraph({attr_name})")
            except Exception as exc:
                print(f"[preflight] warning: could not scan subgraph "
                      f"{attr_name!r}: {exc}", file=sys.stderr)

        if missing:
            lines = [
                f"  {tgt}  [first node: {node}]"
                for tgt, node in sorted(missing.items())
            ]
            raise NotImplementedError(
                f"{len(missing)} op(s) have no sqlite implementation:\n"
                + "\n".join(lines)
            )

    def _resolve(self, arg, env: Dict[str, Any]) -> Any:
        """Resolve an argument: look up node-name strings in env,
        recurse into lists/tuples, return constants as-is."""
        if isinstance(arg, str):
            val = env.get(arg, _SENTINEL)
            if val is _SENTINEL:
                raise KeyError(f"Tensor {arg!r} not found in execution env")
            return val
        if isinstance(arg, list):
            return [self._resolve(a, env) for a in arg]
        if isinstance(arg, tuple):
            return tuple(self._resolve(a, env) for a in arg)
        return arg

    def _exec_subgraph_call(self, instr: _Instr,
                             env: Dict[str, Any]) -> Any:
        """
        Execute wrap_with_set_grad_enabled by running the inner
        sub-GraphModule through a recursive ``_SubgraphExecutor``.
        """
        resolved_args = [self._resolve(a, env) for a in instr.arg_names]
        submod    = resolved_args[1]
        subinputs = resolved_args[2:]
        sub_exec  = _SubgraphExecutor(submod, self._conn)
        return sub_exec.run(*subinputs)


# ---------------------------------------------------------------------------
# Subgraph executor (for inline sub-GraphModules, e.g. RoPE wrapping)
# ---------------------------------------------------------------------------

class _SubgraphExecutor:
    """Lightweight executor for inline sub-GraphModules."""

    def __init__(self, subgm, conn: sqlite3.Connection) -> None:
        self._subgm  = subgm
        self._instrs = _build_subgraph_instrs(subgm)
        self._conn   = conn

    def run(self, *inputs) -> Any:
        env: Dict[str, Any] = {}
        ph_nodes = [i for i in self._instrs if i.op == "placeholder"]
        for ph, val in zip(ph_nodes, inputs):
            if isinstance(val, bytes):
                env[ph.out_name] = val
            elif isinstance(val, np.ndarray):
                env[ph.out_name] = tensor_to_blob(val.astype(np.float32))
            elif isinstance(val, torch.Tensor):
                env[ph.out_name] = tensor_to_blob(
                    val.detach().float().numpy())
            else:
                env[ph.out_name] = val

        output_value = None
        for instr in self._instrs:
            if instr.op == "placeholder":
                continue
            if instr.is_output:
                vals = [_sub_resolve(a, env) for a in instr.output_args]
                output_value = tuple(vals) if len(vals) > 1 else vals[0]
                continue

            args   = [_sub_resolve(a, env) for a in instr.arg_names]
            kwargs = {k: _sub_resolve(v, env)
                      for k, v in instr.kwarg_names.items()}

            fn = SqliteOpRegistry.get(instr.target)
            if fn is None:
                raise NotImplementedError(
                    f"Subgraph: no sqlite impl for {instr.target!r}")
            env[instr.out_name] = fn(self._conn, *args, **kwargs)

        return output_value


def _sub_resolve(arg, env):
    if isinstance(arg, str):
        return env[arg]
    if isinstance(arg, list):
        return [_sub_resolve(a, env) for a in arg]
    if isinstance(arg, tuple):
        return tuple(_sub_resolve(a, env) for a in arg)
    return arg


def _log_value(name: str, val: Any) -> None:
    if isinstance(val, bytes):
        arr = blob_to_numpy(val)
        print(f"  {name:40s} shape={str(arr.shape):20s} dtype={arr.dtype}")
    elif isinstance(val, np.ndarray):
        print(f"  {name:40s} shape={str(val.shape):20s} dtype={val.dtype}")
    elif isinstance(val, tuple):
        shapes = []
        for v in val:
            if isinstance(v, bytes):
                shapes.append(blob_to_numpy(v).shape)
            elif isinstance(v, np.ndarray):
                shapes.append(v.shape)
            else:
                shapes.append(v)
        print(f"  {name:40s} tuple={shapes}")
    else:
        print(f"  {name:40s} value={val}")
