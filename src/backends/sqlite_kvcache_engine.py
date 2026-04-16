"""
SQLite KV Cache graph executor — runs an ``ExportedProgram`` with KV Cache
optimization for efficient autoregressive decoding.

Key Differences from SqliteGraphExecutor
-----------------------------------------
1. **Dual Computation Graphs**: Maintains separate graphs for prefill and
   decoding stages:
   - Prefill: Processes the full prompt, generates first logits + initial KV cache
   - Decoding: Processes one token at a time, reuses cached K/V states

2. **KV Cache Management**: Stores attention key/value states in SQLite BLOBs
   between forward passes, avoiding O(n²) recomputation in attention layers.

3. **Automatic Stage Selection**: Automatically switches between prefill and
   decoding based on whether KV cache exists.

Workflow
--------
1. ``compile(ep_prefill, ep_decode)`` — compiles two exported programs:
   - ep_prefill: Model exported with use_cache=True for initial prompt
   - ep_decode: Model exported with use_cache=True, past_key_values provided

2. ``run(input_ids)`` — automatically chooses prefill or decode based on state:
   - First call: runs prefill graph, stores KV cache
   - Subsequent calls: runs decode graph with cached K/V states

All tensor data flows as BLOB bytes through the SQLite connection.
"""
import sqlite3
import time as _time
from typing import Any, Dict, List, Optional, Tuple

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
    _blob_shape_list,
    _PARAM_TABLE,
)
from src.backends.sqlite_engine import (
    _Instr,
    _node_arg_to_serialisable,
    _build_subgraph_instrs,
    _fuse_silu_mul_instrs,
    _fuse_rope_instrs,
    _fuse_rmsnorm_instrs,
    _fuse_inline_shape_ops,
    _eliminate_aliases,
    _SENTINEL,
    create_connection,
    save_params_to_db,
    load_param_names_from_db,
    preload_params_from_db,
)


# ---------------------------------------------------------------------------
# Compile-time optimisation passes
# ---------------------------------------------------------------------------

# Ops that return their first positional argument unchanged (no-ops).
# These are eliminated at compile time to avoid registry lookup + function
# call overhead during execution.
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


def _eliminate_noop_instrs(instrs: List[_Instr]) -> List[_Instr]:
    """Replace no-op call_function instructions with lightweight aliases.

    A no-op instruction simply copies its first argument to its output name.
    Instead of going through the registry + function call at runtime, we
    rewrite it as an ``alias`` instruction that the executor handles with a
    single dict assignment.
    """
    out: List[_Instr] = []
    for instr in instrs:
        if instr.op == "call_function" and instr.target in _NOOP_TARGETS:
            # The first positional arg is the tensor being passed through.
            src_name = instr.arg_names[0] if instr.arg_names else None
            out.append(_Instr(instr.out_name, "alias", src_name, [], {}))
        else:
            out.append(instr)
    return out


def _preresolve_op_functions(instrs: List[_Instr]) -> None:
    """Pre-resolve ``SqliteOpRegistry`` lookups and store them on each instr.

    After this pass every ``call_function`` instruction has a ``resolved_fn``
    attribute pointing directly at the handler callable, eliminating a dict
    lookup per instruction during execution.  Instructions that already have
    a resolved_fn (e.g. from a fusion pass) are left unchanged.
    """
    for instr in instrs:
        if instr.op == "call_function":
            if instr.resolved_fn is None:
                instr.resolved_fn = SqliteOpRegistry.get(instr.target)
        else:
            instr.resolved_fn = None


# ---------------------------------------------------------------------------
# JSON-serialisation helpers for instruction export
# ---------------------------------------------------------------------------


def _serialise_args(obj):
    """Recursively convert an argument tree to JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _serialise_args(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise_args(a) for a in obj]
    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Forward-SQL export — CTE-based complete SQL generation
# ---------------------------------------------------------------------------

import operator as _operator
import re as _re


def _sanitize_cte_name(name: str) -> str:
    """Convert an instruction name to a valid SQL CTE identifier.

    Prefixes with ``n_`` to avoid collisions with SQL reserved words
    like ``add``, ``select``, ``index``, etc.
    """
    cleaned = _re.sub(r'[^A-Za-z0-9_]', '_', name)
    return f"n_{cleaned}"


def _arg_to_sql_expr(arg, cte_set: set, param_set: set) -> str:
    """Convert a single argument to its SQL expression form.

    * CTE references (including params) → ``(SELECT v FROM <cte_name>)``
    * Numeric literals → their string representation
    * None → ``NULL``
    """
    if isinstance(arg, str):
        if arg in cte_set:
            return f"(SELECT v FROM {_sanitize_cte_name(arg)})"
        if arg in param_set:
            return f"llm_get_param('{arg}')"
        # Fall back to CTE lookup
        return f"(SELECT v FROM {_sanitize_cte_name(arg)})"
    if isinstance(arg, bool):
        return str(int(arg))
    if isinstance(arg, (int, float)):
        return str(arg)
    if arg is None:
        return "NULL"
    return str(arg)


def _list_arg_to_sql_parts(lst, cte_set: set, param_set: set) -> List[str]:
    """Convert a list argument to a list of SQL expression parts."""
    return [_arg_to_sql_expr(a, cte_set, param_set) for a in lst]


# Maps each op target to a function (arg_sqls, kwarg_sqls) → SQL SELECT body.
# arg_sqls: list of SQL expressions for positional args
# kwarg_sqls: dict of SQL expressions for keyword args
# Returns: SQL expression string (without SELECT keyword)

def _is_literal_scalar(sql_expr: str) -> bool:
    """Check whether *sql_expr* is a bare numeric literal (not a CTE ref)."""
    try:
        float(sql_expr)
        return True
    except (ValueError, TypeError):
        return False


def _ensure_blob(sql_expr: str) -> str:
    """Wrap a literal scalar SQL expression in ``llm_full_nd`` if needed.

    FX graphs may use ``aten.add.Tensor`` with a scalar second operand.
    The C extension's ``llm_add_nd`` (and friends) require both arguments
    to be N-D BLOBs, so bare literals like ``0`` or ``1e-06`` must be
    wrapped in ``llm_full_nd(scalar, 1)`` to produce a single-element BLOB.
    """
    if _is_literal_scalar(sql_expr):
        return f"llm_full_nd({sql_expr}, 1)"
    return sql_expr


def _extract_llm_get_param_name(sql_expr: str):
    prefix = "llm_get_param('"
    suffix = "')"
    if sql_expr.startswith(prefix) and sql_expr.endswith(suffix):
        return sql_expr[len(prefix):-len(suffix)]
    return None


def _build_sql_fragment_map():
    """Build the op-target → SQL-fragment-generator mapping lazily."""
    m = {}

    # ── Arithmetic ──
    def _frag_add(args, kw):
        a, b = args[0], args[1]
        alpha = kw.get('alpha', 1)
        if _is_literal_scalar(b):
            b = _ensure_blob(b)
        if alpha != 1:
            return f"llm_add_nd({a}, llm_scale_nd({b}, {alpha}))"
        return f"llm_add_nd({a}, {b})"
    m[torch.ops.aten.add.Tensor] = _frag_add

    def _frag_mul(args, kw):
        a, b = args[0], args[1]
        if _is_literal_scalar(b):
            return f"llm_scale_nd({a}, {b})"
        if _is_literal_scalar(a):
            return f"llm_scale_nd({b}, {a})"
        return f"llm_mul_nd({a}, {b})"
    m[torch.ops.aten.mul.Tensor] = _frag_mul

    def _frag_sub(args, kw):
        a, b = args[0], args[1]
        if _is_literal_scalar(b):
            b = _ensure_blob(b)
        return f"llm_sub_nd({a}, {b})"
    m[torch.ops.aten.sub.Tensor] = _frag_sub

    def _frag_div(args, kw):
        a, b = args[0], args[1]
        if _is_literal_scalar(b):
            return f"llm_scale_nd({a}, 1.0 / {b})"
        return f"llm_div_nd({a}, {b})"
    m[torch.ops.aten.div.Tensor] = _frag_div
    m[torch.ops.aten.neg.default] = lambda a, k: f"llm_neg_nd({a[0]})"
    m[torch.ops.aten.abs.default] = lambda a, k: f"llm_abs_nd({a[0]})"

    def _frag_pow_ts(args, kw):
        return f"llm_pow_nd({args[0]}, {args[1]})"
    m[torch.ops.aten.pow.Tensor_Scalar] = _frag_pow_ts
    m[torch.ops.aten.pow.Scalar] = lambda a, k: f"llm_pow_scalar_nd({a[0]}, {a[1]})"
    m[torch.ops.aten.sqrt.default] = lambda a, k: f"llm_sqrt_nd({a[0]})"
    m[torch.ops.aten.rsqrt.default] = lambda a, k: f"llm_rsqrt_nd({a[0]})"
    m[torch.ops.aten.exp.default] = lambda a, k: f"llm_exp_nd({a[0]})"
    m[torch.ops.aten.log.default] = lambda a, k: f"llm_log_nd({a[0]})"

    # ── Trig ──
    m[torch.ops.aten.sin.default] = lambda a, k: f"llm_sin_nd({a[0]})"
    m[torch.ops.aten.cos.default] = lambda a, k: f"llm_cos_nd({a[0]})"
    m[torch.ops.aten.tanh.default] = lambda a, k: f"llm_tanh_nd({a[0]})"

    # ── Comparison ──
    m[torch.ops.aten.gt.Tensor] = lambda a, k: f"llm_gt_nd({a[0]}, {a[1]})"
    m[torch.ops.aten.gt.Scalar] = lambda a, k: f"llm_gt_nd({a[0]}, {a[1]})"
    m[torch.ops.aten.lt.Tensor] = lambda a, k: f"llm_lt_nd({a[0]}, {a[1]})"
    m[torch.ops.aten.ge.Tensor] = lambda a, k: f"llm_ge_nd({a[0]}, {a[1]})"
    m[torch.ops.aten.le.Tensor] = lambda a, k: f"llm_le_nd({a[0]}, {a[1]})"
    m[torch.ops.aten.eq.Tensor] = lambda a, k: f"llm_eq_nd({a[0]}, {a[1]})"
    m[torch.ops.aten.ne.Tensor] = lambda a, k: f"llm_ne_nd({a[0]}, {a[1]})"

    # ── Reduction ──
    def _frag_mean_dim(args, kw):
        dim_arg = args[1]
        # dim might be a list like [-1]; extract first element
        if isinstance(dim_arg, list):
            dim_arg = dim_arg[0]
        keepdim = args[2] if len(args) > 2 else 0
        return f"llm_mean_nd({args[0]}, {dim_arg}, {keepdim})"
    m[torch.ops.aten.mean.dim] = _frag_mean_dim
    m[torch.ops.aten.mean.default] = lambda a, k: f"llm_reduce_mean_nd({a[0]})"

    def _frag_sum_dim(args, kw):
        dim_arg = args[1]
        if isinstance(dim_arg, list):
            dim_arg = dim_arg[0]
        keepdim = args[2] if len(args) > 2 else 0
        return f"llm_sum_nd({args[0]}, {dim_arg}, {keepdim})"
    m[torch.ops.aten.sum.dim_IntList] = _frag_sum_dim
    m[torch.ops.aten.sum.default] = lambda a, k: f"llm_reduce_sum_nd({a[0]})"

    # ── Linear algebra ──
    m[torch.ops.aten.mm.default] = lambda a, k: f"llm_bmm({a[0]}, {a[1]})"
    m[torch.ops.aten.bmm.default] = lambda a, k: f"llm_bmm({a[0]}, {a[1]})"
    m[torch.ops.aten.matmul.default] = lambda a, k: f"llm_bmm({a[0]}, {a[1]})"

    def _frag_linear(args, kw):
        weight_name = _extract_llm_get_param_name(args[1])
        if weight_name is not None:
            if len(args) >= 3 and args[2] != "NULL":
                return f"llm_linear_param_bias_nd({args[0]}, '{weight_name}', {args[2]})"
            return f"llm_linear_param_nd({args[0]}, '{weight_name}')"
        if len(args) >= 3 and args[2] != "NULL":
            return f"llm_add_nd(llm_linear_nd({args[0]}, {args[1]}), {args[2]})"
        return f"llm_linear_nd({args[0]}, {args[1]})"
    m[torch.ops.aten.linear.default] = _frag_linear

    def _frag_addmm(args, kw):
        return f"llm_add_nd(llm_bmm({args[1]}, {args[2]}), {args[0]})"
    m[torch.ops.aten.addmm.default] = _frag_addmm

    # ── Shape manipulation ──
    def _frag_view(args, kw):
        shape = args[1] if isinstance(args[1], list) else [args[1]]
        parts = ", ".join([args[0]] + [str(s) for s in shape])
        return f"llm_view_nd({parts})"
    m[torch.ops.aten.view.default] = _frag_view
    m[torch.ops.aten._unsafe_view.default] = _frag_view
    m[torch.ops.aten.reshape.default] = _frag_view

    m[torch.ops.aten.unsqueeze.default] = lambda a, k: f"llm_unsqueeze_nd({a[0]}, {a[1]})"
    m[torch.ops.aten.squeeze.dim] = lambda a, k: f"llm_squeeze_nd({a[0]}, {a[1]})"
    m[torch.ops.aten.squeeze.default] = lambda a, k: f"llm_squeeze_all_nd({a[0]})"
    m[torch.ops.aten.transpose.int] = lambda a, k: f"llm_transpose_nd({a[0]}, {a[1]}, {a[2]})"

    def _frag_expand(args, kw):
        size = args[1] if isinstance(args[1], list) else [args[1]]
        parts = ", ".join([args[0]] + [str(s) for s in size])
        return f"llm_expand_nd({parts})"
    m[torch.ops.aten.expand.default] = _frag_expand

    def _frag_permute(args, kw):
        dims = args[1] if isinstance(args[1], list) else [args[1]]
        parts = ", ".join([args[0]] + [str(d) for d in dims])
        return f"llm_permute_nd({parts})"
    m[torch.ops.aten.permute.default] = _frag_permute

    def _frag_slice(args, kw):
        dim = args[1] if len(args) > 1 else 0
        start = args[2] if len(args) > 2 else 0
        end = args[3] if len(args) > 3 else 2147483647
        step = args[4] if len(args) > 4 else 1
        # Clamp end to INT32_MAX to avoid overflow in the C extension
        # (FX graphs may use sys.maxsize = 2**63-1 for "to the end")
        try:
            end_val = int(end)
            if end_val > 2147483647:
                end = 2147483647
        except (ValueError, TypeError):
            pass
        return f"llm_slice_nd({args[0]}, {dim}, {start}, {end}, {step})"
    m[torch.ops.aten.slice.Tensor] = _frag_slice

    def _frag_select(args, kw):
        return (f"llm_squeeze_nd(llm_slice_nd("
                f"{args[0]}, {args[1]}, {args[2]}, {args[2]} + 1), {args[1]})")
    m[torch.ops.aten.select.int] = _frag_select

    def _frag_cat(args, kw):
        tensors = args[0] if isinstance(args[0], list) else [args[0]]
        dim = args[1] if len(args) > 1 else 0
        if len(tensors) == 2:
            a, b = tensors
            # Handle empty-tensor cat: if one operand has a 0-sized dim
            # (e.g. an empty initial KV cache), skip it and return the other.
            return (
                f"CASE WHEN llm_shape({a}, 0) = 0 THEN {b}"
                f" WHEN llm_shape({b}, 0) = 0 THEN {a}"
                f" ELSE llm_cat_multi_nd({dim}, {a}, {b}) END"
            )
        parts = ", ".join([str(dim)] + tensors)
        return f"llm_cat_multi_nd({parts})"
    m[torch.ops.aten.cat.default] = _frag_cat

    def _frag_stack(args, kw):
        tensors = args[0] if isinstance(args[0], list) else [args[0]]
        dim = args[1] if len(args) > 1 else 0
        parts = ", ".join([str(dim)] + tensors)
        return f"llm_stack_nd({parts})"
    m[torch.ops.aten.stack.default] = _frag_stack

    m[torch.ops.aten.flatten.using_ints] = (
        lambda a, k: f"llm_flatten_nd({a[0]}, {a[1]}, {a[2]})")

    # ── Indexing / Embedding ──
    def _frag_embedding(args, kw):
        weight_name = _extract_llm_get_param_name(args[0])
        if weight_name is not None:
            return f"llm_embedding_param_nd('{weight_name}', {args[1]})"
        return f"llm_embedding_nd({args[0]}, {args[1]})"
    m[torch.ops.aten.embedding.default] = _frag_embedding
    m[torch.ops.aten.index_select.default] = (
        lambda a, k: f"llm_index_select_nd({a[0]}, {a[1]}, {a[2]})")
    m[torch.ops.aten.gather.default] = (
        lambda a, k: f"llm_gather_nd({a[0]}, {a[1]}, {a[2]})")

    # ── Masking / Triangular ──
    m[torch.ops.aten.triu.default] = lambda a, k: f"llm_triu_nd({a[0]}, {a[1]})"
    m[torch.ops.aten.tril.default] = lambda a, k: f"llm_tril_nd({a[0]}, {a[1]})"
    m[torch.ops.aten.masked_fill.Scalar] = (
        lambda a, k: f"llm_masked_fill_nd({a[0]}, {a[1]}, {a[2]})")

    # ── Activations ──
    m[torch.ops.aten.silu.default] = lambda a, k: f"llm_silu_nd({a[0]})"
    m[torch.ops.aten.relu.default] = lambda a, k: f"llm_relu_nd({a[0]})"
    m[torch.ops.aten.gelu.default] = lambda a, k: f"llm_gelu_nd({a[0]})"
    m[torch.ops.aten.sigmoid.default] = lambda a, k: f"llm_sigmoid_nd({a[0]})"
    m[torch.ops.aten.softmax.int] = lambda a, k: f"llm_softmax_nd({a[0]}, {a[1]})"
    m["llm_silu_mul_nd"] = lambda a, k: f"llm_silu_mul_nd({a[0]}, {a[1]})"
    m["llm_rope_nd"] = lambda a, k: f"llm_rope_nd({a[0]}, {a[1]}, {a[2]})"
    m["llm_rmsnorm_nd"] = lambda a, k: f"llm_rmsnorm_nd({a[0]}, {a[1]}, {a[2]})"

    # ── Tensor construction ──
    m[torch.ops.aten.arange.default] = lambda a, k: f"llm_arange_nd({a[0]})"
    m[torch.ops.aten.arange.start] = (
        lambda a, k: (
            f"llm_add_nd(llm_arange_nd(({a[1]} - {a[0]})), "
            f"llm_full_nd({a[0]}, ({a[1]} - {a[0]})))"
        )
    )

    def _frag_full(args, kw):
        size = args[0] if isinstance(args[0], list) else [args[0]]
        parts = ", ".join([str(args[1])] + [str(s) for s in size])
        return f"llm_full_nd({parts})"
    m[torch.ops.aten.full.default] = _frag_full

    # ── Conditional ──
    m[torch.ops.aten.where.self] = (
        lambda a, k: f"llm_where_nd({a[0]}, {a[1]}, {a[2]})")

    # ── Clamp ──
    m[torch.ops.aten.clamp.default] = (
        lambda a, k: f"llm_clamp_nd({a[0]}, {a[1]}, {a[2]})")

    # ── SDPA ──
    def _frag_sdpa(args, kw):
        scale = kw.get('scale', 'NULL')
        is_causal = kw.get('is_causal', False)
        attn_mask = args[3] if len(args) > 3 and args[3] != 'NULL' else None

        if not is_causal and attn_mask is None:
            return (f"llm_bmm(llm_softmax_nd(llm_scale_nd("
                    f"llm_bmm({args[0]}, llm_transpose_nd({args[1]}, -2, -1))"
                    f", {scale}), -1), {args[2]})")

        if is_causal and attn_mask is None:
            # Causal SDPA — single nested expression
            return (f"llm_bmm(llm_softmax_nd(llm_add_nd("
                    f"llm_scale_nd(llm_bmm({args[0]}, "
                    f"llm_transpose_nd({args[1]}, -2, -1)), {scale}), "
                    f"llm_masked_fill_nd(llm_full_nd(0.0, "
                    f"llm_shape({args[0]}, -2), llm_shape({args[1]}, -2)), "
                    f"llm_triu_nd(llm_full_nd(1.0, "
                    f"llm_shape({args[0]}, -2), llm_shape({args[1]}, -2)), 1), "
                    f"-1e9)), -1), {args[2]})")

        if attn_mask is not None and not is_causal:
            return (f"llm_bmm(llm_softmax_nd(llm_add_nd("
                    f"llm_scale_nd(llm_bmm({args[0]}, "
                    f"llm_transpose_nd({args[1]}, -2, -1)), {scale}), "
                    f"llm_bool_to_additive_mask_nd({attn_mask}, 1e9)), -1), "
                    f"{args[2]})")

        # Causal + mask
        return (f"llm_bmm(llm_softmax_nd(llm_add_nd(llm_add_nd("
                f"llm_scale_nd(llm_bmm({args[0]}, "
                f"llm_transpose_nd({args[1]}, -2, -1)), {scale}), "
                f"llm_masked_fill_nd(llm_full_nd(0.0, "
                f"llm_shape({args[0]}, -2), llm_shape({args[1]}, -2)), "
                f"llm_triu_nd(llm_full_nd(1.0, "
                f"llm_shape({args[0]}, -2), llm_shape({args[1]}, -2)), 1), "
                f"-1e9)), llm_bool_to_additive_mask_nd({attn_mask}, 1e9)"
                f"), -1), {args[2]})")
    m[torch.ops.aten.scaled_dot_product_attention.default] = _frag_sdpa

    # ── Scalar ops ──
    m[torch.ops.aten.sym_size.int] = lambda a, k: f"llm_shape({a[0]}, {a[1]})"
    m[_operator.add] = lambda a, k: f"({a[0]} + {a[1]})"
    m[_operator.mul] = lambda a, k: f"({a[0]} * {a[1]})"

    # ── Scalar arithmetic ──
    def _frag_add_scalar(args, kw):
        return f"llm_add_nd({args[0]}, llm_full_nd({args[1]}, 1))"
    m[torch.ops.aten.add.Scalar] = _frag_add_scalar

    def _frag_sub_scalar(args, kw):
        return f"llm_sub_nd({args[0]}, llm_full_nd({args[1]}, 1))"
    m[torch.ops.aten.sub.Scalar] = _frag_sub_scalar

    m[torch.ops.aten.mul.Scalar] = lambda a, k: f"llm_scale_nd({a[0]}, {a[1]})"

    def _frag_div_scalar(args, kw):
        return f"llm_scale_nd({args[0]}, 1.0 / {args[1]})"
    m[torch.ops.aten.div.Scalar] = _frag_div_scalar

    # ── Logical ──
    m[torch.ops.aten.logical_not.default] = lambda a, k: f"llm_logical_not_nd({a[0]})"
    m[torch.ops.aten.logical_and.default] = (
        lambda a, k: f"llm_logical_and_nd({a[0]}, {a[1]})")

    # ── Repeat ──
    def _frag_repeat(args, kw):
        repeats = args[1] if isinstance(args[1], list) else [args[1]]
        parts = ", ".join([args[0]] + [str(r) for r in repeats])
        return f"llm_repeat_nd({parts})"
    m[torch.ops.aten.repeat.default] = _frag_repeat

    return m


# Lazily initialised; populated on first call.
_SQL_FRAGMENT_MAP: Optional[Dict] = None


def _get_sql_fragment_map():
    """Return (and lazily build) the op→SQL fragment generator map."""
    global _SQL_FRAGMENT_MAP
    if _SQL_FRAGMENT_MAP is None:
        _SQL_FRAGMENT_MAP = _build_sql_fragment_map()
    return _SQL_FRAGMENT_MAP


def build_forward_sql(
    instrs: List[_Instr],
    param_names: set,
    num_layers: int,
) -> Tuple[str, List[str]]:
    """Generate a complete CTE-based SQL for a forward pass.

    Every placeholder (model parameters **and** runtime inputs like
    ``input_ids`` / ``past_key_values``) becomes a ``VALUES(?)`` CTE.
    Model parameters are supplied by the caller from the database so that
    the generated SQL contains **no** references to ``model_params``,
    avoiding SQLite's per-table reference limit.

    Parameters
    ----------
    instrs : list of _Instr
        The compiled instruction list (with no-ops already eliminated).
    param_names : set of str
        Names of model parameters.
    num_layers : int
        Number of transformer layers (for interpreting past_key_values).

    Returns
    -------
    sql : str
        A complete SQL string using ``WITH … SELECT …`` that computes the
        full forward pass in a single statement.
    bind_names : list of str
        Ordered names of **all** ``?`` bind-parameters in the SQL.  The
        first entries are model parameters (in instruction order), followed
        by runtime inputs (``input_ids``, ``pk0``, ``pv0``, …).
    """
    frag_map = _get_sql_fragment_map()

    # Track which names are CTE nodes
    cte_set: set = set()
    bind_names: List[str] = []   # ordered list of all ? params
    alias_map: Dict[str, str] = {}

    cte_clauses: List[str] = []
    output_cols: List[str] = []

    # All placeholders (params + runtime inputs) become VALUES(?) CTEs.
    for instr in instrs:
        if instr.op == "placeholder":
            cte_name = _sanitize_cte_name(instr.out_name)
            cte_clauses.append(f"{cte_name}(v) AS (VALUES(?))")
            cte_set.add(instr.out_name)
            bind_names.append(instr.out_name)

    for instr in instrs:
        cte_name = _sanitize_cte_name(instr.out_name)

        if instr.op == "placeholder":
            continue  # handled above

        if instr.op == "alias":
            src = instr.target
            while src in alias_map:
                src = alias_map[src]
            alias_map[instr.out_name] = src
            # Do NOT add to cte_set — the alias will be resolved to the
            # original CTE name at reference time.
            continue

        if instr.op == "get_attr":
            continue

        if instr.op == "output":
            for a in instr.output_args:
                ref = _resolve_alias(a, alias_map)
                output_cols.append(
                    _arg_to_sql_expr(ref, cte_set, param_names))
            continue

        if instr.op == "subgraph_call":
            continue

        if instr.op != "call_function":
            continue

        gen = frag_map.get(instr.target)
        if gen is None:
            continue

        arg_sqls = _resolve_arg_list(
            instr.arg_names, cte_set, param_names, alias_map)
        kw_sqls = {
            k: _resolve_kw_val(v, cte_set, param_names, alias_map)
            for k, v in instr.kwarg_names.items()
        }

        try:
            fragment = gen(arg_sqls, kw_sqls)
        except Exception:
            continue

        cte_clauses.append(f"{cte_name}(v) AS (SELECT {fragment})")
        cte_set.add(instr.out_name)

    # Build final SQL
    if not cte_clauses:
        return "SELECT NULL", []

    cte_body = ",\n  ".join(cte_clauses)
    out_exprs = ", ".join(output_cols) if output_cols else "NULL"
    sql = f"WITH\n  {cte_body}\nSELECT {out_exprs}"

    return sql, bind_names


def build_standalone_forward_sql(
    instrs: List[_Instr],
    param_names: set,
    num_layers: int,
) -> Tuple[str, List[str]]:
    """Generate standalone CTE-based SQL with inline model parameter lookups.

    Unlike :func:`build_forward_sql` which uses ``VALUES(?)`` for **all**
    placeholders (including model parameters), this variant fetches model
    parameters directly from the ``model_params`` table via inline
    sub-SELECTs.  Only *runtime* inputs (``input_ids`` and, for the decode
    graph, past key/value cache blobs) remain as ``?`` bind-parameters.

    This makes the generated SQL self-contained w.r.t. model weights — the
    caller only needs to supply the small set of runtime-varying blobs,
    enabling inference without torch or transformers.

    Parameters
    ----------
    instrs : list of _Instr
        The compiled instruction list (with no-ops already eliminated).
    param_names : set of str
        Names of model parameters stored in the ``model_params`` table.
    num_layers : int
        Number of transformer layers.

    Returns
    -------
    sql : str
        A complete ``WITH … SELECT …`` statement.
    bind_names : list of str
        Ordered names of the ``?`` bind-parameters (**runtime inputs only**).
    """
    frag_map = _get_sql_fragment_map()

    cte_set: set = set()
    bind_names: List[str] = []
    alias_map: Dict[str, str] = {}

    cte_clauses: List[str] = []
    output_cols: List[str] = []

    # Placeholders: params → inline llm_get_param(name);
    #               runtime inputs → VALUES(?)
    for instr in instrs:
        if instr.op == "placeholder":
            cte_name = _sanitize_cte_name(instr.out_name)
            if instr.out_name in param_names:
                continue
            else:
                cte_clauses.append(
                    f"{cte_name}(v) AS (SELECT v FROM (SELECT ? AS v LIMIT 1))"
                )
                bind_names.append(instr.out_name)
                cte_set.add(instr.out_name)

    for instr in instrs:
        cte_name = _sanitize_cte_name(instr.out_name)

        if instr.op == "placeholder":
            continue

        if instr.op == "alias":
            src = instr.target
            while src in alias_map:
                src = alias_map[src]
            alias_map[instr.out_name] = src
            continue

        if instr.op == "get_attr":
            continue

        if instr.op == "output":
            for a in instr.output_args:
                ref = _resolve_alias(a, alias_map)
                output_cols.append(
                    _arg_to_sql_expr(ref, cte_set, param_names))
            continue

        if instr.op == "subgraph_call":
            continue

        if instr.op != "call_function":
            continue

        gen = frag_map.get(instr.target)
        if gen is None:
            continue

        arg_sqls = _resolve_arg_list(
            instr.arg_names, cte_set, param_names, alias_map)
        kw_sqls = {
            k: _resolve_kw_val(v, cte_set, param_names, alias_map)
            for k, v in instr.kwarg_names.items()
        }

        try:
            fragment = gen(arg_sqls, kw_sqls)
        except Exception:
            continue

        cte_clauses.append(
            f"{cte_name}(v) AS (SELECT v FROM (SELECT {fragment} AS v LIMIT 1))"
        )
        cte_set.add(instr.out_name)

    if not cte_clauses:
        return "SELECT NULL", []

    cte_body = ",\n  ".join(cte_clauses)
    out_exprs = ", ".join(output_cols) if output_cols else "NULL"
    sql = f"WITH\n  {cte_body}\nSELECT {out_exprs}"

    return sql, bind_names


def _resolve_alias(name, alias_map: Dict[str, str]) -> str:
    """Follow alias chain to find the canonical name."""
    while isinstance(name, str) and name in alias_map:
        name = alias_map[name]
    return name


def _resolve_arg_list(args, cte_set, param_set, alias_map):
    """Resolve a list of instruction arguments to SQL expressions."""
    result = []
    for a in args:
        if isinstance(a, (list, tuple)):
            result.append([
                _resolve_single_arg(x, cte_set, param_set, alias_map)
                for x in a
            ])
        else:
            result.append(
                _resolve_single_arg(a, cte_set, param_set, alias_map))
    return result


def _resolve_single_arg(arg, cte_set, param_set, alias_map):
    """Resolve a single argument to a SQL expression."""
    if isinstance(arg, str):
        arg = _resolve_alias(arg, alias_map)
        return _arg_to_sql_expr(arg, cte_set, param_set)
    return _arg_to_sql_expr(arg, cte_set, param_set)


def _resolve_kw_val(val, cte_set, param_set, alias_map):
    """Resolve a keyword argument value."""
    if isinstance(val, str):
        val = _resolve_alias(val, alias_map)
        return _arg_to_sql_expr(val, cte_set, param_set)
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val
    return val


# ---------------------------------------------------------------------------
# KV Cache storage helper
# ---------------------------------------------------------------------------

class KVCacheState:
    """Stores KV cache states as SQLite BLOBs for efficient reuse.

    Each layer's key and value tensors are stored as bytes (BLOBs).
    Shape is typically (batch, num_kv_heads, seq_len, head_dim).
    """

    def __init__(self):
        # Maps layer_idx -> (key_blob, value_blob)
        self.cache: Dict[int, Tuple[bytes, bytes]] = {}
        self.current_seq_len: int = 0

    def clear(self) -> None:
        """Reset the cache state."""
        self.cache.clear()
        self.current_seq_len = 0

    def is_empty(self) -> bool:
        """Check if cache is empty."""
        return len(self.cache) == 0

    def update(self, layer_idx: int, key: bytes, value: bytes) -> None:
        """Store key/value cache for a specific layer."""
        self.cache[layer_idx] = (key, value)

    def get(self, layer_idx: int) -> Optional[Tuple[bytes, bytes]]:
        """Retrieve cached key/value for a layer."""
        return self.cache.get(layer_idx)


# ---------------------------------------------------------------------------
# Main KV Cache graph executor
# ---------------------------------------------------------------------------

class SqliteKVCacheGraphExecutor:
    """
    Executes ``ExportedProgram`` with KV Cache optimization using SQLite.

    Maintains two separate computation graphs:
    - Prefill graph: processes full prompt, outputs logits + initial KV cache
    - Decode graph: processes single token with previous KV cache

    Usage::

        # Export both prefill and decode graphs
        ep_prefill = export_model_prefill(model)
        ep_decode = export_model_decode(model)

        executor = SqliteKVCacheGraphExecutor()
        executor.compile(ep_prefill, ep_decode)

        # First call: runs prefill
        logits, kv_cache = executor.run(prompt_ids)

        # Subsequent calls: runs decode with cached KV
        logits, kv_cache = executor.run(next_token_ids)
    """

    def __init__(self,
                 extension_path: Optional[str] = None,
                 db_path: str = ':memory:') -> None:
        # Prefill graph state
        self._prefill_instrs: List[_Instr] = []
        self._prefill_ep: Optional[Any] = None

        # Decode graph state
        self._decode_instrs: List[_Instr] = []
        self._decode_ep: Optional[Any] = None

        # Shared state
        self._param_data: Dict[str, Any] = {}  # str param names or bytes BLOBs
        self._submodules: Dict[str, Any] = {}
        self._db_path: str = db_path
        self._conn: sqlite3.Connection = create_connection(extension_path, db_path)

        # KV Cache state
        self._kv_cache = KVCacheState()
        self._num_layers: int = 0

    @property
    def db_path(self) -> str:
        """Path of the underlying SQLite database."""
        return self._db_path

    @property
    def connection(self) -> sqlite3.Connection:
        """The underlying SQLite connection."""
        return self._conn

    @property
    def has_cache(self) -> bool:
        """Check if KV cache exists."""
        return not self._kv_cache.is_empty()

    def clear_cache(self) -> None:
        """Clear the KV cache state."""
        self._kv_cache.clear()

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile(self,
                ep_prefill: torch.export.ExportedProgram,
                ep_decode: torch.export.ExportedProgram) -> None:
        """
        Compile both prefill and decode graphs.

        Parameters
        ----------
        ep_prefill : ExportedProgram
            Exported program for prefill stage (no past_key_values input).
            Should output (logits, kv_cache) tuple.
        ep_decode : ExportedProgram
            Exported program for decode stage (with past_key_values input).
            Should output (logits, kv_cache) tuple.
        """
        self._prefill_ep = ep_prefill
        self._decode_ep = ep_decode

        # Extract and store model parameters (shared between both graphs)
        # Process both graphs to pick up any constants unique to each.
        self._compile_params(ep_prefill)
        self._compile_params(ep_decode)

        # Compile prefill graph
        self._prefill_instrs = self._compile_graph(ep_prefill)

        # Compile decode graph
        self._decode_instrs = self._compile_graph(ep_decode)

        # Detect number of layers from graph structure
        self._detect_num_layers(ep_prefill)

    def _compile_params(self, ep: torch.export.ExportedProgram) -> None:
        """Extract and store model parameters from exported program.

        For file-based databases, parameters are persisted to the
        ``model_params`` table **and** eagerly loaded back into Python
        memory so that inference uses direct BLOB binds (no sub-SELECT
        overhead).
        """
        all_tensors: Dict[str, torch.Tensor] = {}
        all_tensors.update(ep.state_dict)
        all_tensors.update({k: v for k, v in ep.named_buffers()})

        # Also get constant tensors from the exported program
        if hasattr(ep, 'constants') and ep.constants:
            all_tensors.update(ep.constants)

        # Collect parameter node names (including constants)
        param_node_names: List[str] = []
        for spec in ep.graph_signature.input_specs:
            if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER, InputKind.CONSTANT_TENSOR):
                param_node_names.append(spec.arg.name)

        if self._db_path != ':memory:':
            # File DB mode — persist then eagerly preload into memory.
            existing = set(load_param_names_from_db(self._conn))
            needed = set(param_node_names)
            if not (needed and needed.issubset(existing)):
                raw_blobs: Dict[str, bytes] = {}
                for spec in ep.graph_signature.input_specs:
                    if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER, InputKind.CONSTANT_TENSOR):
                        node_name = spec.arg.name
                        tensor = all_tensors.get(spec.target)
                        if tensor is not None:
                            raw_blobs[node_name] = tensor_to_blob(
                                tensor.detach().float().numpy()
                            )
                save_params_to_db(self._conn, raw_blobs)
            # Eagerly load all BLOBs — eliminates sub-SELECT overhead.
            self._param_data.update(preload_params_from_db(self._conn))
        else:
            # In-memory mode
            raw_blobs = {}
            for spec in ep.graph_signature.input_specs:
                if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER, InputKind.CONSTANT_TENSOR):
                    node_name = spec.arg.name
                    tensor = all_tensors.get(spec.target)
                    if tensor is not None:
                        raw_blobs[node_name] = tensor_to_blob(
                            tensor.detach().float().numpy()
                        )
            self._param_data.update(raw_blobs)

    def _compile_graph(self, ep: torch.export.ExportedProgram) -> List[_Instr]:
        """Compile a single exported program into instruction list.

        Applies compile-time optimizations:
        * No-op elimination: instructions that pass through their first
          argument (e.g. ``to.dtype``, ``clone``, ``alias``) are converted
          to lightweight alias instructions, avoiding registry lookup and
          function call overhead at run time.
        * Op function pre-resolution: each ``call_function`` instruction
          stores its resolved handler directly in ``_Instr.resolved_fn``,
          eliminating per-step ``SqliteOpRegistry.get()`` lookups.
        """
        gm = ep.graph_module

        # Collect sub-modules
        for attr_name, submod in gm.named_children():
            self._submodules[attr_name] = submod

        # Build instruction list
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
                args = [_node_arg_to_serialisable(a, node_map)
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
        instrs = _eliminate_noop_instrs(instrs)
        _fuse_silu_mul_instrs(instrs)
        _fuse_rope_instrs(instrs)
        _fuse_rmsnorm_instrs(instrs)
        _fuse_inline_shape_ops(instrs)
        instrs = _eliminate_aliases(instrs)
        _preresolve_op_functions(instrs)
        return instrs

    def _detect_num_layers(self, ep: torch.export.ExportedProgram) -> None:
        """Detect number of transformer layers from model structure."""
        # Look for layer-specific parameters to count layers
        state_dict = ep.state_dict
        layer_indices = set()
        for key in state_dict.keys():
            # Match pattern like "model.layers.N.*"
            if '.layers.' in key:
                parts = key.split('.layers.')
                if len(parts) > 1:
                    layer_idx_str = parts[1].split('.')[0]
                    if layer_idx_str.isdigit():
                        layer_indices.add(int(layer_idx_str))

        self._num_layers = max(layer_indices) + 1 if layer_indices else 0

    # ------------------------------------------------------------------
    # Forward-SQL export
    # ------------------------------------------------------------------

    def export_forward_sql(self, graph: str = 'decode') -> Tuple[str, List[str]]:
        """Export a complete CTE-based SQL for the full forward pass.

        For a fixed graph the sequence of operations is identical on every
        invocation — only the runtime data (input_ids, past_key_values)
        changes.  This method encodes that entire sequence into a single
        SQL ``WITH … SELECT …`` statement using Common Table Expressions.

        Each FX node becomes one CTE clause.  **All** placeholders
        (model parameters and runtime inputs) are represented as ``?``
        bind-parameters so the generated SQL has zero external table
        references and can be executed on any connection with the
        ``llm_ops`` extension loaded.

        Parameters
        ----------
        graph : {'decode', 'prefill'}
            Which compiled graph to export.

        Returns
        -------
        sql : str
            The complete CTE-based SQL statement.
        bind_names : list of str
            Ordered names of **all** ``?`` bind-parameters.  The caller
            must supply the corresponding BLOB values (model parameter
            blobs first, then runtime inputs) when executing the SQL.
        """
        if graph == 'decode':
            instrs = self._decode_instrs
        elif graph == 'prefill':
            instrs = self._prefill_instrs
        else:
            raise ValueError(f"Unknown graph: {graph!r}")

        param_names = set(self._param_data.keys())
        return build_forward_sql(instrs, param_names, self._num_layers)

    def export_standalone_sql(self, graph: str = 'decode') -> Tuple[str, List[str]]:
        """Export standalone SQL with inline model parameter lookups.

        Unlike :meth:`export_forward_sql`, model parameters are fetched
        directly from the ``model_params`` table via inline sub-SELECTs.
        Only runtime inputs (``input_ids``, and for decode graphs the
        past key/value cache blobs) remain as ``?`` bind-parameters.

        Parameters
        ----------
        graph : {'decode', 'prefill'}
            Which compiled graph to export.

        Returns
        -------
        sql : str
            The complete CTE-based SQL statement.
        bind_names : list of str
            Ordered names of runtime-only ``?`` bind-parameters.
        """
        if graph == 'decode':
            instrs = self._decode_instrs
        elif graph == 'prefill':
            instrs = self._prefill_instrs
        else:
            raise ValueError(f"Unknown graph: {graph!r}")

        param_names = set(self._param_data.keys())
        return build_standalone_forward_sql(
            instrs, param_names, self._num_layers)

    def export_instructions_json(self, graph: str = 'decode') -> list:
        """Export compiled instructions as a JSON-serialisable list.

        Each entry is a dict with keys:
        ``name``, ``op``, ``target``, ``args``, ``kwargs``,
        ``is_output``, ``output_args``.

        The ``target`` is serialised as the op's string representation
        (e.g. ``"aten.add.Tensor"``) for ``call_function`` instructions.
        """
        if graph == 'decode':
            instrs = self._decode_instrs
        elif graph == 'prefill':
            instrs = self._prefill_instrs
        else:
            raise ValueError(f"Unknown graph: {graph!r}")

        param_names = set(self._param_data.keys())
        result = []
        # Map fused handler functions to stable export-target strings so that
        # the standalone engine can recognise and use the same fused C calls.
        _FUSED_TARGET_MAP = {
            fused_silu_mul: "llm_silu_mul_nd",
            fused_rope:     "llm_rope_nd",
            fused_rmsnorm:  "llm_rmsnorm_nd",
        }

        for instr in instrs:
            target_str = ""
            if instr.op == "call_function":
                # Prefer a fused-target string when a fused handler is set.
                target_str = _FUSED_TARGET_MAP.get(instr.resolved_fn,
                                                    str(instr.target))
            elif instr.op in ("placeholder", "get_attr", "alias"):
                target_str = str(instr.target) if instr.target else ""

            d = {
                "name": instr.out_name,
                "op": instr.op,
                "target": target_str,
                "args": _serialise_args(instr.arg_names),
                "kwargs": _serialise_args(instr.kwarg_names),
                "is_output": instr.is_output,
                "output_args": _serialise_args(instr.output_args),
                "is_param": instr.out_name in param_names,
            }
            result.append(d)
        return result

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self,
            input_ids: np.ndarray,
            verbose: bool = False,
            profile: bool = False) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Execute the compiled graph with KV cache optimization.

        Automatically selects between prefill and decode based on cache state:
        - If cache is empty: runs prefill graph
        - If cache exists: runs decode graph with cached KV states

        Parameters
        ----------
        input_ids : np.ndarray
            Input token IDs, shape (batch, seqlen)
        verbose : bool
            If True, print execution details
        profile : bool
            When True, collect per-op timing data into ``self.last_profile``.

        Returns
        -------
        logits : np.ndarray
            Output logits, shape (batch, seqlen, vocab_size)
        kv_cache : Optional[Any]
            Updated KV cache state (internal use)
        """
        if self._kv_cache.is_empty():
            # Prefill stage: process full prompt
            if verbose:
                print(f"[KVCache] Prefill stage: processing {input_ids.shape[1]} tokens")
            return self._run_prefill(input_ids, verbose, profile=profile)
        else:
            # Decode stage: process single token with cache
            if verbose:
                print(f"[KVCache] Decode stage: processing 1 token (cache_len={self._kv_cache.current_seq_len})")
            return self._run_decode(input_ids, verbose, profile=profile)

    def _run_prefill(self,
                     input_ids: np.ndarray,
                     verbose: bool = False,
                     profile: bool = False) -> Tuple[np.ndarray, Optional[Any]]:
        """Execute prefill graph without past KV cache."""
        env: Dict[str, Any] = {}
        env.update(self._param_data)

        for attr_name, submod in self._submodules.items():
            env[attr_name] = submod

        output_value = self._execute_instrs(
            self._prefill_instrs, input_ids, env, verbose, profile=profile
        )

        # Parse flat output: (logits, k0, v0, k1, v1, ..., k_{N-1}, v_{N-1})
        logits, past_kv = self._parse_graph_output(output_value)

        if past_kv is not None:
            self._store_kv_cache(past_kv, input_ids.shape[1])

        return logits, past_kv

    def _run_decode(self,
                    input_ids: np.ndarray,
                    verbose: bool = False,
                    profile: bool = False) -> Tuple[np.ndarray, Optional[Any]]:
        """Execute decode graph with past KV cache."""
        env: Dict[str, Any] = {}
        env.update(self._param_data)

        for attr_name, submod in self._submodules.items():
            env[attr_name] = submod

        # Build past_key_values tuple from cache
        # Format: tuple of (key_blob, value_blob) for each layer
        past_kv_tuple = []
        for layer_idx in range(self._num_layers):
            kv = self._kv_cache.get(layer_idx)
            if kv is not None:
                past_kv_tuple.append(kv)  # (key_blob, value_blob)
            else:
                raise RuntimeError(f"Missing cache for layer {layer_idx}")
        past_kv_tuple = tuple(past_kv_tuple)

        # Pass both input_ids and past_kv to the decode graph
        output_value = self._execute_instrs(
            self._decode_instrs, input_ids, env, verbose,
            past_key_values=past_kv_tuple,
            profile=profile
        )

        # Parse flat output and update KV cache
        logits, past_kv = self._parse_graph_output(output_value)

        if past_kv is not None:
            self._store_kv_cache(past_kv, self._kv_cache.current_seq_len + 1)

        return logits, past_kv

    def _parse_graph_output(self, output_value: Tuple) -> Tuple[np.ndarray, Optional[Tuple]]:
        """Parse flat graph output into (logits, kv_cache_tuple).

        The exported graph produces a flat tuple:
        ``(logits, k0, v0, k1, v1, …, k_{N-1}, v_{N-1})``

        This method restructures it into:
        ``(logits, ((k0, v0), (k1, v1), …))``
        """
        if not isinstance(output_value, tuple) or len(output_value) < 2:
            val = output_value[0] if isinstance(output_value, tuple) else output_value
            return val, None

        logits = output_value[0]
        kv_flat = output_value[1:]

        # Flat KV outputs: k0, v0, k1, v1, ...
        if len(kv_flat) >= 2 and len(kv_flat) % 2 == 0:
            num_layers = len(kv_flat) // 2
            past_kv = tuple(
                (kv_flat[2 * i], kv_flat[2 * i + 1])
                for i in range(num_layers)
            )
            return logits, past_kv

        return logits, None

    def _execute_instrs(self,
                       instrs: List[_Instr],
                       input_ids: np.ndarray,
                       env: Dict[str, Any],
                       verbose: bool = False,
                       past_key_values: Optional[Tuple] = None,
                       profile: bool = False) -> Tuple:
        """Execute instruction list (shared by prefill and decode).

        Optimised hot-loop:
        * Alias instructions are eliminated at compile time.
        * ``call_function`` instructions use the pre-resolved handler
          (``instr.resolved_fn``) to skip the per-step registry lookup.
        * Argument resolution is inlined for the no-kwargs fast path.

        When *profile* is ``True``, per-op timing data is stored in
        ``self.last_profile``.
        """
        output_value = None
        op_timings: Optional[Dict[str, float]] = {} if profile else None
        sql_call_count = 0

        # Local variable bindings for hot-path speed
        conn = self._conn
        _resolve = self._resolve

        # Count user inputs to handle flattened past_key_values
        # USER_INPUT placeholders come after parameters/buffers/constants
        user_input_count = 0

        for instr in instrs:
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
                    self._log_value(instr.out_name, result)

            elif op == "placeholder":
                if instr.target not in env:
                    # First user input: input_ids
                    if user_input_count == 0:
                        ids = np.asarray(input_ids, dtype=np.int64)
                        env[instr.out_name] = tensor_to_blob(
                            ids.astype(np.float32))
                        user_input_count += 1
                    # Subsequent user inputs: flattened past_key_values
                    # Format: past_key_values_layer_key/value
                    elif past_key_values is not None and user_input_count > 0:
                        # Extract layer_idx and kv_idx from flattened past_kv
                        # past_key_values is tuple of (k,v) tuples per layer
                        # We need to map to the flattened inputs
                        kv_idx = user_input_count - 1  # 0-based index into flattened kv
                        layer_idx = kv_idx // 2
                        is_key = (kv_idx % 2 == 0)

                        if layer_idx < len(past_key_values):
                            kv_pair = past_key_values[layer_idx]
                            if is_key:
                                env[instr.out_name] = kv_pair[0]  # key blob
                            else:
                                env[instr.out_name] = kv_pair[1]  # value blob
                        user_input_count += 1

            elif op == "output":
                vals = []
                for i, a in enumerate(instr.output_args):
                    v = _resolve(a, env)
                    # Only convert the first output (logits) to numpy;
                    # keep KV cache outputs as blobs to avoid redundant
                    # blob→numpy→blob round-trips in _store_kv_cache.
                    if i == 0 and isinstance(v, bytes):
                        v = blob_to_numpy(v)
                    vals.append(v)
                output_value = tuple(vals)

            elif op == "subgraph_call":
                from src.backends.sqlite_engine import _SubgraphExecutor
                resolved_args = [_resolve(a, env) for a in instr.arg_names]
                submod = resolved_args[1]
                subinputs = resolved_args[2:]
                sub_exec = _SubgraphExecutor(submod, conn)
                result = sub_exec.run(*subinputs)
                env[instr.out_name] = result
                if verbose:
                    self._log_value(instr.out_name, result)

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

    def _store_kv_cache(self, past_kv: Any, seq_len: int) -> None:
        """Store KV cache from graph output."""
        # past_kv is typically a tuple of tuples: ((k0, v0), (k1, v1), ...)
        # where each (k, v) is for one layer
        if isinstance(past_kv, (tuple, list)):
            for layer_idx, layer_kv in enumerate(past_kv):
                if isinstance(layer_kv, (tuple, list)) and len(layer_kv) == 2:
                    key, value = layer_kv
                    # Convert to bytes if needed
                    if isinstance(key, np.ndarray):
                        key = tensor_to_blob(key.astype(np.float32))
                    if isinstance(value, np.ndarray):
                        value = tensor_to_blob(value.astype(np.float32))

                    self._kv_cache.update(layer_idx, key, value)

        self._kv_cache.current_seq_len = seq_len

    def _resolve(self, arg, env: Dict[str, Any]) -> Any:
        """Resolve an argument from the environment."""
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

    def _log_value(self, name: str, val: Any) -> None:
        """Log value for verbose mode."""
        if isinstance(val, bytes):
            shape = tuple(_blob_shape_list(val))
            print(f"  {name:40s} shape={str(shape):20s} dtype=float32")
        elif isinstance(val, np.ndarray):
            print(f"  {name:40s} shape={str(val.shape):20s} dtype={val.dtype}")
        elif isinstance(val, tuple):
            shapes = []
            for v in val:
                if isinstance(v, bytes):
                    shapes.append(tuple(_blob_shape_list(v)))
                elif isinstance(v, np.ndarray):
                    shapes.append(v.shape)
                else:
                    shapes.append(v)
            print(f"  {name:40s} tuple={shapes}")
        else:
            print(f"  {name:40s} value={val}")
