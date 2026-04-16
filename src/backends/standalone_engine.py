"""Standalone SQLite inference engine — **no** torch / transformers / numpy.

This module implements a minimal autoregressive text generation pipeline
that relies exclusively on:

* Python standard library: ``sqlite3``, ``struct``, ``os``
* The ``llm_ops.so`` C extension (loaded at runtime via ``sqlite3``)

It reads model weights from the ``model_params`` table, tokenizer
vocabulary from the ``vocab`` table, and forward-pass instructions from JSON files exported by exported model.

Typical usage::

    engine = StandaloneEngine("/tmp/llm_export")
    text = engine.generate("hello", max_tokens=20)
    print(text)
"""

from __future__ import annotations

import json as _json
import os
import struct
import sqlite3
import time as _time
import ctypes as _ctypes
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ND_MAGIC = 0x80000000
_INT8_MAGIC = 0x80000008
_INT32_MAX = 2147483647
_NEG_INF_MASK = -1e9

# Dispatch type codes for the _execute_instrs hot loop.
# Using module-level integer constants (not dict lookups) for zero overhead.
_DT_NORMAL = 0       # Normal call_function (resolve args, invoke handler)
_DT_FUSED = 1        # Fused op (silu_mul, rope, rmsnorm)
_DT_LAZY = 2         # Precompiled lazy shape op (SqlExpr)
_DT_PLACEHOLDER = 3  # Input / KV-cache placeholder
_DT_ALIAS = 4        # Alias (env[name] = env[target])
_DT_OUTPUT = 5       # Output collection node
_DT_SKIP = 6         # Dead instruction (handler=None)

# Pre-compiled struct objects for common header fields
_STRUCT_MAGIC = struct.Struct('<I')
_STRUCT_I32 = struct.Struct('<i')
_STRUCT_MAGIC_I32 = struct.Struct('<Ii')   # magic + ndim header

# Regex to extract transformer layer index from parameter names.
# Handles both dotted (layers.0.) and underscored (layers_0_) formats.
import re as _re
_LAYER_RE = _re.compile(r"layers[._](\d+)[._]")
_PAST_KEY_VALUES_RE = _re.compile(r"past_key_values_(\d+)_(\d+)")

try:
    _LIBC = _ctypes.CDLL(None)
    _LIBC.malloc_trim.argtypes = [_ctypes.c_size_t]
    _LIBC.malloc_trim.restype = _ctypes.c_int
except (AttributeError, OSError):
    _LIBC = None


def _maybe_malloc_trim() -> None:
    """Return freed libc heap pages to the OS when available."""
    if _LIBC is not None:
        try:
            _LIBC.malloc_trim(0)
        except OSError:
            pass


def _is_int8_blob(blob: bytes) -> bool:
    """Return True if *blob* uses the INT8 quantized format."""
    if blob and len(blob) >= 4:
        return _STRUCT_MAGIC.unpack_from(blob, 0)[0] == _INT8_MAGIC
    return False


# ---------------------------------------------------------------------------
# N-D BLOB encoding / decoding  (mirrors sqlite_ops.tensor_to_blob / blob_to_numpy)
# ---------------------------------------------------------------------------

# LRU caches for struct.Struct objects indexed by element count
_struct_cache_int: Dict[int, struct.Struct] = {}
_struct_cache_float: Dict[int, struct.Struct] = {}


def _get_struct_int(n: int) -> struct.Struct:
    """Return a cached Struct for packing/unpacking *n* int32 values."""
    s = _struct_cache_int.get(n)
    if s is None:
        s = struct.Struct(f'<{n}i')
        _struct_cache_int[n] = s
    return s


def _get_struct_float(n: int) -> struct.Struct:
    """Return a cached Struct for packing/unpacking *n* float32 values."""
    s = _struct_cache_float.get(n)
    if s is None:
        s = struct.Struct(f'<{n}f')
        _struct_cache_float[n] = s
    return s


def encode_nd_blob(flat_data: List[float], shape: List[int]) -> bytes:
    """Encode a list of float32 values into an N-D BLOB.

    Format::

        [magic:u32=0x80000000][ndim:i32][shape[0]:i32…][data:f32…]
    """
    ndim = len(shape)
    n_elems = 1
    for s in shape:
        n_elems *= s
    assert len(flat_data) == n_elems, (
        f"flat_data length {len(flat_data)} != product of shape {shape}"
    )
    hdr = _STRUCT_MAGIC_I32.pack(_ND_MAGIC, ndim)
    hdr += _get_struct_int(ndim).pack(*shape)
    hdr += _get_struct_float(n_elems).pack(*flat_data)
    return hdr


def decode_nd_blob(blob: bytes) -> Tuple[List[int], List[float]]:
    """Decode an N-D BLOB into ``(shape, flat_data)``."""
    magic = _STRUCT_MAGIC.unpack_from(blob, 0)[0]
    if magic == _ND_MAGIC:
        ndim = _STRUCT_I32.unpack_from(blob, 4)[0]
        shape = list(_get_struct_int(ndim).unpack_from(blob, 8))
        hdr_size = 8 + 4 * ndim
        n_elems = 1
        for s in shape:
            n_elems *= s
        data = list(_get_struct_float(n_elems).unpack_from(blob, hdr_size))
        return shape, data
    # Legacy 1-D format
    n = _STRUCT_I32.unpack_from(blob, 0)[0]
    data = list(_get_struct_float(n).unpack_from(blob, 4))
    return [n], data


def blob_shape(blob: bytes) -> List[int]:
    """Return the shape of an N-D BLOB without decoding data."""
    magic = _STRUCT_MAGIC.unpack_from(blob, 0)[0]
    if magic in (_ND_MAGIC, _INT8_MAGIC):
        ndim = _STRUCT_I32.unpack_from(blob, 4)[0]
        return list(_get_struct_int(ndim).unpack_from(blob, 8))
    n = _STRUCT_I32.unpack_from(blob, 0)[0]
    return [n]


def _blob_hdr_size(ndim: int) -> int:
    """Return the header size in bytes for an N-D blob with *ndim* dims."""
    return 8 + 4 * ndim


def cat_nd_blobs(a: bytes, b: bytes, dim: int) -> bytes:
    """Fast Python-side concatenation of two N-D blobs along *dim*.

    Avoids a SQL round-trip by directly building the output blob.
    Uses ``memoryview`` and ``bytearray`` for zero-copy where possible.

    For the KV-cache hot path (dim=2 of 4-D tensors), this eliminates
    the ``llm_cat_nd`` / ``llm_cat_multi_nd`` SQL call entirely.
    """
    shape_a = blob_shape(a)
    shape_b = blob_shape(b)
    ndim = len(shape_a)
    if dim < 0:
        dim += ndim

    hdr_a = _blob_hdr_size(ndim)
    hdr_b = _blob_hdr_size(ndim)

    # Build output shape
    out_shape = list(shape_a)
    out_shape[dim] = shape_a[dim] + shape_b[dim]
    out_total = 1
    for s in out_shape:
        out_total *= s

    # If concat along last dim, data is interleaved per "row"
    # If concat along first dim, data is just appended
    # General case: use stride-based copy

    # Compute strides for a, b, and output
    def _strides(shape):
        st = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            st[i] = st[i + 1] * shape[i + 1]
        return st

    sa = _strides(shape_a)
    sb = _strides(shape_b)

    # Special fast path: concat along last dim (contiguous append per slice)
    # or any dim where we can do block copies
    # Use the general block-copy approach: iterate over "outer" dims

    # Number of independent "blocks" = product of all dims before 'dim'
    outer = 1
    for i in range(dim):
        outer *= shape_a[i]

    # Inner block size = product of all dims after 'dim'
    inner = 1
    for i in range(dim + 1, ndim):
        inner *= shape_a[i]

    a_dim = shape_a[dim]
    b_dim = shape_b[dim]

    # Build output blob
    out_hdr = _STRUCT_MAGIC_I32.pack(_ND_MAGIC, ndim)
    out_hdr += _get_struct_int(ndim).pack(*out_shape)
    out = bytearray(len(out_hdr) + out_total * 4)
    out[:len(out_hdr)] = out_hdr

    # Block copy: for each outer slice, copy a's dim-block then b's dim-block
    a_data = memoryview(a)[hdr_a:]
    b_data = memoryview(b)[hdr_b:]
    out_data = memoryview(out)[len(out_hdr):]

    a_block = a_dim * inner * 4  # bytes per block from a
    b_block = b_dim * inner * 4  # bytes per block from b
    o_block = (a_dim + b_dim) * inner * 4  # bytes per block in output

    for i in range(outer):
        a_off = i * a_block
        b_off = i * b_block
        o_off = i * o_block
        out_data[o_off:o_off + a_block] = a_data[a_off:a_off + a_block]
        out_data[o_off + a_block:o_off + a_block + b_block] = \
            b_data[b_off:b_off + b_block]

    return bytes(out)


# ---------------------------------------------------------------------------
# Python-side shape ops (zero SQL round-trip)
# ---------------------------------------------------------------------------
# These rewrite only the BLOB header, reusing the data payload.
# For shape ops that don't change the data layout (view/reshape/unsqueeze
# when it doesn't require padding), this eliminates SQL round-trips entirely.

def _py_unsqueeze(blob: bytes, dim: int) -> bytes:
    """Insert a size-1 dimension at *dim*. Header-only rewrite, no data copy."""
    shape = blob_shape(blob)
    ndim = len(shape)
    if dim < 0:
        dim += ndim + 1
    new_shape = shape[:dim] + [1] + shape[dim:]
    new_ndim = ndim + 1
    # Build new header + reuse data payload
    hdr = _STRUCT_MAGIC_I32.pack(_ND_MAGIC, new_ndim)
    hdr += _get_struct_int(new_ndim).pack(*new_shape)
    old_hdr = _blob_hdr_size(ndim)
    return hdr + blob[old_hdr:]


def _py_view(blob: bytes, new_shape: List[int]) -> bytes:
    """Reshape (view) — header-only rewrite when total elements match."""
    shape = blob_shape(blob)
    ndim = len(shape)
    total = 1
    for s in shape:
        total *= s
    # Resolve -1 dim
    neg_idx = -1
    known = 1
    for i, s in enumerate(new_shape):
        if s == -1:
            neg_idx = i
        else:
            known *= s
    if neg_idx >= 0:
        new_shape = list(new_shape)
        new_shape[neg_idx] = total // known
    new_ndim = len(new_shape)
    hdr = _STRUCT_MAGIC_I32.pack(_ND_MAGIC, new_ndim)
    hdr += _get_struct_int(new_ndim).pack(*new_shape)
    old_hdr = _blob_hdr_size(ndim)
    return hdr + blob[old_hdr:]


def _py_expand(blob: bytes, sizes: List[int]) -> bytes:
    """Expand — for dims that are already the target size or size-1 → target.

    Only handles the trivial case where no actual data broadcast is needed
    (all expanded dims are size-1 → target or already match). If actual
    broadcast is needed, returns None and the caller should fall back to SQL.
    """
    shape = blob_shape(blob)
    ndim = len(shape)
    new_ndim = len(sizes)
    # If ndim mismatch, fall back to SQL
    if ndim != new_ndim:
        return None
    # Check if any actual broadcast is needed
    new_shape = list(sizes)
    for i in range(ndim):
        if new_shape[i] == -1:
            new_shape[i] = shape[i]
        elif shape[i] == 1 and new_shape[i] != 1:
            # Actual broadcast needed — fall back to SQL
            return None
        elif shape[i] != new_shape[i]:
            return None
    # No broadcast needed — just rewrite header
    hdr = _STRUCT_MAGIC_I32.pack(_ND_MAGIC, new_ndim)
    hdr += _get_struct_int(new_ndim).pack(*new_shape)
    old_hdr = _blob_hdr_size(ndim)
    return hdr + blob[old_hdr:]


def _py_squeeze(blob: bytes, dim: int) -> bytes:
    """Remove a size-1 dimension at *dim*. Header-only rewrite."""
    shape = blob_shape(blob)
    ndim = len(shape)
    if dim < 0:
        dim += ndim
    if shape[dim] != 1:
        return blob  # nothing to squeeze
    new_shape = shape[:dim] + shape[dim + 1:]
    new_ndim = ndim - 1
    hdr = _STRUCT_MAGIC_I32.pack(_ND_MAGIC, new_ndim)
    hdr += _get_struct_int(new_ndim).pack(*new_shape)
    old_hdr = _blob_hdr_size(ndim)
    return hdr + blob[old_hdr:]


# ---------------------------------------------------------------------------
# Argmax helper
# ---------------------------------------------------------------------------

def argmax_last_row(logits_blob: bytes, vocab_size: int) -> int:
    """Return argmax over the last *vocab_size* floats of *logits_blob*.

    For logits of shape ``(1, seq_len, vocab_size)`` this picks the token
    with the highest logit at the last sequence position.

    Only unpacks the last *vocab_size* float32 values, avoiding
    deserialization of the entire logits tensor.
    """
    # Only unpack the tail of the blob (last vocab_size floats).
    # Each float32 is 4 bytes; the last row starts at:
    #   len(blob) - vocab_size * 4
    tail_offset = len(logits_blob) - vocab_size * 4
    last_row = _get_struct_float(vocab_size).unpack_from(
        logits_blob, tail_offset)
    best_idx = 0
    best_val = last_row[0]
    for i in range(1, vocab_size):
        if last_row[i] > best_val:
            best_val = last_row[i]
            best_idx = i
    return best_idx


# ---------------------------------------------------------------------------
# Extension loader
# ---------------------------------------------------------------------------

_EXTENSION_NAME = "llm_ops"


def _find_extension() -> str:
    """Locate ``llm_ops.so`` next to this file or in ``sqlite-llm/``."""
    candidates = [
        os.path.join(os.path.dirname(__file__), f"{_EXTENSION_NAME}.so"),
        os.path.join(os.path.dirname(__file__), "..", "..", "sqlite-llm",
                     f"{_EXTENSION_NAME}.so"),
    ]
    for p in candidates:
        rp = os.path.realpath(p)
        if os.path.isfile(rp):
            return rp
    raise FileNotFoundError(
        f"Cannot find {_EXTENSION_NAME}.so in {candidates}")


def open_db(
    db_path: str,
    extension_path: Optional[str] = None,
    cache_size_kb: Optional[int] = None,
    disable_mmap: bool = False,
) -> sqlite3.Connection:
    """Open a SQLite connection with the llm_ops extension loaded.

    Applies inference-mode PRAGMAs for optimal LLM read performance:

    * ``journal_mode=OFF``         – no journal (DB is read-only at inference)
    * ``synchronous=OFF``          – skip fsync
    * ``locking_mode=EXCLUSIVE``   – avoid per-statement lock overhead
    * ``temp_store=MEMORY``        – temp results in RAM
    * ``cache_size``               – configurable page cache budget
    * ``mmap_size``                – memory-map the file for zero-copy reads

    Parameters
    ----------
    cache_size_kb : int, optional
        Override the default PRAGMA cache_size value (negative = KiB).
        E.g. ``-51200`` for 50 MiB.
    disable_mmap : bool
        When ``True``, set ``PRAGMA mmap_size=0`` to force all reads
        through the SQLite page cache (useful for pcache benchmarking).
    """
    from src.backends.llm_memory_manager import apply_inference_pragmas

    ext = extension_path or _find_extension()
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    try:
        conn.load_extension(ext, entrypoint="sqlite3_llm_ops_init")
    except TypeError:
        conn.load_extension(ext)

    kwargs = {}
    if cache_size_kb is not None:
        kwargs["cache_size_kb"] = cache_size_kb
    apply_inference_pragmas(conn, db_path, **kwargs)

    if disable_mmap:
        conn.execute("PRAGMA mmap_size=0")

    return conn


# ---------------------------------------------------------------------------
# Tokenizer (pure SQL)
# ---------------------------------------------------------------------------

class StandaloneTokenizer:
    """Tokenizer backed by the ``vocab`` table in a DB.

    When a ``tokenizer.json`` file is available (produced by
    ``scripts/export_pretrained.py``), encoding uses the HuggingFace
    ``tokenizers`` library for correct BPE tokenization.  Otherwise
    falls back to greedy character-level matching (sufficient for the
    mock 100-token vocabulary).

    Only uses ``sqlite3`` + optionally ``tokenizers`` — no torch or
    transformers required.
    """

    EOS_ID = 0
    IM_START_ID = 1
    IM_END_ID = 2
    PAD_ID = 3

    def __init__(
        self,
        conn: sqlite3.Connection,
        tokenizer_json_path: Optional[str] = None,
    ) -> None:
        self._conn = conn
        self._hf_tokenizer = None  # Optional[tokenizers.Tokenizer]

        # Try to load HuggingFace fast tokenizer for BPE encoding
        if tokenizer_json_path and os.path.isfile(tokenizer_json_path):
            try:
                from tokenizers import Tokenizer as HFTokenizer
                self._hf_tokenizer = HFTokenizer.from_file(tokenizer_json_path)
            except ImportError:
                pass  # tokenizers library not installed — fall back

        # Cache id↔token mappings from DB for decoding
        self._id2tok: Dict[int, str] = {}
        self._tok2id: Dict[str, int] = {}
        self._specials: List[Tuple[str, int]] = []
        self._special_ids: set = set()

        rows = conn.execute(
            "SELECT token_id, token, is_special FROM vocab"
        ).fetchall()
        for tid, text, is_sp in rows:
            self._id2tok[tid] = text
            self._tok2id[text] = tid
            if is_sp:
                self._specials.append((text, tid))
                self._special_ids.add(tid)
        # Sort specials longest-first for greedy matching
        self._specials.sort(key=lambda x: len(x[0]), reverse=True)
        self._vocab_size = len(rows)

        # Read EOS token id from metadata if available
        try:
            row = conn.execute(
                "SELECT value FROM metadata WHERE key = 'eos_token_id'"
            ).fetchone()
            if row:
                val = _json.loads(row[0])
                if isinstance(val, list) and val:
                    val = val[0]
                self.EOS_ID = int(val)
        except (sqlite3.OperationalError, ValueError, TypeError):
            pass

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.

        Uses the HuggingFace ``tokenizers`` BPE encoder when available,
        otherwise falls back to greedy character-level matching.
        """
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.encode(text).ids

        # Fallback: character-level greedy matching
        ids: List[int] = []
        pos = 0
        n = len(text)
        while pos < n:
            matched = False
            for sp_text, sp_id in self._specials:
                if text[pos: pos + len(sp_text)] == sp_text:
                    ids.append(sp_id)
                    pos += len(sp_text)
                    matched = True
                    break
            if matched:
                continue
            ch = text[pos]
            tid = self._tok2id.get(ch)
            if tid is not None:
                ids.append(tid)
            pos += 1
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back to text.

        Uses the HuggingFace ``tokenizers`` decoder when available for
        correct BPE detokenization, otherwise joins DB token texts.
        """
        if self._hf_tokenizer is not None:
            if skip_special:
                ids = [tid for tid in ids if tid not in self._special_ids]
            return self._hf_tokenizer.decode(ids, skip_special_tokens=skip_special)

        # Fallback: character-level join
        parts: List[str] = []
        for tid in ids:
            if skip_special and tid in self._special_ids:
                continue
            tok = self._id2tok.get(tid)
            if tok is not None:
                parts.append(tok)
        return "".join(parts)


# ---------------------------------------------------------------------------
# Standalone engine
# ---------------------------------------------------------------------------

# Maps target string → callable(conn, args, kwargs) → result blob.
# Each function executes a single SQL call, matching the Python executor.
_SQL_OPS: Dict[str, object] = {}


class SqlExpr:
    """Deferred SQL expression that can be inlined into a consumer's SQL.

    Shape ops (unsqueeze, transpose, view …) that are consumed by exactly
    one downstream instruction return a ``SqlExpr`` instead of executing
    immediately.  The consumer's ``_sql1`` call inlines the fragment,
    collapsing two SQL round-trips into one::

        -- Before (2 round-trips):
        SELECT llm_unsqueeze_nd(?, ?)       -- materialises intermediate BLOB
        SELECT llm_add_nd(<intermediate>, ?)

        -- After (1 round-trip):
        SELECT llm_add_nd(llm_unsqueeze_nd(?, ?), ?)
    """
    __slots__ = ('sql_fragment', 'params')

    def __init__(self, sql_fragment: str, params: list) -> None:
        self.sql_fragment = sql_fragment
        self.params = params


def _sql_text_literal(value: str) -> SqlExpr:
    """Inline a SQL text literal without rewriting it as llm_get_param()."""
    return SqlExpr(f"'{_sql_quote(value)}'", [])


def _resolve_sql_params(sql: str, params: list):
    """Inline ``SqlExpr`` objects and parameter-name refs into SQL.

    Each ``?`` placeholder corresponding to a ``SqlExpr`` parameter is
    replaced by the expression's SQL fragment and its bind-parameters are
    spliced into the new parameter list.  Plain ``str`` parameters are
    treated as model-parameter names and rewritten to ``llm_get_param()``
    calls so Python never binds the actual BLOB bytes.

    Returns ``(new_sql, new_params)``.
    """
    has_special = False
    for p in params:
        if isinstance(p, (SqlExpr, str)):
            has_special = True
            break
    if not has_special:
        return sql, params

    parts = sql.split("?")
    new_parts: list = []
    new_params: list = []
    for i, p in enumerate(params):
        new_parts.append(parts[i])
        if isinstance(p, SqlExpr):
            new_parts.append(p.sql_fragment)
            new_params.extend(p.params)
        elif isinstance(p, str):
            new_parts.append(f"llm_get_param('{_sql_quote(p)}')")
        else:
            new_parts.append("?")
            new_params.append(p)
    if len(parts) > len(params):
        new_parts.append(parts[len(params)])
    return "".join(new_parts), new_params


def _sql1_lazy(sql_fragment: str, params: list) -> SqlExpr:
    """Create a deferred ``SqlExpr``, flattening any nested expressions."""
    has_nested = False
    for p in params:
        if isinstance(p, (SqlExpr, str)):
            has_nested = True
            break
    if has_nested:
        sql_fragment, params = _resolve_sql_params(sql_fragment, params)
    return SqlExpr(sql_fragment, params)


def _sql1(conn: sqlite3.Connection, sql: str, params: list) -> bytes:
    """Execute a SQL statement that returns a single BLOB value.

    ``SqlExpr`` objects in *params* are transparently inlined into the
    SQL string before execution.
    """
    # Fast path: skip _resolve_sql_params entirely when no SqlExpr or
    # parameter-name strings are present.  Checking type() directly is
    # ~3× faster than isinstance() in CPython for the common case.
    for p in params:
        if type(p) is SqlExpr or type(p) is str:
            sql, params = _resolve_sql_params(sql, params)
            break
    row = conn.execute(sql, params).fetchone()
    return row[0] if row else b""


def _ensure_op_blob(x):
    """Convert a Python scalar to a 1-element BLOB if needed.

    ``SqlExpr`` objects are passed through unchanged for later inlining.
    """
    if isinstance(x, (bytes, SqlExpr)):
        return x
    if isinstance(x, (int, float)):
        return encode_nd_blob([float(x)], [1])
    return x


def _build_sql_ops():
    """Build the target→handler mapping for standalone execution."""
    m: Dict[str, object] = {}

    # ── Arithmetic ──
    def _add(conn, args, kw):
        a, b = _ensure_op_blob(args[0]), _ensure_op_blob(args[1])
        alpha = kw.get("alpha", 1)
        if isinstance(alpha, str):
            try:
                alpha = float(alpha)
            except ValueError:
                alpha = 1
        # Identity: add 0 → return unchanged
        if isinstance(b, (int, float)) and b == 0 and alpha == 1:
            return a
        if alpha != 1:
            return _sql1(conn, "SELECT llm_add_nd(?, llm_scale_nd(?, ?))",
                         [a, b, float(alpha)])
        return _sql1(conn, "SELECT llm_add_nd(?, ?)", [a, b])
    m["aten.add.Tensor"] = _add

    def _add_scalar(conn, args, kw):
        a = _ensure_op_blob(args[0])
        b = float(args[1])
        # Identity: add 0.0 → return unchanged
        if b == 0.0:
            return a
        return _sql1(conn, "SELECT llm_add_nd(?, llm_full_nd(?, 1))", [a, b])
    m["aten.add.Scalar"] = _add_scalar

    def _sub(conn, args, kw):
        return _sql1(conn, "SELECT llm_sub_nd(?, ?)",
                     [_ensure_op_blob(args[0]), _ensure_op_blob(args[1])])
    m["aten.sub.Tensor"] = _sub

    def _mul(conn, args, kw):
        a, b = args[0], args[1]
        # Identity: mul by 1.0 → return unchanged (avoids SQL round-trip)
        if isinstance(b, (int, float)):
            if b == 1.0 or b == 1:
                return _ensure_op_blob(a)
            return _sql1(conn, "SELECT llm_scale_nd(?, ?)",
                         [_ensure_op_blob(a), float(b)])
        if isinstance(a, (int, float)):
            if a == 1.0 or a == 1:
                return _ensure_op_blob(b)
            return _sql1(conn, "SELECT llm_scale_nd(?, ?)",
                         [_ensure_op_blob(b), float(a)])
        return _sql1(conn, "SELECT llm_mul_nd(?, ?)", [a, b])
    m["aten.mul.Tensor"] = _mul

    m["aten.mul.Scalar"] = lambda c, a, k: _sql1(
        c, "SELECT llm_scale_nd(?, ?)", [a[0], float(a[1])])

    def _div(conn, args, kw):
        a, b = args[0], args[1]
        if isinstance(b, (int, float)) and b != 0:
            return _sql1(conn, "SELECT llm_scale_nd(?, ?)",
                         [_ensure_op_blob(a), 1.0 / float(b)])
        return _sql1(conn, "SELECT llm_div_nd(?, ?)",
                     [_ensure_op_blob(a), _ensure_op_blob(b)])
    m["aten.div.Tensor"] = _div

    def _div_scalar(conn, args, kw):
        divisor = float(args[1])
        if divisor != 0:
            return _sql1(conn, "SELECT llm_scale_nd(?, ?)",
                         [args[0], 1.0 / divisor])
        return _sql1(conn, "SELECT llm_div_nd(?, ?)",
                     [args[0], _ensure_op_blob(args[1])])
    m["aten.div.Scalar"] = _div_scalar

    m["aten.neg.default"] = lambda c, a, k: _sql1(
        c, "SELECT llm_neg_nd(?)", [a[0]])
    m["aten.abs.default"] = lambda c, a, k: _sql1(
        c, "SELECT llm_abs_nd(?)", [a[0]])

    m["aten.pow.Tensor_Scalar"] = lambda c, a, k: _sql1(
        c, "SELECT llm_pow_nd(?, ?)", [a[0], float(a[1])])
    m["aten.sqrt.default"] = lambda c, a, k: _sql1(
        c, "SELECT llm_sqrt_nd(?)", [a[0]])
    m["aten.rsqrt.default"] = lambda c, a, k: _sql1(
        c, "SELECT llm_rsqrt_nd(?)", [a[0]])
    m["aten.exp.default"] = lambda c, a, k: _sql1(
        c, "SELECT llm_exp_nd(?)", [a[0]])
    m["aten.log.default"] = lambda c, a, k: _sql1(
        c, "SELECT llm_log_nd(?)", [a[0]])

    # ── Trig (with result caching for position encoding) ──
    # Position encoding ops (cos/sin of position embeddings) produce the
    # same results for the same input blobs across decode steps.  Cache
    # by input blob identity to avoid redundant SQL calls.
    # Limit cache size to prevent unbounded memory growth.
    _TRIG_CACHE_MAX = 256
    _trig_cache: Dict[Tuple[str, bytes], bytes] = {}

    def _cos_cached(conn, args, kw):
        inp = args[0]
        if isinstance(inp, bytes):
            key = ("cos", inp)
            cached = _trig_cache.get(key)
            if cached is not None:
                return cached
            result = _sql1(conn, "SELECT llm_cos_nd(?)", [inp])
            if len(_trig_cache) >= _TRIG_CACHE_MAX:
                _trig_cache.clear()
            _trig_cache[key] = result
            return result
        return _sql1(conn, "SELECT llm_cos_nd(?)", [inp])
    m["aten.cos.default"] = _cos_cached

    def _sin_cached(conn, args, kw):
        inp = args[0]
        if isinstance(inp, bytes):
            key = ("sin", inp)
            cached = _trig_cache.get(key)
            if cached is not None:
                return cached
            result = _sql1(conn, "SELECT llm_sin_nd(?)", [inp])
            if len(_trig_cache) >= _TRIG_CACHE_MAX:
                _trig_cache.clear()
            _trig_cache[key] = result
            return result
        return _sql1(conn, "SELECT llm_sin_nd(?)", [inp])
    m["aten.sin.default"] = _sin_cached

    m["aten.tanh.default"] = lambda c, a, k: _sql1(
        c, "SELECT llm_tanh_nd(?)", [a[0]])

    # ── Comparison ──
    for name, fn in [("gt", "llm_gt_nd"), ("lt", "llm_lt_nd"),
                     ("ge", "llm_ge_nd"), ("le", "llm_le_nd"),
                     ("eq", "llm_eq_nd"), ("ne", "llm_ne_nd")]:
        for suffix in (".Tensor", ".Scalar"):
            tgt = f"aten.{name}{suffix}"
            sql_fn = fn
            m[tgt] = (lambda sf: lambda c, a, k: _sql1(
                c, f"SELECT {sf}(?, ?)",
                [_ensure_op_blob(a[0]), _ensure_op_blob(a[1])]))(sql_fn)

    # ── Reduction ──
    def _mean_dim(conn, args, kw):
        dim = args[1]
        if isinstance(dim, list):
            dim = dim[0]
        keepdim = int(args[2]) if len(args) > 2 else 0
        return _sql1(conn, "SELECT llm_mean_nd(?, ?, ?)",
                     [args[0], int(dim), int(keepdim)])
    m["aten.mean.dim"] = _mean_dim
    m["aten.mean.default"] = lambda c, a, k: _sql1(
        c, "SELECT llm_reduce_mean_nd(?)", [a[0]])

    def _sum_dim(conn, args, kw):
        dim = args[1]
        if isinstance(dim, list):
            dim = dim[0]
        keepdim = int(args[2]) if len(args) > 2 else 0
        return _sql1(conn, "SELECT llm_sum_nd(?, ?, ?)",
                     [args[0], int(dim), int(keepdim)])
    m["aten.sum.dim_IntList"] = _sum_dim

    # ── Linear algebra ──
    m["aten.mm.default"] = lambda c, a, k: _sql1(
        c, "SELECT llm_bmm(?, ?)", [a[0], a[1]])
    m["aten.bmm.default"] = lambda c, a, k: _sql1(
        c, "SELECT llm_bmm(?, ?)", [a[0], a[1]])
    m["aten.matmul.default"] = lambda c, a, k: _sql1(
        c, "SELECT llm_bmm(?, ?)", [a[0], a[1]])

    def _linear(conn, args, kw):
        weight = args[1]
        if isinstance(weight, str):
            if len(args) >= 3 and args[2] is not None:
                return _sql1(
                    conn,
                    "SELECT llm_linear_param_bias_nd(?, ?, ?)",
                    [args[0], _sql_text_literal(weight), args[2]],
                )
            return _sql1(
                conn,
                "SELECT llm_linear_param_nd(?, ?)",
                [args[0], _sql_text_literal(weight)],
            )
        if isinstance(weight, (bytes, memoryview)) and _is_int8_blob(weight):
            if len(args) >= 3 and args[2] is not None:
                return _sql1(conn,
                             "SELECT llm_linear_int8_bias_nd(?, ?, ?)",
                             [args[0], weight, args[2]])
            return _sql1(conn, "SELECT llm_linear_int8_nd(?, ?)",
                         [args[0], weight])
        if len(args) >= 3 and args[2] is not None:
            return _sql1(conn,
                         "SELECT llm_add_nd(llm_linear_nd(?, ?), ?)",
                         [args[0], args[1], args[2]])
        return _sql1(conn, "SELECT llm_linear_nd(?, ?)",
                     [args[0], args[1]])
    m["aten.linear.default"] = _linear

    m["aten.addmm.default"] = lambda c, a, k: _sql1(
        c, "SELECT llm_add_nd(llm_bmm(?, ?), ?)", [a[1], a[2], a[0]])

    # ── Shape manipulation ──
    # Cache SQL strings by argument count to avoid repeated f-string builds
    _sql_cache: Dict[Tuple[str, int], str] = {}

    def _cached_sql(fn_name: str, nargs: int) -> str:
        key = (fn_name, nargs)
        s = _sql_cache.get(key)
        if s is None:
            ph = ", ".join(["?"] * nargs)
            s = f"SELECT {fn_name}({ph})"
            _sql_cache[key] = s
        return s

    def _view(conn, args, kw):
        shape = args[1] if isinstance(args[1], list) else [args[1]]
        shape_ints = [int(s) for s in shape]
        # Python fast path: header-only rewrite (no SQL round-trip)
        x = args[0]
        if isinstance(x, bytes):
            result = _py_view(x, shape_ints)
            if result is not None:
                return result
        return _sql1(conn, _cached_sql("llm_view_nd", 1 + len(shape_ints)),
                     [x] + shape_ints)
    m["aten.view.default"] = _view
    m["aten._unsafe_view.default"] = _view
    m["aten.reshape.default"] = _view

    def _unsqueeze(conn, args, kw):
        x = args[0]
        dim = int(args[1])
        if isinstance(x, bytes):
            return _py_unsqueeze(x, dim)
        return _sql1(conn, "SELECT llm_unsqueeze_nd(?, ?)", [x, dim])
    m["aten.unsqueeze.default"] = _unsqueeze
    def _squeeze_dim(conn, args, kw):
        x = args[0]
        dim = int(args[1])
        if isinstance(x, bytes):
            return _py_squeeze(x, dim)
        return _sql1(conn, "SELECT llm_squeeze_nd(?, ?)", [x, dim])
    m["aten.squeeze.dim"] = _squeeze_dim
    m["aten.squeeze.default"] = lambda c, a, k: _sql1(
        c, "SELECT llm_squeeze_all_nd(?)", [a[0]])
    m["aten.transpose.int"] = lambda c, a, k: _sql1(
        c, "SELECT llm_transpose_nd(?, ?, ?)",
        [a[0], int(a[1]), int(a[2])])

    def _expand(conn, args, kw):
        size = args[1] if isinstance(args[1], list) else [args[1]]
        size_ints = [int(s) for s in size]
        # Python fast path: no-broadcast expand is header-only rewrite
        x = args[0]
        if isinstance(x, bytes):
            result = _py_expand(x, size_ints)
            if result is not None:
                return result
        return _sql1(conn, _cached_sql("llm_expand_nd", 1 + len(size_ints)),
                     [x] + size_ints)
    m["aten.expand.default"] = _expand

    def _permute(conn, args, kw):
        dims = args[1] if isinstance(args[1], list) else [args[1]]
        dims_ints = [int(d) for d in dims]
        return _sql1(conn,
                     _cached_sql("llm_permute_nd", 1 + len(dims_ints)),
                     [args[0]] + dims_ints)
    m["aten.permute.default"] = _permute

    def _slice(conn, args, kw):
        dim = int(args[1]) if len(args) > 1 else 0
        start = int(args[2]) if len(args) > 2 else 0
        end = int(args[3]) if len(args) > 3 else _INT32_MAX
        step = int(args[4]) if len(args) > 4 else 1
        if end > _INT32_MAX:
            end = _INT32_MAX
        # No-op slice: taking all elements
        if start == 0 and end >= _INT32_MAX and step == 1:
            return args[0]
        return _sql1(conn, "SELECT llm_slice_nd(?, ?, ?, ?, ?)",
                     [args[0], dim, start, end, step])
    m["aten.slice.Tensor"] = _slice

    def _select(conn, args, kw):
        dim = int(args[1])
        idx = int(args[2])
        return _sql1(conn, "SELECT llm_select_nd(?, ?, ?)",
                     [args[0], dim, idx])
    m["aten.select.int"] = _select

    def _cat(conn, args, kw):
        tensors = args[0] if isinstance(args[0], list) else [args[0]]
        dim = int(args[1]) if len(args) > 1 else 0
        # Filter empty tensors (only check bytes blobs, SqlExpr are never empty)
        non_empty = []
        for t in tensors:
            if isinstance(t, bytes):
                s = blob_shape(t)
                if any(d == 0 for d in s):
                    continue
            non_empty.append(t)
        if len(non_empty) == 0:
            return tensors[0]
        if len(non_empty) == 1:
            return non_empty[0]
        # Handle negative dim: try Python-side, fall back to C extension
        if dim < 0:
            # Find a bytes tensor to get ndim
            for t in non_empty:
                if isinstance(t, bytes):
                    ndim = len(blob_shape(t))
                    dim = dim % ndim
                    break
            # If no bytes tensor found, C extension handles negative dim
        # Use 2-arg cat_nd for exactly 2 tensors
        if len(non_empty) == 2:
            return _sql1(conn, "SELECT llm_cat_nd(?, ?, ?)",
                         [non_empty[0], non_empty[1], dim])
        return _sql1(conn,
                     _cached_sql("llm_cat_multi_nd", 1 + len(non_empty)),
                     [dim] + non_empty)
    m["aten.cat.default"] = _cat

    def _stack(conn, args, kw):
        tensors = args[0] if isinstance(args[0], list) else [args[0]]
        dim = int(args[1]) if len(args) > 1 else 0
        if dim < 0:
            ndim = len(blob_shape(tensors[0]))
            dim = dim % (ndim + 1)
        return _sql1(conn,
                     _cached_sql("llm_stack_nd", 1 + len(tensors)),
                     [dim] + list(tensors))
    m["aten.stack.default"] = _stack

    m["aten.flatten.using_ints"] = lambda c, a, k: _sql1(
        c, "SELECT llm_flatten_nd(?, ?, ?)",
        [a[0], int(a[1]), int(a[2])])

    # ── Indexing / Embedding ──
    def _embedding(conn, args, kw):
        weight = args[0]
        if isinstance(weight, str):
            return _sql1(
                conn,
                "SELECT llm_embedding_param_nd(?, ?)",
                [_sql_text_literal(weight), args[1]],
            )
        if isinstance(weight, (bytes, memoryview)) and _is_int8_blob(weight):
            return _sql1(conn, "SELECT llm_embedding_int8_nd(?, ?)",
                         [weight, args[1]])
        return _sql1(conn, "SELECT llm_embedding_nd(?, ?)",
                     [weight, args[1]])
    m["aten.embedding.default"] = _embedding
    m["aten.index_select.default"] = lambda c, a, k: _sql1(
        c, "SELECT llm_index_select_nd(?, ?, ?)",
        [a[0], int(a[1]), a[2]])
    m["aten.gather.default"] = lambda c, a, k: _sql1(
        c, "SELECT llm_gather_nd(?, ?, ?)", [a[0], int(a[1]), a[2]])

    # ── Masking / Triangular ──
    m["aten.triu.default"] = lambda c, a, k: _sql1(
        c, "SELECT llm_triu_nd(?, ?)", [a[0], int(a[1])])
    m["aten.tril.default"] = lambda c, a, k: _sql1(
        c, "SELECT llm_tril_nd(?, ?)", [a[0], int(a[1])])
    m["aten.masked_fill.Scalar"] = lambda c, a, k: _sql1(
        c, "SELECT llm_masked_fill_nd(?, ?, ?)",
        [a[0], a[1], float(a[2])])

    # ── Activations ──
    m["aten.silu.default"] = lambda c, a, k: _sql1(
        c, "SELECT llm_silu_nd(?)", [a[0]])
    m["aten.relu.default"] = lambda c, a, k: _sql1(
        c, "SELECT llm_relu_nd(?)", [a[0]])
    m["aten.gelu.default"] = lambda c, a, k: _sql1(
        c, "SELECT llm_gelu_nd(?)", [a[0]])
    m["aten.sigmoid.default"] = lambda c, a, k: _sql1(
        c, "SELECT llm_sigmoid_nd(?)", [a[0]])
    m["aten.softmax.int"] = lambda c, a, k: _sql1(
        c, "SELECT llm_softmax_nd(?, ?)", [a[0], int(a[1])])

    # ── Fused ops (encoded as target strings in exported JSON) ──
    m["llm_silu_mul_nd"] = lambda c, a, k: _sql1(
        c, "SELECT llm_silu_mul_nd(?, ?)", [a[0], a[1]])
    m["llm_rope_nd"] = lambda c, a, k: _sql1(
        c, "SELECT llm_rope_nd(?, ?, ?)", [a[0], a[1], a[2]])
    m["llm_rmsnorm_nd"] = lambda c, a, k: _sql1(
        c, "SELECT llm_rmsnorm_nd(?, ?, ?)", [a[0], a[1], float(a[2])])

    # ── Tensor construction (with caching for repeated calls) ──
    # Limit cache size to prevent unbounded memory growth.
    _ARANGE_CACHE_MAX = 256
    _arange_cache: Dict[int, bytes] = {}

    def _arange_cached(conn, args, kw):
        n = int(args[0])
        cached = _arange_cache.get(n)
        if cached is not None:
            return cached
        result = _sql1(conn, "SELECT llm_arange_nd(?)", [n])
        if len(_arange_cache) >= _ARANGE_CACHE_MAX:
            _arange_cache.clear()
        _arange_cache[n] = result
        return result
    m["aten.arange.default"] = _arange_cached

    def _full(conn, args, kw):
        size = args[0] if isinstance(args[0], list) else [args[0]]
        val = float(args[1])
        ph = ", ".join(["?"] * (1 + len(size)))
        return _sql1(conn, f"SELECT llm_full_nd({ph})",
                     [val] + [int(s) for s in size])
    m["aten.full.default"] = _full

    # ── Conditional / Clamp ──
    m["aten.where.self"] = lambda c, a, k: _sql1(
        c, "SELECT llm_where_nd(?, ?, ?)", [a[0], a[1], a[2]])
    m["aten.clamp.default"] = lambda c, a, k: _sql1(
        c, "SELECT llm_clamp_nd(?, ?, ?)", [a[0], a[1], a[2]])

    # ── Logical ──
    m["aten.logical_not.default"] = lambda c, a, k: _sql1(
        c, "SELECT llm_logical_not_nd(?)", [a[0]])

    # ── SDPA ──
    def _sdpa(conn, args, kw):
        q, k_t, v = args[0], args[1], args[2]
        attn_mask = args[3] if len(args) > 3 and args[3] is not None else None
        scale = kw.get("scale")
        if scale is None:
            # Default scale
            head_dim = blob_shape(q)[-1]
            scale = 1.0 / (head_dim ** 0.5)
        else:
            scale = float(scale)
        is_causal = kw.get("is_causal", False)

        # Fused C path: single SQL call for all cases
        if attn_mask is None:
            return _sql1(conn,
                         "SELECT llm_sdpa_nd(?, ?, ?, ?, ?)",
                         [q, k_t, v, float(scale),
                          1 if is_causal else 0])
        # With boolean mask: pass as 6th argument
        return _sql1(conn,
                     "SELECT llm_sdpa_nd(?, ?, ?, ?, ?, ?)",
                     [q, k_t, v, float(scale),
                      1 if is_causal else 0, attn_mask])
    m["aten.scaled_dot_product_attention.default"] = _sdpa

    # ── Scalar ops ──
    def _sym_size(conn, args, kw):
        """Return tensor dimension size, computed from blob header in Python."""
        x = args[0]
        dim = int(args[1])
        if isinstance(x, bytes):
            shape = blob_shape(x)
            if dim < 0:
                dim += len(shape)
            return shape[dim]
        # Fallback to SQL for non-bytes (SqlExpr)
        return _sql1(conn, "SELECT llm_shape(?, ?)", [x, dim])
    m["aten.sym_size.int"] = _sym_size

    # ── Python operator (scalar arithmetic for dynamic shapes) ──
    m["<built-in function add>"] = lambda c, a, k: int(a[0]) + int(a[1])
    m["<built-in function mul>"] = lambda c, a, k: int(a[0]) * int(a[1])

    # ── Repeat ──
    def _repeat(conn, args, kw):
        repeats = args[1] if isinstance(args[1], list) else [args[1]]
        repeats_ints = [int(r) for r in repeats]
        return _sql1(conn,
                     _cached_sql("llm_repeat_nd", 1 + len(repeats_ints)),
                     [args[0]] + repeats_ints)
    m["aten.repeat.default"] = _repeat

    # ── Repeat KV (fused unsqueeze→expand→reshape for GQA) ──
    m["llm_repeat_kv_nd"] = lambda c, a, k: _sql1(
        c, "SELECT llm_repeat_kv_nd(?, ?)", [a[0], int(a[1])])

    # ── Fused view + transpose(1,2) for QKV head reshape ──
    m["llm_view_t12_nd"] = lambda c, a, k: _sql1(
        c, _cached_sql("llm_view_t12_nd", 1 + len(a) - 1),
        [a[0]] + [int(s) for s in a[1:]])

    return m


# ---------------------------------------------------------------------------
# Single-SQL graph compiler
# ---------------------------------------------------------------------------


def _sql_quote(value: str) -> str:
    """Return *value* quoted for safe embedding in a SQL string literal."""
    return value.replace("'", "''")


def _compiled_cte_name(name: str) -> str:
    """Convert an instruction name into a valid SQL CTE identifier."""
    cleaned = _re.sub(r"[^A-Za-z0-9_]", "_", name)
    return f"n_{cleaned}"


_COMPILED_PARAM_CTE = "n_model_params_cache"


def _compiled_param_expr(name: str) -> str:
    """Return an inline subquery that fetches a parameter BLOB by name."""
    return (
        "(SELECT data FROM model_params "
        f"WHERE name = '{_sql_quote(name)}')"
    )


def _compiled_resolve_alias(name, alias_map: Dict[str, Any]):
    """Follow alias chains until the canonical source value is reached."""
    while isinstance(name, str) and name in alias_map:
        name = alias_map[name]
    return name


def _compiled_arg_to_sql(arg, cte_set: set, param_set: set) -> str:
    """Convert an instruction argument into its SQL expression form."""
    if isinstance(arg, str):
        if arg in cte_set:
            return f"(SELECT v FROM {_compiled_cte_name(arg)})"
        if arg in param_set:
            return f"(SELECT v FROM {_compiled_cte_name(arg)})"
        return f"(SELECT v FROM {_compiled_cte_name(arg)})"
    if isinstance(arg, bool):
        return str(int(arg))
    if isinstance(arg, (int, float)):
        return str(arg)
    if arg is None:
        return "NULL"
    return str(arg)


def _compiled_resolve_arg_list(args, cte_set, param_set, alias_map):
    """Resolve a positional argument list into SQL expressions."""
    result = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            result.append([
                _compiled_arg_to_sql(
                    _compiled_resolve_alias(item, alias_map),
                    cte_set,
                    param_set,
                )
                for item in arg
            ])
        else:
            result.append(
                _compiled_arg_to_sql(
                    _compiled_resolve_alias(arg, alias_map),
                    cte_set,
                    param_set,
                )
            )
    return result


def _compiled_resolve_kw_val(value, cte_set, param_set, alias_map):
    """Resolve a keyword argument value into its SQL expression."""
    if isinstance(value, str):
        value = _compiled_resolve_alias(value, alias_map)
    return _compiled_arg_to_sql(value, cte_set, param_set)


def _compiled_is_literal_scalar(sql_expr: str) -> bool:
    """Return True when *sql_expr* is a bare numeric literal."""
    try:
        float(sql_expr)
        return True
    except (TypeError, ValueError):
        return False


def _compiled_ensure_blob(sql_expr: str) -> str:
    """Wrap a scalar SQL expression in ``llm_full_nd`` when needed."""
    if _compiled_is_literal_scalar(sql_expr):
        return f"llm_full_nd({sql_expr}, 1)"
    return sql_expr


_COMPILED_SQL_FRAGMENTS: Dict[str, object] = {}


def _build_compiled_sql_fragments() -> Dict[str, object]:
    """Build the target→SQL-fragment mapping for single-statement execution."""
    fragments: Dict[str, object] = {}

    def _frag_add(args, kw):
        a, b = args[0], args[1]
        alpha = kw.get("alpha", 1)
        b_expr = _compiled_ensure_blob(b)
        if alpha != 1:
            return f"llm_add_nd({a}, llm_scale_nd({b_expr}, {alpha}))"
        return f"llm_add_nd({a}, {b_expr})"

    def _frag_mul(args, kw):
        a, b = args[0], args[1]
        if _compiled_is_literal_scalar(b):
            return f"llm_scale_nd({a}, {b})"
        if _compiled_is_literal_scalar(a):
            return f"llm_scale_nd({b}, {a})"
        return f"llm_mul_nd({a}, {b})"

    def _frag_sub(args, kw):
        a, b = args[0], args[1]
        return f"llm_sub_nd({a}, {_compiled_ensure_blob(b)})"

    def _frag_div(args, kw):
        a, b = args[0], args[1]
        if _compiled_is_literal_scalar(b):
            return f"llm_scale_nd({a}, 1.0 / {b})"
        return f"llm_div_nd({a}, {b})"

    def _frag_mean_dim(args, kw):
        dim = args[1]
        if isinstance(dim, list):
            dim = dim[0]
        keepdim = args[2] if len(args) > 2 else 0
        return f"llm_mean_nd({args[0]}, {dim}, {keepdim})"

    def _frag_sum_dim(args, kw):
        dim = args[1]
        if isinstance(dim, list):
            dim = dim[0]
        keepdim = args[2] if len(args) > 2 else 0
        return f"llm_sum_nd({args[0]}, {dim}, {keepdim})"

    def _frag_linear(args, kw):
        x_expr = args[0]
        w_expr = args[1]
        if len(args) >= 3 and args[2] != "NULL":
            bias_expr = args[2]
            return (
                "CASE "
                f"WHEN llm_is_int8_nd({w_expr}) "
                f"THEN llm_linear_int8_bias_nd({x_expr}, {w_expr}, {bias_expr}) "
                f"ELSE llm_add_nd(llm_linear_nd({x_expr}, {w_expr}), {bias_expr}) "
                "END"
            )
        return (
            "CASE "
            f"WHEN llm_is_int8_nd({w_expr}) "
            f"THEN llm_linear_int8_nd({x_expr}, {w_expr}) "
            f"ELSE llm_linear_nd({x_expr}, {w_expr}) "
            "END"
        )

    def _frag_view(args, kw):
        shape = args[1] if isinstance(args[1], list) else [args[1]]
        parts = ", ".join([args[0]] + [str(size) for size in shape])
        return f"llm_view_nd({parts})"

    def _frag_expand(args, kw):
        size = args[1] if isinstance(args[1], list) else [args[1]]
        parts = ", ".join([args[0]] + [str(dim) for dim in size])
        return f"llm_expand_nd({parts})"

    def _frag_permute(args, kw):
        dims = args[1] if isinstance(args[1], list) else [args[1]]
        parts = ", ".join([args[0]] + [str(dim) for dim in dims])
        return f"llm_permute_nd({parts})"

    def _frag_slice(args, kw):
        dim = args[1] if len(args) > 1 else 0
        start = args[2] if len(args) > 2 else 0
        end = args[3] if len(args) > 3 else _INT32_MAX
        step = args[4] if len(args) > 4 else 1
        try:
            end_val = int(end)
            if end_val > _INT32_MAX:
                end = _INT32_MAX
        except (TypeError, ValueError):
            pass
        return f"llm_slice_nd({args[0]}, {dim}, {start}, {end}, {step})"

    def _frag_select(args, kw):
        return f"llm_select_nd({args[0]}, {args[1]}, {args[2]})"

    def _frag_cat(args, kw):
        tensors = args[0] if isinstance(args[0], list) else [args[0]]
        dim = args[1] if len(args) > 1 else 0
        if len(tensors) == 1:
            return tensors[0]
        if len(tensors) == 2:
            a_expr, b_expr = tensors
            return (
                "CASE "
                f"WHEN llm_shape({a_expr}, {dim}) = 0 THEN {b_expr} "
                f"WHEN llm_shape({b_expr}, {dim}) = 0 THEN {a_expr} "
                f"ELSE llm_cat_nd({a_expr}, {b_expr}, {dim}) "
                "END"
            )
        parts = ", ".join([str(dim)] + tensors)
        return f"llm_cat_multi_nd({parts})"

    def _frag_stack(args, kw):
        tensors = args[0] if isinstance(args[0], list) else [args[0]]
        dim = args[1] if len(args) > 1 else 0
        parts = ", ".join([str(dim)] + tensors)
        return f"llm_stack_nd({parts})"

    def _frag_embedding(args, kw):
        weight_expr = args[0]
        index_expr = args[1]
        return (
            "CASE "
            f"WHEN llm_is_int8_nd({weight_expr}) "
            f"THEN llm_embedding_int8_nd({weight_expr}, {index_expr}) "
            f"ELSE llm_embedding_nd({weight_expr}, {index_expr}) "
            "END"
        )

    def _frag_sdpa(args, kw):
        scale = kw.get("scale", "NULL")
        is_causal = kw.get("is_causal", False)
        attn_mask = args[3] if len(args) > 3 and args[3] != "NULL" else None

        if not is_causal and attn_mask is None:
            return (
                f"llm_bmm(llm_softmax_nd(llm_scale_nd("
                f"llm_bmm({args[0]}, llm_transpose_nd({args[1]}, -2, -1)), "
                f"{scale}), -1), {args[2]})"
            )

        if is_causal and attn_mask is None:
            return (
                f"llm_bmm(llm_softmax_nd(llm_add_nd("
                f"llm_scale_nd(llm_bmm({args[0]}, "
                f"llm_transpose_nd({args[1]}, -2, -1)), {scale}), "
                f"llm_masked_fill_nd(llm_full_nd(0.0, "
                f"llm_shape({args[0]}, -2), llm_shape({args[1]}, -2)), "
                f"llm_triu_nd(llm_full_nd(1.0, "
                f"llm_shape({args[0]}, -2), llm_shape({args[1]}, -2)), 1), "
                f"{_NEG_INF_MASK})), -1), {args[2]})"
            )

        if attn_mask is not None and not is_causal:
            return (
                f"llm_bmm(llm_softmax_nd(llm_add_nd("
                f"llm_scale_nd(llm_bmm({args[0]}, "
                f"llm_transpose_nd({args[1]}, -2, -1)), {scale}), "
                f"llm_bool_to_additive_mask_nd({attn_mask}, 1e9)), -1), "
                f"{args[2]})"
            )

        return (
            f"llm_bmm(llm_softmax_nd(llm_add_nd(llm_add_nd("
            f"llm_scale_nd(llm_bmm({args[0]}, "
            f"llm_transpose_nd({args[1]}, -2, -1)), {scale}), "
            f"llm_masked_fill_nd(llm_full_nd(0.0, "
            f"llm_shape({args[0]}, -2), llm_shape({args[1]}, -2)), "
            f"llm_triu_nd(llm_full_nd(1.0, "
            f"llm_shape({args[0]}, -2), llm_shape({args[1]}, -2)), 1), "
            f"{_NEG_INF_MASK})), llm_bool_to_additive_mask_nd({attn_mask}, 1e9)"
            f"), -1), {args[2]})"
        )

    def _frag_full(args, kw):
        size = args[0] if isinstance(args[0], list) else [args[0]]
        parts = ", ".join([str(args[1])] + [str(dim) for dim in size])
        return f"llm_full_nd({parts})"

    def _frag_repeat(args, kw):
        repeats = args[1] if isinstance(args[1], list) else [args[1]]
        parts = ", ".join([args[0]] + [str(rep) for rep in repeats])
        return f"llm_repeat_nd({parts})"

    fragments["aten.add.Tensor"] = _frag_add
    fragments["aten.add.Scalar"] = (
        lambda args, kw: f"llm_add_nd({args[0]}, llm_full_nd({args[1]}, 1))"
    )
    fragments["aten.sub.Tensor"] = _frag_sub
    fragments["aten.sub.Scalar"] = (
        lambda args, kw: f"llm_sub_nd({args[0]}, llm_full_nd({args[1]}, 1))"
    )
    fragments["aten.mul.Tensor"] = _frag_mul
    fragments["aten.mul.Scalar"] = (
        lambda args, kw: f"llm_scale_nd({args[0]}, {args[1]})"
    )
    fragments["aten.div.Tensor"] = _frag_div
    fragments["aten.div.Scalar"] = (
        lambda args, kw: f"llm_scale_nd({args[0]}, 1.0 / {args[1]})"
    )
    fragments["aten.neg.default"] = lambda args, kw: f"llm_neg_nd({args[0]})"
    fragments["aten.abs.default"] = lambda args, kw: f"llm_abs_nd({args[0]})"
    fragments["aten.pow.Tensor_Scalar"] = (
        lambda args, kw: f"llm_pow_nd({args[0]}, {args[1]})"
    )
    fragments["aten.sqrt.default"] = lambda args, kw: f"llm_sqrt_nd({args[0]})"
    fragments["aten.rsqrt.default"] = lambda args, kw: f"llm_rsqrt_nd({args[0]})"
    fragments["aten.exp.default"] = lambda args, kw: f"llm_exp_nd({args[0]})"
    fragments["aten.log.default"] = lambda args, kw: f"llm_log_nd({args[0]})"
    fragments["aten.sin.default"] = lambda args, kw: f"llm_sin_nd({args[0]})"
    fragments["aten.cos.default"] = lambda args, kw: f"llm_cos_nd({args[0]})"
    fragments["aten.tanh.default"] = lambda args, kw: f"llm_tanh_nd({args[0]})"

    for target_name, sql_fn in (
        ("aten.gt.Tensor", "llm_gt_nd"),
        ("aten.gt.Scalar", "llm_gt_nd"),
        ("aten.lt.Tensor", "llm_lt_nd"),
        ("aten.lt.Scalar", "llm_lt_nd"),
        ("aten.ge.Tensor", "llm_ge_nd"),
        ("aten.ge.Scalar", "llm_ge_nd"),
        ("aten.le.Tensor", "llm_le_nd"),
        ("aten.le.Scalar", "llm_le_nd"),
        ("aten.eq.Tensor", "llm_eq_nd"),
        ("aten.eq.Scalar", "llm_eq_nd"),
        ("aten.ne.Tensor", "llm_ne_nd"),
        ("aten.ne.Scalar", "llm_ne_nd"),
    ):
        fragments[target_name] = (
            lambda sql_name: (
                lambda args, kw: f"{sql_name}({_compiled_ensure_blob(args[0])}, {_compiled_ensure_blob(args[1])})"
            )
        )(sql_fn)

    fragments["aten.mean.dim"] = _frag_mean_dim
    fragments["aten.mean.default"] = (
        lambda args, kw: f"llm_reduce_mean_nd({args[0]})"
    )
    fragments["aten.sum.dim_IntList"] = _frag_sum_dim
    fragments["aten.sum.default"] = (
        lambda args, kw: f"llm_reduce_sum_nd({args[0]})"
    )

    fragments["aten.mm.default"] = lambda args, kw: f"llm_bmm({args[0]}, {args[1]})"
    fragments["aten.bmm.default"] = lambda args, kw: f"llm_bmm({args[0]}, {args[1]})"
    fragments["aten.matmul.default"] = lambda args, kw: f"llm_bmm({args[0]}, {args[1]})"
    fragments["aten.linear.default"] = _frag_linear
    fragments["aten.addmm.default"] = (
        lambda args, kw: f"llm_add_nd(llm_bmm({args[1]}, {args[2]}), {args[0]})"
    )

    fragments["aten.view.default"] = _frag_view
    fragments["aten._unsafe_view.default"] = _frag_view
    fragments["aten.reshape.default"] = _frag_view
    fragments["aten.unsqueeze.default"] = (
        lambda args, kw: f"llm_unsqueeze_nd({args[0]}, {args[1]})"
    )
    fragments["aten.squeeze.dim"] = (
        lambda args, kw: f"llm_squeeze_nd({args[0]}, {args[1]})"
    )
    fragments["aten.squeeze.default"] = (
        lambda args, kw: f"llm_squeeze_all_nd({args[0]})"
    )
    fragments["aten.transpose.int"] = (
        lambda args, kw: f"llm_transpose_nd({args[0]}, {args[1]}, {args[2]})"
    )
    fragments["aten.expand.default"] = _frag_expand
    fragments["aten.permute.default"] = _frag_permute
    fragments["aten.slice.Tensor"] = _frag_slice
    fragments["aten.select.int"] = _frag_select
    fragments["aten.cat.default"] = _frag_cat
    fragments["aten.stack.default"] = _frag_stack
    fragments["aten.flatten.using_ints"] = (
        lambda args, kw: f"llm_flatten_nd({args[0]}, {args[1]}, {args[2]})"
    )

    fragments["aten.embedding.default"] = _frag_embedding
    fragments["aten.index_select.default"] = (
        lambda args, kw: f"llm_index_select_nd({args[0]}, {args[1]}, {args[2]})"
    )
    fragments["aten.gather.default"] = (
        lambda args, kw: f"llm_gather_nd({args[0]}, {args[1]}, {args[2]})"
    )

    fragments["aten.triu.default"] = (
        lambda args, kw: f"llm_triu_nd({args[0]}, {args[1]})"
    )
    fragments["aten.tril.default"] = (
        lambda args, kw: f"llm_tril_nd({args[0]}, {args[1]})"
    )
    fragments["aten.masked_fill.Scalar"] = (
        lambda args, kw: f"llm_masked_fill_nd({args[0]}, {args[1]}, {args[2]})"
    )

    fragments["aten.silu.default"] = lambda args, kw: f"llm_silu_nd({args[0]})"
    fragments["aten.relu.default"] = lambda args, kw: f"llm_relu_nd({args[0]})"
    fragments["aten.gelu.default"] = lambda args, kw: f"llm_gelu_nd({args[0]})"
    fragments["aten.sigmoid.default"] = (
        lambda args, kw: f"llm_sigmoid_nd({args[0]})"
    )
    fragments["aten.softmax.int"] = (
        lambda args, kw: f"llm_softmax_nd({args[0]}, {args[1]})"
    )
    fragments["llm_silu_mul_nd"] = (
        lambda args, kw: f"llm_silu_mul_nd({args[0]}, {args[1]})"
    )
    fragments["llm_rope_nd"] = (
        lambda args, kw: f"llm_rope_nd({args[0]}, {args[1]}, {args[2]})"
    )
    fragments["llm_rmsnorm_nd"] = (
        lambda args, kw: f"llm_rmsnorm_nd({args[0]}, {args[1]}, {args[2]})"
    )
    fragments["llm_repeat_kv_nd"] = (
        lambda args, kw: f"llm_repeat_kv_nd({args[0]}, {args[1]})"
    )
    fragments["llm_view_t12_nd"] = (
        lambda args, kw: f"llm_view_t12_nd({', '.join(str(part) for part in args)})"
    )

    fragments["aten.arange.default"] = (
        lambda args, kw: f"llm_arange_nd({args[0]})"
    )
    fragments["aten.full.default"] = _frag_full
    fragments["aten.where.self"] = (
        lambda args, kw: f"llm_where_nd({args[0]}, {args[1]}, {args[2]})"
    )
    fragments["aten.clamp.default"] = (
        lambda args, kw: f"llm_clamp_nd({args[0]}, {args[1]}, {args[2]})"
    )

    fragments["aten.scaled_dot_product_attention.default"] = _frag_sdpa
    fragments["aten.sym_size.int"] = (
        lambda args, kw: f"llm_shape({args[0]}, {args[1]})"
    )
    fragments["<built-in function add>"] = (
        lambda args, kw: f"({args[0]} + {args[1]})"
    )
    fragments["<built-in function mul>"] = (
        lambda args, kw: f"({args[0]} * {args[1]})"
    )
    fragments["aten.logical_not.default"] = (
        lambda args, kw: f"llm_logical_not_nd({args[0]})"
    )
    fragments["aten.logical_and.default"] = (
        lambda args, kw: f"llm_logical_and_nd({args[0]}, {args[1]})"
    )
    fragments["aten.repeat.default"] = _frag_repeat

    return fragments


def _get_compiled_sql_fragments() -> Dict[str, object]:
    """Return the lazily built compiled-fragment mapping."""
    global _COMPILED_SQL_FRAGMENTS
    if not _COMPILED_SQL_FRAGMENTS:
        _COMPILED_SQL_FRAGMENTS = _build_compiled_sql_fragments()
    return _COMPILED_SQL_FRAGMENTS


def _compiled_expr_arg(arg, env_sql: Dict[str, str], param_names: set) -> str:
    """Resolve an argument into a nested SQL expression."""
    if isinstance(arg, str):
        if arg in env_sql:
            return env_sql[arg]
        if arg in param_names:
            return f"llm_get_param('{_sql_quote(arg)}')"
        return arg
    if isinstance(arg, bool):
        return str(int(arg))
    if isinstance(arg, (int, float)):
        return str(arg)
    if arg is None:
        return "NULL"
    return str(arg)


def _compiled_expr_arg_list(args, env_sql: Dict[str, str], param_names: set):
    """Resolve a positional argument list into nested SQL expressions."""
    result = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            result.append([
                _compiled_expr_arg(item, env_sql, param_names)
                for item in arg
            ])
        else:
            result.append(_compiled_expr_arg(arg, env_sql, param_names))
    return result


def _build_compiled_forward_sql(
    instrs: list,
    param_names: set,
) -> Tuple[str, List[str]]:
    """Compile an instruction list into one SQL statement.

    Runtime inputs (``input_ids`` and decode-time KV cache blobs) become
    ``VALUES(?)`` CTEs.  Model parameters are fetched inline from the
    ``model_params`` table so Python never binds parameter BLOBs.
    """
    fragments = _get_compiled_sql_fragments()

    bind_names: List[str] = []
    cte_clauses: List[str] = []
    env_sql: Dict[str, str] = {}
    output_cols: List[str] = []

    for instr in instrs:
        if instr.get("op") != "placeholder":
            continue
        name = instr["name"]
        if name in param_names:
            env_sql[name] = f"llm_get_param('{_sql_quote(name)}')"
            continue
        cte_name = _compiled_cte_name(name)
        cte_clauses.append(f"{cte_name}(v) AS (VALUES(?))")
        bind_names.append(name)
        env_sql[name] = f"(SELECT v FROM {cte_name})"

    for instr in instrs:
        name = instr.get("name")
        op = instr.get("op")
        if op == "placeholder" or op == "get_attr" or op == "subgraph_call":
            continue
        if op == "alias":
            env_sql[name] = _compiled_expr_arg(
                instr.get("target"),
                env_sql,
                param_names,
            )
            continue
        if op == "output":
            for output_arg in instr.get("output_args", []):
                output_cols.append(
                    _compiled_expr_arg(output_arg, env_sql, param_names)
                )
            continue
        if op != "call_function":
            continue

        target = instr.get("target")
        fragment_builder = fragments.get(target)
        if fragment_builder is None:
            raise RuntimeError(
                f"single_sql_mode does not support target {target!r}"
            )

        arg_sqls = _compiled_expr_arg_list(
            instr.get("args", []),
            env_sql,
            param_names,
        )
        kw_sqls = {
            key: _compiled_expr_arg(value, env_sql, param_names)
            for key, value in instr.get("kwargs", {}).items()
        }
        env_sql[name] = fragment_builder(arg_sqls, kw_sqls)

    if not output_cols:
        raise RuntimeError("single_sql_mode graph has no output node")

    select_sql = "SELECT " + ", ".join(output_cols)
    if cte_clauses:
        sql = "WITH\n  " + ",\n  ".join(cte_clauses) + "\n" + select_sql
    else:
        sql = select_sql
    return sql, bind_names


# ---------------------------------------------------------------------------
# Lazy shape-op handlers (return SqlExpr for SQL nesting)
# ---------------------------------------------------------------------------


def _lazy_unsqueeze(conn, args, kw):
    """Return ``SqlExpr`` for ``llm_unsqueeze_nd(x, dim)``."""
    return _sql1_lazy("llm_unsqueeze_nd(?, ?)", [args[0], int(args[1])])


def _lazy_view(conn, args, kw):
    """Return ``SqlExpr`` for ``llm_view_nd(x, d0, d1, …)``."""
    shape = args[1] if isinstance(args[1], list) else [args[1]]
    shape_ints = [int(s) for s in shape]
    ph = ", ".join(["?"] * (1 + len(shape_ints)))
    return _sql1_lazy(f"llm_view_nd({ph})", [args[0]] + shape_ints)


def _lazy_reshape(conn, args, kw):
    """Return ``SqlExpr`` for reshape (same SQL as view)."""
    return _lazy_view(conn, args, kw)


def _lazy_expand(conn, args, kw):
    """Return ``SqlExpr`` for ``llm_expand_nd(x, s0, s1, …)``."""
    size = args[1] if isinstance(args[1], list) else [args[1]]
    size_ints = [int(s) for s in size]
    ph = ", ".join(["?"] * (1 + len(size_ints)))
    return _sql1_lazy(f"llm_expand_nd({ph})", [args[0]] + size_ints)


def _lazy_transpose(conn, args, kw):
    """Return ``SqlExpr`` for ``llm_transpose_nd(x, d0, d1)``."""
    return _sql1_lazy("llm_transpose_nd(?, ?, ?)",
                      [args[0], int(args[1]), int(args[2])])


def _lazy_slice(conn, args, kw):
    """Hybrid handler: no-op pass-through or SqlExpr for deferred."""
    dim = int(args[1]) if len(args) > 1 else 0
    start = int(args[2]) if len(args) > 2 else 0
    end = int(args[3]) if len(args) > 3 else _INT32_MAX
    step = int(args[4]) if len(args) > 4 else 1
    if end > _INT32_MAX:
        end = _INT32_MAX
    # No-op slice: taking all elements
    if start == 0 and end >= _INT32_MAX and step == 1:
        return args[0]
    return _sql1_lazy("llm_slice_nd(?, ?, ?, ?, ?)",
                      [args[0], dim, start, end, step])


def _lazy_permute(conn, args, kw):
    """Return ``SqlExpr`` for ``llm_permute_nd(x, d0, d1, …)``."""
    dims = args[1] if isinstance(args[1], list) else [args[1]]
    dims_ints = [int(d) for d in dims]
    ph = ", ".join(["?"] * (1 + len(dims_ints)))
    return _sql1_lazy(f"llm_permute_nd({ph})", [args[0]] + dims_ints)


def _lazy_squeeze(conn, args, kw):
    """Return ``SqlExpr`` for ``llm_squeeze_nd(x, dim)``."""
    return _sql1_lazy("llm_squeeze_nd(?, ?)", [args[0], int(args[1])])


def _lazy_view_t12(conn, args, kw):
    """Return ``SqlExpr`` for fused view + transpose(1,2)."""
    shape = args[1:] if len(args) > 1 else []
    shape_ints = [int(s) for s in shape]
    ph = ", ".join(["?"] * (1 + len(shape_ints)))
    return _sql1_lazy(f"llm_view_t12_nd({ph})", [args[0]] + shape_ints)


# Target string → lazy handler, used by _fuse_inline_shape_ops
_LAZY_SHAPE_OPS: Dict[str, object] = {
    "aten.unsqueeze.default":    _lazy_unsqueeze,
    "aten.view.default":         _lazy_view,
    "aten._unsafe_view.default": _lazy_view,
    "aten.reshape.default":      _lazy_reshape,
    "aten.expand.default":       _lazy_expand,
    "aten.transpose.int":        _lazy_transpose,
    "aten.slice.Tensor":         _lazy_slice,
    "aten.permute.default":      _lazy_permute,
    "aten.squeeze.dim":          _lazy_squeeze,
    "llm_view_t12_nd":           _lazy_view_t12,
}


def _precompile_lazy(handler_name: str, args: list):
    """Pre-build SQL fragment and static params for a lazy shape op.

    Returns ``(sql_fragment, static_params)`` or ``None`` if the op
    cannot be precompiled (e.g., dynamic args that are string refs).
    """
    def _is_static(v):
        return isinstance(v, (int, float))

    if handler_name == '_lazy_unsqueeze':
        if not _is_static(args[1]):
            return None
        dim = int(args[1])
        return ("llm_unsqueeze_nd(?, ?)", [dim])
    if handler_name in ('_lazy_view', '_lazy_reshape'):
        shape = args[1] if isinstance(args[1], list) else [args[1]]
        if not all(_is_static(s) for s in shape):
            return None  # dynamic shape (e.g., sym_size ref)
        shape_ints = [int(s) for s in shape]
        ph = ", ".join(["?"] * (1 + len(shape_ints)))
        return (f"llm_view_nd({ph})", shape_ints)
    if handler_name == '_lazy_expand':
        size = args[1] if isinstance(args[1], list) else [args[1]]
        if not all(_is_static(s) for s in size):
            return None  # dynamic expand size
        size_ints = [int(s) for s in size]
        ph = ", ".join(["?"] * (1 + len(size_ints)))
        return (f"llm_expand_nd({ph})", size_ints)
    if handler_name == '_lazy_transpose':
        if not (_is_static(args[1]) and _is_static(args[2])):
            return None
        d0, d1 = int(args[1]), int(args[2])
        return ("llm_transpose_nd(?, ?, ?)", [d0, d1])
    if handler_name == '_lazy_slice':
        dim = int(args[1]) if len(args) > 1 and _is_static(args[1]) else None
        if dim is None:
            return None
        start = int(args[2]) if len(args) > 2 else 0
        end = int(args[3]) if len(args) > 3 else _INT32_MAX
        step = int(args[4]) if len(args) > 4 else 1
        if end > _INT32_MAX:
            end = _INT32_MAX
        return ("llm_slice_nd(?, ?, ?, ?, ?)", [dim, start, end, step])
    if handler_name == '_lazy_permute':
        dims = args[1] if isinstance(args[1], list) else [args[1]]
        if not all(_is_static(d) for d in dims):
            return None
        dims_ints = [int(d) for d in dims]
        ph = ", ".join(["?"] * (1 + len(dims_ints)))
        return (f"llm_permute_nd({ph})", dims_ints)
    if handler_name == '_lazy_squeeze':
        if not _is_static(args[1]):
            return None
        dim = int(args[1])
        return ("llm_squeeze_nd(?, ?)", [dim])
    if handler_name == '_lazy_view_t12':
        shape = args[1:] if len(args) > 1 else []
        if not all(_is_static(s) for s in shape):
            return None
        shape_ints = [int(s) for s in shape]
        ph = ", ".join(["?"] * (1 + len(shape_ints)))
        return (f"llm_view_t12_nd({ph})", shape_ints)
    return None


def _eliminate_identities(instrs: list) -> None:
    """Compile-time elimination of identity ops and dead code.

    - ``mul(x, 1.0)`` / ``mul(1.0, x)`` → alias to x
    - ``add(x, 0)`` → alias to x
    - Dead code (ref_count=0) → disable handler

    Converting these to alias ops avoids handler dispatch entirely.
    """
    # Build ref counts
    ref_counts: Dict[str, int] = {}
    for instr in instrs:
        for arg_val in _flat_string_refs(instr):
            ref_counts[arg_val] = ref_counts.get(arg_val, 0) + 1

    for instr in instrs:
        if instr["op"] != "call_function":
            continue
        target = instr.get("target", "")
        args = instr.get("args", [])
        name = instr["name"]

        # Dead code: never referenced by any other instruction
        if ref_counts.get(name, 0) == 0:
            instr["_handler"] = None
            continue

        # Identity: mul(x, 1.0) or mul(1.0, x) → alias
        if target == "aten.mul.Tensor" and len(args) >= 2:
            a0, a1 = args[0], args[1]
            if isinstance(a1, (int, float)) and (a1 == 1.0 or a1 == 1):
                instr["op"] = "alias"
                instr["_op"] = "alias"
                instr["target"] = a0
                instr["_alias_target"] = a0
                instr["_handler"] = None
                continue
            if isinstance(a0, (int, float)) and (a0 == 1.0 or a0 == 1):
                instr["op"] = "alias"
                instr["_op"] = "alias"
                instr["target"] = a1
                instr["_alias_target"] = a1
                instr["_handler"] = None
                continue

        # Identity: add(x, 0) → alias
        if target == "aten.add.Tensor" and len(args) >= 2:
            a0, a1 = args[0], args[1]
            if isinstance(a1, (int, float)) and a1 == 0:
                instr["op"] = "alias"
                instr["_op"] = "alias"
                instr["target"] = a0
                instr["_alias_target"] = a0
                instr["_handler"] = None
                continue

        # No-op slice: slice(x, dim, 0, INT_MAX[, 1]) → alias
        if target == "aten.slice.Tensor" and len(args) >= 2:
            _start = args[2] if len(args) > 2 else 0
            _end = args[3] if len(args) > 3 else _INT32_MAX
            _step = args[4] if len(args) > 4 else 1
            if (isinstance(_start, (int, float)) and int(_start) == 0
                    and isinstance(_end, (int, float))
                    and int(_end) >= _INT32_MAX
                    and isinstance(_step, (int, float)) and int(_step) == 1):
                instr["op"] = "alias"
                instr["_op"] = "alias"
                instr["target"] = args[0]
                instr["_alias_target"] = args[0]
                instr["_handler"] = None
                continue


def _eliminate_common_subexpressions(instrs: list) -> None:
    """Compile-time Common Subexpression Elimination (CSE).

    Instructions with identical ``(target, args)`` produce the same
    result.  The second occurrence is converted to an alias for the first,
    eliminating the redundant SQL call.

    Typical savings:
    - Repeated ``arange(1)`` calls (4→1 in decode step)
    - Repeated ``unsqueeze(cos, 1)`` across layers
    """
    # Resolve aliases: map name → canonical name
    alias_map: Dict[str, str] = {}
    for instr in instrs:
        if instr["op"] == "alias":
            target_name = instr.get("_alias_target")
            if target_name:
                # Follow alias chain
                while target_name in alias_map:
                    target_name = alias_map[target_name]
                alias_map[instr["name"]] = target_name

    def _canonical(arg):
        """Resolve an arg through alias chain."""
        if isinstance(arg, str):
            return alias_map.get(arg, arg)
        if isinstance(arg, list):
            return tuple(_canonical(a) for a in arg)
        return arg

    # Side-effect-free ops that are safe for CSE.
    # Exclude: cat (appends to KV cache → different inputs each step)
    # Exclude: linear, matmul (depends on runtime input)
    # Include: pure shape ops, arange, cos, sin, le, embedding
    _CSE_SAFE_TARGETS = {
        "aten.arange.default",
        "aten.unsqueeze.default",
        "aten.view.default",
        "aten._unsafe_view.default",
        "aten.reshape.default",
        "aten.expand.default",
        "aten.transpose.int",
        "aten.slice.Tensor",
        "aten.permute.default",
        "aten.squeeze.dim",
        "aten.squeeze.default",
        "<built-in function add>",
        "<built-in function mul>",
    }

    seen: Dict[tuple, str] = {}  # canonical_key → first instruction name

    for instr in instrs:
        if instr["op"] != "call_function":
            continue
        handler = instr.get("_handler")
        if handler is None:
            continue
        target = instr.get("target", "")
        if target not in _CSE_SAFE_TARGETS:
            continue
        args = instr.get("args", [])
        # Build canonical key
        key = (target, tuple(_canonical(a) for a in args))
        if key in seen:
            # This instruction produces the same result as a previous one
            first_name = seen[key]
            instr["op"] = "alias"
            instr["_op"] = "alias"
            instr["target"] = first_name
            instr["_alias_target"] = first_name
            instr["_handler"] = None
            alias_map[instr["name"]] = first_name
        else:
            seen[key] = instr["name"]


def _eliminate_decode_mask(instrs: list) -> None:
    """Remove causal mask from decode SDPA — it's always all-True.

    For single-token decode, the causal mask ``le(positions, past_len)``
    is always all-True because the current position is ≤ all KV positions.
    Removing the mask arg from SDPA and setting ``is_causal=True`` makes
    the entire mask computation chain unreferenced dead code.
    """
    # Find SDPA instructions with a mask (4th arg)
    name_to_instr = {i["name"]: i for i in instrs}
    modified = False
    for instr in instrs:
        if instr.get("target") != "aten.scaled_dot_product_attention.default":
            continue
        args = instr.get("args", [])
        if len(args) < 4 or args[3] is None:
            continue
        mask_name = args[3]
        if not isinstance(mask_name, str):
            continue
        # Trace the mask back: must be expand → le pattern
        mask_instr = name_to_instr.get(mask_name)
        if mask_instr is None:
            continue
        # Follow alias
        while (mask_instr and mask_instr.get("op") == "alias"):
            alias_tgt = mask_instr.get("_alias_target")
            mask_instr = name_to_instr.get(alias_tgt)
        if mask_instr is None:
            continue
        mtgt = mask_instr.get("target", "")
        if mtgt != "aten.expand.default":
            continue
        exp_args = mask_instr.get("args", [])
        if not exp_args or not isinstance(exp_args[0], str):
            continue
        le_name = exp_args[0]
        le_instr = name_to_instr.get(le_name)
        if le_instr is None:
            continue
        while (le_instr and le_instr.get("op") == "alias"):
            alias_tgt = le_instr.get("_alias_target")
            le_instr = name_to_instr.get(alias_tgt)
        if le_instr is None:
            continue
        if le_instr.get("target") not in ("aten.le.Tensor", "aten.le.Scalar"):
            continue
        # Pattern confirmed: expand(le(...)) is a standard causal mask.
        # Remove the mask arg and set is_causal=True.
        instr["args"] = args[:3]
        kw = instr.get("kwargs", {})
        kw["is_causal"] = True
        instr["kwargs"] = kw
        modified = True

    if modified:
        # Iterative dead code elimination: removing the mask arg
        # creates a cascade of unreferenced instructions.
        # Only count refs from live instructions (not dead/alias).
        while True:
            ref_counts: Dict[str, int] = {}
            for i in instrs:
                # Skip dead instructions and aliases when counting refs
                if i.get("_handler") is None and i["op"] != "alias":
                    continue
                if i["op"] == "alias":
                    # Alias instructions reference their target
                    tgt = i.get("_alias_target")
                    if tgt:
                        ref_counts[tgt] = ref_counts.get(tgt, 0) + 1
                    continue
                for arg_val in _flat_string_refs(i):
                    ref_counts[arg_val] = ref_counts.get(arg_val, 0) + 1
            # Also count output refs
            for i in instrs:
                if i["op"] == "output":
                    for arg_val in _flat_string_refs(i):
                        ref_counts[arg_val] = ref_counts.get(arg_val, 0) + 1
            changed = False
            for i in instrs:
                if i["op"] != "call_function" or i.get("_handler") is None:
                    continue
                if ref_counts.get(i["name"], 0) == 0:
                    i["_handler"] = None
                    changed = True
            # Also eliminate now-orphaned aliases
            for i in instrs:
                if i["op"] == "alias":
                    if ref_counts.get(i["name"], 0) == 0:
                        i["op"] = "dead"
                        i["_op"] = "dead"
                        changed = True
            if not changed:
                break


def _fuse_inline_shape_ops(instrs: list) -> None:
    """Replace single-use shape ops with lazy SQL-expression wrappers.

    Shape ops consumed by exactly one downstream instruction are
    converted to return ``SqlExpr`` objects.  The consumer's SQL call
    then inlines the fragment, collapsing two round-trips into one.

    Must run **after** all other fusion passes so that fused
    intermediates are already resolved.
    """
    # Ops whose handlers inspect blob_shape() at runtime — they cannot
    # receive SqlExpr arguments, so we must not inline shape ops that
    # feed into them.
    # NOTE: cat modified to accept SqlExpr (negative dim resolved via
    # C extension). SDPA always has scale in kwargs so no blob_shape().
    _SHAPE_INSPECTING_OPS = {
        "aten.stack.default",
    }

    # Map instruction name → instruction dict, and build ref counts
    name_to_instr: Dict[str, dict] = {}
    ref_counts: Dict[str, int] = {}
    for instr in instrs:
        name_to_instr[instr["name"]] = instr
        for arg_val in _flat_string_refs(instr):
            ref_counts[arg_val] = ref_counts.get(arg_val, 0) + 1

    # Find the single consumer for each instruction's output
    consumers: Dict[str, str] = {}  # shape_op_name → consumer_target
    for instr in instrs:
        target = instr.get("target", "")
        for arg_val in _flat_string_refs(instr):
            if arg_val in name_to_instr:
                consumers[arg_val] = target

    for instr in instrs:
        if instr["op"] != "call_function":
            continue
        # Skip already-fused ops
        if instr.get("_fused_type") is not None:
            continue
        target = instr.get("target", "")
        lazy_fn = _LAZY_SHAPE_OPS.get(target)
        if lazy_fn is None:
            continue
        # Only inline if the output is consumed exactly once
        if ref_counts.get(instr["name"], 0) != 1:
            continue
        # Don't inline if the consumer inspects blob_shape()
        consumer_target = consumers.get(instr["name"], "")
        if consumer_target in _SHAPE_INSPECTING_OPS:
            continue
        instr["_handler"] = lazy_fn


def _flat_string_refs(instr: dict) -> list:
    """Return all string argument references from an instruction (flat)."""
    result = []
    def _collect(obj):
        if isinstance(obj, str):
            result.append(obj)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _collect(item)
        elif isinstance(obj, dict):
            for v in obj.values():
                _collect(v)
    _collect(instr.get("args", []))
    _collect(instr.get("kwargs", {}))
    _collect(instr.get("output_args", []))
    return result


class StandaloneEngine:
    """Torch-free autoregressive inference engine.

    Reads exported instruction lists (JSON) and model parameters from
    a SQLite database, executing each operation as an individual SQL
    call through the ``llm_ops.so`` C extension.

    Parameters
    ----------
    export_dir : str
        Directory containing ``model.db``, ``prefill.json``, and
        ``decode.json`` files produced by ``scripts/export_model.py``.
    extension_path : str, optional
        Explicit path to ``llm_ops.so``.  Auto-detected if omitted.
    cache_size_kb : int, optional
        Override PRAGMA cache_size (negative = KiB, e.g. ``-51200`` for 50 MiB).
        When ``pcache`` is provided and this is ``None``, it is
        automatically set to match the pcache memory budget.
    pcache : object, optional
        An ``LLMPcache`` instance for layer-aware page cache.  When
        provided the engine **fully takes over** SQLite page-cache
        management: mmap is disabled automatically (so all reads go
        through the custom pcache) and ``cache_size`` is aligned with
        the pcache budget.  Layer hints (``set_active_layer`` /
        ``mark_layer_done``) are sent during inference.
    lazy_params : bool
        When ``True``, model parameters are **not** pre-loaded into
        Python memory.  Instead, they are read from the SQLite DB on
        every forward pass.  This is the mode where the custom pcache
        makes a real difference: completed-layer pages are evicted
        first to keep upcoming layers warm.
    disable_mmap : bool
        When ``True``, disable memory-mapped I/O (``PRAGMA mmap_size=0``)
        to force all reads through the SQLite page cache.  Automatically
        set to ``True`` when ``pcache`` is provided.
    single_sql_mode : bool
        When ``True``, pre-compile both exported graphs into a single SQL
        statement each.  Python only binds runtime inputs and never loads
        model parameter BLOBs into the Python heap.
    """

    def __init__(
        self,
        export_dir: str,
        extension_path: Optional[str] = None,
        cache_size_kb: Optional[int] = None,
        pcache: Optional[Any] = None,
        lazy_params: bool = False,
        disable_mmap: bool = False,
        db_filename: str = "model.db",
        overlap_load: bool = False,
        use_param_manager: bool = False,
        single_sql_mode: bool = False,
    ) -> None:
        import json as _json

        self._dir = export_dir
        db_path = os.path.join(export_dir, db_filename)

        # ── pcache takeover ─────────────────────────────────────
        # When a custom pcache is provided, automatically:
        # 1. Disable mmap (mmap bypasses the page cache entirely)
        # 2. Set cache_size to match the pcache budget so SQLite
        #    allocates the right number of page slots.
        if pcache is not None:
            disable_mmap = True  # force reads through page cache
            if cache_size_kb is None and hasattr(pcache, '_budget_bytes'):
                budget = pcache._budget_bytes
                if budget > 0:
                    cache_size_kb = -int(budget // 1024)  # negative = KiB

        self._conn = open_db(db_path, extension_path,
                             cache_size_kb=cache_size_kb,
                             disable_mmap=disable_mmap)

        # Native page cache (layer hints)
        self._pcache = pcache
        self._single_sql_mode = single_sql_mode

        # Read metadata (values are stored as JSON strings)
        self._meta: Dict[str, Any] = {}
        try:
            for key, val in self._conn.execute(
                "SELECT key, value FROM metadata"
            ).fetchall():
                try:
                    self._meta[key] = _json.loads(val)
                except (ValueError, TypeError):
                    self._meta[key] = val
        except sqlite3.OperationalError:
            pass

        self._vocab_size: int = self._meta.get("vocab_size", 100)
        self._num_layers: int = self._meta.get("num_layers", 0)

        # Tokenizer — look for tokenizer.json for BPE encoding
        tok_json = os.path.join(export_dir, "tokenizer.json")
        self._tokenizer = StandaloneTokenizer(self._conn, tok_json)

        # Pre-load model parameter BLOBs from the DB
        self._lazy_params = lazy_params
        self._use_param_manager = use_param_manager
        self._param_manager: Optional[Any] = None
        self._param_blobs: Dict[str, bytes] = {}
        self._zero_param_names: set = set()
        self._layer_param_names: Dict[int, List[str]] = {}
        self._layer_param_sql: Dict[int, str] = {}
        self._compiled_prefill_sql: Optional[str] = None
        self._compiled_decode_sql: Optional[str] = None
        self._compiled_prefill_bind_names: List[str] = []
        self._compiled_decode_bind_names: List[str] = []
        self._compiled_sql_enabled = False

        if single_sql_mode:
            # Keep only parameter names in Python.  All parameter BLOB reads
            # happen inside SQLite during compiled statement execution.
            try:
                rows = self._conn.execute(
                    "SELECT name FROM model_params"
                ).fetchall()
                for (name,) in rows:
                    self._param_blobs[name] = b""
            except sqlite3.OperationalError:
                pass
            try:
                rows = self._conn.execute(
                    "SELECT name FROM model_params WHERE length(data) <= 12"
                ).fetchall()
                self._zero_param_names = {name for (name,) in rows}
            except sqlite3.OperationalError:
                self._zero_param_names = set()
            self._async_loader = None

        elif lazy_params and use_param_manager:
            # Use simplified parameter cache (Python-based)
            try:
                from src.backends.param_manager_simple import ParamManager
                budget_mb = 0  # unlimited by default
                if pcache and hasattr(pcache, '_budget_bytes'):
                    budget_mb = int(pcache._budget_bytes / (1024 * 1024))
                # Pass the connection so cache uses the same connection
                self._param_manager = ParamManager(
                    self._conn, self._num_layers, budget_mb
                )
                # Still need to build param name index for _resolve
                try:
                    rows = self._conn.execute(
                        "SELECT name FROM model_params"
                    ).fetchall()
                    for (name,) in rows:
                        self._param_blobs[name] = b""  # placeholder
                except sqlite3.OperationalError:
                    pass
                # Build layer→param_names index
                for pname in self._param_blobs:
                    m = _LAYER_RE.search(pname)
                    lid = int(m.group(1)) if m else -1
                    self._layer_param_names.setdefault(lid, []).append(pname)
                # Prepare per-layer SQL with placeholders for batch fetch
                for lid, names in self._layer_param_names.items():
                    ph = ",".join(["?"] * len(names))
                    self._layer_param_sql[lid] = (
                        f"SELECT name, data FROM model_params "
                        f"WHERE name IN ({ph})"
                    )
            except ImportError as e:
                print(f"Warning: Failed to load ParamManager: {e}")
                print("Falling back to traditional lazy loading")
                use_param_manager = False
                self._use_param_manager = False

        if not single_sql_mode and lazy_params and not use_param_manager:
            # Lazy mode: only load param names (not data).
            # Params are loaded per-layer during inference to avoid
            # bulk-loading all 289 MB into the Python heap at once.
            try:
                rows = self._conn.execute(
                    "SELECT name FROM model_params"
                ).fetchall()
                for (name,) in rows:
                    self._param_blobs[name] = b""  # placeholder
            except sqlite3.OperationalError:
                pass
            # Build layer→param_names index for per-layer streaming.
            self._layer_param_names: Dict[int, List[str]] = {}
            for pname in self._param_blobs:
                m = _LAYER_RE.search(pname)
                lid = int(m.group(1)) if m else -1
                self._layer_param_names.setdefault(lid, []).append(pname)
            # Prepare per-layer SQL with placeholders for batch fetch
            self._layer_param_sql: Dict[int, str] = {}
            for lid, names in self._layer_param_names.items():
                ph = ",".join(["?"] * len(names))
                self._layer_param_sql[lid] = (
                    f"SELECT name, data FROM model_params "
                    f"WHERE name IN ({ph})"
                )
            # ── Async overlap loader ──────────────────────────────
            # When overlap_load=True, a background thread pre-fetches
            # the next layer's params while the main thread computes
            # the current layer, hiding SQL I/O behind computation.
            if overlap_load:
                from src.backends.async_param_loader import AsyncParamLoader
                self._async_loader: Optional[AsyncParamLoader] = \
                    AsyncParamLoader(
                        db_path=db_path,
                        extension_path=extension_path,
                        layer_param_names=self._layer_param_names,
                        layer_param_sql=self._layer_param_sql,
                        pcache=pcache,
                        lookahead=1,
                    )
                self._async_loader.start()
            else:
                self._async_loader = None
        elif not single_sql_mode:
            try:
                rows = self._conn.execute(
                    "SELECT name, data FROM model_params"
                ).fetchall()
                for name, data in rows:
                    self._param_blobs[name] = data
            except sqlite3.OperationalError:
                pass
            self._async_loader = None

        # ── Custom LLM memory manager ──
        # Initialise layer-aware cache tracking and background prefetcher.
        from src.backends.llm_memory_manager import LLMMemoryManager
        if (not self._single_sql_mode
                and self._num_layers > 0 and db_path != ":memory:"):
            self._memory_mgr: Optional["LLMMemoryManager"] = \
                LLMMemoryManager(
                    db_path,
                    num_layers=self._num_layers,
                    extension_path=extension_path,
                    disable_mmap=disable_mmap,   # align with main connection
                )
            self._memory_mgr.begin_forward_pass(
                list(self._param_blobs.keys()))
        else:
            self._memory_mgr = None

        # Load instruction lists
        with open(os.path.join(export_dir, "prefill.json"), "r") as f:
            self._prefill_instrs = _json.load(f)
        with open(os.path.join(export_dir, "decode.json"), "r") as f:
            self._decode_instrs = _json.load(f)

        # Build op dispatch table
        self._ops = _build_sql_ops()

        # Pre-resolve handlers and fuse op patterns
        self._preresolve_handlers(self._prefill_instrs)
        self._preresolve_handlers(self._decode_instrs)

        # Compile-time identity/dead-code elimination
        _eliminate_identities(self._prefill_instrs)
        _eliminate_identities(self._decode_instrs)

        # Common Subexpression Elimination (must run after identity elimination)
        _eliminate_common_subexpressions(self._prefill_instrs)
        _eliminate_common_subexpressions(self._decode_instrs)

        # Inline single-use shape ops (must run after preresolve)
        _fuse_inline_shape_ops(self._prefill_instrs)
        _fuse_inline_shape_ops(self._decode_instrs)

        # ── Eliminate causal mask for decode (single token) ──
        # For single-token decode, the causal mask is always all-True
        # (position ≤ all past positions). Removing the mask arg from
        # SDPA makes the entire mask computation chain dead code.
        _eliminate_decode_mask(self._decode_instrs)

        # Eliminate cat([empty_tensor, x], dim) → alias to x
        # Zero-size parameter blobs make the cat a no-op.
        if self._single_sql_mode:
            zero_params = set(self._zero_param_names)
        else:
            zero_params = {
                name for name, blob in self._param_blobs.items()
                if len(blob) <= 12  # header-only → zero-element tensor
            }
        if zero_params:
            for instrs in (self._prefill_instrs, self._decode_instrs):
                for instr in instrs:
                    if (instr.get("target") != "aten.cat.default"
                            or instr.get("_handler") is None):
                        continue
                    args = instr.get("args", [])
                    if not args or not isinstance(args[0], list):
                        continue
                    tensor_list = args[0]
                    non_empty = [t for t in tensor_list
                                 if not (isinstance(t, str)
                                         and t in zero_params)]
                    if len(non_empty) == 1 and isinstance(non_empty[0], str):
                        instr["op"] = "alias"
                        instr["_op"] = "alias"
                        instr["target"] = non_empty[0]
                        instr["_alias_target"] = non_empty[0]
                        instr["_handler"] = None

        # Strip placeholder instructions whose names are already present
        # in _param_blobs — they would just hit `if name not in env` and
        # be skipped at runtime.  Removing them shrinks the per-step
        # iteration count (typically −20 %).
        param_names = set(self._param_blobs)
        self._prefill_instrs = [
            i for i in self._prefill_instrs
            if not (i["op"] == "placeholder" and i["name"] in param_names)
        ]
        self._decode_instrs = [
            i for i in self._decode_instrs
            if not (i["op"] == "placeholder" and i["name"] in param_names)
        ]

        # Strip dead instructions (handler=None call_functions and dead ops).
        # Keep aliases (needed for runtime env resolution), output, and
        # remaining placeholders.
        def _is_live(i):
            op = i["op"]
            if op == "call_function":
                return i.get("_handler") is not None
            return op in ("alias", "output", "placeholder")

        self._prefill_instrs = [i for i in self._prefill_instrs if _is_live(i)]
        self._decode_instrs = [i for i in self._decode_instrs if _is_live(i)]

        # Pre-extract instruction fields for the hot loop.
        # Each instruction dict gets cached tuple fields to avoid
        # repeated dict.__getitem__ calls (~3-5% overhead reduction).
        # Also precompile lazy shape ops: pre-build SQL fragments and
        # static parameters to avoid int() conversions and list building
        # at runtime.
        _lazy_names = {
            '_lazy_unsqueeze', '_lazy_view', '_lazy_reshape',
            '_lazy_expand', '_lazy_transpose', '_lazy_slice',
            '_lazy_permute', '_lazy_squeeze', '_lazy_view_t12',
        }
        for instrs in (self._prefill_instrs, self._decode_instrs):
            for instr in instrs:
                # Pre-store frequently accessed fields
                instr["_op"] = instr["op"]
                instr["_name"] = instr["name"]
                if instr["op"] == "call_function":
                    instr["_args"] = instr.get("args", [])
                    instr["_kwargs"] = instr.get("kwargs", {})
                    instr["_has_kwargs"] = bool(instr.get("kwargs"))
                    handler = instr.get("_handler")
                    if handler is None:
                        instr["_dt"] = _DT_SKIP
                        continue
                    # Check fused ops
                    if instr.get("_fused_type") is not None:
                        instr["_dt"] = _DT_FUSED
                        continue
                    # Precompile lazy shape ops
                    if handler is not None:
                        hname = getattr(handler, '__name__', '')
                        if hname in _lazy_names:
                            args = instr.get("args", [])
                            blob_ref = args[0] if args else None
                            pre = _precompile_lazy(hname, args)
                            if pre is not None and blob_ref is not None:
                                instr["_pre_sql"] = pre[0]
                                instr["_pre_static"] = pre[1]
                                instr["_pre_blob"] = blob_ref
                                instr["_dt"] = _DT_LAZY
                                continue
                    instr["_dt"] = _DT_NORMAL
                elif instr["op"] == "output":
                    instr["_output_args"] = instr.get("output_args", [])
                    instr["_dt"] = _DT_OUTPUT
                elif instr["op"] == "alias":
                    instr["_alias_target"] = instr.get("target")
                    instr["_dt"] = _DT_ALIAS
                elif instr["op"] == "placeholder":
                    instr["_dt"] = _DT_PLACEHOLDER
                else:
                    instr["_dt"] = _DT_SKIP

        # Pre-bind parameter blobs directly into instruction args.
        # This eliminates ~50 env.get() lookups per decode step for
        # args that always resolve to the same param blob.
        # Skip in lazy_params mode since blobs are loaded from DB each step.
        if not self._lazy_params and not self._single_sql_mode:
            param_blobs = self._param_blobs
            for instrs in (self._prefill_instrs, self._decode_instrs):
                for instr in instrs:
                    dt = instr.get("_dt")
                    if dt == _DT_NORMAL:
                        raw = instr["_args"]
                        for idx, a in enumerate(raw):
                            if isinstance(a, str) and a in param_blobs:
                                raw[idx] = param_blobs[a]
                            elif isinstance(a, list):
                                for j, sub in enumerate(a):
                                    if isinstance(sub, str) and sub in param_blobs:
                                        a[j] = param_blobs[sub]
                    elif dt == _DT_FUSED:
                        fargs = instr.get("_fused_args")
                        if fargs:
                            new_fargs = []
                            for a in fargs:
                                if isinstance(a, str) and a in param_blobs:
                                    new_fargs.append(param_blobs[a])
                                else:
                                    new_fargs.append(a)
                            instr["_fused_args"] = tuple(new_fargs)

        # Pre-annotate instructions with their transformer layer index.
        # Used by the native pcache to send layer-change hints.
        for instrs in (self._prefill_instrs, self._decode_instrs):
            for instr in instrs:
                lid = -1  # default: non-layer (embedding, norm, lm_head)
                for ref in _flat_string_refs(instr):
                    m = _LAYER_RE.search(ref)
                    if m:
                        lid = int(m.group(1))
                        break
                instr["_layer"] = lid

        self._compiled_sql_enabled = False

    # ── Blob helpers ──

    @staticmethod
    def _make_input_ids_blob(token_ids: List[int]) -> bytes:
        """Encode token IDs as a (1, seq_len) float32 BLOB."""
        seq_len = len(token_ids)
        return encode_nd_blob(
            [float(tid) for tid in token_ids],
            [1, seq_len],
        )

    def _load_layer_params(
        self, layer_idx: int, env: dict,
    ) -> None:
        """Load parameters for a single transformer layer into *env*.

        Only used in ``lazy_params`` mode.  Loads *layer_idx* params via
        a targeted ``SELECT … WHERE name IN (…)`` instead of bulk-loading
        the entire 289 MB table.  When a pcache is active, sends
        ``set_active_layer`` before the query so new pages are tagged
        with the correct layer for priority eviction.  Also sends a
        prefetch hint for the *next* layer so the pcache knows to
        protect those pages during eviction.
        """
        # Use C-layer parameter manager if available
        if self._param_manager:
            # Load params directly into env (no double caching)
            names = self._layer_param_names.get(layer_idx, [])
            if not names:
                return

            # Check if params are already loaded
            already_loaded = all(name in env and isinstance(env.get(name), bytes) and len(env[name]) > 0
                                for name in names[:1])  # Check first param as indicator
            if already_loaded:
                # Params already in env, just track for eviction
                self._param_manager.load_layer(layer_idx)
                return

            # Load only the params we need (same as traditional)
            sql = self._layer_param_sql[layer_idx]
            try:
                rows = self._conn.execute(sql, names).fetchall()
                for name, data in rows:
                    env[name] = data
            except Exception:
                pass

            # Track loading for eviction strategy
            self._param_manager.load_layer(layer_idx)
            return

        names = self._layer_param_names.get(layer_idx)
        if not names:
            return
        pcache = self._pcache
        if pcache is not None and layer_idx >= 0:
            pcache.set_active_layer(layer_idx)
            # Prefetch hint: mark next layer(s) as upcoming so their
            # pages are not evicted while we read the current layer.
            next_layer = layer_idx + 1
            if next_layer < self._num_layers:
                pcache.prefetch_layer(next_layer)
        sql = self._layer_param_sql[layer_idx]
        try:
            rows = self._conn.execute(sql, names).fetchall()
            for name, data in rows:
                env[name] = data
        except sqlite3.OperationalError:
            pass

    def _evict_layer_params(
        self, layer_idx: int, env: dict,
    ) -> None:
        """Remove parameters for *layer_idx* from *env* to free heap.

        Also notifies the pcache that the layer is done so its pages
        become eligible for priority eviction.
        """
        # Use parameter manager if available
        if self._param_manager:
            # Simple strategy: evict current layer only
            names = self._layer_param_names.get(layer_idx, [])
            for name in names:
                env.pop(name, None)
            return

        names = self._layer_param_names.get(layer_idx)
        if names:
            for name in names:
                env.pop(name, None)
        pcache = self._pcache
        if pcache is not None and layer_idx >= 0:
            pcache.mark_layer_done(layer_idx)

    # ── Instruction pre-resolution ──

    def _preresolve_handlers(self, instrs: list) -> None:
        """Attach resolved handler directly to each instruction dict.

        Eliminates a ``dict.get`` lookup per instruction at runtime.

        Also detects ``silu → mul`` pairs (MLP gate pattern) and replaces
        them with the fused ``llm_silu_mul_nd`` C function — saving one
        SQL round-trip per pair.

        Also detects the 7-instruction RoPE pattern and fuses it into
        a single ``llm_rope_nd`` call.
        """
        # Sentinel used for fused instructions so _execute_instrs doesn't skip
        # them via the "if handler is None: continue" guard.
        _FUSED = lambda c, a, k: None  # noqa: E731

        # Index instructions by name for quick lookup
        name_to_idx: Dict[str, int] = {}
        for idx, instr in enumerate(instrs):
            name_to_idx[instr["name"]] = idx

        for instr in instrs:
            if instr["op"] == "call_function":
                instr["_handler"] = self._ops.get(instr["target"])
            else:
                instr["_handler"] = None

        # ── Fuse silu → mul into llm_silu_mul_nd ──
        for idx, instr in enumerate(instrs):
            if (instr.get("target") != "aten.mul.Tensor"
                    or len(instr.get("args", [])) < 2):
                continue
            a0, a1 = instr["args"][0], instr["args"][1]
            # Check if either arg is the output of a silu instruction
            silu_name = None
            other_arg_pos = None
            for pos, arg in enumerate([a0, a1]):
                if isinstance(arg, str) and arg in name_to_idx:
                    src = instrs[name_to_idx[arg]]
                    if src.get("target") == "aten.silu.default":
                        silu_name = arg
                        other_arg_pos = 1 - pos
                        break
            if silu_name is None:
                continue
            # Replace mul handler with fused silu_mul
            silu_instr = instrs[name_to_idx[silu_name]]
            gate_arg = silu_instr["args"][0]
            up_arg = instr["args"][other_arg_pos]
            # Store fused info on the mul instruction
            instr["_fused_type"] = "silu_mul"
            instr["_fused_args"] = (gate_arg, up_arg)
            instr["_handler"] = _FUSED
            # Mark silu instruction as skippable (its output may still be
            # referenced by other instructions, but in Qwen2 MLP it is only
            # consumed by this mul)
            silu_instr["_handler"] = None

        # ── Fuse 7-op RoPE pattern into llm_rope_nd ──
        for instr in instrs:
            if instr.get("target") != "aten.add.Tensor":
                continue
            args = instr.get("args", [])
            if len(args) < 2:
                continue
            mul_x_name, mul_sin_name = args[0], args[1]
            if not (isinstance(mul_x_name, str) and isinstance(mul_sin_name, str)):
                continue
            if mul_x_name not in name_to_idx or mul_sin_name not in name_to_idx:
                continue
            mul_x_i = instrs[name_to_idx[mul_x_name]]
            mul_sin_i = instrs[name_to_idx[mul_sin_name]]
            if (mul_x_i.get("target") != "aten.mul.Tensor"
                    or mul_sin_i.get("target") != "aten.mul.Tensor"):
                continue
            # mul_x: [x, cos]
            mx_args = mul_x_i.get("args", [])
            if len(mx_args) < 2 or not (isinstance(mx_args[0], str)
                                         and isinstance(mx_args[1], str)):
                continue
            x_name, cos_name = mx_args[0], mx_args[1]
            # mul_sin: find which arg is cat([neg, slice_a])
            ms_args = mul_sin_i.get("args", [])
            if len(ms_args) < 2:
                continue
            cat_name, sin_name = None, None
            for pos in range(2):
                cand = ms_args[pos]
                if not isinstance(cand, str) or cand not in name_to_idx:
                    continue
                if instrs[name_to_idx[cand]].get("target") == "aten.cat.default":
                    cat_name = cand
                    sin_name = ms_args[1 - pos]
                    break
            if cat_name is None or not isinstance(sin_name, str):
                continue
            # cat: args = [[neg_name, slice_a_name], -1]
            cat_i = instrs[name_to_idx[cat_name]]
            cat_args = cat_i.get("args", [])
            if (len(cat_args) < 2 or not isinstance(cat_args[0], list)
                    or len(cat_args[0]) != 2 or cat_args[1] != -1):
                continue
            neg_name, slice_a_name = cat_args[0][0], cat_args[0][1]
            if not (isinstance(neg_name, str) and isinstance(slice_a_name, str)):
                continue
            if neg_name not in name_to_idx or slice_a_name not in name_to_idx:
                continue
            # neg: [slice_b]
            neg_i = instrs[name_to_idx[neg_name]]
            if neg_i.get("target") != "aten.neg.default":
                continue
            neg_args = neg_i.get("args", [])
            if not neg_args or not isinstance(neg_args[0], str):
                continue
            slice_b_name = neg_args[0]
            if slice_b_name not in name_to_idx:
                continue
            # slice_a, slice_b must both slice x
            sa_i = instrs[name_to_idx[slice_a_name]]
            sb_i = instrs[name_to_idx[slice_b_name]]
            if (sa_i.get("target") != "aten.slice.Tensor"
                    or sb_i.get("target") != "aten.slice.Tensor"):
                continue
            sa_args = sa_i.get("args", [])
            sb_args = sb_i.get("args", [])
            if not sa_args or not sb_args:
                continue
            if sa_args[0] != x_name or sb_args[0] != x_name:
                continue
            # Pattern confirmed.  Check that intermediates are only used
            # within the pattern (safe to disable).
            skip_candidates = {mul_x_name, slice_a_name, slice_b_name,
                                neg_name, cat_name, mul_sin_name}
            # Build a flat set of all arg strings across all instructions
            external_refs: set = set()
            add_name = instr["name"]
            pattern_names = skip_candidates | {add_name}
            for other in instrs:
                if other["name"] in pattern_names:
                    continue
                for ref in self._flat_refs(other):
                    if ref in skip_candidates:
                        external_refs.add(ref)
            # Only disable intermediates that have no external references
            safe_to_skip = skip_candidates - external_refs
            # Fuse add → llm_rope_nd
            instr["_fused_type"] = "rope"
            instr["_fused_args"] = (x_name, cos_name, sin_name)
            # Use sentinel so _execute_instrs reaches the _fused_rope check.
            instr["_handler"] = _FUSED
            for skip_name in safe_to_skip:
                instrs[name_to_idx[skip_name]]["_handler"] = None

        # ── Fuse 6-op RMSNorm pattern into llm_rmsnorm_nd ──
        for instr in instrs:
            if instr.get("target") != "aten.mul.Tensor":
                continue
            args = instr.get("args", [])
            if len(args) < 2:
                continue
            # The anchor is mul(norm_result, weight).
            # norm_result traces through alias→mul(x, rsqrt)→rsqrt→add→mean→pow
            for pos in range(2):
                norm_arg = args[pos]
                weight_arg = args[1 - pos]
                if not isinstance(norm_arg, str) or norm_arg not in name_to_idx:
                    continue
                # Follow alias chain (to.dtype → mul(x, rsqrt))
                norm_i = instrs[name_to_idx[norm_arg]]
                actual_mul_name = None
                if norm_i.get("op") == "alias":
                    src = norm_i.get("target")
                    if isinstance(src, str) and src in name_to_idx:
                        actual_mul_name = src
                elif norm_i.get("target") == "aten.mul.Tensor":
                    actual_mul_name = norm_arg
                if actual_mul_name is None:
                    continue
                mul_norm_i = instrs[name_to_idx[actual_mul_name]]
                if mul_norm_i.get("target") != "aten.mul.Tensor":
                    continue
                mn_args = mul_norm_i.get("args", [])
                if len(mn_args) < 2:
                    continue
                # Find rsqrt arg
                rsqrt_name = None
                x_name = None
                for mpos in range(2):
                    cand = mn_args[mpos]
                    if isinstance(cand, str) and cand in name_to_idx:
                        ci = instrs[name_to_idx[cand]]
                        if ci.get("target") == "aten.rsqrt.default":
                            rsqrt_name = cand
                            x_name = mn_args[1 - mpos]
                            break
                if rsqrt_name is None or not isinstance(x_name, str):
                    continue
                rsqrt_i = instrs[name_to_idx[rsqrt_name]]
                r_args = rsqrt_i.get("args", [])
                if not r_args or not isinstance(r_args[0], str):
                    continue
                add_name = r_args[0]
                if add_name not in name_to_idx:
                    continue
                add_i = instrs[name_to_idx[add_name]]
                if add_i.get("target") != "aten.add.Tensor":
                    continue
                add_args = add_i.get("args", [])
                if len(add_args) < 2:
                    continue
                mean_name = add_args[0]
                eps_val = add_args[1]
                if not isinstance(mean_name, str) or mean_name not in name_to_idx:
                    continue
                mean_i = instrs[name_to_idx[mean_name]]
                if mean_i.get("target") != "aten.mean.dim":
                    continue
                m_args = mean_i.get("args", [])
                if not m_args or not isinstance(m_args[0], str):
                    continue
                pow_name = m_args[0]
                if pow_name not in name_to_idx:
                    continue
                pow_i = instrs[name_to_idx[pow_name]]
                if (pow_i.get("target") != "aten.pow.Tensor_Scalar"
                        or len(pow_i.get("args", [])) < 2
                        or pow_i["args"][1] != 2):
                    continue
                pow_x = pow_i["args"][0]
                # Resolve through alias chains
                resolved_pow_x = pow_x
                while isinstance(resolved_pow_x, str) and resolved_pow_x in name_to_idx:
                    ai = instrs[name_to_idx[resolved_pow_x]]
                    if ai.get("op") == "alias" and isinstance(ai.get("target"), str):
                        resolved_pow_x = ai["target"]
                    else:
                        break
                resolved_x = x_name
                while isinstance(resolved_x, str) and resolved_x in name_to_idx:
                    ai = instrs[name_to_idx[resolved_x]]
                    if ai.get("op") == "alias" and isinstance(ai.get("target"), str):
                        resolved_x = ai["target"]
                    else:
                        break
                if resolved_pow_x != resolved_x:
                    continue
                # Pattern confirmed!
                instr["_fused_type"] = "rmsnorm"
                instr["_fused_args"] = (x_name, weight_arg, eps_val)
                instr["_handler"] = _FUSED
                # Disable intermediates with external-reference safety check
                skip_candidates = {pow_name, mean_name, add_name,
                                   rsqrt_name, actual_mul_name}
                external_refs_rms: set = set()
                pattern_names_rms = skip_candidates | {instr["name"]}
                for other in instrs:
                    if other["name"] in pattern_names_rms:
                        continue
                    for ref in self._flat_refs(other):
                        if ref in skip_candidates:
                            external_refs_rms.add(ref)
                safe_to_skip_rms = skip_candidates - external_refs_rms
                for skip_name in safe_to_skip_rms:
                    instrs[name_to_idx[skip_name]]["_handler"] = None
                break  # Found pattern for this mul

        # ── Fuse unsqueeze→expand→reshape into llm_repeat_kv_nd ──
        # Detects the repeat_kv pattern used in GQA:
        #   unsqueeze(kv, 2) → [alias] → expand(dim2, n_rep) → reshape(merge)
        def _follow_alias(nm):
            """Follow alias chain to find the canonical instruction."""
            visited = set()
            while nm in name_to_idx:
                i = instrs[name_to_idx[nm]]
                if i.get("op") == "alias":
                    tgt = i.get("_alias_target") or i.get("target")
                    if tgt in visited or tgt == nm:
                        break
                    visited.add(nm)
                    nm = tgt
                else:
                    break
            return nm

        for instr in instrs:
            if instr.get("target") != "aten.reshape.default":
                continue
            rsh_args = instr.get("args", [])
            if len(rsh_args) < 2 or not isinstance(rsh_args[0], str):
                continue
            expand_name = _follow_alias(rsh_args[0])
            if expand_name not in name_to_idx:
                continue
            expand_i = instrs[name_to_idx[expand_name]]
            if expand_i.get("target") != "aten.expand.default":
                continue
            exp_args = expand_i.get("args", [])
            if len(exp_args) < 2 or not isinstance(exp_args[0], str):
                continue
            unsq_name = _follow_alias(exp_args[0])
            if unsq_name not in name_to_idx:
                continue
            unsq_i = instrs[name_to_idx[unsq_name]]
            # Handle optional no-op slice between unsqueeze and expand
            extra_skip = set()
            if unsq_i.get("target") == "aten.slice.Tensor":
                slice_args = unsq_i.get("args", [])
                if (len(slice_args) >= 2
                        and isinstance(slice_args[0], str)
                        and slice_args[0] in name_to_idx):
                    extra_skip.add(unsq_name)
                    unsq_name = slice_args[0]
                    unsq_i = instrs[name_to_idx[unsq_name]]
            if unsq_i.get("target") != "aten.unsqueeze.default":
                continue
            usq_args = unsq_i.get("args", [])
            if len(usq_args) < 2:
                continue
            unsq_dim = usq_args[1]
            if not isinstance(unsq_dim, (int, float)) or int(unsq_dim) != 2:
                continue
            kv_input = usq_args[0]
            # Determine num_repeats from expand sizes
            exp_sizes = exp_args[1]
            if not isinstance(exp_sizes, list) or len(exp_sizes) < 3:
                continue
            # exp_sizes should be [batch, kv_heads, num_repeats, seq_len, hd]
            num_repeats = exp_sizes[2]
            if not isinstance(num_repeats, (int, float)):
                continue
            num_repeats = int(num_repeats)
            if num_repeats <= 0:
                continue
            # Verify reshape merges dims 1,2: output_shape[1] = kv_heads * n_rep
            rsh_shape = rsh_args[1]
            if not isinstance(rsh_shape, list) or len(rsh_shape) < 2:
                continue
            # Check intermediates are safe to disable
            skip_candidates = {expand_name, unsq_name} | extra_skip
            external_refs_rkv: set = set()
            pattern_names_rkv = skip_candidates | {instr["name"]}
            for other in instrs:
                if other["name"] in pattern_names_rkv:
                    continue
                for ref in self._flat_refs(other):
                    if ref in skip_candidates:
                        external_refs_rkv.add(ref)
            safe_to_skip_rkv = skip_candidates - external_refs_rkv
            # Fuse: replace reshape with llm_repeat_kv_nd
            instr["target"] = "llm_repeat_kv_nd"
            instr["args"] = [kv_input, num_repeats]
            instr["_handler"] = self._ops.get("llm_repeat_kv_nd")
            for skip_name in safe_to_skip_rkv:
                instrs[name_to_idx[skip_name]]["_handler"] = None

        # ── Fuse view → transpose(1, 2) into llm_view_t12_nd ──
        # Detects: view(x, [shape...]) → transpose(result, 1, 2)
        for instr in instrs:
            if instr.get("target") != "aten.transpose.int":
                continue
            tr_args = instr.get("args", [])
            if len(tr_args) < 3:
                continue
            d0, d1 = tr_args[1], tr_args[2]
            if not (isinstance(d0, (int, float)) and isinstance(d1, (int, float))):
                continue
            if (int(d0), int(d1)) != (1, 2) and (int(d0), int(d1)) != (2, 1):
                continue
            view_name = tr_args[0]
            if not isinstance(view_name, str) or view_name not in name_to_idx:
                continue
            view_i = instrs[name_to_idx[view_name]]
            if view_i.get("target") not in (
                    "aten.view.default", "aten.reshape.default",
                    "aten._unsafe_view.default"):
                continue
            v_args = view_i.get("args", [])
            if len(v_args) < 2 or not isinstance(v_args[0], str):
                continue
            # Check view has no external refs (safe to skip)
            vname = view_i["name"]
            ext_count = 0
            for other in instrs:
                if other["name"] == instr["name"]:
                    continue
                for ref in self._flat_refs(other):
                    if ref == vname:
                        ext_count += 1
            if ext_count > 0:
                continue  # view used elsewhere
            view_shape = v_args[1] if isinstance(v_args[1], list) else [v_args[1]]
            # Fuse: transpose becomes llm_view_t12_nd(x, shape...)
            instr["target"] = "llm_view_t12_nd"
            instr["args"] = [v_args[0]] + view_shape
            instr["_handler"] = self._ops.get("llm_view_t12_nd")
            view_i["_handler"] = None

    @staticmethod
    def _flat_refs(instr: dict) -> list:
        """Return all string argument references from an instruction."""
        result = []
        def _collect(obj):
            if isinstance(obj, str):
                result.append(obj)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    _collect(item)
            elif isinstance(obj, dict):
                for v in obj.values():
                    _collect(v)
        _collect(instr.get("args", []))
        _collect(instr.get("kwargs", {}))
        _collect(instr.get("output_args", []))
        return result

    # ── Instruction execution ──

    def _resolve(self, arg, env: Dict[str, object]) -> object:
        """Resolve an argument from the execution environment."""
        if isinstance(arg, str):
            if arg in env:
                val = env[arg]
                # If using ParamManager and value is empty placeholder, fetch from C layer
                if self._param_manager and val == b"":
                    param_data = self._param_manager.get_param(arg)
                    if param_data is not None:
                        # Convert memoryview to bytes for compatibility
                        return bytes(param_data)
                return val
            # Try to get from ParamManager if available
            if self._param_manager:
                param_data = self._param_manager.get_param(arg)
                if param_data is not None:
                    # Convert memoryview to bytes for compatibility
                    return bytes(param_data)
            return arg  # might be a literal string
        if isinstance(arg, list):
            return [self._resolve(a, env) for a in arg]
        if isinstance(arg, tuple):
            return tuple(self._resolve(a, env) for a in arg)
        return arg

    def _execute_instrs(
        self,
        instrs: list,
        input_ids_blob: bytes,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        """Execute an instruction list, returning the output tuple."""
        # In lazy_params mode with per-layer streaming, load params
        # one layer at a time to avoid allocating all 289 MB at once.
        if self._lazy_params:
            # Start with non-layer params only (embed, norm, lm_head).
            # These are small (~4 MB for H=1024) and always needed.
            env: dict = {}
            self._load_layer_params(-1, env)   # non-layer params
        else:
            # In SQL-downcall mode keep parameters out of Python entirely.
            env = {} if self._single_sql_mode else dict(self._param_blobs)
        # Cache instance attrs as locals to avoid repeated attribute lookups
        conn = self._conn
        _resolve = self._resolve
        _env_get = env.get  # method-local for speed
        _trim = _maybe_malloc_trim if self._single_sql_mode else None

        # Native pcache layer hints
        pcache = self._pcache
        cur_layer = -2  # sentinel: no layer set yet

        # Per-layer streaming state (lazy mode only)
        _lazy = self._lazy_params
        _async_loader = self._async_loader if _lazy else None
        _load_layer = self._load_layer_params if _lazy else None
        _evict_layer = self._evict_layer_params if _lazy else None

        # If async overlap is active, pre-request the first two layers now
        # so the background thread starts loading while we set up.
        if _async_loader is not None:
            _async_loader.request_layer(0)
            if self._num_layers > 1:
                _async_loader.request_layer(1)

        user_input_count = 0
        output_value = None
        _t_layer_start = _time.perf_counter()

        for instr in instrs:
            dt = instr["_dt"]
            name = instr["_name"]

            # ── Per-layer streaming: load/evict params on layer change ──
            instr_layer = instr.get("_layer", -1)
            if instr_layer != cur_layer and instr_layer >= 0:
                if _lazy:
                    if _async_loader is not None:
                        # Record compute time of the completed layer
                        _t_now = _time.perf_counter()
                        if cur_layer >= 0:
                            _async_loader.record_compute_time(
                                _t_now - _t_layer_start)
                            # Evict the done layer's env entries
                            _evict_layer(cur_layer, env)
                        _t_layer_start = _t_now

                        # Get next layer params (blocks until ready,
                        # but background thread was pre-loading them)
                        params = _async_loader.get_layer_params(instr_layer)
                        env.update(params)

                        # Pre-request the layer after next (lookahead)
                        la = _async_loader.suggested_lookahead()
                        for ahead in range(1, la + 1):
                            next_req = instr_layer + ahead
                            if next_req < self._num_layers:
                                _async_loader.request_layer(next_req)
                    else:
                        # Synchronous per-layer load (original path)
                        if cur_layer >= 0:
                            _evict_layer(cur_layer, env)
                        _load_layer(instr_layer, env)

                    # Refresh the fast lookup after env changes
                    _env_get = env.get
                elif pcache is not None:
                    # Non-lazy: just send pcache hints
                    if cur_layer >= 0:
                        pcache.mark_layer_done(cur_layer)
                    pcache.set_active_layer(instr_layer)
                cur_layer = instr_layer

            # Normal call_function is the most frequent path
            if dt == _DT_NORMAL:
                raw_args = instr["_args"]
                if instr["_has_kwargs"]:
                    args = [_resolve(a, env) for a in raw_args]
                    kwargs = {k: _resolve(v, env)
                              for k, v in instr["_kwargs"].items()}
                else:
                    # Fast path: no kwargs (most ops) — inline resolve
                    args = []
                    for a in raw_args:
                        if isinstance(a, str):
                            args.append(_env_get(a, a))
                        elif isinstance(a, list):
                            args.append(_resolve(a, env))
                        else:
                            args.append(a)
                    kwargs = {}
                try:
                    env[name] = instr["_handler"](conn, args, kwargs)
                    if _trim is not None:
                        _trim()
                except Exception as exc:
                    raise RuntimeError(
                        f"Error executing {name} ({instr['target']}): {exc}"
                    ) from exc
                continue

            if dt == _DT_FUSED:
                fused_type = instr["_fused_type"]
                fused_args = instr["_fused_args"]
                if fused_type == "silu_mul":
                    env[name] = _sql1(conn,
                                      "SELECT llm_silu_mul_nd(?, ?)",
                                      [_resolve(fused_args[0], env),
                                       _resolve(fused_args[1], env)])
                elif fused_type == "rope":
                    env[name] = _sql1(conn,
                                      "SELECT llm_rope_nd(?, ?, ?)",
                                      [_resolve(fused_args[0], env),
                                       _resolve(fused_args[1], env),
                                       _resolve(fused_args[2], env)])
                elif fused_type == "rmsnorm":
                    env[name] = _sql1(conn,
                                      "SELECT llm_rmsnorm_nd(?, ?, ?)",
                                      [_resolve(fused_args[0], env),
                                       _resolve(fused_args[1], env),
                                       float(fused_args[2])])
                if _trim is not None:
                    _trim()
                continue

            if dt == _DT_LAZY:
                blob_ref = instr["_pre_blob"]
                blob = _env_get(blob_ref, blob_ref) if isinstance(
                    blob_ref, str) else blob_ref
                env[name] = _sql1_lazy(
                    instr["_pre_sql"], [blob] + instr["_pre_static"])
                if _trim is not None:
                    _trim()
                continue

            if dt == _DT_PLACEHOLDER:
                if name not in env:
                    if user_input_count == 0:
                        env[name] = input_ids_blob
                        user_input_count += 1
                    elif past_key_values is not None:
                        kv_idx = user_input_count - 1
                        layer_idx = kv_idx // 2
                        is_key = (kv_idx % 2 == 0)
                        if layer_idx < len(past_key_values):
                            kv_pair = past_key_values[layer_idx]
                            env[name] = kv_pair[0] if is_key else kv_pair[1]
                        user_input_count += 1
                continue

            if dt == _DT_ALIAS:
                env[name] = _env_get(instr["_alias_target"])
                continue

            if dt == _DT_OUTPUT:
                vals = []
                for a in instr["_output_args"]:
                    vals.append(_resolve(a, env))
                output_value = tuple(vals)
                continue

            # dt == _DT_SKIP or unknown — skip

        # Evict the last layer's params and mark done
        if _lazy and cur_layer >= 0:
            if _async_loader is not None:
                _async_loader.release_layer(cur_layer)
            _evict_layer(cur_layer, env)
        elif pcache is not None and cur_layer >= 0:
            pcache.mark_layer_done(cur_layer)

        if output_value is None:
            raise RuntimeError("Graph had no output node")
        return output_value

    # ── Inference ──

    def _execute_single_sql(
        self,
        graph_name: str,
        input_ids_blob: bytes,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        """Execute a pre-compiled graph SQL statement in one SQLite call."""
        if graph_name == "prefill":
            sql = self._compiled_prefill_sql
            bind_names = self._compiled_prefill_bind_names
        elif graph_name == "decode":
            sql = self._compiled_decode_sql
            bind_names = self._compiled_decode_bind_names
        else:
            raise ValueError(f"Unknown graph name: {graph_name!r}")

        if sql is None:
            raise RuntimeError(f"single_sql_mode graph {graph_name!r} is not compiled")

        params = []
        for name in bind_names:
            if name == "input_ids":
                params.append(input_ids_blob)
                continue
            match = _PAST_KEY_VALUES_RE.fullmatch(name)
            if match is None:
                raise RuntimeError(
                    f"Unsupported runtime placeholder {name!r} in single_sql_mode"
                )
            if past_key_values is None:
                raise RuntimeError(
                    f"Missing past_key_values for placeholder {name!r}"
                )
            layer_idx = int(match.group(1))
            value_idx = int(match.group(2))
            if layer_idx >= len(past_key_values):
                raise RuntimeError(
                    f"KV cache layer {layer_idx} missing for placeholder {name!r}"
                )
            params.append(past_key_values[layer_idx][value_idx])

        row = self._conn.execute(sql, params).fetchone()
        if row is None:
            raise RuntimeError(f"single_sql_mode graph {graph_name!r} returned no row")
        return tuple(row)

    def _execute_graph(
        self,
        graph_name: str,
        input_ids_blob: bytes,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        """Execute either the compiled single-SQL graph or the per-op graph."""
        if self._compiled_sql_enabled:
            return self._execute_single_sql(
                graph_name,
                input_ids_blob,
                past_key_values=past_key_values,
            )
        if graph_name == "prefill":
            instrs = self._prefill_instrs
        elif graph_name == "decode":
            instrs = self._decode_instrs
        else:
            raise ValueError(f"Unknown graph name: {graph_name!r}")
        return self._execute_instrs(instrs, input_ids_blob, past_key_values)

    def _execute_instrs_profiled(
        self,
        instrs: list,
        input_ids_blob: bytes,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        """Like ``_execute_instrs`` but collects per-op timing.

        After this call, ``self._last_op_timings`` contains a dict
        mapping op name → total microseconds, and
        ``self._last_sql_call_count`` holds the number of SQL calls made.
        """
        from collections import defaultdict
        op_timings: Dict[str, float] = defaultdict(float)
        sql_call_count = 0

        if self._lazy_params:
            env: dict = {}
            self._load_layer_params(-1, env)
        else:
            env = {} if self._single_sql_mode else dict(self._param_blobs)
        conn = self._conn
        _resolve = self._resolve
        _perf = _time.perf_counter
        _trim = _maybe_malloc_trim if self._single_sql_mode else None

        _lazy = self._lazy_params
        _load_layer = self._load_layer_params if _lazy else None
        _evict_layer = self._evict_layer_params if _lazy else None
        cur_layer = -2

        user_input_count = 0
        output_value = None

        for instr in instrs:
            op = instr["op"]
            name = instr["name"]
            t0 = _perf()

            # Per-layer streaming (same as _execute_instrs)
            instr_layer = instr.get("_layer", -1)
            if instr_layer != cur_layer and instr_layer >= 0:
                if _lazy:
                    if cur_layer >= 0:
                        _evict_layer(cur_layer, env)
                    _load_layer(instr_layer, env)
                cur_layer = instr_layer

            if op == "call_function":
                handler = instr["_handler"]
                if handler is None:
                    continue

                fused_type = instr.get("_fused_type")
                if fused_type is not None:
                    fused_args = instr["_fused_args"]
                    if fused_type == "silu_mul":
                        env[name] = _sql1(conn,
                                          "SELECT llm_silu_mul_nd(?, ?)",
                                          [_resolve(fused_args[0], env),
                                           _resolve(fused_args[1], env)])
                    elif fused_type == "rope":
                        env[name] = _sql1(conn,
                                          "SELECT llm_rope_nd(?, ?, ?)",
                                          [_resolve(fused_args[0], env),
                                           _resolve(fused_args[1], env),
                                           _resolve(fused_args[2], env)])
                    elif fused_type == "rmsnorm":
                        env[name] = _sql1(conn,
                                          "SELECT llm_rmsnorm_nd(?, ?, ?)",
                                          [_resolve(fused_args[0], env),
                                           _resolve(fused_args[1], env),
                                           float(fused_args[2])])
                    if _trim is not None:
                        _trim()
                    elapsed = (_perf() - t0) * 1e6
                    op_timings[f"fused_{fused_type}"] += elapsed
                    sql_call_count += 1
                    continue

                raw_args = instr["args"]
                raw_kwargs = instr["kwargs"]
                if raw_kwargs:
                    args = [_resolve(a, env) for a in raw_args]
                    kwargs = {k: _resolve(v, env)
                              for k, v in raw_kwargs.items()}
                else:
                    args = []
                    for a in raw_args:
                        if isinstance(a, str):
                            args.append(env.get(a, a))
                        elif isinstance(a, list):
                            args.append(_resolve(a, env))
                        else:
                            args.append(a)
                    kwargs = {}

                result = handler(conn, args, kwargs)
                env[name] = result
                if _trim is not None and not isinstance(result, SqlExpr):
                    _trim()

                elapsed = (_perf() - t0) * 1e6
                tgt = instr.get("target", "")
                if isinstance(tgt, str) and "aten." in tgt:
                    tgt = tgt.split("aten.")[1].split(".")[0]
                # Lazy ops (SqlExpr) don't issue SQL
                if not isinstance(result, SqlExpr):
                    sql_call_count += 1
                    op_timings[str(tgt)] += elapsed
                else:
                    op_timings[f"lazy_{tgt}"] += elapsed
                continue

            if op == "placeholder":
                if name not in env:
                    if user_input_count == 0:
                        env[name] = input_ids_blob
                        user_input_count += 1
                    elif past_key_values is not None:
                        kv_idx = user_input_count - 1
                        layer_idx = kv_idx // 2
                        is_key = (kv_idx % 2 == 0)
                        if layer_idx < len(past_key_values):
                            kv_pair = past_key_values[layer_idx]
                            env[name] = kv_pair[0] if is_key else kv_pair[1]
                        user_input_count += 1
                elapsed = (_perf() - t0) * 1e6
                op_timings["placeholder"] += elapsed
                continue

            if op == "alias":
                env[name] = env.get(instr["target"])
                elapsed = (_perf() - t0) * 1e6
                op_timings["alias"] += elapsed
                continue

            if op == "output":
                vals = []
                for a in instr["output_args"]:
                    vals.append(_resolve(a, env))
                output_value = tuple(vals)
                elapsed = (_perf() - t0) * 1e6
                op_timings["output"] += elapsed
                continue

        self._last_op_timings = dict(op_timings)
        self._last_sql_call_count = sql_call_count

        # Evict last layer in lazy mode
        if _lazy and cur_layer >= 0:
            _evict_layer(cur_layer, env)

        if output_value is None:
            raise RuntimeError("Graph had no output node")
        return output_value

    def generate(
        self,
        prompt: str,
        max_tokens: int = 20,
        verbose: bool = False,
        profile: bool = False,
    ) -> str:
        """Generate text from a prompt.

        Parameters
        ----------
        prompt : str
            Input text.
        max_tokens : int
            Maximum number of tokens to generate.
        verbose : bool
            Print per-step details.
        profile : bool
            When ``True``, collect timing and memory metrics and store
            them in ``self.last_profile`` (a dict) after generation.
            The decode loop always uses the fast (non-profiled) execution
            path so that ``tokens_per_sec`` reflects real throughput.
            A single extra profiled step is run afterwards to populate
            ``op_timings_us`` and ``sql_call_count``.

        Returns the generated text (excluding the prompt).
        """
        # Lazy imports for profiling utilities (only stdlib otherwise)
        if profile:
            from src.utils.perf_timer import PerfTimer
            from src.utils.memory_profiler import MemoryTracker
            timer = PerfTimer()
            mem = MemoryTracker()
            decode_step_ms: List[float] = []
            mem.snapshot("start")
        else:
            timer = None  # type: ignore[assignment]
            mem = None     # type: ignore[assignment]

        prompt_ids = self._tokenizer.encode(prompt)
        if not prompt_ids:
            raise ValueError("Prompt produced no tokens")

        if verbose:
            print(f"[standalone] prompt ids ({len(prompt_ids)} tokens)")

        # Always use the fast execution path for actual generation so
        # throughput numbers are not polluted by per-op timing overhead.
        _exec = self._execute_graph

        # ── Prefill ──────────────────────────────────────────────
        input_blob = self._make_input_ids_blob(prompt_ids)

        # Trigger prefetch for early layers before prefill starts
        if self._memory_mgr is not None:
            self._memory_mgr.begin_forward_pass()
            self._memory_mgr.prefetch_layer(0)

        # Reset native pcache layer states for new forward pass
        if self._pcache is not None:
            self._pcache.reset_layers()
        if self._async_loader is not None:
            self._async_loader.reset()
            self._async_loader.set_prefill_mode()  # prefill: lookahead=1

        prefill_t0 = _time.perf_counter()
        output = _exec("prefill", input_blob)
        prefill_ms = (_time.perf_counter() - prefill_t0) * 1000

        if profile:
            timer.record_duration("prefill", prefill_ms / 1000)
            mem.snapshot("after_prefill")

        logits_blob = output[0]
        kv_flat = output[1:]
        kv_cache: Optional[Tuple] = None
        if len(kv_flat) >= 2 and len(kv_flat) % 2 == 0:
            n_layers = len(kv_flat) // 2
            kv_cache = tuple(
                (kv_flat[2 * i], kv_flat[2 * i + 1])
                for i in range(n_layers)
            )

        first_token = argmax_last_row(logits_blob, self._vocab_size)
        generated_ids: List[int] = [first_token]

        if verbose:
            ch = self._tokenizer.decode([first_token])
            print(f"  [prefill]  token={first_token:3d}  char={ch!r}")

        # ── Autoregressive decode ────────────────────────────────
        if self._async_loader is not None:
            self._async_loader.set_decode_mode()  # decode: lookahead=2
        for step in range(1, max_tokens):
            tok_blob = self._make_input_ids_blob([generated_ids[-1]])

            # Reset async loader state for this decode step
            if self._async_loader is not None:
                self._async_loader.reset()

            # Trigger prefetch for decode step
            if self._memory_mgr is not None:
                self._memory_mgr.begin_forward_pass()
                self._memory_mgr.prefetch_layer(0)

            step_t0 = _time.perf_counter()
            output = _exec(
                "decode", tok_blob,
                past_key_values=kv_cache)
            step_ms = (_time.perf_counter() - step_t0) * 1000

            if profile:
                decode_step_ms.append(step_ms)

            logits_blob = output[0]
            kv_flat = output[1:]
            if len(kv_flat) >= 2 and len(kv_flat) % 2 == 0:
                n_layers = len(kv_flat) // 2
                kv_cache = tuple(
                    (kv_flat[2 * i], kv_flat[2 * i + 1])
                    for i in range(n_layers)
                )

            next_token = argmax_last_row(logits_blob, self._vocab_size)
            generated_ids.append(next_token)

            if verbose:
                ch = self._tokenizer.decode([next_token])
                print(f"  [decode {step:2d}]  token={next_token:3d}  char={ch!r}")

            if next_token == self._tokenizer.EOS_ID:
                break

        output_text = self._tokenizer.decode(generated_ids)

        # ── Build profile report ─────────────────────────────────
        if profile:
            total_decode_ms = sum(decode_step_ms)
            timer.record_duration("decode_total", total_decode_ms / 1000)
            mem.snapshot("end")

            num_decode = len(decode_step_ms)
            avg_decode = total_decode_ms / num_decode if num_decode else 0
            tokens_per_sec = (num_decode / (total_decode_ms / 1000)
                              if total_decode_ms > 0 else 0)

            # Run one extra step to collect SQL call counts without
            # affecting the throughput numbers.  In legacy mode we also
            # collect per-op timings from that replay step.
            if num_decode > 0 and kv_cache is not None:
                replay_blob = self._make_input_ids_blob(
                    [generated_ids[-1]])
                if self._compiled_sql_enabled:
                    self._execute_single_sql(
                        "decode", replay_blob,
                        past_key_values=kv_cache,
                    )
                    self._last_op_timings = {}
                    self._last_sql_call_count = 1
                else:
                    self._execute_instrs_profiled(
                        self._decode_instrs, replay_blob,
                        past_key_values=kv_cache)

            self.last_profile = {
                "prompt_tokens": len(prompt_ids),
                "generated_tokens": len(generated_ids),
                "prefill_ms": prefill_ms,
                "decode_steps": num_decode,
                "decode_total_ms": total_decode_ms,
                "decode_avg_ms": avg_decode,
                "decode_step_ms": decode_step_ms,
                "tokens_per_sec": tokens_per_sec,
                "timer_rows": timer.to_rows(),
                "memory_rows": mem.to_rows(),
                "memory_summary": mem.summary(),
                "op_timings_us": getattr(self, "_last_op_timings", {}),
                "sql_call_count": getattr(
                    self, "_last_sql_call_count", 0),
            }

        return output_text

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the underlying SQLite connection."""
        return self._conn

    def set_threads(self, num_threads: int) -> None:
        """Set the llm_ops runtime thread count for parallel kernels."""
        self._conn.execute("SELECT llm_set_threads(?)", [int(num_threads)]).fetchone()

    def close(self) -> None:
        """Close the database connection and memory manager."""
        if getattr(self, "_async_loader", None) is not None:
            self._async_loader.stop()
            self._async_loader = None
        if getattr(self, "_memory_mgr", None) is not None:
            self._memory_mgr.close()
            self._memory_mgr = None
        if getattr(self, "_param_manager", None) is not None:
            self._param_manager.close()
            self._param_manager = None
        self._conn.close()
