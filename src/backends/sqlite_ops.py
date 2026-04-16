"""
SQLite operator registry — SQL-driven implementations of torch aten ops.

Each entry maps a torch op target to a Python callable that accepts
``(conn, *args, **kwargs)`` and returns either a BLOB (bytes) representing
the result tensor, or a Python scalar / tuple thereof.

``conn`` is a ``sqlite3.Connection`` with the ``llm_ops.so`` extension
loaded, so all ``llm_*`` SQL functions are available.

The registry is a plain dict keyed by the *actual target object* so we can
do exact-match lookups without stringification.
"""
import operator
import struct
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SqliteOpRegistry: Dict[Any, Callable] = {}

_TORCH_SLICE_END_SENTINEL: int = 2 ** 62
_SQLITE_SLICE_END_SENTINEL: int = (2 ** 31) - 1

_SOFTMAX_EPS: float = 1e-9

# Header size for 1-D N-D blobs: 4 bytes magic + 4 bytes ndim + 4 bytes shape[0] = 12
# For general N-D: 8 + ndim * 4
_ND_HDR_BASE: int = 8   # magic(4) + ndim(4)


def _reg(target: Any) -> Callable[[Callable], Callable]:
    """Decorator: register ``fn`` under ``target`` in SqliteOpRegistry."""
    def decorator(fn: Callable) -> Callable:
        SqliteOpRegistry[target] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Blob encode / decode helpers
# ---------------------------------------------------------------------------

_LLM_ND_MAGIC = struct.unpack('<i', struct.pack('<I', 0x80000000))[0]
_LLM_INT8_MAGIC = struct.unpack('<i', struct.pack('<I', 0x80000008))[0]


def is_int8_blob(blob: bytes) -> bool:
    """Return True if *blob* uses the INT8 quantized format."""
    if blob and len(blob) >= 4:
        return struct.unpack_from('<i', blob, 0)[0] == _LLM_INT8_MAGIC
    return False


def int8_blob_to_numpy(blob: bytes) -> np.ndarray:
    """Decode an INT8 quantized BLOB to a dequantized float32 numpy array.

    INT8 blob layout::

        [magic:i32=0x80000008][ndim:i32][shape[0]:i32…][scales:f32×nrows][data:i8×total]

    Per-row dequantization: ``float_val = int8_val * scale[row]``
    """
    if not blob or len(blob) < 8:
        return np.array([], dtype=np.float32)
    ndim = struct.unpack_from('<i', blob, 4)[0]
    hdr_sz = 8 + ndim * 4
    shape = list(struct.unpack_from(f'<{ndim}i', blob, 8))
    total = 1
    for s in shape:
        total *= s
    row_len = shape[-1] if ndim >= 1 else total
    nrows = total // row_len if row_len > 0 else 0

    # Parse scales (one float32 per row)
    scales_offset = hdr_sz
    scales = np.frombuffer(
        blob[scales_offset:scales_offset + nrows * 4], dtype=np.float32
    ).copy()

    # Parse int8 data
    data_offset = scales_offset + nrows * 4
    qdata = np.frombuffer(
        blob[data_offset:data_offset + total], dtype=np.int8
    ).copy().astype(np.float32)

    # Dequantize: multiply each row by its scale
    qdata = qdata.reshape(nrows, row_len)
    qdata *= scales[:, np.newaxis]
    return qdata.reshape(shape)


def tensor_to_blob(arr: np.ndarray) -> bytes:
    """Encode a numpy array as an N-D tensor BLOB (always uses N-D format)."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    ndim = arr.ndim
    if ndim == 0:
        arr = arr.reshape(1)
        ndim = 1
    shape = list(arr.shape)
    header = (struct.pack('<i', _LLM_ND_MAGIC) +
              struct.pack('<i', ndim) +
              struct.pack(f'<{ndim}i', *shape))
    return header + arr.tobytes()


def blob_to_numpy(blob: bytes) -> np.ndarray:
    """Decode a BLOB to a numpy float32 array.
    Supports N-D tensor blobs, INT8 quantized blobs, legacy vector, and
    legacy matrix formats.
    """
    if not blob or len(blob) < 4:
        return np.array([], dtype=np.float32)
    first_int = struct.unpack_from('<i', blob, 0)[0]
    if first_int == _LLM_INT8_MAGIC:
        return int8_blob_to_numpy(blob)
    if first_int == _LLM_ND_MAGIC:
        ndim = struct.unpack_from('<i', blob, 4)[0]
        header_sz = _ND_HDR_BASE + ndim * 4
        shape = list(struct.unpack_from(f'<{ndim}i', blob, _ND_HDR_BASE))
        total = 1
        for s in shape:
            total *= s
        data = np.frombuffer(blob[header_sz:header_sz + total * 4],
                             dtype=np.float32).copy()
        return data.reshape(shape)
    n = first_int
    vec_size = 4 + n * 4
    if n >= 0 and len(blob) == vec_size:
        return np.frombuffer(blob[4:vec_size], dtype=np.float32).copy()
    if len(blob) >= 8:
        rows = n
        cols = struct.unpack_from('<i', blob, 4)[0]
        if rows >= 0 and cols >= 0:
            mat_size = 8 + rows * cols * 4
            if len(blob) >= mat_size:
                data = np.frombuffer(blob[8:mat_size], dtype=np.float32)
                return data.reshape(rows, cols).copy()
    return np.frombuffer(blob[4:], dtype=np.float32).copy()


def _ensure_blob(x):
    """Convert a numpy array or scalar to blob if needed.

    Also accepts ``str`` (a parameter-name reference for disk-based
    parameter lookup) and ``SqlExpr`` (a lazy SQL expression from the
    inline shape-op fusion pass) and passes them through unchanged so
    that ``_sql1`` can resolve them later.
    """
    if isinstance(x, (bytes, SqlExpr)):
        return x
    if isinstance(x, str):
        # Parameter-name reference – resolved at SQL execution time.
        return x
    if isinstance(x, np.ndarray):
        return tensor_to_blob(x)
    if isinstance(x, (int, float)):
        return tensor_to_blob(np.array([x], dtype=np.float32))
    raise TypeError(f"Cannot convert {type(x)} to blob")


# ---------------------------------------------------------------------------
# Lazy SQL expression for nested SQL fusion
# ---------------------------------------------------------------------------

class SqlExpr:
    """A deferred SQL expression that can be inlined into a consumer's SQL.

    Instead of executing a shape op (unsqueeze, transpose, …) immediately
    and materialising the intermediate BLOB, we store the SQL fragment and
    its parameters.  When the consumer op calls ``_sql1``, the expression
    is expanded inline, turning two round-trips into one nested SQL call::

        -- Before (2 calls):
        SELECT llm_unsqueeze_nd(?, ?)       -- intermediate
        SELECT llm_add_nd(intermediate, ?)

        -- After (1 call):
        SELECT llm_add_nd(llm_unsqueeze_nd(?, ?), ?)

    This is constructed by ``_sql1_lazy`` and consumed transparently by
    ``_sql1``'s parameter resolution.
    """
    __slots__ = ('sql_fragment', 'params')

    def __init__(self, sql_fragment: str, params: list):
        self.sql_fragment = sql_fragment
        self.params = list(params)


# ---------------------------------------------------------------------------
# Disk-parameter resolution helpers
# ---------------------------------------------------------------------------

# Table name used for persisted model parameters.  The table is created
# by ``save_params_to_db`` in ``sqlite_engine.py``; ops only *read* from it.
_PARAM_TABLE = "model_params"
_PARAM_SUBQUERY = f"(SELECT data FROM {_PARAM_TABLE} WHERE name = ?)"


def _resolve_sql_params(sql, params):
    """Replace ``?`` placeholders with inline sub-queries / SQL expressions.

    Handles two kinds of special parameters:

    * ``str`` — a parameter-name reference: the ``?`` is replaced by an
      inline ``SELECT`` from the ``model_params`` table.
    * ``SqlExpr`` — a lazy SQL expression: the ``?`` is replaced by the
      expression's SQL fragment and its own parameters are spliced in.

    Returns ``(new_sql, new_params)`` ready for ``conn.execute()``.
    """
    if not params:
        return sql, params

    # Fast path: nothing to rewrite when no special params present.
    has_special = False
    for p in params:
        if isinstance(p, (str, SqlExpr)):
            has_special = True
            break
    if not has_special:
        return sql, params

    # Build the new SQL and params by scanning through the placeholders.
    parts = sql.split("?")
    new_parts = []
    new_params = []
    for i, p in enumerate(params):
        new_parts.append(parts[i])
        if isinstance(p, SqlExpr):
            new_parts.append(p.sql_fragment)
            new_params.extend(p.params)
        elif isinstance(p, str):
            new_parts.append(_PARAM_SUBQUERY)
            new_params.append(p)
        else:
            new_parts.append("?")
            new_params.append(p)
    if len(parts) > len(params):
        new_parts.append(parts[len(params)])
    return "".join(new_parts), new_params


def _sql1(conn, sql, params=None):
    """Execute *sql* and return a single scalar result.

    Special parameter types:
    * ``str``     → parameter-name reference (inlined as sub-SELECT)
    * ``SqlExpr`` → lazy SQL expression (inlined as nested SQL fragment)
    """
    if params:
        sql, params = _resolve_sql_params(sql, params)
    return conn.execute(sql, params or []).fetchone()[0]


def _sql1_lazy(sql, params):
    """Return a ``SqlExpr`` wrapping *sql* and *params* for deferred execution.

    The expression will be inlined into the consumer's SQL call by
    ``_resolve_sql_params``.  Any nested ``SqlExpr`` objects in *params*
    are recursively flattened.
    """
    # Flatten any nested SqlExpr in params first
    if any(isinstance(p, (str, SqlExpr)) for p in params):
        sql, params = _resolve_sql_params(sql, params)
    return SqlExpr(sql, params)


def _blob_ndim(blob: bytes) -> int:
    """Read the number of dimensions directly from an N-D tensor blob header.

    This avoids an SQL round-trip when the blob is already in Python memory.
    Expects the standard N-D format: [magic:i32][ndim:i32][shape…][data…].
    """
    return struct.unpack_from('<i', blob, 4)[0]


def _blob_shape(blob: bytes, dim: int) -> int:
    """Read the size of *dim* directly from an N-D tensor blob header.

    Supports negative indexing (``dim=-1`` → last dimension).
    Avoids an SQL round-trip when the blob is already in Python memory.
    """
    ndim = _blob_ndim(blob)
    if dim < 0:
        dim += ndim
    return struct.unpack_from('<i', blob, _ND_HDR_BASE + dim * 4)[0]


def _blob_shape_list(blob: bytes) -> list:
    """Read the full shape ``[d0, d1, …]`` directly from an N-D blob header.

    Replaces the previous approach of issuing N+1 SQL queries
    (1 for ndim, N for each dimension), turning an O(ndim) SQL cost
    into an O(1) in-process struct unpack.
    """
    ndim = _blob_ndim(blob)
    return list(struct.unpack_from(f'<{ndim}i', blob, _ND_HDR_BASE))


def _sql_ndim(conn, blob) -> int:
    """Return the number of dimensions of a tensor blob.

    Uses direct header parsing for ``bytes`` blobs (zero SQL cost) and
    falls back to the ``llm_ndim`` SQL function for string parameter
    references that must be resolved at query time.
    """
    if isinstance(blob, bytes):
        return _blob_ndim(blob)
    return int(_sql1(conn, "SELECT llm_ndim(?)", [blob]))


def _sql_shape(conn, blob, dim: int) -> int:
    """Return the size of *dim* for a tensor blob.

    Uses direct header parsing for ``bytes`` blobs and falls back to
    ``llm_shape`` SQL function for string parameter references.
    """
    if isinstance(blob, bytes):
        return _blob_shape(blob, dim)
    return int(_sql1(conn, "SELECT llm_shape(?, ?)", [blob, dim]))


def _sql_shape_list(conn, blob) -> list:
    """Return the full shape list ``[d0, d1, …]`` of a tensor blob.

    For ``bytes`` blobs the shape is read directly from the header in a
    single ``struct.unpack_from`` call — **no SQL round-trips**.  String
    parameter references still go through SQL.
    """
    if isinstance(blob, bytes):
        return _blob_shape_list(blob)
    ndim = _sql_ndim(conn, blob)
    return [_sql_shape(conn, blob, i) for i in range(ndim)]


def _normalize_sql_slice_end(end) -> int:
    """Map Torch's open-ended slice sentinel to a SQLite-safe int."""
    if end is None:
        return _SQLITE_SLICE_END_SENTINEL
    end_i = int(end)
    if end_i >= _TORCH_SLICE_END_SENTINEL:
        return _SQLITE_SLICE_END_SENTINEL
    return end_i


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

def _torch_dtype_to_numpy(dtype) -> np.dtype:
    """Map a ``torch.dtype`` to the corresponding ``numpy.dtype``.

    The SQLite backend stores all tensors as float32, so unknown or
    ``None`` dtypes default to ``np.float32``.  ``bfloat16`` is also
    promoted to ``float32`` since NumPy does not natively support it.
    """
    _MAP = {
        torch.float16:  np.float16,
        torch.float32:  np.float32,
        torch.float64:  np.float64,
        torch.int8:     np.int8,
        torch.int16:    np.int16,
        torch.int32:    np.int32,
        torch.int64:    np.int64,
        torch.bool:     np.bool_,
        torch.bfloat16: np.float32,
    }
    if dtype is None:
        return np.float32
    return _MAP.get(dtype, np.float32)


# ---------------------------------------------------------------------------
# Tensor construction
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten.arange.default)
def _arange_default(conn, end, **kwargs):
    """Create a 1-D tensor ``[0, 1, …, end-1]`` via ``llm_arange_nd``."""
    return _sql1(conn, "SELECT llm_arange_nd(?)", [int(end)])


@_reg(torch.ops.aten.arange.start)
def _arange_start(conn, start, end, **kwargs):
    """Create a 1-D tensor ``[start, start+1, …, end-1]``.

    When *start* is non-zero the arange, fill, and add are fused into a
    single nested SQL call (was 2 sequential calls).
    """
    n = int(end) - int(start)
    if n <= 0:
        return _full(conn, [0], 0.0)
    if int(start) == 0:
        return _sql1(conn, "SELECT llm_arange_nd(?)", [n])
    # Fuse arange + offset into one SQL round-trip (was 2)
    return _sql1(conn,
                 "SELECT llm_add_nd(llm_arange_nd(?), llm_full_nd(?, ?))",
                 [n, float(start), n])


@_reg(torch.ops.aten.full.default)
def _full(conn, size, fill_value, **kwargs):
    """Create a tensor of shape *size* filled with *fill_value*.

    Uses the variadic ``llm_full_nd(fill, d0, d1, …)`` SQL function.
    A 0-D (scalar) tensor is stored as a 1-D tensor of length 1.
    """
    if isinstance(size, (list, tuple)) and len(size) == 0:
        # 0-D scalar tensor — encode as 1-D of length 1 then view to scalar
        return _sql1(conn, "SELECT llm_full_nd(?, ?)",
                     [float(fill_value), 1])
    args = [float(fill_value)] + [int(s) for s in size]
    placeholders = ", ".join(["?"] * len(args))
    return _sql1(conn, f"SELECT llm_full_nd({placeholders})", args)


@_reg(torch.ops.aten.zeros.default)
def _zeros(conn, size, **kwargs):
    """Create a tensor of shape *size* filled with zeros."""
    return _full(conn, size, 0.0)


@_reg(torch.ops.aten.ones.default)
def _ones(conn, size, **kwargs):
    """Create a tensor of shape *size* filled with ones."""
    return _full(conn, size, 1.0)


@_reg(torch.ops.aten.new_ones.default)
def _new_ones(conn, x, size, **kwargs):
    """Create a new tensor of shape *size* filled with ones (same dtype as *x*)."""
    return _full(conn, size, 1.0)


# ---------------------------------------------------------------------------
# Assertion / validation (no-ops)
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten._assert_tensor_metadata.default)
def _assert_tensor_metadata(conn, x, *args, **kwargs):
    """Tensor metadata assertion — no-op in the SQLite backend."""
    return x


# ---------------------------------------------------------------------------
# Logical ops
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten.__and__.Tensor)
def _and_tensor(conn, a, b):
    """Element-wise logical AND with broadcasting support.

    If one operand is a scalar (0-D), broadcasts it before applying AND.
    """
    a_blob = _ensure_blob(a)
    b_blob = _ensure_blob(b)

    a_ndim = _blob_ndim(a_blob)
    b_ndim = _blob_ndim(b_blob)

    # Handle scalar broadcasting: expand scalar to match the other tensor
    if a_ndim == 0 or (a_ndim == 1 and _blob_shape(a_blob, 0) == 1):
        b_shape = _blob_shape_list(b_blob)
        a_blob = _sql1(conn, f"SELECT llm_expand_nd(?, {', '.join('?' * len(b_shape))})",
                        [a_blob] + b_shape)
    elif b_ndim == 0 or (b_ndim == 1 and _blob_shape(b_blob, 0) == 1):
        a_shape = _blob_shape_list(a_blob)
        b_blob = _sql1(conn, f"SELECT llm_expand_nd(?, {', '.join('?' * len(a_shape))})",
                        [b_blob] + a_shape)

    return _sql1(conn, "SELECT llm_logical_and_nd(?, ?)", [a_blob, b_blob])


# ---------------------------------------------------------------------------
# Type casting
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten._to_copy.default)
def _to_copy(conn, x, **kwargs):
    """Type-cast / copy operation — no-op in the SQLite backend.

    All tensors are stored as float32 blobs, so dtype conversion is
    unnecessary; the original blob is returned unchanged.
    """
    return x


@_reg(torch.ops.aten.to.dtype)
def _to_dtype(conn, x, dtype, **kwargs):
    """Type conversion to specified dtype — no-op in the SQLite backend.

    All tensors are stored as float32 blobs, so dtype conversion is
    unnecessary; the original blob is returned unchanged.
    """
    return x


@_reg(torch.ops.aten.to.dtype_layout)
def _to_dtype_layout(conn, x, **kwargs):
    """Type conversion with layout specification — no-op in the SQLite backend.

    All tensors are stored as float32 blobs, so dtype conversion is
    unnecessary; the original blob is returned unchanged.
    """
    return x


@_reg(torch.ops.aten.lift_fresh_copy.default)
def _lift_fresh_copy(conn, x):
    """Lift fresh copy operation — no-op in the SQLite backend.

    This operation is used to create a fresh copy of a tensor.
    Since we don't have aliasing concerns in the SQLite backend,
    we can just return the input unchanged.
    """
    return x


@_reg(torch.ops.aten.detach_.default)
def _detach_(conn, x):
    """Detach operation (in-place) — no-op in the SQLite backend.

    Detaches a tensor from the computation graph. Since we don't
    track gradients in the SQLite backend, this is a no-op.
    """
    return x


# ---------------------------------------------------------------------------
# Shape manipulation
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten.view.default)
def _view(conn, x, shape):
    """Reshape tensor *x* to *shape* via ``llm_view_nd``.

    Accepts ``-1`` in one dimension to infer the size.  Uses the variadic
    ``llm_view_nd(blob, d0, d1, …)`` SQL function — a single SQL call.
    """
    shape = [int(s) for s in shape]
    args = [x] + shape
    placeholders = ", ".join(["?"] * len(args))
    return _sql1(conn, f"SELECT llm_view_nd({placeholders})", args)


@_reg(torch.ops.aten._unsafe_view.default)
def _unsafe_view(conn, x, shape):
    """Alias for ``_view`` — no safety distinction in the SQLite backend."""
    return _view(conn, x, shape)


@_reg(torch.ops.aten.reshape.default)
def _reshape(conn, x, shape):
    """Reshape tensor — delegates to ``_view`` (same semantics for contiguous data)."""
    return _view(conn, x, shape)


@_reg(torch.ops.aten.unsqueeze.default)
def _unsqueeze(conn, x, dim):
    """Insert a length-1 dimension at position *dim*."""
    dim = int(dim)

    # Handle negative dimensions - convert to positive
    if dim < 0:
        ndim = _sql_ndim(conn, x)  # 0 SQL cost for bytes blobs
        dim = dim % (ndim + 1)  # unsqueeze adds one dimension

    return _sql1(conn, "SELECT llm_unsqueeze_nd(?, ?)", [x, dim])


@_reg(torch.ops.aten.squeeze.dim)
def _squeeze_dim(conn, x, dim):
    """Remove the dimension at position *dim* if its size is 1."""
    dim = int(dim)

    # Handle negative dimensions - convert to positive
    if dim < 0:
        ndim = _sql_ndim(conn, x)  # 0 SQL cost for bytes blobs
        dim = dim % ndim

    return _sql1(conn, "SELECT llm_squeeze_nd(?, ?)", [x, dim])


@_reg(torch.ops.aten.squeeze.default)
def _squeeze(conn, x):
    """Remove all length-1 dimensions from *x*."""
    return _sql1(conn, "SELECT llm_squeeze_all_nd(?)", [x])


@_reg(torch.ops.aten.expand.default)
def _expand(conn, x, size):
    """Broadcast *x* to the given *size*.

    Passes ``-1`` entries through to the C extension, which interprets
    them as "keep the source dimension unchanged".
    """
    args = [x] + [int(s) for s in size]
    placeholders = ", ".join(["?"] * len(args))
    return _sql1(conn, f"SELECT llm_expand_nd({placeholders})", args)


@_reg(torch.ops.aten.transpose.int)
def _transpose(conn, x, dim0, dim1):
    """Swap dimensions *dim0* and *dim1*."""
    return _sql1(conn, "SELECT llm_transpose_nd(?, ?, ?)",
                 [x, int(dim0), int(dim1)])


@_reg(torch.ops.aten.permute.default)
def _permute(conn, x, dims):
    """Reorder dimensions of *x* according to *dims*.

    Uses the variadic ``llm_permute_nd(blob, d0, d1, …)`` SQL function.
    """
    dims = [int(d) for d in dims]
    args = [x] + dims
    placeholders = ", ".join(["?"] * len(args))
    return _sql1(conn, f"SELECT llm_permute_nd({placeholders})", args)


@_reg(torch.ops.aten.contiguous.default)
def _contiguous(conn, x, **kwargs):
    """Ensure tensor is contiguous in memory — no-op (blobs are always contiguous)."""
    return x


@_reg(torch.ops.aten.clone.default)
def _clone(conn, x, **kwargs):
    """Clone tensor — no-op (blobs are immutable ``bytes`` objects)."""
    return x


@_reg(torch.ops.aten.slice.Tensor)
def _slice(conn, x, dim=0, start=None, end=None, step=1):
    """Extract a slice ``x[start:end:step]`` along *dim*.

    Torch's open-ended sentinel (2**62) is capped to a SQLite-safe int.
    """
    s = int(start) if start is not None else 0
    e = _normalize_sql_slice_end(end)
    return _sql1(conn, "SELECT llm_slice_nd(?, ?, ?, ?, ?)",
                 [x, int(dim), s, e, int(step)])


@_reg(torch.ops.aten.select.int)
def _select(conn, x, dim, index):
    """Select a single index along *dim*, removing that dimension.

    Negative *dim* / *index* are normalized via blob header parsing
    (zero SQL cost for ``bytes`` inputs).  The operation is fused into
    a single nested SQL expression: ``squeeze(slice(x, dim, i, i+1))``.
    """
    dim = int(dim)
    index = int(index)
    # Normalize negative dim/index via blob header (no SQL round-trip)
    if isinstance(x, bytes):
        ndim = _blob_ndim(x)
        if dim < 0:
            dim += ndim
        if index < 0:
            index += _blob_shape(x, dim)
    else:
        ndim = _sql_ndim(conn, x)
        if dim < 0:
            dim += ndim
        if index < 0:
            index += _sql_shape(conn, x, dim)
    # Fuse slice + squeeze into a single nested SQL expression (1 call)
    return _sql1(
        conn,
        "SELECT llm_squeeze_nd(llm_slice_nd(?, ?, ?, ?), ?)",
        [x, dim, index, index + 1, dim],
    )


@_reg(torch.ops.aten.cat.default)
def _cat(conn, tensors, dim=0):
    """Concatenate tensors along *dim* via ``llm_cat_multi_nd``.

    Uses the variadic ``llm_cat_multi_nd(dim, t0, t1, …)`` SQL function
    so that all tensors are concatenated in a single SQL call.

    Empty tensors (0 elements) are filtered out before concatenation,
    matching PyTorch semantics where ``cat([empty, x], dim)`` returns ``x``.
    """
    dim = int(dim)

    # Filter out empty (0-element) tensors — PyTorch cat skips them.
    non_empty = []
    for t in tensors:
        if isinstance(t, bytes):
            shape = _blob_shape_list(t)
            if any(s == 0 for s in shape):
                continue
        elif isinstance(t, str):
            # Param reference — check shape via SQL
            shape = _sql_shape_list(conn, t)
            if any(s == 0 for s in shape):
                continue
        non_empty.append(t)

    if len(non_empty) == 0:
        # All tensors empty — return the first one
        return tensors[0]
    if len(non_empty) == 1:
        return non_empty[0]

    # Handle negative dimensions - convert to positive
    if dim < 0:
        first_tensor = non_empty[0]
        ndim = _sql_ndim(conn, first_tensor)
        dim = dim % ndim  # Convert negative to positive

    args = [dim] + non_empty
    placeholders = ", ".join(["?"] * len(args))
    return _sql1(conn, f"SELECT llm_cat_multi_nd({placeholders})", args)


@_reg(torch.ops.aten.stack.default)
def _stack(conn, tensors, dim=0):
    """Stack tensors along a new dimension *dim* via ``llm_stack_nd``.

    Uses the variadic ``llm_stack_nd(dim, t0, t1, …)`` SQL function —
    a single SQL call regardless of the number of inputs.
    """
    dim = int(dim)

    # Handle negative dimensions - convert to positive
    if dim < 0:
        # Get the rank of the first tensor, then add 1 since stack adds a dimension
        first_tensor = tensors[0]
        ndim = _sql_ndim(conn, first_tensor)  # 0 SQL cost for bytes blobs
        dim = dim % (ndim + 1)  # Stack adds one dimension

    args = [dim] + list(tensors)
    placeholders = ", ".join(["?"] * len(args))
    return _sql1(conn, f"SELECT llm_stack_nd({placeholders})", args)


@_reg(torch.ops.aten.flatten.using_ints)
def _flatten(conn, x, start_dim=0, end_dim=-1):
    """Flatten dimensions ``[start_dim .. end_dim]`` into a single dimension."""
    return _sql1(conn, "SELECT llm_flatten_nd(?, ?, ?)",
                 [x, int(start_dim), int(end_dim)])


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten.add.Tensor)
def _add(conn, a, b, alpha=1):
    """Element-wise addition: ``a + alpha * b``.

    When *alpha* ≠ 1, the scale and add are fused into a single nested
    SQL call: ``llm_add_nd(a, llm_scale_nd(b, alpha))``.
    Broadcasting is handled by the C extension.
    """
    a_blob = _ensure_blob(a)
    b_blob = _ensure_blob(b)
    if alpha != 1:
        return _sql1(conn,
                     "SELECT llm_add_nd(?, llm_scale_nd(?, ?))",
                     [a_blob, b_blob, float(alpha)])
    return _sql1(conn, "SELECT llm_add_nd(?, ?)", [a_blob, b_blob])


@_reg(torch.ops.aten.sub.Tensor)
def _sub(conn, a, b, alpha=1):
    """Element-wise subtraction: ``a - alpha * b``.

    When *alpha* ≠ 1, the scale and sub are fused into a single nested
    SQL call: ``llm_sub_nd(a, llm_scale_nd(b, alpha))``.
    """
    a_blob = _ensure_blob(a)
    b_blob = _ensure_blob(b)
    if alpha != 1:
        return _sql1(conn,
                     "SELECT llm_sub_nd(?, llm_scale_nd(?, ?))",
                     [a_blob, b_blob, float(alpha)])
    return _sql1(conn, "SELECT llm_sub_nd(?, ?)", [a_blob, b_blob])


@_reg(torch.ops.aten.mul.Tensor)
def _mul(conn, a, b):
    """Element-wise multiplication."""
    return _sql1(conn, "SELECT llm_mul_nd(?, ?)",
                 [_ensure_blob(a), _ensure_blob(b)])


@_reg(torch.ops.aten.div.Tensor)
def _div(conn, a, b):
    """Element-wise division."""
    return _sql1(conn, "SELECT llm_div_nd(?, ?)",
                 [_ensure_blob(a), _ensure_blob(b)])


@_reg(torch.ops.aten.div.Scalar)
def _div_scalar(conn, a, b):
    """Divide tensor *a* by scalar *b*.

    Uses ``llm_scale_nd(a, 1/b)`` for non-zero scalars to avoid
    creating a full tensor for the divisor.
    """
    if isinstance(b, (int, float)) and b != 0:
        return _sql1(conn, "SELECT llm_scale_nd(?, 1.0 / ?)",
                      [_ensure_blob(a), float(b)])
    return _div(conn, a, _ensure_blob(b))


@_reg(torch.ops.aten.neg.default)
def _neg(conn, x):
    """Element-wise negation: ``-x``."""
    return _sql1(conn, "SELECT llm_neg_nd(?)", [x])


@_reg(torch.ops.aten.abs.default)
def _abs(conn, x):
    """Element-wise absolute value."""
    return _sql1(conn, "SELECT llm_abs_nd(?)", [x])


@_reg(torch.ops.aten.pow.Tensor_Scalar)
def _pow_ts(conn, x, exponent):
    """Raise every element of *x* to *exponent*."""
    return _sql1(conn, "SELECT llm_pow_nd(?, ?)",
                 [x, float(exponent)])


@_reg(torch.ops.aten.pow.Scalar)
def _pow_s(conn, base, exponent):
    """Raise scalar *base* to tensor *exponent* element-wise."""
    return _sql1(conn, "SELECT llm_pow_scalar_nd(?, ?)",
                 [float(base), exponent])


@_reg(torch.ops.aten.sqrt.default)
def _sqrt(conn, x):
    """Element-wise square root."""
    return _sql1(conn, "SELECT llm_sqrt_nd(?)", [x])


@_reg(torch.ops.aten.rsqrt.default)
def _rsqrt(conn, x):
    """Element-wise reciprocal square root: ``1 / sqrt(x)``."""
    return _sql1(conn, "SELECT llm_rsqrt_nd(?)", [x])


@_reg(torch.ops.aten.exp.default)
def _exp(conn, x):
    """Element-wise exponential."""
    return _sql1(conn, "SELECT llm_exp_nd(?)", [x])


@_reg(torch.ops.aten.log.default)
def _log(conn, x):
    """Element-wise natural logarithm."""
    return _sql1(conn, "SELECT llm_log_nd(?)", [x])


@_reg(torch.ops.aten.sin.default)
def _sin(conn, x):
    """Element-wise sine."""
    return _sql1(conn, "SELECT llm_sin_nd(?)", [x])


@_reg(torch.ops.aten.cos.default)
def _cos(conn, x):
    """Element-wise cosine."""
    return _sql1(conn, "SELECT llm_cos_nd(?)", [x])


@_reg(torch.ops.aten.tanh.default)
def _tanh(conn, x):
    """Element-wise hyperbolic tangent."""
    return _sql1(conn, "SELECT llm_tanh_nd(?)", [x])


# ---------------------------------------------------------------------------
# Comparison (return boolean-like tensor: 1.0 = True, 0.0 = False)
# ---------------------------------------------------------------------------

def _cmp_op(conn, a, b, op):
    """Dispatch an element-wise comparison to the appropriate ``llm_*_nd`` function.

    Returns a float32 tensor where 1.0 = True and 0.0 = False.
    """
    sql_fn = {
        'gt': 'llm_gt_nd',
        'lt': 'llm_lt_nd',
        'ge': 'llm_ge_nd',
        'le': 'llm_le_nd',
        'eq': 'llm_eq_nd',
        'ne': 'llm_ne_nd',
    }.get(op)
    if sql_fn is None:
        raise ValueError(f"Unknown cmp op: {op}")
    return _sql1(conn, f"SELECT {sql_fn}(?, ?)", [_ensure_blob(a), _ensure_blob(b)])


@_reg(torch.ops.aten.gt.Tensor)
def _gt(conn, a, b):
    """Element-wise greater-than (tensor vs tensor)."""
    return _cmp_op(conn, a, b, 'gt')


@_reg(torch.ops.aten.gt.Scalar)
def _gt_scalar(conn, a, b):
    """Element-wise greater-than (tensor vs scalar)."""
    return _cmp_op(conn, a, b, 'gt')


@_reg(torch.ops.aten.lt.Tensor)
def _lt(conn, a, b):
    """Element-wise less-than (tensor vs tensor)."""
    return _cmp_op(conn, a, b, 'lt')


@_reg(torch.ops.aten.ge.Tensor)
def _ge(conn, a, b):
    """Element-wise greater-than-or-equal (tensor vs tensor)."""
    return _cmp_op(conn, a, b, 'ge')


@_reg(torch.ops.aten.le.Tensor)
def _le(conn, a, b):
    """Element-wise less-than-or-equal (tensor vs tensor)."""
    return _cmp_op(conn, a, b, 'le')


@_reg(torch.ops.aten.eq.Tensor)
def _eq(conn, a, b):
    """Element-wise equality (tensor vs tensor)."""
    return _cmp_op(conn, a, b, 'eq')


@_reg(torch.ops.aten.ne.Tensor)
def _ne(conn, a, b):
    """Element-wise not-equal (tensor vs tensor)."""
    return _cmp_op(conn, a, b, 'ne')


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten.mean.default)
def _mean(conn, x):
    """Global mean reduction — returns a scalar tensor."""
    return _sql1(conn, "SELECT llm_reduce_mean_nd(?)", [x])


@_reg(torch.ops.aten.mean.dim)
def _mean_dim(conn, x, dim, keepdim=False):
    """Mean along *dim*.  If *dim* is a list, only the first element is used."""
    if isinstance(dim, (list, tuple)):
        dim = dim[0]
    return _sql1(conn, "SELECT llm_mean_nd(?, ?, ?)",
                 [x, int(dim), int(keepdim)])


@_reg(torch.ops.aten.sum.default)
def _sum(conn, x):
    """Global sum reduction — returns a scalar tensor."""
    return _sql1(conn, "SELECT llm_reduce_sum_nd(?)", [x])


@_reg(torch.ops.aten.sum.dim_IntList)
def _sum_dim(conn, x, dim, keepdim=False, **kwargs):
    """Sum along *dim*.  If *dim* is a list, only the first element is used."""
    if isinstance(dim, (list, tuple)):
        dim = dim[0]
    return _sql1(conn, "SELECT llm_sum_nd(?, ?, ?)",
                 [x, int(dim), int(keepdim)])


@_reg(torch.ops.aten.max.default)
def _max(conn, x):
    """Global max reduction — returns a scalar tensor."""
    return _sql1(conn, "SELECT llm_reduce_max_nd(?)", [x])


@_reg(torch.ops.aten.max.dim)
def _max_dim(conn, x, dim, keepdim=False):
    """Max along *dim*.  Returns ``(values, indices)`` tuple."""
    val = _sql1(conn, "SELECT llm_max_dim_nd(?, ?, ?)",
                [x, int(dim), int(keepdim)])
    idx = _sql1(conn, "SELECT llm_argmax_dim_nd(?, ?, ?)",
                [x, int(dim), int(keepdim)])
    return (val, idx)


@_reg(torch.ops.aten.min.default)
def _min(conn, x):
    """Global min reduction — returns a scalar tensor."""
    return _sql1(conn, "SELECT llm_reduce_min_nd(?)", [x])


@_reg(torch.ops.aten.argmax.default)
def _argmax(conn, x, dim=None, keepdim=False):
    """Argmax — global (returns Python int) or along *dim* (returns tensor)."""
    if dim is None:
        return int(_sql1(conn, "SELECT llm_reduce_argmax_nd(?)", [x]))
    return _sql1(conn, "SELECT llm_argmax_nd(?, ?)", [x, int(dim)])


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten.mm.default)
def _mm(conn, a, b):
    """2-D matrix multiplication via ``llm_bmm``."""
    return _sql1(conn, "SELECT llm_bmm(?, ?)", [a, b])


@_reg(torch.ops.aten.matmul.default)
def _matmul(conn, a, b):
    """General matrix multiplication supporting 1-D through N-D tensors.

    * 1D × 1D  → dot product (fused view+bmm+view in 1 SQL call)
    * 1D × ND  → vector-matrix (fused view+bmm in 1 call + reshape)
    * ND × 1D  → matrix-vector (fused bmm+view in 1 call + reshape)
    * ND × ND  → batched matmul via ``llm_bmm`` (1 SQL call)

    Shape metadata is read from blob headers (0 SQL cost).
    """
    a_ndim = _sql_ndim(conn, a)  # 0 SQL for bytes
    b_ndim = _sql_ndim(conn, b)  # 0 SQL for bytes
    # Handle different dimension combinations for matmul
    if a_ndim == 1 and b_ndim == 1:
        # Dot product: (K,) x (K,) -> scalar
        # Fuse view+bmm+view into 1 nested SQL call (was 4 calls)
        k = _sql_shape(conn, a, 0)
        return _sql1(conn, """
            SELECT llm_view_nd(
                llm_bmm(llm_view_nd(?, 1, ?), llm_view_nd(?, ?, 1)),
                1)
        """, [a, k, b, k])
    if a_ndim == 1:
        # (K,) x (K, N) -> (N,)
        # Fuse view+bmm into 1 nested SQL call (was 2 calls)
        k = _sql_shape(conn, a, 0)
        r = _sql1(conn, "SELECT llm_bmm(llm_view_nd(?, 1, ?), ?)",
                  [a, k, b])
        # result shape is (..., 1, N) — squeeze the inserted dim
        r_shape = _sql_shape_list(conn, r)  # 0 SQL for bytes
        new_shape = r_shape[:-2] + r_shape[-1:]
        args = [r] + new_shape
        ph = ", ".join(["?"] * len(args))
        return _sql1(conn, f"SELECT llm_view_nd({ph})", args)
    if b_ndim == 1:
        # (M, K) x (K,) -> (M,)
        # Fuse view+bmm into 1 nested SQL call (was 2 calls)
        k = _sql_shape(conn, b, 0)
        r = _sql1(conn, "SELECT llm_bmm(?, llm_view_nd(?, ?, 1))",
                  [a, b, k])
        # result shape is (..., M, 1) — drop last dim
        r_shape = _sql_shape_list(conn, r)  # 0 SQL for bytes
        new_shape = r_shape[:-1]
        args = [r] + new_shape
        ph = ", ".join(["?"] * len(args))
        return _sql1(conn, f"SELECT llm_view_nd({ph})", args)
    # General N-D matmul with batch dims via bmm
    return _sql1(conn, "SELECT llm_bmm(?, ?)", [a, b])


@_reg(torch.ops.aten.bmm.default)
def _bmm(conn, a, b):
    """Batched matrix multiplication: ``(B, M, K) × (B, K, N) → (B, M, N)``."""
    return _sql1(conn, "SELECT llm_bmm(?, ?)", [a, b])


@_reg(torch.ops.aten.linear.default)
def _linear(conn, x, weight, bias=None):
    """Fused linear layer: ``x @ weight.T + bias``.

    When *bias* is provided, the addition is nested into the same SQL
    expression as the linear transform (1 SQL call instead of 2).
    """
    if bias is not None:
        # Fuse linear + bias add into 1 nested SQL call (was 2 calls)
        return _sql1(conn,
                     "SELECT llm_add_nd(llm_linear_nd(?, ?), ?)",
                     [x, weight, bias])
    return _sql1(conn, "SELECT llm_linear_nd(?, ?)", [x, weight])


@_reg(torch.ops.aten.addmm.default)
def _addmm(conn, bias, a, b, beta=1, alpha=1):
    """``beta * bias + alpha * (a @ b)``.

    Common case (alpha=beta=1) is fused into a single nested SQL call.
    """
    if alpha == 1 and beta == 1:
        # Common fast path: result = bmm(a, b) + bias — fuse into 1 call
        return _sql1(conn,
                     "SELECT llm_add_nd(llm_bmm(?, ?), ?)",
                     [a, b, _ensure_blob(bias)])
    r = _sql1(conn, "SELECT llm_bmm(?, ?)", [a, b])
    if alpha != 1:
        r = _sql1(conn, "SELECT llm_scale_nd(?, ?)", [r, float(alpha)])
    if beta == 1:
        r = _sql1(conn, "SELECT llm_add_nd(?, ?)", [r, _ensure_blob(bias)])
    elif beta != 0:
        bias_scaled = _sql1(conn, "SELECT llm_scale_nd(?, ?)",
                             [_ensure_blob(bias), float(beta)])
        r = _sql1(conn, "SELECT llm_add_nd(?, ?)", [r, bias_scaled])
    return r


# ---------------------------------------------------------------------------
# Indexing / Embedding
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten.embedding.default)
def _embedding(conn, weight, indices, padding_idx=-1, **kwargs):
    """Embedding lookup: select rows from *weight* by *indices*.

    Both *weight* and *indices* are passed as blobs to ``llm_embedding_nd``.
    """
    idx_blob = _ensure_blob(indices)
    return _sql1(conn, "SELECT llm_embedding_nd(?, ?)", [weight, idx_blob])


@_reg(torch.ops.aten.index.Tensor)
def _index(conn, x, indices):
    """Advanced indexing (``x[idx0, idx1, …]``).

    * 0 tensor indices → identity.
    * 1 tensor index → ``llm_index_select_nd`` (single SQL call).
    * Multiple tensor indices with broadcasting → numpy-based fallback
      to handle arbitrary shape combinations.
    """
    # Determine which dims have tensor indices (non-None)
    tensor_dims = []
    for i, idx in enumerate(indices):
        if idx is not None:
            tensor_dims.append(i)

    if len(tensor_dims) == 0:
        return x

    if len(tensor_dims) == 1:
        # Simple case: single dimension indexing
        d = tensor_dims[0]
        idx_blob = _ensure_blob(indices[d])
        idx_ndim = _sql_ndim(conn, idx_blob)
        if idx_ndim == 1:
            return _sql1(conn, "SELECT llm_index_select_nd(?, ?, ?)",
                         [x, d, idx_blob])
        else:
            return _sql1(conn, "SELECT llm_gather_nd(?, ?, ?)",
                         [x, d, idx_blob])

    # Multi-dim advanced indexing with broadcasting: use numpy fallback
    x_arr = blob_to_numpy(x) if isinstance(x, bytes) else blob_to_numpy(_ensure_blob(x))
    np_indices = []
    for idx in indices:
        if idx is None:
            np_indices.append(slice(None))
        else:
            idx_blob = _ensure_blob(idx)
            idx_arr = blob_to_numpy(idx_blob).astype(np.intp)
            np_indices.append(idx_arr)
    result = x_arr[tuple(np_indices)]
    return tensor_to_blob(result.astype(np.float32))


# ---------------------------------------------------------------------------
# Masking / Triangular
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten.triu.default)
def _triu(conn, x, diagonal=0):
    """Upper triangular mask: zero out elements below *diagonal*."""
    return _sql1(conn, "SELECT llm_triu_nd(?, ?)", [x, int(diagonal)])


@_reg(torch.ops.aten.tril.default)
def _tril(conn, x, diagonal=0):
    """Lower triangular mask: zero out elements above *diagonal*."""
    return _sql1(conn, "SELECT llm_tril_nd(?, ?)", [x, int(diagonal)])


@_reg(torch.ops.aten.masked_fill.Scalar)
def _masked_fill(conn, x, mask, value):
    """Fill positions where *mask* is non-zero with *value*."""
    return _sql1(conn, "SELECT llm_masked_fill_nd(?, ?, ?)",
                 [x, _ensure_blob(mask), float(value)])


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten.silu.default)
def _silu(conn, x):
    """SiLU (Swish) activation: ``x * sigmoid(x)``."""
    return _sql1(conn, "SELECT llm_silu_nd(?)", [x])


def fused_silu_mul(conn, gate, up):
    """Fused ``silu(gate) * up`` — 1 SQL round-trip via ``llm_silu_mul_nd``."""
    return _sql1(conn, "SELECT llm_silu_mul_nd(?, ?)", [gate, up])


def fused_rope(conn, x, cos, sin):
    """Fused RoPE: ``x*cos + rotate_half(x)*sin`` via ``llm_rope_nd``."""
    return _sql1(conn, "SELECT llm_rope_nd(?, ?, ?)", [x, cos, sin])


def fused_rmsnorm(conn, x, weight, eps):
    """Fused RMSNorm: ``x * rsqrt(mean(x², dim=-1, keepdim=True) + eps) * weight``.

    Replaces the 6-op pattern (pow → mean → add → rsqrt → mul → mul) with
    a single ``llm_rmsnorm_nd`` call.
    """
    return _sql1(conn, "SELECT llm_rmsnorm_nd(?, ?, ?)", [x, weight, float(eps)])


# ---------------------------------------------------------------------------
# Lazy shape-op wrappers for SQL nesting fusion
# ---------------------------------------------------------------------------
# These return ``SqlExpr`` objects instead of executing immediately.
# The compile-time fusion pass swaps the resolved_fn to one of these
# for single-use shape ops, so the shape SQL is nested inside the
# consumer's SQL call (saving one SQL round-trip per instance).

def lazy_unsqueeze(conn, x, dim):
    """Return a ``SqlExpr`` for ``llm_unsqueeze_nd(x, dim)``."""
    dim = int(dim)
    if dim < 0 and isinstance(x, bytes):
        ndim = _blob_ndim(x)
        dim = dim % (ndim + 1)
    return _sql1_lazy("llm_unsqueeze_nd(?, ?)", [x, dim])


def lazy_view(conn, x, shape):
    """Return a ``SqlExpr`` for ``llm_view_nd(x, d0, d1, …)``."""
    shape = [int(s) for s in shape]
    args = [x] + shape
    placeholders = ", ".join(["?"] * len(args))
    return _sql1_lazy(f"llm_view_nd({placeholders})", args)


def lazy_reshape(conn, x, shape):
    """Return a ``SqlExpr`` for ``llm_view_nd(x, d0, d1, …)`` (reshape = view)."""
    return lazy_view(conn, x, shape)


def lazy_expand(conn, x, size):
    """Return a ``SqlExpr`` for ``llm_expand_nd(x, s0, s1, …)``."""
    args = [x] + [int(s) for s in size]
    placeholders = ", ".join(["?"] * len(args))
    return _sql1_lazy(f"llm_expand_nd({placeholders})", args)


def lazy_transpose(conn, x, dim0, dim1):
    """Return a ``SqlExpr`` for ``llm_transpose_nd(x, dim0, dim1)``."""
    return _sql1_lazy("llm_transpose_nd(?, ?, ?)", [x, int(dim0), int(dim1)])


# Map from aten target → lazy wrapper for the compile-time fusion pass.
_LAZY_SHAPE_OPS = {
    torch.ops.aten.unsqueeze.default:    lazy_unsqueeze,
    torch.ops.aten.view.default:         lazy_view,
    torch.ops.aten._unsafe_view.default: lazy_view,
    torch.ops.aten.reshape.default:      lazy_reshape,
    torch.ops.aten.expand.default:       lazy_expand,
    torch.ops.aten.transpose.int:        lazy_transpose,
}


@_reg(torch.ops.aten.relu.default)
def _relu(conn, x):
    """ReLU activation: ``max(0, x)``."""
    return _sql1(conn, "SELECT llm_relu_nd(?)", [x])


@_reg(torch.ops.aten.gelu.default)
def _gelu(conn, x, approximate='none'):
    """GELU activation (tanh approximation in the C extension)."""
    return _sql1(conn, "SELECT llm_gelu_nd(?)", [x])


@_reg(torch.ops.aten.sigmoid.default)
def _sigmoid(conn, x):
    """Sigmoid activation: ``1 / (1 + exp(-x))``."""
    return _sql1(conn, "SELECT llm_sigmoid_nd(?)", [x])


@_reg(torch.ops.aten.softmax.int)
def _softmax(conn, x, dim, **kwargs):
    """Softmax along *dim*."""
    return _sql1(conn, "SELECT llm_softmax_nd(?, ?)", [x, int(dim)])


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten.native_layer_norm.default)
def _native_layer_norm(conn, x, normalized_shape, weight, bias, eps):
    """Layer norm via SQL: y = (x - mean) / sqrt(var + eps) * weight + bias.
    Returns (result, mean, rstd) — mean and rstd are computed via SQL for
    compatibility but are rarely consumed in inference.

    Optimised: the layernorm result is computed in 1 SQL call, and the
    mean/rstd are fused into a single CTE with the final reshape included
    in the SELECT clause — total 2 SQL round-trips (was 4).
    """
    if weight is None:
        # Create all-ones weight of the right shape
        args = [1.0] + [int(s) for s in normalized_shape]
        ph = ", ".join(["?"] * len(args))
        weight = _sql1(conn, f"SELECT llm_full_nd({ph})", args)

    result = _sql1(conn, "SELECT llm_layernorm_nd(?, ?, ?, ?)",
                   [x, weight, bias, float(eps)])

    # ── shape metadata from blob header (0 SQL round-trips) ──
    norm_size = 1
    for s in normalized_shape:
        norm_size *= s
    shape = _sql_shape_list(conn, x)
    outer_shape = shape[:-len(normalized_shape)]
    if not outer_shape:
        outer_shape = [1]
    outer_total = 1
    for s in outer_shape:
        outer_total *= s

    # ── Fused mean + rstd via CTE with reshape included (2 SQL total) ──
    # The SQL ? placeholders are consumed left-to-right:
    #   xf:   view_nd(x, outer_total, norm_size)
    #   diff: expand_nd(mu, outer_total, norm_size)
    #   SELECT col1: view_nd(mu.v, *out_shape)
    #   SELECT col2: full_nd(eps, outer_total, 1) then view_nd(rstd, *out_shape)
    out_shape = outer_shape + [1]
    mean_view_ph = ", ".join(["?"] * len(out_shape))
    rstd_view_ph = ", ".join(["?"] * len(out_shape))

    sql_params = ([x, outer_total, norm_size,
                   outer_total, norm_size]
                  + list(out_shape)
                  + [float(eps), outer_total, 1]
                  + list(out_shape))

    sql_text = f"""
        WITH
          xf(v)   AS (SELECT llm_view_nd(?, ?, ?)),
          mu(v)   AS (SELECT llm_mean_nd(xf.v, 1, 1) FROM xf),
          diff(v) AS (SELECT llm_sub_nd(xf.v,
                             llm_expand_nd(mu.v, ?, ?)) FROM xf, mu)
        SELECT
          llm_view_nd(mu.v, {mean_view_ph}),
          llm_view_nd(
            llm_rsqrt_nd(llm_add_nd(
                llm_mean_nd(llm_mul_nd(diff.v, diff.v), 1, 1),
                llm_full_nd(?, ?, ?))),
            {rstd_view_ph})
        FROM mu, diff
    """
    sql_text, sql_params = _resolve_sql_params(sql_text, sql_params)
    row = conn.execute(sql_text, sql_params).fetchone()
    mean_out = row[0]
    rstd_out = row[1]

    return (result, mean_out, rstd_out)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten.scaled_dot_product_attention.default)
def _sdpa(conn, query, key, value, attn_mask=None,
          dropout_p=0.0, is_causal=False, scale=None):
    """Scaled dot-product attention, fully delegated to nested SQL.

    All code paths now produce the result in a **single SQL round-trip**:

    * **Non-causal, no mask** (1 SQL call):
      ``bmm(softmax(scale(bmm(Q, Kᵀ), s), -1), V)``
    * **Causal** (1 SQL call):
      score + causal-mask + softmax + bmm all nested.
    * **With attn_mask** (1 SQL call):
      mask conversion + addition + softmax + bmm all nested.

    Scale defaults to ``1/sqrt(d_k)`` computed in Python from the blob
    header (0 SQL cost).
    """
    if scale is None:
        d_k = _sql_shape(conn, query, -1)  # 0 SQL for bytes
        scale = 1.0 / (float(d_k) ** 0.5)  # compute in Python, avoid SQL

    # ── Non-causal, no mask: fuse entire attention into 1 SQL call ──
    if not is_causal and attn_mask is None:
        return _sql1(conn, """
            SELECT llm_bmm(
                llm_softmax_nd(
                    llm_scale_nd(
                        llm_bmm(?, llm_transpose_nd(?, -2, -1)),
                        ?), -1),
                ?)
        """, [query, key, float(scale), value])

    # ── Causal, no mask: fuse scores+mask+softmax+bmm into 1 SQL call ──
    if is_causal and attn_mask is None:
        seq_q = _sql_shape(conn, query, -2)  # 0 SQL for bytes
        seq_k = _sql_shape(conn, key, -2)    # 0 SQL for bytes
        return _sql1(conn, """
            SELECT llm_bmm(
                llm_softmax_nd(
                    llm_add_nd(
                        llm_scale_nd(
                            llm_bmm(?, llm_transpose_nd(?, -2, -1)),
                            ?),
                        llm_masked_fill_nd(
                            llm_full_nd(0.0, ?, ?),
                            llm_triu_nd(llm_full_nd(1.0, ?, ?), 1),
                            -1e9)),
                    -1),
                ?)
        """, [query, key, float(scale), seq_q, seq_k, seq_q, seq_k, value])

    # ── With attn_mask (non-causal): fuse mask+softmax+bmm into 1 call ──
    if attn_mask is not None and not is_causal:
        attn_mask = _ensure_blob(attn_mask)
        return _sql1(conn, """
            SELECT llm_bmm(
                llm_softmax_nd(
                    llm_add_nd(
                        llm_scale_nd(
                            llm_bmm(?, llm_transpose_nd(?, -2, -1)),
                            ?),
                        llm_bool_to_additive_mask_nd(?, ?)),
                    -1),
                ?)
        """, [query, key, float(scale), attn_mask, 1e9, value])

    # ── Causal + attn_mask: fuse everything into 1 SQL call ──
    attn_mask = _ensure_blob(attn_mask)
    seq_q = _sql_shape(conn, query, -2)
    seq_k = _sql_shape(conn, key, -2)
    return _sql1(conn, """
        SELECT llm_bmm(
            llm_softmax_nd(
                llm_add_nd(
                    llm_add_nd(
                        llm_scale_nd(
                            llm_bmm(?, llm_transpose_nd(?, -2, -1)),
                            ?),
                        llm_masked_fill_nd(
                            llm_full_nd(0.0, ?, ?),
                            llm_triu_nd(llm_full_nd(1.0, ?, ?), 1),
                            -1e9)),
                    llm_bool_to_additive_mask_nd(?, ?)),
                -1),
            ?)
    """, [query, key, float(scale),
          seq_q, seq_k, seq_q, seq_k,
          attn_mask, 1e9, value])


# ---------------------------------------------------------------------------
# Symbolic sizes
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten.sym_size.int)
def _sym_size(conn, x, dim):
    """Return the symbolic (integer) size of *dim* — uses blob header if possible."""
    return _sql_shape(conn, x, int(dim))


# ---------------------------------------------------------------------------
# Python builtins
# ---------------------------------------------------------------------------

@_reg(operator.add)
def _py_add(conn, a, b):
    """Python ``+`` operator: scalar arithmetic via SQL, or tensor add."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return _sql1(conn, "SELECT ? + ?", [a, b])
    return _add(conn, a, b)


@_reg(operator.mul)
def _py_mul(conn, a, b):
    """Python ``*`` operator: scalar arithmetic via SQL, or tensor mul."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return _sql1(conn, "SELECT ? * ?", [a, b])
    return _mul(conn, a, b)


@_reg(operator.getitem)
def _py_getitem(conn, x, idx):
    """Python ``[]`` operator for tensors, lists, and tuples.

    Dispatches to the appropriate SQL operation based on index type:
    * ``int`` → ``_select`` (removes the indexed dimension)
    * ``slice`` → ``llm_slice_nd``
    * ``tuple`` → chained select/slice per element
    * ``bytes`` (tensor index) → ``llm_index_select_nd``
    * ``list``/``tuple`` (Python container) → plain Python indexing
    """
    if isinstance(x, (list, tuple)):
        return x[idx]
    if isinstance(x, bytes):
        if isinstance(idx, int):
            return _select(conn, x, 0, idx)
        if isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = _normalize_sql_slice_end(idx.stop)
            step = idx.step if idx.step is not None else 1
            return _sql1(conn, "SELECT llm_slice_nd(?, ?, ?, ?, ?)",
                         [x, 0, int(start), int(stop), int(step)])
        # tuple of ints → chain select on dim 0 for each scalar index
        if isinstance(idx, tuple):
            result = x
            for i, ix in enumerate(idx):
                if isinstance(ix, int):
                    result = _select(conn, result, 0, ix)
                elif isinstance(ix, slice):
                    start = ix.start if ix.start is not None else 0
                    stop = _normalize_sql_slice_end(ix.stop)
                    step = ix.step if ix.step is not None else 1
                    result = _sql1(
                        conn, "SELECT llm_slice_nd(?, ?, ?, ?, ?)",
                        [result, 0, int(start), int(stop), int(step)])
            return result
        # Fallback for other index types — use SQL ops
        if isinstance(idx, bytes):
            # Tensor indexing on dim 0
            return _sql1(conn, "SELECT llm_index_select_nd(?, ?, ?)",
                         [x, 0, idx])
        raise TypeError(f"Unsupported getitem index type: {type(idx)}")
    return x[idx]


# ---------------------------------------------------------------------------
# Scalar variants of arithmetic / comparison
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten.add.Scalar)
def _add_scalar(conn, a, b):
    """Add scalar *b* to tensor *a*: ``a + b``.

    Fused into a single SQL call by inlining the scalar blob creation.
    """
    return _sql1(conn,
                 "SELECT llm_add_nd(?, llm_full_nd(?, 1))",
                 [_ensure_blob(a), float(b)])


@_reg(torch.ops.aten.sub.Scalar)
def _sub_scalar(conn, a, b):
    """Subtract scalar *b* from tensor *a*: ``a - b``.

    Fused into a single SQL call by inlining the scalar blob creation.
    """
    return _sql1(conn,
                 "SELECT llm_sub_nd(?, llm_full_nd(?, 1))",
                 [_ensure_blob(a), float(b)])


@_reg(torch.ops.aten.mul.Scalar)
def _mul_scalar(conn, a, b):
    """Multiply tensor *a* by scalar *b* via ``llm_scale_nd``."""
    return _sql1(conn, "SELECT llm_scale_nd(?, ?)",
                 [_ensure_blob(a), float(b)])


@_reg(torch.ops.aten.div.Tensor_mode)
def _div_tensor_mode(conn, a, b, rounding_mode=None):
    """Tensor division with optional rounding (``'trunc'`` or ``'floor'``)."""
    r = _div(conn, a, b)
    if rounding_mode == 'trunc':
        return _sql1(conn, "SELECT llm_trunc_nd(?)", [r])
    if rounding_mode == 'floor':
        return _sql1(conn, "SELECT llm_floor_nd(?)", [r])
    return r


@_reg(torch.ops.aten.div.Scalar_mode)
def _div_scalar_mode(conn, a, b, rounding_mode=None):
    """Scalar division with optional rounding (``'trunc'`` or ``'floor'``)."""
    r = _div_scalar(conn, a, b)
    if rounding_mode == 'trunc':
        return _sql1(conn, "SELECT llm_trunc_nd(?)", [r])
    if rounding_mode == 'floor':
        return _sql1(conn, "SELECT llm_floor_nd(?)", [r])
    return r


@_reg(torch.ops.aten.floor_divide.default)
def _floor_divide(conn, a, b):
    """Element-wise floor division: ``floor(a / b)``."""
    return _sql1(conn, "SELECT llm_floor_div_nd(?, ?)",
                 [_ensure_blob(a), _ensure_blob(b)])


@_reg(torch.ops.aten.remainder.Scalar)
def _remainder_scalar(conn, a, b):
    """Element-wise remainder of tensor by scalar."""
    return _sql1(conn, "SELECT llm_remainder_nd(?, ?)",
                 [a, float(b)])


@_reg(torch.ops.aten.remainder.Tensor)
def _remainder_tensor(conn, a, b):
    """Element-wise remainder of tensor by tensor."""
    return _sql1(conn, "SELECT llm_remainder_tensor_nd(?, ?)", [a, b])


@_reg(torch.ops.aten.ne.Scalar)
def _ne_scalar(conn, a, b):
    """Element-wise not-equal (tensor vs scalar)."""
    return _cmp_op(conn, a, float(b), 'ne')


@_reg(torch.ops.aten.lt.Scalar)
def _lt_scalar(conn, a, b):
    """Element-wise less-than (tensor vs scalar)."""
    return _cmp_op(conn, a, float(b), 'lt')


@_reg(torch.ops.aten.le.Scalar)
def _le_scalar(conn, a, b):
    """Element-wise less-than-or-equal (tensor vs scalar)."""
    return _cmp_op(conn, a, float(b), 'le')


@_reg(torch.ops.aten.ge.Scalar)
def _ge_scalar(conn, a, b):
    """Element-wise greater-than-or-equal (tensor vs scalar)."""
    return _cmp_op(conn, a, float(b), 'ge')


@_reg(torch.ops.aten.eq.Scalar)
def _eq_scalar(conn, a, b):
    """Element-wise equality (tensor vs scalar)."""
    return _cmp_op(conn, a, float(b), 'eq')


# ---------------------------------------------------------------------------
# Conditional selection
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten.where.self)
def _where(conn, cond, x, y):
    """Element-wise conditional: ``cond ? x : y``."""
    return _sql1(conn, "SELECT llm_where_nd(?, ?, ?)",
                 [_ensure_blob(cond), _ensure_blob(x), _ensure_blob(y)])


# ---------------------------------------------------------------------------
# Clamp / Like
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten.clamp.default)
def _clamp(conn, x, min_val=None, max_val=None):
    """Clamp elements to ``[min_val, max_val]`` (scalar bounds)."""
    mn = float(min_val) if min_val is not None else None
    mx = float(max_val) if max_val is not None else None
    return _sql1(conn, "SELECT llm_clamp_nd(?, ?, ?)", [x, mn, mx])


@_reg(torch.ops.aten.clamp.Tensor)
def _clamp_tensor(conn, x, min_val=None, max_val=None):
    """Clamp elements to tensor bounds via nested where expressions.

    Fuses lt+where (and gt+where) into single nested SQL calls:
    ``where(lt(x, min), min, x)`` and ``where(gt(x, max), max, x)``.
    When both bounds are provided, 2 SQL calls (was 4).
    """
    result = x
    if min_val is not None:
        result = _sql1(conn,
                       "SELECT llm_where_nd(llm_lt_nd(?, ?), ?, ?)",
                       [result, min_val, min_val, result])
    if max_val is not None:
        result = _sql1(conn,
                       "SELECT llm_where_nd(llm_gt_nd(?, ?), ?, ?)",
                       [result, max_val, max_val, result])
    return result


@_reg(torch.ops.aten.clamp_min.default)
def _clamp_min(conn, x, min_val):
    """Clamp tensor from below: ``max(x, min_val)``."""
    return _clamp(conn, x, min_val=min_val)


@_reg(torch.ops.aten.clamp_max.default)
def _clamp_max(conn, x, max_val):
    """Clamp tensor from above: ``min(x, max_val)``."""
    return _clamp(conn, x, max_val=max_val)


@_reg(torch.ops.aten.zeros_like.default)
def _zeros_like(conn, x, **kwargs):
    """Create a zero tensor with the same shape as *x*."""
    shape = _sql_shape_list(conn, x)
    return _full(conn, shape, 0.0)


@_reg(torch.ops.aten.ones_like.default)
def _ones_like(conn, x, **kwargs):
    """Create an all-ones tensor with the same shape as *x*."""
    shape = _sql_shape_list(conn, x)
    return _full(conn, shape, 1.0)


@_reg(torch.ops.aten.full_like.default)
def _full_like(conn, x, fill_value, **kwargs):
    """Create a tensor filled with *fill_value* matching *x*'s shape."""
    shape = _sql_shape_list(conn, x)
    return _full(conn, shape, float(fill_value))


@_reg(torch.ops.aten.empty_like.default)
def _empty_like(conn, x, **kwargs):
    """Create an uninitialized tensor matching *x*'s shape (filled with 0)."""
    shape = _sql_shape_list(conn, x)
    return _full(conn, shape, 0.0)


@_reg(torch.ops.aten.empty.memory_format)
def _empty(conn, size, **kwargs):
    """Create an uninitialized tensor of *size* (filled with 0)."""
    return _full(conn, [int(s) for s in size], 0.0)


# ---------------------------------------------------------------------------
# Logical / bitwise
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten.logical_not.default)
def _logical_not(conn, x):
    """Element-wise logical NOT."""
    return _sql1(conn, "SELECT llm_logical_not_nd(?)", [x])


@_reg(torch.ops.aten.logical_and.default)
def _logical_and(conn, a, b):
    """Element-wise logical AND."""
    return _sql1(conn, "SELECT llm_logical_and_nd(?, ?)", [a, b])


@_reg(torch.ops.aten.logical_or.default)
def _logical_or(conn, a, b):
    """Element-wise logical OR."""
    return _sql1(conn, "SELECT llm_logical_or_nd(?, ?)", [a, b])


@_reg(torch.ops.aten.bitwise_not.default)
def _bitwise_not(conn, x):
    """Element-wise bitwise NOT (treats float bits as integers)."""
    return _sql1(conn, "SELECT llm_bitwise_not_nd(?)", [x])


@_reg(torch.ops.aten.bitwise_and.Tensor)
def _bitwise_and_tensor(conn, a, b):
    """Element-wise bitwise AND (tensor × tensor)."""
    return _sql1(conn, "SELECT llm_bitwise_and_nd(?, ?)", [a, b])


@_reg(torch.ops.aten.bitwise_and.Scalar)
def _bitwise_and_scalar(conn, a, b):
    """Element-wise bitwise AND (tensor × scalar)."""
    return _sql1(conn, "SELECT llm_bitwise_and_scalar_nd(?, ?)",
                 [a, int(b)])


@_reg(torch.ops.aten.bitwise_or.Tensor)
def _bitwise_or_tensor(conn, a, b):
    """Element-wise bitwise OR (tensor × tensor)."""
    return _sql1(conn, "SELECT llm_bitwise_or_nd(?, ?)", [a, b])


# ---------------------------------------------------------------------------
# Element-wise math
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten.floor.default)
def _floor(conn, x):
    """Element-wise floor."""
    return _sql1(conn, "SELECT llm_floor_nd(?)", [x])


@_reg(torch.ops.aten.ceil.default)
def _ceil(conn, x):
    """Element-wise ceiling."""
    return _sql1(conn, "SELECT llm_ceil_nd(?)", [x])


@_reg(torch.ops.aten.round.default)
def _round(conn, x):
    """Element-wise rounding to nearest integer."""
    return _sql1(conn, "SELECT llm_round_nd(?)", [x])


@_reg(torch.ops.aten.trunc.default)
def _trunc(conn, x):
    """Element-wise truncation toward zero."""
    return _sql1(conn, "SELECT llm_trunc_nd(?)", [x])


@_reg(torch.ops.aten.sign.default)
def _sign(conn, x):
    """Element-wise sign: -1, 0, or +1."""
    return _sql1(conn, "SELECT llm_sign_nd(?)", [x])


@_reg(torch.ops.aten.cumsum.default)
def _cumsum(conn, x, dim, **kwargs):
    """Cumulative sum along *dim*."""
    return _sql1(conn, "SELECT llm_cumsum_nd(?, ?)", [x, int(dim)])


@_reg(torch.ops.aten.cumprod.default)
def _cumprod(conn, x, dim, **kwargs):
    """Cumulative product along *dim*."""
    return _sql1(conn, "SELECT llm_cumprod_nd(?, ?)", [x, int(dim)])


@_reg(torch.ops.aten.diff.default)
def _diff(conn, x, n=1, dim=-1, prepend=None, append=None):
    """N-th finite difference along *dim*.

    Optional *prepend*/*append* tensors are concatenated before
    differencing.  Each iteration fuses ``slice(data,1:) - slice(data,:-1)``
    into a single nested SQL call (was 3 separate calls per iteration).
    """
    dim = int(dim)
    ndim = _sql_ndim(conn, x)
    if dim < 0:
        dim += ndim
    data = x
    if prepend is not None:
        data = _sql1(conn, "SELECT llm_cat_nd(?, ?, ?)",
                      [prepend, data, dim])
    if append is not None:
        data = _sql1(conn, "SELECT llm_cat_nd(?, ?, ?)",
                      [data, append, dim])
    for _ in range(int(n)):
        dim_sz = _sql_shape(conn, data, dim)
        # Fuse slice+slice+sub into 1 nested SQL call (was 3)
        data = _sql1(conn,
                     "SELECT llm_sub_nd("
                     "  llm_slice_nd(?, ?, ?, ?),"
                     "  llm_slice_nd(?, ?, ?, ?))",
                     [data, dim, 1, dim_sz,
                      data, dim, 0, dim_sz - 1])
    return data


# ---------------------------------------------------------------------------
# Split / Chunk
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten.split.Tensor)
def _split(conn, x, split_size, dim=0):
    """Split tensor into chunks of *split_size* along *dim*.

    Returns a list of blob tensors.  Each chunk is obtained via
    ``llm_slice_nd`` in one SQL call.  Shape metadata is read from
    the blob header (0 SQL cost).
    """
    dim = int(dim)
    ndim = _sql_ndim(conn, x)
    if dim < 0:
        dim += ndim
    total_dim = _sql_shape(conn, x, dim)
    ss = int(split_size)
    chunks = []
    start = 0
    while start < total_dim:
        end = min(start + ss, total_dim)
        c = _sql1(conn, "SELECT llm_slice_nd(?, ?, ?, ?)",
                  [x, dim, start, end])
        chunks.append(c)
        start = end
    return chunks


@_reg(torch.ops.aten.split_with_sizes.default)
def _split_with_sizes(conn, x, sizes, dim=0):
    """Split tensor into chunks with the given *sizes* along *dim*.

    Returns a list of blob tensors.
    """
    dim = int(dim)
    ndim = _sql_ndim(conn, x)
    if dim < 0:
        dim += ndim
    chunks = []
    start = 0
    for s in sizes:
        end = start + int(s)
        c = _sql1(conn, "SELECT llm_slice_nd(?, ?, ?, ?)",
                  [x, dim, start, end])
        chunks.append(c)
        start = end
    return chunks


@_reg(torch.ops.aten.chunk.default)
def _chunk(conn, x, chunks, dim=0):
    """Split tensor into *chunks* approximately equal pieces along *dim*.

    Uses ceiling division for chunk size (same as ``torch.chunk``).
    """
    dim = int(dim)
    ndim = _sql_ndim(conn, x)
    if dim < 0:
        dim += ndim
    total_dim = _sql_shape(conn, x, dim)
    n = int(chunks)
    chunk_size = (total_dim + n - 1) // n
    result = []
    start = 0
    while start < total_dim:
        end = min(start + chunk_size, total_dim)
        c = _sql1(conn, "SELECT llm_slice_nd(?, ?, ?, ?)",
                  [x, dim, start, end])
        result.append(c)
        start = end
    return result


# ---------------------------------------------------------------------------
# Repeat / Interleave
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten.repeat.default)
def _repeat(conn, x, repeats):
    """Repeat tensor along each dimension as specified by *repeats*."""
    args = [x] + [int(r) for r in repeats]
    ph = ", ".join(["?"] * len(args))
    return _sql1(conn, f"SELECT llm_repeat_nd({ph})", args)


@_reg(torch.ops.aten.repeat_interleave.self_int)
def _repeat_interleave_int(conn, x, repeats, dim=None, **kwargs):
    """Repeat each element *repeats* times along *dim*.

    When *dim* is ``None``, the tensor is first flattened to 1-D.
    """
    if dim is None:
        total = 1
        for s in _sql_shape_list(conn, x):
            total *= s
        x = _view(conn, x, [total])
        dim = 0
    return _sql1(conn, "SELECT llm_repeat_interleave_nd(?, ?, ?)",
                 [x, int(repeats), int(dim)])


@_reg(torch.ops.aten.repeat_interleave.self_Tensor)
def _repeat_interleave_tensor(conn, x, repeats, dim=None, **kwargs):
    """Repeat each element by per-element *repeats* tensor along *dim*.

    When *dim* is ``None``, the tensor is first flattened to 1-D.
    """
    if dim is None:
        total = 1
        for s in _sql_shape_list(conn, x):
            total *= s
        x = _view(conn, x, [total])
        dim = 0
    return _sql1(conn, "SELECT llm_repeat_interleave_tensor_nd(?, ?, ?)",
                 [x, _ensure_blob(repeats), int(dim)])


# ---------------------------------------------------------------------------
# Roll / Flip
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten.roll.default)
def _roll(conn, x, shifts, dims=None):
    """Circular shift of elements along specified dimensions.

    When *dims* is ``None``, the tensor is flattened, rolled, then
    reshaped.  Multiple ``(shift, dim)`` pairs are applied sequentially.
    """
    if isinstance(shifts, (list, tuple)):
        shifts_list = [int(s) for s in shifts]
    else:
        shifts_list = [int(shifts)]
    if dims is not None:
        if isinstance(dims, (list, tuple)):
            dims_list = [int(d) for d in dims]
        else:
            dims_list = [int(dims)]
    else:
        shape = _sql_shape_list(conn, x)
        total = 1
        for s in shape:
            total *= s
        x_flat = _view(conn, x, [total])
        result = _sql1(conn, "SELECT llm_roll_nd(?, ?, ?)",
                       [x_flat, shifts_list[0], 0])
        return _view(conn, result, shape)
    result = x
    for s, d in zip(shifts_list, dims_list):
        result = _sql1(conn, "SELECT llm_roll_nd(?, ?, ?)",
                       [result, s, d])
    return result


@_reg(torch.ops.aten.flip.default)
def _flip(conn, x, dims):
    """Reverse elements along each dimension in *dims* sequentially."""
    result = x
    for d in dims:
        result = _sql1(conn, "SELECT llm_flip_nd(?, ?)",
                       [result, int(d)])
    return result


# ---------------------------------------------------------------------------
# Gather / Scatter / Fill
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten.gather.default)
def _gather(conn, x, dim, index, **kwargs):
    """Gather elements from *x* along *dim* using *index* tensor."""
    return _sql1(conn, "SELECT llm_gather_nd(?, ?, ?)",
                 [x, int(dim), index])


@_reg(torch.ops.aten.index_select.default)
def _index_select(conn, x, dim, index):
    """Select slices from *x* along *dim* using 1-D *index* tensor."""
    return _sql1(conn, "SELECT llm_index_select_nd(?, ?, ?)",
                 [x, int(dim), index])


@_reg(torch.ops.aten.index_put.default)
def _index_put(conn, x, indices, values, accumulate=False):
    """Put *values* into *x* at positions given by *indices*.

    Uses the variadic ``llm_index_put_nd(x, values, accum, idx0, …)``
    SQL function.  ``None`` indices are passed as SQL NULL.
    """
    vals_blob = _ensure_blob(values)
    args = [x, vals_blob, int(bool(accumulate))]
    for idx_entry in indices:
        if idx_entry is None:
            args.append(None)
        else:
            args.append(_ensure_blob(idx_entry))
    placeholders = ", ".join(["?"] * len(args))
    return _sql1(conn, f"SELECT llm_index_put_nd({placeholders})", args)


@_reg(torch.ops.aten.scatter.value)
def _scatter(conn, x, dim, index, value):
    """Scatter scalar *value* into *x* at positions given by *index* along *dim*."""
    return _sql1(conn, "SELECT llm_scatter_nd(?, ?, ?, ?)",
                 [x, int(dim), index, float(value)])


@_reg(torch.ops.aten.fill.Scalar)
def _fill(conn, x, value):
    """Fill tensor *x* with scalar *value* (preserves shape)."""
    shape = _sql_shape_list(conn, x)
    return _full(conn, shape, float(value))


@_reg(torch.ops.aten.masked_fill.Tensor)
def _masked_fill_tensor(conn, x, mask, value):
    """Fill *x* at positions where *mask* is true with *value*.

    *value* may be a scalar ``float`` or a tensor ``bytes``.
    """
    if isinstance(value, bytes):
        return _sql1(conn, "SELECT llm_where_nd(?, ?, ?)",
                     [_ensure_blob(mask), _ensure_blob(value),
                      _ensure_blob(x)])
    return _sql1(conn, "SELECT llm_masked_fill_nd(?, ?, ?)",
                 [x, _ensure_blob(mask), float(value)])


# ---------------------------------------------------------------------------
# Copy / Alias
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten.copy.default)
def _copy(conn, dst, src, **kwargs):
    """Copy *src* into *dst* — returns *src* (blobs are immutable)."""
    return src


@_reg(torch.ops.aten.alias.default)
def _alias(conn, x):
    """Create an alias — no-op (blobs are immutable ``bytes``)."""
    return x


# ---------------------------------------------------------------------------
# Norm
# ---------------------------------------------------------------------------

@_reg(torch.ops.aten.norm.ScalarOpt_dim)
def _norm(conn, x, p=2, dim=None, keepdim=False):
    """Lp norm (default p=2) along *dim* or globally (dim=None).

    Multi-dim norms are applied sequentially from highest dim to lowest.
    Global norm uses the sentinel ``dim=-999``.
    """
    p_val = float(p) if p is not None else 2.0
    if dim is None:
        return _sql1(conn, "SELECT llm_norm_nd(?, ?, ?, ?)",
                     [x, p_val, -999, int(keepdim)])
    if isinstance(dim, (list, tuple)):
        dims = sorted([int(d) for d in dim], reverse=True)
        result = x
        for d in dims:
            result = _sql1(conn, "SELECT llm_norm_nd(?, ?, ?, ?)",
                           [result, p_val, d, int(keepdim)])
        return result
    return _sql1(conn, "SELECT llm_norm_nd(?, ?, ?, ?)",
                 [x, p_val, int(dim), int(keepdim)])


@_reg(torch.ops.aten.linalg_vector_norm.default)
def _linalg_vector_norm(conn, x, ord=2, dim=None, keepdim=False, **kwargs):
    """``torch.linalg.vector_norm`` — delegates to ``_norm``."""
    return _norm(conn, x, p=ord, dim=dim, keepdim=keepdim)
