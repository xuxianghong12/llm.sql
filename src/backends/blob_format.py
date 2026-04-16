from __future__ import annotations

import struct
from typing import Dict, List, Optional, Tuple

ND_MAGIC = 0x80000000
INT8_MAGIC = 0x80000008
ND_HDR_BASE = 8

_STRUCT_MAGIC = struct.Struct('<I')
_STRUCT_I32 = struct.Struct('<i')
_STRUCT_MAGIC_I32 = struct.Struct('<Ii')

_struct_cache_int: Dict[int, struct.Struct] = {}
_struct_cache_float: Dict[int, struct.Struct] = {}


def _get_struct_int(n: int) -> struct.Struct:
    s = _struct_cache_int.get(n)
    if s is None:
        s = struct.Struct(f'<{n}i')
        _struct_cache_int[n] = s
    return s


def _get_struct_float(n: int) -> struct.Struct:
    s = _struct_cache_float.get(n)
    if s is None:
        s = struct.Struct(f'<{n}f')
        _struct_cache_float[n] = s
    return s


def is_int8_blob(blob: bytes) -> bool:
    if blob and len(blob) >= 4:
        return _STRUCT_MAGIC.unpack_from(blob, 0)[0] == INT8_MAGIC
    return False


def encode_nd_blob(flat_data: List[float], shape: List[int]) -> bytes:
    ndim = len(shape)
    n_elems = 1
    for size in shape:
        n_elems *= size
    assert len(flat_data) == n_elems, (
        f"flat_data length {len(flat_data)} != product of shape {shape}"
    )
    header = _STRUCT_MAGIC_I32.pack(ND_MAGIC, ndim)
    header += _get_struct_int(ndim).pack(*shape)
    header += _get_struct_float(n_elems).pack(*flat_data)
    return header


def decode_nd_blob(blob: bytes) -> Tuple[List[int], List[float]]:
    magic = _STRUCT_MAGIC.unpack_from(blob, 0)[0]
    if magic == ND_MAGIC:
        ndim = _STRUCT_I32.unpack_from(blob, 4)[0]
        shape = list(_get_struct_int(ndim).unpack_from(blob, ND_HDR_BASE))
        header_size = blob_header_size(ndim)
        n_elems = 1
        for size in shape:
            n_elems *= size
        data = list(_get_struct_float(n_elems).unpack_from(blob, header_size))
        return shape, data
    n = _STRUCT_I32.unpack_from(blob, 0)[0]
    data = list(_get_struct_float(n).unpack_from(blob, 4))
    return [n], data


def blob_ndim(blob: bytes) -> int:
    magic = _STRUCT_MAGIC.unpack_from(blob, 0)[0]
    if magic in (ND_MAGIC, INT8_MAGIC):
        return _STRUCT_I32.unpack_from(blob, 4)[0]
    return 1


def blob_shape_at(blob: bytes, dim: int) -> int:
    ndim = blob_ndim(blob)
    if dim < 0:
        dim += ndim
    return _STRUCT_I32.unpack_from(blob, ND_HDR_BASE + dim * 4)[0]


def blob_shape(blob: bytes) -> List[int]:
    magic = _STRUCT_MAGIC.unpack_from(blob, 0)[0]
    if magic in (ND_MAGIC, INT8_MAGIC):
        ndim = _STRUCT_I32.unpack_from(blob, 4)[0]
        return list(_get_struct_int(ndim).unpack_from(blob, ND_HDR_BASE))
    n = _STRUCT_I32.unpack_from(blob, 0)[0]
    return [n]


def blob_header_size(ndim: int) -> int:
    return ND_HDR_BASE + 4 * ndim


def py_unsqueeze(blob: bytes, dim: int) -> bytes:
    shape = blob_shape(blob)
    ndim = len(shape)
    if dim < 0:
        dim += ndim + 1
    new_shape = shape[:dim] + [1] + shape[dim:]
    new_ndim = ndim + 1
    header = _STRUCT_MAGIC_I32.pack(ND_MAGIC, new_ndim)
    header += _get_struct_int(new_ndim).pack(*new_shape)
    return header + blob[blob_header_size(ndim):]


def py_view(blob: bytes, new_shape: List[int]) -> bytes:
    shape = blob_shape(blob)
    ndim = len(shape)
    total = 1
    for size in shape:
        total *= size
    neg_idx = -1
    known = 1
    for i, size in enumerate(new_shape):
        if size == -1:
            neg_idx = i
        else:
            known *= size
    if neg_idx >= 0:
        new_shape = list(new_shape)
        new_shape[neg_idx] = total // known
    new_ndim = len(new_shape)
    header = _STRUCT_MAGIC_I32.pack(ND_MAGIC, new_ndim)
    header += _get_struct_int(new_ndim).pack(*new_shape)
    return header + blob[blob_header_size(ndim):]


def py_expand(blob: bytes, sizes: List[int]) -> Optional[bytes]:
    shape = blob_shape(blob)
    ndim = len(shape)
    new_ndim = len(sizes)
    if ndim != new_ndim:
        return None
    new_shape = list(sizes)
    for i in range(ndim):
        if new_shape[i] == -1:
            new_shape[i] = shape[i]
        elif shape[i] == 1 and new_shape[i] != 1:
            return None
        elif shape[i] != new_shape[i]:
            return None
    header = _STRUCT_MAGIC_I32.pack(ND_MAGIC, new_ndim)
    header += _get_struct_int(new_ndim).pack(*new_shape)
    return header + blob[blob_header_size(ndim):]


def py_squeeze(blob: bytes, dim: int) -> bytes:
    shape = blob_shape(blob)
    ndim = len(shape)
    if dim < 0:
        dim += ndim
    if shape[dim] != 1:
        return blob
    new_shape = shape[:dim] + shape[dim + 1:]
    new_ndim = ndim - 1
    header = _STRUCT_MAGIC_I32.pack(ND_MAGIC, new_ndim)
    header += _get_struct_int(new_ndim).pack(*new_shape)
    return header + blob[blob_header_size(ndim):]


def argmax_last_row(logits_blob: bytes, vocab_size: int) -> int:
    tail_offset = len(logits_blob) - vocab_size * 4
    last_row = _get_struct_float(vocab_size).unpack_from(logits_blob, tail_offset)
    best_idx = 0
    best_val = last_row[0]
    for i in range(1, vocab_size):
        if last_row[i] > best_val:
            best_val = last_row[i]
            best_idx = i
    return best_idx