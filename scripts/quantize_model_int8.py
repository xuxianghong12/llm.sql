#!/usr/bin/env python
"""Quantize exported model weights to per-row symmetric INT8.

Usage:
    python scripts/quantize_model_int8.py --export-dir EXPORT_DIR [--output-db PATH]

Examples:
    # Quantize the exported model in-place (writes model_int8.db next to model.db)
    python scripts/quantize_model_int8.py --export-dir /tmp/llm_sql_qwen2_05b

    # Quantize and write to a specific output file
    python scripts/quantize_model_int8.py --export-dir /tmp/llm_sql_qwen2_05b --output-db /tmp/model_int8.db

Description:
    Connects to the SQLite file at EXPORT_DIR/model.db, quantizes eligible
    parameter rows to symmetric per-row int8 using the sqlite-llm extension,
    updates rows in-place, and runs VACUUM to reclaim freed pages. Prints a
    summary with compression statistics.

Requirements:
    - sqlite-llm extension built and available at sqlite-llm/llm_ops.so
    - Python's sqlite3 module with load_extension support
    - Sufficient disk space for a copy of the DB (if output path differs)

"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import struct
import sys
import time


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


_ND_MAGIC = 0x80000000
_INT8_MAGIC = 0x80000008
_STRUCT_MAGIC = struct.Struct("<I")
_STRUCT_I32 = struct.Struct("<i")
_MIN_ELEMENTS_FOR_QUANT = 64


def _blob_info(blob: bytes):
    if not blob or len(blob) < 8:
        return None, 0, [], 0
    magic = _STRUCT_MAGIC.unpack_from(blob, 0)[0]
    if magic not in (_ND_MAGIC, _INT8_MAGIC):
        return magic, 0, [], 0
    ndim = _STRUCT_I32.unpack_from(blob, 4)[0]
    if ndim <= 0 or len(blob) < 8 + ndim * 4:
        return magic, ndim, [], 0
    shape = list(struct.unpack_from(f"<{ndim}i", blob, 8))
    total = 1
    for size in shape:
        total *= size
    return magic, ndim, shape, total


def _should_quantize(blob: bytes) -> bool:
    magic, ndim, _, total = _blob_info(blob)
    if magic == _INT8_MAGIC:
        return False
    if magic != _ND_MAGIC:
        return False
    if ndim < 2:
        return False
    if total < _MIN_ELEMENTS_FOR_QUANT:
        return False
    return True


def _find_extension() -> str:
    path = os.path.join(_PROJECT_ROOT, "sqlite-llm", "llm_ops.so")
    if os.path.isfile(path):
        return os.path.abspath(path)
    raise FileNotFoundError("Build sqlite-llm first: cd sqlite-llm && make")


def _open_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    ext = _find_extension()
    try:
        conn.load_extension(ext, entrypoint="sqlite3_llm_ops_init")
    except TypeError:
        conn.load_extension(ext)
    conn.enable_load_extension(False)
    return conn


def _db_storage_stats(conn: sqlite3.Connection, db_path: str) -> dict:
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    page_count = conn.execute("PRAGMA page_count").fetchone()[0]
    freelist_count = conn.execute("PRAGMA freelist_count").fetchone()[0]
    return {
        "page_size": page_size,
        "page_count": page_count,
        "freelist_count": freelist_count,
        "file_bytes": os.path.getsize(db_path),
    }


def quantize_db(db_path: str) -> dict:
    conn = _open_db(db_path)
    names = [name for (name,) in conn.execute("SELECT name FROM model_params ORDER BY name")]

    original_bytes = 0
    quantized_bytes = 0
    quantized = 0
    skipped = 0
    start = time.perf_counter()

    for name in names:
        blob = conn.execute(
            "SELECT data FROM model_params WHERE name = ?",
            [name],
        ).fetchone()[0]
        original_bytes += len(blob)
        if not _should_quantize(blob):
            skipped += 1
            quantized_bytes += len(blob)
            continue
        q_blob = conn.execute("SELECT llm_quantize_int8_nd(?)", [blob]).fetchone()[0]
        conn.execute(
            "UPDATE model_params SET data = ?, dtype = 'int8' WHERE name = ?",
            [q_blob, name],
        )
        quantized += 1
        quantized_bytes += len(q_blob)

    conn.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ["quantization", '"int8"'],
    )
    conn.commit()

    before_vacuum = _db_storage_stats(conn, db_path)
    conn.execute("VACUUM")
    after_vacuum = _db_storage_stats(conn, db_path)
    conn.close()

    return {
        "quantized": quantized,
        "skipped": skipped,
        "original_bytes": original_bytes,
        "quantized_bytes": quantized_bytes,
        "compression_ratio": original_bytes / quantized_bytes if quantized_bytes else 0.0,
        "db_file_bytes_before_vacuum": before_vacuum["file_bytes"],
        "db_file_bytes_after_vacuum": after_vacuum["file_bytes"],
        "db_page_size": after_vacuum["page_size"],
        "db_page_count_before_vacuum": before_vacuum["page_count"],
        "db_page_count_after_vacuum": after_vacuum["page_count"],
        "freelist_pages_before_vacuum": before_vacuum["freelist_count"],
        "freelist_pages_after_vacuum": after_vacuum["freelist_count"],
        "elapsed_sec": time.perf_counter() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize model.db to model_int8.db")
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output-db", default=None)
    args = parser.parse_args()

    source = os.path.join(args.export_dir, "model.db")
    if not os.path.isfile(source):
        raise FileNotFoundError(source)

    target = args.output_db or os.path.join(args.export_dir, "model_int8.db")
    if os.path.abspath(source) != os.path.abspath(target):
        shutil.copy2(source, target)

    summary = quantize_db(target)
    print("=" * 60)
    print("INT8 quantization complete")
    print("=" * 60)
    print(f"db               : {target}")
    print(f"quantized params : {summary['quantized']}")
    print(f"skipped params   : {summary['skipped']}")
    print(f"compression      : {summary['compression_ratio']:.2f}x")
    print(f"elapsed          : {summary['elapsed_sec']:.2f}s")


if __name__ == "__main__":
    main()
