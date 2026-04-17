import json
import os
import sqlite3

import numpy as np

from scripts.quantize_model_int8 import quantize_db
from src.backends.sqlite_ops import tensor_to_blob


def _create_param_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE model_params (
            name TEXT PRIMARY KEY,
            data BLOB NOT NULL,
            ndim INTEGER NOT NULL,
            shape TEXT NOT NULL,
            dtype TEXT NOT NULL DEFAULT 'float32'
        )
        """
    )

    rows = []
    base = np.linspace(-2.0, 2.0, num=256 * 256, dtype=np.float32).reshape(256, 256)
    for idx in range(8):
        arr = base + idx
        rows.append(
            (
                f"weight_{idx}",
                tensor_to_blob(arr),
                arr.ndim,
                json.dumps(list(arr.shape)),
                "float32",
            )
        )

    conn.executemany(
        "INSERT INTO model_params (name, data, ndim, shape, dtype) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_quantize_db_vacuums_reclaimed_pages(tmp_path) -> None:
    db_path = tmp_path / "model_int8.db"
    _create_param_db(str(db_path))
    original_size = os.path.getsize(db_path)

    summary = quantize_db(str(db_path))

    assert summary["quantized"] == 8
    assert summary["compression_ratio"] > 1.5
    assert summary["freelist_pages_before_vacuum"] > 0
    assert summary["freelist_pages_after_vacuum"] == 0
    assert (
        summary["db_file_bytes_before_vacuum"]
        >= summary["db_file_bytes_after_vacuum"]
    )
    assert summary["db_file_bytes_after_vacuum"] < original_size