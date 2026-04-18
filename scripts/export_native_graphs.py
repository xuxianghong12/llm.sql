#!/usr/bin/env python

"""Export native per-op SQL graphs for the C/C++ demo runtime.

Usage:
    python scripts/export_native_graphs.py --model-dir MODEL_DIR [--db-filename FILE] [--output-dir DIR]

Example:
    python scripts/export_native_graphs.py --model-dir /tmp/llm_sql_qwen2_05b

Description:
    Reads the exported model DB (default model.db) and prefill/decode JSON,
    builds per-op SQL graphs for the C/C++ demo runtime, and writes
    prefill.native.json and decode.native.json to the output directory.

Requirements:
    - The exported model directory produced by `export_qwen2_05b.py`.

"""

from __future__ import annotations

import argparse
import os
import sys


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.backends.native_graph_export import export_native_graphs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export native per-op SQL graphs for the C/C++ demo runtime.",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing model.db plus prefill/decode.json.",
    )
    parser.add_argument(
        "--db-filename",
        default="model.db",
        help="Runtime DB to inspect for parameter names.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write prefill.native.json and decode.native.json. Defaults to model-dir.",
    )
    args = parser.parse_args()

    counts = export_native_graphs(
        export_dir=args.model_dir,
        db_filename=args.db_filename,
        output_dir=args.output_dir,
    )
    output_dir = args.output_dir or args.model_dir
    print("native graph export complete")
    print(f"output dir : {output_dir}")
    print(f"prefill    : prefill.native.json ({counts['prefill']} instructions)")
    print(f"decode     : decode.native.json ({counts['decode']} instructions)")


if __name__ == "__main__":
    main()