#!/usr/bin/env python
"""Export Qwen2.5-0.5B-Instruct to SQLite instruction graphs and build INT8 weights."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import warnings


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_DEFAULT_MODEL_PATH = "/path/to/Qwen/Qwen2.5-0.5B-Instruct"
_DEFAULT_MODEL = (
    _DEFAULT_MODEL_PATH
    if os.path.isdir(_DEFAULT_MODEL_PATH)
    else "Qwen/Qwen2.5-0.5B-Instruct"
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils.memory_profiler import StagePeakMemory, StagePeakMemoryTracker


def _write_manifest(output_dir: str, payload: dict) -> None:
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_standalone_sql(output_dir: str, executor) -> None:
    for graph_name in ("prefill", "decode"):
        sql, bind_names = executor.export_standalone_sql(graph=graph_name)
        with open(os.path.join(output_dir, f"{graph_name}.sql"), "w") as f:
            f.write(sql)
        with open(os.path.join(output_dir, f"{graph_name}.binds.json"), "w") as f:
            json.dump(bind_names, f, separators=(",", ":"))


def _remove_stale_standalone_sql(output_dir: str) -> None:
    for name in (
        "prefill.sql",
        "decode.sql",
        "prefill.binds.json",
        "decode.binds.json",
    ):
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            os.remove(path)


def _format_stage_peak(record: StagePeakMemory) -> str:
    return (
        "[export][memory] "
        f"{record.label:<16s} "
        f"elapsed={record.elapsed_sec:.2f}s "
        f"peak_rss={record.peak_rss_kb / 1024.0:.1f}MiB "
        f"peak_vms={record.peak_vmsize_kb / 1024.0:.1f}MiB "
        f"peak_swap={record.peak_swap_kb / 1024.0:.1f}MiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Qwen/Qwen2.5-0.5B-Instruct to llm.sql standalone artifacts.",
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help="HuggingFace model id or local path.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/llm_sql_qwen2_05b",
        help="Directory for exported artifacts.",
    )
    parser.add_argument(
        "--example-seq-len",
        type=int,
        default=8,
        help="Tracing sequence length for export.",
    )
    parser.add_argument(
        "--skip-int8",
        action="store_true",
        help="Skip generating model_int8.db.",
    )
    parser.add_argument(
        "--emit-standalone-sql",
        action="store_true",
        help="Also emit prefill.sql/decode.sql and bind metadata for the raw C demo.",
    )
    args = parser.parse_args()

    import torch
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer, Qwen2ForCausalLM

    from src.backends.native_graph_export import export_native_graphs
    from scripts.quantize_model_int8 import quantize_db
    from src.backends.sqlite_kvcache_engine import SqliteKVCacheGraphExecutor
    from src.backends.sqlite_tokenizer import SqliteTokenizer
    from src.frontend.export_kvcache import export_decode_graph, export_prefill_graph

    stage_tracker = StagePeakMemoryTracker()
    os.makedirs(args.output_dir, exist_ok=True)
    db_path = os.path.join(args.output_dir, "model.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    model = None
    ep_prefill = None
    ep_decode = None
    executor = None
    hf_tokenizer = None
    metadata = None
    native_counts = None
    default_db = "model.db"
    quant_summary = None

    try:
        with stage_tracker.track("load_model"):
            print(f"[export] loading {args.model}")
            model = Qwen2ForCausalLM.from_pretrained(
                args.model,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            model.eval()
        print(_format_stage_peak(stage_tracker.records[-1]))

        with stage_tracker.track("trace_graphs"):
            print("[export] tracing prefill/decode graphs")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ep_prefill = export_prefill_graph(
                    model,
                    example_seq_len=args.example_seq_len,
                )
                ep_decode = export_decode_graph(
                    model,
                    example_cache_seq_len=args.example_seq_len,
                )
        print(_format_stage_peak(stage_tracker.records[-1]))

        with stage_tracker.track("build_artifacts"):
            print("[export] compiling SQLite graph executor")
            executor = SqliteKVCacheGraphExecutor(db_path=db_path)
            executor.compile(ep_prefill, ep_decode)

            metadata = {
                "model_name": args.model,
                "vocab_size": model.config.vocab_size,
                "hidden_size": model.config.hidden_size,
                "num_hidden_layers": model.config.num_hidden_layers,
                "num_attention_heads": model.config.num_attention_heads,
                "num_key_value_heads": model.config.num_key_value_heads,
                "intermediate_size": model.config.intermediate_size,
                "num_layers": executor._num_layers,
            }

            # The compiled executor no longer needs the original model or
            # exported programs; dropping them here lowers the next stages'
            # steady-state memory.
            del ep_prefill
            del ep_decode
            del model
            ep_prefill = None
            ep_decode = None
            model = None
            gc.collect()

            print("[export] storing tokenizer vocabulary in SQLite")
            hf_tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                trust_remote_code=True,
            )
            tokenizer = SqliteTokenizer.from_pretrained(args.model, db_path=db_path)
            tokenizer.close()

            tokenizers_json = os.path.join(args.output_dir, "tokenizer.json")
            try:
                fast_tokenizer = Tokenizer.from_pretrained(args.model)
                fast_tokenizer.save(tokenizers_json)
            except Exception:
                hf_tokenizer.save_pretrained(args.output_dir)
                generated = os.path.join(args.output_dir, "tokenizer.json")
                if generated != tokenizers_json and os.path.isfile(generated):
                    shutil.copy2(generated, tokenizers_json)

            conn = executor.connection
            conn.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            metadata["eos_token_id"] = hf_tokenizer.eos_token_id
            conn.executemany(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                [(key, json.dumps(value)) for key, value in metadata.items()],
            )
            conn.commit()

            for graph_name in ("prefill", "decode"):
                instrs = executor.export_instructions_json(graph=graph_name)
                with open(os.path.join(args.output_dir, f"{graph_name}.json"), "w") as f:
                    json.dump(instrs, f, separators=(",", ":"))

            print("[export] lowering native per-op SQL graphs")
            native_counts = export_native_graphs(
                export_dir=args.output_dir,
                db_filename=default_db,
            )

            if args.emit_standalone_sql:
                print("[export] compiling standalone SQL files")
                _write_standalone_sql(args.output_dir, executor)
            else:
                _remove_stale_standalone_sql(args.output_dir)

            executor.connection.close()
        print(_format_stage_peak(stage_tracker.records[-1]))

        if not args.skip_int8:
            with stage_tracker.track("quantize_int8"):
                int8_db = os.path.join(args.output_dir, "model_int8.db")
                shutil.copy2(db_path, int8_db)
                quant_summary = quantize_db(int8_db)
            print(_format_stage_peak(stage_tracker.records[-1]))
        else:
            print("[export][memory] quantize_int8     skipped")

        _write_manifest(
            args.output_dir,
            {
                "model_name": args.model,
                "default_db": default_db,
                "num_layers": executor._num_layers,
                "vocab_size": metadata["vocab_size"],
                "files": {
                    "model_db": "model.db",
                    "model_int8_db": "model_int8.db" if not args.skip_int8 else None,
                    "prefill_json": "prefill.json",
                    "decode_json": "decode.json",
                    "prefill_native_json": "prefill.native.json",
                    "decode_native_json": "decode.native.json",
                    "prefill_sql": "prefill.sql" if args.emit_standalone_sql else None,
                    "decode_sql": "decode.sql" if args.emit_standalone_sql else None,
                    "prefill_binds": "prefill.binds.json" if args.emit_standalone_sql else None,
                    "decode_binds": "decode.binds.json" if args.emit_standalone_sql else None,
                    "tokenizer_json": "tokenizer.json",
                },
                "native_graph_counts": native_counts,
                "quantization": quant_summary,
                "memory_profile": stage_tracker.to_rows(),
            },
        )

        print("=" * 60)
        print("Export complete")
        print("=" * 60)
        print(f"output dir  : {args.output_dir}")
        print(f"default db  : {default_db}")
        if args.emit_standalone_sql:
            print("artifacts   : model.db, prefill/decode.json, prefill/decode.native.json, prefill/decode.sql, tokenizer.json")
        else:
            print("artifacts   : model.db, prefill/decode.json, prefill/decode.native.json, tokenizer.json")
        if quant_summary is not None:
            print(f"int8 ratio  : {quant_summary['compression_ratio']:.2f}x")
            print(f"int8 db     : {quant_summary['db_file_bytes_after_vacuum'] / (1024 * 1024):.1f} MiB")
    finally:
        if stage_tracker.records:
            print(stage_tracker.summary())


if __name__ == "__main__":
    main()
