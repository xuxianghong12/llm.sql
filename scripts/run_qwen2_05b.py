#!/usr/bin/env python
"""Run Qwen2.5-0.5B generation through the llm.sql public API.

Usage:
    python scripts/run_qwen2_05b.py [--model-dir DIR] [--prompt TEXT] [--max-tokens N]
        [--mode single_sql|eager|layer_stream] [--db-filename FILE] [--threads N]

Example:
    python scripts/run_qwen2_05b.py --model-dir /tmp/llm_sql_qwen2_05b --prompt "Hello" --max-tokens 64 --mode eager

Description:
    Loads an exported model directory (manifest + model DB) and runs a single
    generation. Prints the completion and optional profile info if `--profile`
    is supplied.

Requirements:
    - The exported model directory produced by `export_qwen2_05b.py`.

"""

from __future__ import annotations

import argparse
import json
import os
import sys


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from llmsql import InferenceSession, SessionConfig


def _manifest_default_db(model_dir: str) -> str | None:
    manifest_path = os.path.join(model_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        return None
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    default_db = manifest.get("default_db")
    return default_db if isinstance(default_db, str) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run llm.sql Qwen2.5-0.5B inference")
    parser.add_argument("--model-dir", default="/tmp/llm_sql_qwen2_05b")
    parser.add_argument("--prompt", default="Explain why SQLite is a useful LLM backend.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--mode",
        choices=["single_sql", "eager", "layer_stream"],
        default="single_sql",
    )
    parser.add_argument("--db-filename", default="model_int8.db")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--pcache-budget-mb", type=float, default=None)
    parser.add_argument("--use-pcache", action="store_true")
    parser.add_argument("--pcache-policy", choices=["lru", "opt"], default="opt")
    parser.add_argument("--pcache-prefetch-window", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = SessionConfig(
        export_dir=args.model_dir,
        mode=args.mode,
        db_filename=args.db_filename,
        num_threads=args.threads,
        pcache_budget_mb=args.pcache_budget_mb,
        use_pcache=args.use_pcache,
        pcache_policy=args.pcache_policy,
        pcache_prefetch_window=args.pcache_prefetch_window,
    )

    with InferenceSession(config) as session:
        text = session.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
            profile=args.profile,
        )
        profile = session.last_profile
        resolved_db_filename = session.db_filename
    manifest_default_db = _manifest_default_db(args.model_dir)

    print("=" * 60)
    print("Runtime")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Database: {resolved_db_filename}")
    if (
        args.db_filename is None
        and manifest_default_db is not None
        and manifest_default_db != resolved_db_filename
    ):
        print(
            "Manifest default_db: "
            f"{manifest_default_db} "
            f"(ignored; using {resolved_db_filename})"
        )
    print()

    print("=" * 60)
    print("Prompt")
    print("=" * 60)
    print(args.prompt)
    print()
    print("=" * 60)
    print("Completion")
    print("=" * 60)
    print(text)

    if args.profile and profile is not None:
        print()
        print("=" * 60)
        print("Profile")
        print("=" * 60)
        print(f"Mode: {profile.get('mode', args.mode)}")
        print(f"Database: {profile.get('db_filename', resolved_db_filename)}")
        print(f"Prompt tokens: {profile.get('prompt_tokens', 0)}")
        print(f"Generated tokens: {profile.get('generated_tokens', 0)}")
        print(f"Prefill latency: {profile.get('prefill_ms', 0.0):.2f} ms")
        print(f"Decode steps: {profile.get('decode_steps', 0)}")
        print(f"Decode total: {profile.get('decode_total_ms', 0.0):.2f} ms")
        print(f"Decode avg/step: {profile.get('decode_avg_ms', 0.0):.2f} ms")
        print(f"Throughput: {profile.get('tokens_per_sec', 0.0):.2f} tokens/sec")
        print(f"Peak RSS: {profile.get('peak_rss_mb', 0.0):.1f} MB")
        if config.use_pcache or config.mode == "layer_stream":
            print(f"pcache policy: {config.pcache_policy}")
            print(f"pcache prefetch window: {config.pcache_prefetch_window}")

        # Additional diagnostics
        sql_calls = profile.get("sql_call_count")
        if sql_calls is not None:
            print(f"SQL call count: {sql_calls}")

        op_timings = profile.get("op_timings_us")
        if isinstance(op_timings, dict) and op_timings:
            try:
                # Print top 5 longest op timings
                items = sorted(op_timings.items(), key=lambda kv: kv[1], reverse=True)
                print("Top op timings (us):")
                for name, us in items[:5]:
                    print(f"  {name:<30s} {us:10.2f}")
            except Exception:
                pass

        mem_summary = profile.get("memory_summary")
        if mem_summary:
            print()
            print("Memory summary:")
            print(mem_summary)


if __name__ == "__main__":
    main()
