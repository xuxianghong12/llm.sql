<!--
 * @SPDX-License-Identifier: Apache-2.0
 * @Author: xuxianghong12
 * @Date: 2026-04-19 12:05:00
 * @LastEditors: xuxianghong12
 * @LastEditTime: 2026-04-19 16:27:04
-->
# Roadmap

[Home](index.md) | [中文](../zh/roadmap.html) | [Features](features.md)

## Near-Term Priorities

- Reduce peak RAM during model export so writing SQLite BLOB tables does not require large temporary memory spikes.
- Remove the remaining `tokenizers` dependency from the lightweight Python inference path.
- Tighten SQLite page-cache upper bounds so runtime memory stays predictable under pressure.
- Improve x86 operator performance and continue cleaning the graph-to-runtime integration path.

## Planned Expansions

- Add ISA-aware acceleration paths for ARM NEON and RISC-V RVV, alongside stronger x86 specializations.
- Improve INT8 storage efficiency and extend quantization support toward smaller formats such as INT4.
- Expand model coverage beyond the current Qwen2.5-0.5B export and runtime workflow.
- Package the exported assets into a more self-contained distribution that is easier to ship and reuse.

## Developer Tooling

- Add clearer tracing and profiling so bottlenecks can be attributed to layers, operators, and swap behavior.
- Improve the C/C++ path with text-friendly I/O and more complete runtime instrumentation.
- Add SQL-native prompt helpers and better streaming decode ergonomics for application developers.
- Grow language bindings beyond Python toward Go, Java, Rust, and other direct integrations.

## Longer-Horizon Runtime Work

- Persist KV cache state for fault-tolerant resume and task-specific prefilling workflows.
- Load tokenizer and embedding data on demand to reduce baseline memory cost.
- Support multi-model scheduling when several LLM runtimes compete for the same RAM budget.
- Keep the SQL-facing API stable while opening room for NPU or GPU-backed operator bundles underneath.