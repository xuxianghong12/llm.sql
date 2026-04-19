<!--
 * @SPDX-License-Identifier: Apache-2.0
 * @Author: xuxianghong12
 * @Date: 2026-04-19 15:49:57
 * @LastEditTime: 2026-04-19 16:26:32
 * @LastEditors: xuxianghong12
-->
# Features

[Home](index.md) | [中文](../zh/features.html) | [Roadmap](roadmap.md)

## Implemented Today

| Area | Status | Notes |
| --- | --- | --- |
| Model path | Implemented | The repository already contains export, quantization, runtime, and native-graph scripts for Qwen2.5-0.5B-Instruct. |
| SQLite runtime | Implemented | Operator execution is wired through the SQLite extension and can be driven from SQL-backed runtime code. |
| Runtime modes | Implemented | The Python runner exposes both eager full loading and layer-by-layer streaming loading. |
| Storage format | Implemented | Model artifacts are persisted as SQLite databases together with graph, manifest, and tokenizer metadata. |
| Memory control | Implemented | The codebase includes full loading, loading-on-demand, and page-cache related components for bounded execution. |
| Language access | Implemented / PoC | Python is the main supported path today, while the C/C++ demos cover lower-level token-id-based experiments. |
| Observability | Implemented | Runtime flags plus utility modules provide lightweight timing and memory inspection hooks. |

## What You Can Do Right Now

- Build the SQLite extension and run Qwen2.5-0.5B inference from exported artifacts.
- Switch between `eager` and `layer_stream` to trade throughput for lower peak memory.
- Export SQLite-native artifacts such as `model.db`, `model_int8.db`, graph JSON, and tokenizer files.
- Reuse the same exported assets across the Python path and the current C/C++ demo path.
- Profile the runtime with the existing script flags and helper utilities before changing backend code.

## Current Boundaries

- The public happy path is centered on Qwen2.5-0.5B-Instruct.
- Lightweight inference still depends on the `tokenizers` package for BPE handling.
- The C/C++ demos are useful for runtime validation, but they are not yet the primary end-user interface.