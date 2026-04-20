<!--
 * @Author: xuxianghong12
 * @Date: 2026-04-14 12:23:20
 * @LastEditors: xuxianghong12
 * @LastEditTime: 2026-04-16 20:33:41
 * @SPDX-License-Identifier: Apache-2.0
-->
# Demo: Run Qwen2.5-0.5B-Instruct on llm.sql from C and C++

This README is the verified demo path for Ubuntu-22.04.

Fixed inputs used below:

- model: `Qwen2.5-0.5B-Instruct`
- prompt: `hello`

The commands below are written for Linux and should be run from the repository root.

## 1. Build the SQLite extensions

```bash
# Core LLM tensor ops
make -C sqlite-llm llm_ops.so

# BPE tokenizer (enables string prompt input/output)
make -C sqlite-llm llm_tokenizer.so
```

## 2. Export Qwen2.5-0.5B-Instruct
Refer to [README.md](../README.md) to export model yourself of directly download exported models.


## 3. Run Qwen on llm.sql from C and C++

All four demos now accept **plain-text string prompts** and produce
decoded text output.  Tokenization is handled by the
`llm_tokenizer.so` SQLite extension — no C tokenizer library is
linked into the binaries.

Build all binaries:

```bash
make -C demo c_qwen_graph_tokens cpp_qwen_graph_tokens \
             c_qwen_graph_string cpp_qwen_graph_string
```

Run the C demo with a plain-text prompt and `max_tokens=5`:

```bash
LLM_SQL_EXTENSION_PATH=./sqlite-llm/llm_ops.so \
  ./demo/c_qwen_graph_tokens /path/to/exported/model "hello" 5 model_int8.db
```

Run the C++ demo:

```bash
LLM_SQL_EXTENSION_PATH=./sqlite-llm/llm_ops.so \
  ./demo/cpp_qwen_graph_tokens /path/to/exported/model "hello" 5 model_int8.db
```

The same commands work for the `_string` variants:

```bash
LLM_SQL_EXTENSION_PATH=./sqlite-llm/llm_ops.so \
  ./demo/c_qwen_graph_string /path/to/exported/model "hello" 5 model_int8.db

LLM_SQL_EXTENSION_PATH=./sqlite-llm/llm_ops.so \
  ./demo/cpp_qwen_graph_string /path/to/exported/model "hello" 5 model_int8.db
```

The output is decoded text instead of token IDs.

> **Note:** The tokenizer extension (`llm_tokenizer.so`) is
> automatically located relative to the `llm_ops.so` path.  You can
> override it with the `LLM_TOKENIZER_EXTENSION_PATH` environment
> variable.

<p align="center">
  <img src="figures/c-cpp-demo.png" alt="C and C++ Demo" width="100%">
  <br>
  <small>C/C++ Demo: Running Qwen-2.5-0.5B-Instruct on llm.sql</small>
</p>

## 4. Run tokenizer extension tests

```bash
make -C sqlite-llm test-tokenizer
```
