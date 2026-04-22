<!--
 * @Author: xuxianghong12
 * @Date: 2026-04-14 12:23:20
 * @LastEditors: xuxianghong12
 * @LastEditTime: 2026-04-23 00:01:57
 * @SPDX-License-Identifier: Apache-2.0
-->
# Demo: load tokenizer SQLite plugin from C and C++

This README documents the current demo path:

- tokenizer lives in `/home/runner/work/llm.sql/llm.sql/sqlite-llm/llm_tokenizer.c`
- demo binaries call tokenizer through SQLite extension loading
- no standalone tokenizer C library is needed under `demo/`

All commands below are run from the repository root.

## 1. Build the SQLite plugins

```bash
make -C sqlite-llm llm_ops.so llm_tokenizer.so
```

## 2. Build the C/C++ demos

```bash
make -C demo all
```

## 3. Tokenize / detokenize directly in SQLite (WIP)

The tokenizer plugin exports these SQL functions:

- `llm_tokenize(text, tokenizer_json_path) -> BLOB`
- `llm_detokenize(token_blob) -> TEXT`
- `llm_token_count(token_blob) -> INTEGER`
- `llm_token_at(token_blob, index) -> INTEGER`

Example in the SQLite shell:

```sql
.load ./sqlite-llm/llm_tokenizer
SELECT hex(llm_tokenize('hello', '/path/to/exported/model/tokenizer.json'));
SELECT llm_detokenize(llm_tokenize('hello', '/path/to/exported/model/tokenizer.json'));
SELECT llm_token_count(llm_tokenize('hello', '/path/to/exported/model/tokenizer.json'));
SELECT llm_token_at(llm_tokenize('hello', '/path/to/exported/model/tokenizer.json'), 0);
```

## 4. C / C++ token-in token-out usage

Run the C demo with the verified hello token id (14990) and max_tokens=5:

```bash
LLM_SQL_EXTENSION_PATH=./sqlite-llm/llm_ops.so \
./demo/c_qwen_graph_tokens /path/to/exported/model 14990 5 model_int8.db
```

Run the C++ demo:

```bash
LLM_SQL_EXTENSION_PATH=./sqlite-llm/llm_ops.so \
./demo/cpp_qwen_graph_tokens /path/to/exported/model 14990 5 model_int8.db
```
<p align="center">
  <img src="figures/c-cpp-demo.png" alt="C and C++ Demo" width="100%">
  <br>
  <small>C/C++ Demo: Running Qwen-2.5-0.5B-Instruct on llm.sql</small>
</p>


---

`--profile` prints runtime profile section with prompt/generated token counts, prefill latency, decode total and average latency, per-token decode latency, throughput, peak RSS, SQL call count, top op timings, and a small RSS memory summary.

Run the C/C++ demo with profile and max_tokens=64:

```bash
# C with profile
LLM_SQL_EXTENSION_PATH=./sqlite-llm/llm_ops.so \
./demo/c_qwen_graph_tokens /path/to/exported/model 14990 64 model_int8.db --profile 
```

<p align="center">
  <img src="figures/c-profile.png" alt="C w/ profile" width="100%">
  <br>
  <small>C w/ profile: Running Qwen-2.5-0.5B-Instruct on llm.sql</small>
</p>


```bash
# C++ with profile
LLM_SQL_EXTENSION_PATH=./sqlite-llm/llm_ops.so \
./demo/cpp_qwen_graph_tokens /path/to/exported/model 14990 64 model_int8.db --profile 
```
<p align="center">
  <img src="figures/cpp-profile.png" alt="C++ w/ profile" width="100%">
  <br>
  <small>C++ w/ profile: Running Qwen-2.5-0.5B-Instruct on llm.sql</small>
</p>


**Highlight:** Running the Qwen2.5-0.5B-INT8 model (about **640MB**) on llm.sql in C/C++ with a maximum of 64 tokens achieves approximately **7.25 tokens/s**, with a peak memory usage (Peak RSS) of around **210MB**.


## 5. C / C++ string prompt demo (WIP)

`c_qwen_graph_string` and `cpp_qwen_graph_string` are the string-prompt
variants and use the same SQLite tokenizer plugin internally. They print
both the generated token IDs and the detokenized human-readable text.

```bash
LLM_SQL_EXTENSION_PATH=./sqlite-llm/llm_ops.so \
LLM_TOKENIZER_EXTENSION_PATH=./sqlite-llm/llm_tokenizer.so \
./demo/c_qwen_graph_string /path/to/exported/model "hello" 5 model_int8.db

LLM_SQL_EXTENSION_PATH=./sqlite-llm/llm_ops.so \
LLM_TOKENIZER_EXTENSION_PATH=./sqlite-llm/llm_tokenizer.so \
./demo/cpp_qwen_graph_string /path/to/exported/model "hello" 5 model_int8.db
```

If `LLM_TOKENIZER_EXTENSION_PATH` is not set, the demos try to locate
`llm_tokenizer.so` next to `llm_ops.so`.


## 6. Run tokenizer tests

```bash
make -C sqlite-llm test-tokenizer
```
