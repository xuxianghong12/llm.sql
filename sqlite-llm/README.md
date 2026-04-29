<!--
 * @SPDX-License-Identifier: Apache-2.0
 * @Author: xuxianghong12
 * @Date: 2026-04-30 00:05:54
 * @LastEditTime: 2026-04-30 00:05:54
 * @LastEditors: xuxianghong12
-->
# sqlite-llm

`sqlite-llm` is a small collection of SQLite-facing native components for LLM inference workloads.
It is not a full model runner by itself. Instead, it supplies the low-level pieces that make a SQLite-backed LLM runtime practical:

- an `ops` extension that executes tensor, matrix, normalization, transformer, and int8 kernels inside SQLite
- a custom `pcache` library that teaches SQLite's page cache about transformer layer order and memory budgets
- a `param manager` library that loads model parameters by name and evicts them by layer
- a `tokenizer` extension that exposes GPT-style BPE tokenization and detokenization as SQL functions

The directory is organized so these parts can be built independently and then combined by a higher-level runtime in Python, C, or another host language.

## What This Directory Builds

From `sqlite-llm/Makefile`:

- `make` builds `llm_ops.so`, `llm_pcache.so`, `llm_param_mgr.so`, and `llm_tokenizer.so`
- `make arm` builds `llm_ops_arm.so`
- `make test` runs the x86 operator test binary
- `make test-arm` runs the ARM operator test binary
- `make test-tokenizer` runs the tokenizer test binary

The operator extension has two back ends with the same SQL surface:

- `x86/` contains the x86 implementation, with AVX2/FMA fast paths where available
- `arm/` contains the ARM implementation, with scalar fallbacks and simple NEON hot paths gated by `LLM_HAVE_NEON`

The non-operator components live at the top level of `sqlite-llm/` because they are shared infrastructure rather than ISA-specific kernels.

## Component Summary

| Component | Artifact | Interface style | Purpose |
| --- | --- | --- | --- |
| Ops | `llm_ops.so`, `llm_ops_arm.so` | SQLite extension | Execute tensor and model primitives directly from SQL |
| Page cache | `llm_pcache.so` | C API | Replace SQLite's global page cache with a layer-aware cache |
| Parameter manager | `llm_param_mgr.so` | C API | Fetch model parameters by name with layer-aware lifetime control |
| Tokenizer | `llm_tokenizer.so` | SQLite extension | Tokenize and detokenize text using GPT/BPE rules |

## Ops Extension

The ops extension is the largest part of this directory. It registers `160` SQL entry points in the current tree and is intended to cover the common primitives needed by a transformer forward pass.

Those functions are split into these categories:

| Category | Count | What it covers |
| --- | --- | --- |
| Parameter lookup | 4 | Read named parameters and run parameter-backed linear and embedding ops |
| Construct / inspect | 8 | Create vectors and matrices and inspect shapes or values |
| Arithmetic | 10 | Add, subtract, multiply, divide, scale, and dot products |
| Element-wise / activations | 11 | Unary math plus ReLU, GELU, SiLU, sigmoid, and softmax |
| Reductions | 4 | Sum, mean, max, and min |
| Normalization | 2 | RMSNorm and LayerNorm |
| Matrix | 6 | GEMM, matrix add, transpose, and matrix-vector products |
| Shape / indexing | 6 | Reshape, flatten, slice, concat, row extract, and gather |
| Transformer | 1 | RoPE |
| N-D core | 49 | General tensor layout, broadcast, algebra, reductions, embeddings, and masking |
| N-D extra | 51 | Logical ops, bitwise ops, cumulative ops, selection, scatter/gather, stacking, attention helpers, and structural transforms |
| INT8 | 6 | Quantize, dequantize, int8 linear layers, int8 embeddings, and format inspection |
| Runtime config | 2 | Set and query worker thread count |

In practice, this means the extension covers both a legacy 1-D or 2-D blob API and a newer N-D tensor API. The newer API is what you would use for most transformer graphs because it supports broadcast semantics, shape transforms, attention helpers, int8 weights, and parameter-backed lookups.

### Ops Plugin Layout

The operator code is split by responsibility rather than placed in one monolithic source file:

- `llm_ops_*_common.c`: shared kernels and helper routines
- `llm_ops_*_blob.c`: legacy vector and matrix blob wrappers
- `llm_ops_*_elementwise.c`: unary math and activation functions
- `llm_ops_*_reduce.c`: reduction kernels
- `llm_ops_*_norm.c`: RMSNorm and LayerNorm
- `llm_ops_*_matrix.c`: GEMM, transpose, and matvec paths
- `llm_ops_*_shape.c`: reshape and indexing helpers for legacy blobs
- `llm_ops_*_transformer.c`: transformer-specific kernels such as RoPE
- `llm_ops_*_nd_*.c`: the newer N-D tensor surface, split by layout, algebra, stats, pointwise ops, reductions, sequence ops, utilities, and model helpers
- `llm_ops_*_int8.c`: quantization, dequantization, and int8 inference kernels
- `llm_ops_*_runtime.c`: runtime thread controls
- `llm_ops_*_lookup.c`: parameter-backed SQL wrappers

### What The Ops Extension Is For

The extension is meant to make SQLite blobs usable as model tensors.
Typical use cases include:

- running small inference graphs directly in SQL for demos and experiments
- exporting model subgraphs into SQL-callable operators
- validating tensor programs against SQLite-hosted data
- mixing relational queries with tensor execution in one process

Representative SQL entry points include:

- `llm_add_simd`, `llm_mul_nd`, `llm_bmm`, `llm_softmax_nd`
- `llm_rmsnorm`, `llm_layernorm_nd`, `llm_rope_nd`
- `llm_linear_nd`, `llm_linear_int8_nd`, `llm_embedding_nd`, `llm_embedding_param_nd`
- `llm_sdpa_nd`, `llm_bool_to_additive_mask_nd`

## Page Cache Library

`llm_pcache.so` is not a SQL extension. It is a shared library that installs a custom `sqlite3_pcache_methods2` implementation through `sqlite3_config()`.

Its job is to adapt SQLite's page cache to the access pattern of transformer inference, where the runtime tends to walk parameters layer by layer.

The public C API currently exposes `15` entry points, including:

- installation and removal of the custom cache
- policy selection between baseline LRU and a transformer-oriented OPT-style policy
- active-layer and done-layer notifications
- a prefetch window for future layers
- precise begin or end bracketing of layer-specific I/O
- statistics and per-layer residency queries

In concrete terms, this component is responsible for:

- enforcing a hard memory budget for SQLite pages
- preferring eviction of completed-layer pages before active or near-future layers
- tracking cache hits, misses, evictions, and per-layer page residency
- giving the inference engine a place to express layer-order hints without changing SQLite itself

This is the component that makes a SQLite-backed parameter store behave more like a model-serving cache and less like a generic database cache.

## Parameter Manager Library

`llm_param_mgr.so` is also a C library rather than a SQL extension. It builds on SQLite plus the custom page cache to provide layer-aware parameter access.

The public C API currently exposes `8` functions:

- `param_manager_create`
- `param_manager_destroy`
- `param_manager_get`
- `param_manager_load_layer`
- `param_manager_evict_layer`
- `param_manager_reset`
- `param_manager_get_stats`
- `param_manager_set_active_window`

Its responsibilities are:

- opening a SQLite database that contains a `model_params` table
- indexing parameter names up front
- extracting transformer layer IDs from parameter names such as `layers.0.*`
- loading parameter blobs lazily on demand
- enforcing a memory budget at the parameter level
- cooperating with `llm_pcache` so completed layers can be dropped under pressure

This is the piece that turns a raw parameter table into something a runtime can query by symbolic name without keeping the entire model resident in host memory.

## Tokenizer Extension

`llm_tokenizer.so` is a SQLite extension focused on GPT-style byte-pair encoding.
It currently registers `4` SQL scalar functions:

- `llm_tokenize(text, tokenizer_json_path)`
- `llm_detokenize(token_ids_blob)`
- `llm_token_count(text)`
- `llm_token_at(token_ids_blob, index)`

The tokenizer implementation includes:

- byte-to-Unicode and Unicode-to-byte mapping compatible with GPT-family tokenizers
- UTF-8 helpers for encoding and decoding code points
- a compact in-memory hash table for token lookup
- merge-rule loading from HuggingFace-style `tokenizer.json`
- vocabulary lookup from the active SQLite connection

This extension exists so tokenization can happen in the same SQLite-centered workflow as parameter lookup and tensor execution.

## How The Pieces Fit Together

At a high level, the intended workflow is:

1. Store model parameters in SQLite.
2. Install `llm_pcache.so` before opening the inference database if layer-aware caching is needed.
3. Use `llm_param_mgr.so` to load or prefetch parameter blobs by symbolic name and manage their lifetime across layers.
4. Load `llm_ops.so` or `llm_ops_arm.so` into SQLite to execute tensor operators against those blobs.
5. Load `llm_tokenizer.so` when text needs to enter or leave the system as token IDs.

That separation is deliberate:

- `ops` does math
- `pcache` controls page residency
- `param manager` controls named parameter lifetime
- `tokenizer` bridges text and token IDs

## Minimal Build Example

```bash
cd sqlite-llm
make
make test
make test-tokenizer
```

For the ARM operator build:

```bash
cd sqlite-llm
make arm
make test-arm
```

## Minimal SQLite Usage Example

```sql
.load ./llm_ops
.load ./llm_tokenizer

SELECT llm_token_count('hello world');
SELECT llm_str(llm_vec(4, 1.0));
SELECT llm_sum(llm_add_simd(llm_vec(4, 1.0), llm_vec(4, 2.0)));
```

For Python or another host runtime, `llm_pcache.so` and `llm_param_mgr.so` are intended to be loaded as shared libraries and called through their C APIs rather than through `load_extension()`.
