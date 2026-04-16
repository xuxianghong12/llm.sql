/*
 * llm_pcache.h — Layer-aware page cache for LLM inference
 *
 * Custom SQLite page cache (sqlite3_pcache_methods2) optimised for the
 * sequential, layer-by-layer access pattern of transformer inference:
 *
 *   1. **Layer-priority eviction** — pages belonging to completed layers
 *      are evicted before pages of the current or future layers.
 *   2. **Memory budget** — a hard cap (in bytes) on total page-cache
 *      memory prevents OOM under constrained environments.
 *   3. **Layer hints** — the inference engine notifies the cache which
 *      layer is active; eviction policy adapts accordingly.
 *   4. **Statistics** — hit/miss/eviction counters and per-layer stats
 *      for profiling and tuning.
 *
 * Usage from Python (via ctypes):
 *
 *     lib = ctypes.CDLL("llm_pcache.so")
 *     lib.llm_pcache_install(budget_bytes, num_layers)
 *     ...
 *     lib.llm_pcache_set_active_layer(layer_idx)
 *     lib.llm_pcache_mark_layer_done(layer_idx)
 *     ...
 *     lib.llm_pcache_get_stats(...)
 *     lib.llm_pcache_uninstall()
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef LLM_PCACHE_H
#define LLM_PCACHE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LLM_PCACHE_POLICY_LRU 0
#define LLM_PCACHE_POLICY_OPT 1

/* ── Installation / lifecycle ────────────────────────────────────────────── */

/*
 * Install the custom LLM page cache as SQLite's global page cache.
 *
 * Must be called BEFORE any sqlite3_open() / sqlite3_initialize().
 *
 *   budget_bytes : maximum total memory the cache may use (0 = unlimited)
 *   num_layers   : number of transformer layers (for priority tracking)
 *
 * Eviction policy defaults to LRU for backwards compatibility. Call
 * llm_pcache_set_policy(LLM_PCACHE_POLICY_OPT) before install to enable
 * deterministic layer-order eviction for transformer streaming workloads.
 *
 * Returns 0 on success, -1 if already installed, or a positive SQLite
 * error code (e.g. SQLITE_MISUSE) if sqlite3_config() fails.
 */
int llm_pcache_install(int64_t budget_bytes, int num_layers);

/*
 * Set or query the active eviction policy.
 *
 * This may be called before or after install. Invalid values fall back to LRU.
 * Returns the normalized policy value.
 */
int llm_pcache_set_policy(int policy);
int llm_pcache_get_policy(void);

/*
 * Configure how many future layers belong to the explicit prefetch window.
 *
 * A value of 0 disables the window and reverts to pure distance-based future
 * ordering. Values are clamped to [0, LLM_PCACHE_MAX_LAYERS].
 */
int llm_pcache_set_prefetch_window(int window);
int llm_pcache_get_prefetch_window(void);

/*
 * Uninstall the custom page cache and restore SQLite's default.
 * Returns 0 on success.
 */
int llm_pcache_uninstall(void);

/* ── Layer hints (called by the inference engine) ────────────────────────── */

/*
 * Set the currently active layer.  Pages touched while this layer is
 * active receive elevated priority and are evicted last.
 *
 * layer_idx = -1 means "no specific layer" (global params, embeddings).
 */
void llm_pcache_set_active_layer(int layer_idx);

/*
 * Mark a layer as done.  Its pages become low-priority eviction
 * candidates (they will be evicted before current/future layer pages).
 */
void llm_pcache_mark_layer_done(int layer_idx);

/*
 * Mark an upcoming layer as "active" so its pages are protected from
 * eviction.  Use this as a prefetch hint before the layer is actually
 * being processed.
 */
void llm_pcache_prefetch_layer(int layer_idx);

/*
 * Reset all layer states for a new forward pass.  All layers return
 * to "unknown" priority.
 */
void llm_pcache_reset_layers(void);

/* ── Precise layer I/O bracketing ────────────────────────────────────────── */

/*
 * Begin a layer-specific I/O region.  While a layer I/O region is active:
 *   - New pages (cache misses) are tagged with layer_idx.
 *   - Existing pages (cache hits) keep their current layer tag
 *     (avoids tag drift from shared B-tree internal / index pages).
 *
 * This is more precise than set_active_layer(), which updates the tag
 * on every hit.  Use begin/end around the SQL query that loads a
 * specific layer's parameters.
 *
 * Calls may be nested or overlapping across threads; the layer_io
 * state is per-thread (stored in a thread-local).
 */
void llm_pcache_begin_layer_io(int layer_idx);

/*
 * End the current layer I/O region.  Subsequent cache misses revert
 * to using the global active_layer for tagging.
 */
void llm_pcache_end_layer_io(void);

/* ── Statistics ──────────────────────────────────────────────────────────── */

typedef struct {
    int64_t cache_hits;      /* xFetch found the page                      */
    int64_t cache_misses;    /* xFetch did not find the page               */
    int64_t pages_current;   /* pages currently in cache                   */
    int64_t pages_pinned;    /* pages currently pinned                     */
    int64_t evictions;       /* total pages evicted                        */
    int64_t evictions_layer; /* evictions of done-layer pages (good)       */
    int64_t evictions_other; /* evictions of active/future pages (bad)     */
    int64_t memory_used;     /* current memory usage in bytes              */
    int64_t memory_budget;   /* configured budget in bytes                 */
    int active_layer;        /* current active layer index                 */
    int num_layers;          /* total number of layers                     */
    int policy;              /* LLM_PCACHE_POLICY_*                        */
    int prefetch_window;     /* protected future-layer window size         */
} llm_pcache_stats_t;

/*
 * Fill *stats with current cache statistics.
 */
void llm_pcache_get_stats(llm_pcache_stats_t *stats);

/*
 * Reset all statistics counters to zero.
 */
void llm_pcache_reset_stats(void);

/*
 * Return the number of pages currently held in the cache for layer_idx.
 * Useful for verifying that prefetch loaded the expected pages.
 * Returns 0 if layer_idx is out of range.
 */
int64_t llm_pcache_layer_pages(int layer_idx);

#ifdef __cplusplus
}
#endif

#endif /* LLM_PCACHE_H */
