/*
 * llm_param_mgr.h - Layer-aware Parameter Manager for LLM Inference
 *
 * Provides zero-copy parameter loading with layer-aware memory management.
 * Eliminates Python-side memory accumulation by managing parameters in C.
 *
 * Key features:
 * - Zero-copy parameter access via memoryview
 * - Layer-aware eviction (completed layers evicted first)
 * - Memory budget enforcement
 * - Integration with custom pcache
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef LLM_PARAM_MGR_H
#define LLM_PARAM_MGR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
typedef struct param_manager param_manager_t;
typedef struct sqlite3 sqlite3;

/* Export macro for shared library */
#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef __GNUC__
#define PARAM_MGR_API __attribute__((dllexport))
#else
#define PARAM_MGR_API __declspec(dllexport)
#endif
#elif defined(__GNUC__) && __GNUC__ >= 4
#define PARAM_MGR_API __attribute__((visibility("default")))
#else
#define PARAM_MGR_API
#endif

/* ── Parameter Manager Lifecycle ────────────────────────────────────────── */

/**
 * Create a new parameter manager.
 *
 * @param db_path       Path to the SQLite database containing model_params
 * @param num_layers    Number of transformer layers in the model
 * @param budget_mb     Memory budget in megabytes (0 = unlimited)
 * @return              Parameter manager handle, or NULL on error
 */
PARAM_MGR_API param_manager_t *
param_manager_create(const char *db_path, int num_layers, size_t budget_mb);

/**
 * Destroy a parameter manager and free all resources.
 */
PARAM_MGR_API void param_manager_destroy(param_manager_t *mgr);

/* ── Parameter Access ───────────────────────────────────────────────────── */

/**
 * Get a parameter by name (zero-copy).
 *
 * Returns a pointer to the parameter data in the page cache.
 * The pointer remains valid until the parameter is evicted.
 *
 * @param mgr           Parameter manager
 * @param name          Parameter name (e.g.,
 * "layers.0.self_attn.q_proj.weight")
 * @param out_size      Output: size of the parameter in bytes
 * @return              Pointer to parameter data, or NULL if not found
 */
PARAM_MGR_API const void *
param_manager_get(param_manager_t *mgr, const char *name, size_t *out_size);

/**
 * Load all parameters for a specific layer.
 *
 * Loads parameters matching "layers.<layer_id>.*" pattern.
 * Notifies pcache to set the active layer.
 *
 * @param mgr           Parameter manager
 * @param layer_id      Layer index (0-based), or -1 for non-layer params
 * @return              0 on success, -1 on error
 */
PARAM_MGR_API int param_manager_load_layer(param_manager_t *mgr, int layer_id);

/**
 * Mark a layer as completed and eligible for eviction.
 *
 * Does not immediately evict; eviction happens when memory pressure occurs.
 * Notifies pcache that the layer is done.
 *
 * @param mgr           Parameter manager
 * @param layer_id      Layer index to mark as done
 */
PARAM_MGR_API void param_manager_evict_layer(param_manager_t *mgr,
                                             int layer_id);

/**
 * Reset all layer states for a new forward pass.
 *
 * Clears the "done" flags but keeps cached parameters.
 */
PARAM_MGR_API void param_manager_reset(param_manager_t *mgr);

/* ── Memory Management ──────────────────────────────────────────────────── */

/**
 * Get current memory usage statistics.
 *
 * @param mgr           Parameter manager
 * @param out_used_mb   Output: currently used memory in MB
 * @param out_budget_mb Output: memory budget in MB
 */
PARAM_MGR_API void param_manager_get_stats(param_manager_t *mgr,
                                           double *out_used_mb,
                                           double *out_budget_mb);

/**
 * Force eviction of all parameters outside the active window.
 *
 * Keeps only parameters for layers [current_layer, current_layer + lookahead).
 *
 * @param mgr           Parameter manager
 * @param current_layer Current active layer
 * @param lookahead     Number of layers to keep ahead
 */
PARAM_MGR_API void param_manager_set_active_window(param_manager_t *mgr,
                                                   int current_layer,
                                                   int lookahead);

#ifdef __cplusplus
}
#endif

#endif /* LLM_PARAM_MGR_H */
