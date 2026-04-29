/*
 * llm_ops_x86_runtime.c - Runtime control wrappers for x86 operators
 *
 * This file owns SQL functions that adjust or inspect runtime behavior rather
 * than perform tensor math directly. Today that is the thread-count control
 * surface used by parallel int8 paths.
 *
 * Maintenance guidance:
 *   - Keep global runtime knobs centralized here so side effects are easy to
 *     audit.
 *   - Clamp or validate user-provided runtime values at the boundary.
 *   - If a setting affects multiple operator families, expose it here instead
 *     of hiding it inside one compute-oriented split file.
 */
#include "llm_ops_x86_internal.h"
SQLITE_EXTENSION_INIT3

void sql_set_threads(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    int n = sqlite3_value_int(argv[0]);
    if (n < 1)
        n = 1;
    if (n > 256)
        n = 256;
    g_llm_num_threads = n;
    sqlite3_result_int(ctx, g_llm_num_threads);
}

void sql_get_threads(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    (void)argv;
    sqlite3_result_int(ctx, g_llm_num_threads);
}