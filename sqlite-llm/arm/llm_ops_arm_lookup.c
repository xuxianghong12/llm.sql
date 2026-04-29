/*
 * llm_ops_arm_lookup.c - Parameter lookup helpers backed by SQLite storage
 *
 * This file owns SQL functions and helper state used to fetch model parameter
 * blobs from a file-backed SQLite database. It is the bridge between operator
 * registration and the parameter storage layer, not a math kernel file.
 *
 * Maintenance guidance:
 *   - Keep connection lifecycle, prepared statement reuse, and cleanup logic
 *     together here so lookup state remains easy to reason about.
 *   - Surface user-facing database errors at this boundary instead of letting
 *     lower-level SQLite failures leak out unannotated.
 *   - New parameter-fetch SQL helpers should usually start in this file.
 */
#include "llm_ops_arm_internal.h"
SQLITE_EXTENSION_INIT3

struct llm_param_lookup_t {
    sqlite3 *lookup_db;
    sqlite3_stmt *lookup_stmt;
};

static void llm_param_lookup_destroy(void *raw) {
    llm_param_lookup_t *lookup = (llm_param_lookup_t *)raw;
    if (!lookup)
        return;
    if (lookup->lookup_stmt)
        sqlite3_finalize(lookup->lookup_stmt);
    if (lookup->lookup_db)
        sqlite3_close(lookup->lookup_db);
    sqlite3_free(lookup);
}

int llm_register_lookup_function(sqlite3 *db,
                                 const char *name,
                                 int narg,
                                 void (*fn)(sqlite3_context *,
                                            int,
                                            sqlite3_value **)) {
    llm_param_lookup_t *lookup =
        (llm_param_lookup_t *)sqlite3_malloc(sizeof(llm_param_lookup_t));
    if (!lookup)
        return SQLITE_NOMEM;
    lookup->lookup_db = NULL;
    lookup->lookup_stmt = NULL;

    int rc = sqlite3_create_function_v2(db,
                                        name,
                                        narg,
                                        SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                                        lookup,
                                        fn,
                                        NULL,
                                        NULL,
                                        llm_param_lookup_destroy);
    if (rc != SQLITE_OK)
        llm_param_lookup_destroy(lookup);
    return rc;
}

static int llm_param_lookup_ensure(llm_param_lookup_t *lookup,
                                   sqlite3 *db,
                                   sqlite3_context *ctx) {
    if (lookup->lookup_stmt)
        return SQLITE_OK;

    const char *db_path = sqlite3_db_filename(db, "main");
    if (!db_path || db_path[0] == '\0' || strcmp(db_path, ":memory:") == 0) {
        sqlite3_result_error(
            ctx, "llm_get_param requires a file-backed SQLite database", -1);
        return SQLITE_ERROR;
    }

    int rc = sqlite3_open_v2(db_path,
                             &lookup->lookup_db,
                             SQLITE_OPEN_READONLY | SQLITE_OPEN_NOMUTEX,
                             NULL);
    if (rc != SQLITE_OK) {
        sqlite3_result_error(
            ctx,
            sqlite3_errmsg(lookup->lookup_db ? lookup->lookup_db : db),
            -1);
        if (lookup->lookup_db) {
            sqlite3_close(lookup->lookup_db);
            lookup->lookup_db = NULL;
        }
        return rc;
    }

    sqlite3_exec(lookup->lookup_db, "PRAGMA mmap_size=0", NULL, NULL, NULL);
    rc = sqlite3_prepare_v2(lookup->lookup_db,
                            "SELECT data FROM model_params WHERE name = ?1",
                            -1,
                            &lookup->lookup_stmt,
                            NULL);
    if (rc != SQLITE_OK) {
        sqlite3_result_error(ctx, sqlite3_errmsg(lookup->lookup_db), -1);
        sqlite3_close(lookup->lookup_db);
        lookup->lookup_db = NULL;
        lookup->lookup_stmt = NULL;
    }
    return rc;
}

int llm_param_lookup_step_blob(llm_param_lookup_t *lookup,
                               sqlite3 *db,
                               sqlite3_context *ctx,
                               const char *name,
                               const void **out_blob,
                               int *out_size) {
    if (!name || !out_blob || !out_size) {
        sqlite3_result_null(ctx);
        return SQLITE_MISUSE;
    }
    if (llm_param_lookup_ensure(lookup, db, ctx) != SQLITE_OK)
        return SQLITE_ERROR;

    sqlite3_reset(lookup->lookup_stmt);
    sqlite3_clear_bindings(lookup->lookup_stmt);
    sqlite3_bind_text(lookup->lookup_stmt, 1, name, -1, SQLITE_TRANSIENT);

    int rc = sqlite3_step(lookup->lookup_stmt);
    if (rc == SQLITE_ROW) {
        *out_blob = sqlite3_column_blob(lookup->lookup_stmt, 0);
        *out_size = sqlite3_column_bytes(lookup->lookup_stmt, 0);
        return SQLITE_OK;
    }

    if (rc == SQLITE_DONE) {
        sqlite3_result_error(ctx, "llm parameter not found", -1);
    } else {
        sqlite3_result_error(ctx, sqlite3_errmsg(lookup->lookup_db), -1);
    }
    sqlite3_reset(lookup->lookup_stmt);
    sqlite3_clear_bindings(lookup->lookup_stmt);
    return rc;
}

void llm_param_lookup_release(llm_param_lookup_t *lookup) {
    if (!lookup || !lookup->lookup_stmt)
        return;
    sqlite3_reset(lookup->lookup_stmt);
    sqlite3_clear_bindings(lookup->lookup_stmt);
}

void sql_get_param(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    llm_param_lookup_t *lookup = (llm_param_lookup_t *)sqlite3_user_data(ctx);
    sqlite3 *db = sqlite3_context_db_handle(ctx);
    const unsigned char *name = sqlite3_value_text(argv[0]);

    if (!lookup || !name) {
        sqlite3_result_null(ctx);
        return;
    }

    const void *blob = NULL;
    int size = 0;
    if (llm_param_lookup_step_blob(
            lookup, db, ctx, (const char *)name, &blob, &size) != SQLITE_OK) {
        return;
    }
    sqlite3_result_blob(ctx, blob, size, SQLITE_TRANSIENT);
    llm_param_lookup_release(lookup);
}