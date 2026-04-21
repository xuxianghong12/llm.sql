/*
 * C demo: run Qwen inference with string prompt input/output.
 *
 * Usage:
 *   c_qwen_graph_string <model_dir> <prompt> <max_tokens> [db]
 *
 * Tokenization is performed via the llm_tokenizer SQLite extension
 * (loaded alongside llm_ops), so no C tokenizer library is linked.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "native_graph_runtime.h"

#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* ── path helpers ──────────────────────────────────────────── */

static char *join_path(
    const char *dir,
    const char *name
) {
    size_t dlen = strlen(dir);
    size_t nlen = strlen(name);
    char *buf = (char *)malloc(dlen + 1 + nlen + 1);
    if (buf == NULL) {
        return NULL;
    }
    memcpy(buf, dir, dlen);
    if (dlen > 0 && dir[dlen - 1] != '/') {
        buf[dlen] = '/';
        ++dlen;
    }
    memcpy(buf + dlen, name, nlen + 1);
    return buf;
}

static const char *resolve_db(
    const char *model_dir,
    const char *explicit_db
) {
    struct stat st;
    char *path;
    if (explicit_db != NULL && explicit_db[0] != '\0') {
        return explicit_db;
    }
    path = join_path(model_dir, "model.db");
    if (path != NULL && stat(path, &st) == 0) {
        free(path);
        return "model.db";
    }
    free(path);
    path = join_path(model_dir, "model_int8.db");
    if (path != NULL && stat(path, &st) == 0) {
        free(path);
        return "model_int8.db";
    }
    free(path);
    return "model.db";
}

/* ── tokenizer helpers via SQL ─────────────────────────────── */

/*
 * Open the model db, load llm_ops + llm_tokenizer extensions,
 * then use SQL to encode/decode.
 */

static sqlite3 *open_model_db(
    const char *db_path,
    const char *ext_path,
    const char *tok_ext_path
) {
    sqlite3 *db = NULL;
    char *err = NULL;

    if (sqlite3_open_v2(
            db_path,
            &db,
            SQLITE_OPEN_READONLY,
            NULL) != SQLITE_OK) {
        if (db != NULL) {
            sqlite3_close(db);
        }
        return NULL;
    }
    sqlite3_enable_load_extension(db, 1);

    /* Load llm_ops (needed by the runtime) */
    if (sqlite3_load_extension(
            db,
            ext_path,
            "sqlite3_llm_ops_init",
            &err) != SQLITE_OK) {
        fprintf(stderr,
                "load llm_ops: %s\n",
                err ? err : "unknown");
        sqlite3_free(err);
        sqlite3_close(db);
        return NULL;
    }

    /* Load llm_tokenizer */
    if (sqlite3_load_extension(
            db,
            tok_ext_path,
            "sqlite3_llm_tokenizer_init",
            &err) != SQLITE_OK) {
        fprintf(stderr,
                "load llm_tokenizer: %s\n",
                err ? err : "unknown");
        sqlite3_free(err);
        sqlite3_close(db);
        return NULL;
    }

    return db;
}

/* Encode text to int32 BLOB via SQL, then extract token IDs. */
static int sql_encode(
    sqlite3 *db,
    const char *text,
    const char *json_path,
    int **out_ids,
    int *out_count
) {
    sqlite3_stmt *stmt = NULL;
    int rc;

    rc = sqlite3_prepare_v2(
        db,
        "SELECT llm_tokenize(?1, ?2)",
        -1,
        &stmt,
        NULL);
    if (rc != SQLITE_OK) {
        return 0;
    }
    sqlite3_bind_text(stmt, 1, text, -1, SQLITE_STATIC);
    sqlite3_bind_text(
        stmt, 2, json_path, -1, SQLITE_STATIC);

    if (sqlite3_step(stmt) != SQLITE_ROW ||
        sqlite3_column_type(stmt, 0) != SQLITE_BLOB) {
        sqlite3_finalize(stmt);
        return 0;
    }

    {
        int nbytes = sqlite3_column_bytes(stmt, 0);
        int count = nbytes / (int)sizeof(int32_t);
        const int32_t *blob =
            (const int32_t *)sqlite3_column_blob(
                stmt, 0);
        int *ids =
            (int *)malloc((size_t)count * sizeof(int));
        if (ids == NULL) {
            sqlite3_finalize(stmt);
            return 0;
        }
        for (int i = 0; i < count; ++i) {
            ids[i] = (int)blob[i];
        }
        *out_ids = ids;
        *out_count = count;
    }
    sqlite3_finalize(stmt);
    return 1;
}

/* Decode int32 token IDs to text via SQL. */
static char *sql_decode(
    sqlite3 *db,
    const int *ids,
    int count
) {
    sqlite3_stmt *stmt = NULL;
    int32_t *blob;
    char *result = NULL;
    int rc;

    blob = (int32_t *)malloc(
        (size_t)count * sizeof(int32_t));
    if (blob == NULL) {
        return NULL;
    }
    for (int i = 0; i < count; ++i) {
        blob[i] = (int32_t)ids[i];
    }

    rc = sqlite3_prepare_v2(
        db,
        "SELECT llm_detokenize(?1)",
        -1,
        &stmt,
        NULL);
    if (rc != SQLITE_OK) {
        free(blob);
        return NULL;
    }
    sqlite3_bind_blob(
        stmt,
        1,
        blob,
        count * (int)sizeof(int32_t),
        SQLITE_STATIC);

    if (sqlite3_step(stmt) == SQLITE_ROW &&
        sqlite3_column_type(stmt, 0) == SQLITE_TEXT) {
        const char *text =
            (const char *)sqlite3_column_text(
                stmt, 0);
        if (text != NULL) {
            result = strdup(text);
        }
    }
    sqlite3_finalize(stmt);
    free(blob);
    return result;
}

/* ── main ──────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    const char *extension_path =
        getenv("LLM_SQL_EXTENSION_PATH");
    const char *tok_ext_env =
        getenv("LLM_TOKENIZER_EXTENSION_PATH");
    const char *thread_env = getenv("LLM_SQL_THREADS");
    const char *db_filename;
    const char *resolved_db;
    char *db_path = NULL;
    char *json_path = NULL;
    char *tok_ext_path = NULL;
    sqlite3 *tok_db = NULL;
    llmsql_generation generation;
    char error[512] = {0};
    int *prompt_ids = NULL;
    int prompt_len = 0;
    int max_tokens;
    int num_threads = 4;

    if (argc < 4 || argc > 5) {
        fprintf(
            stderr,
            "usage: %s <model_dir> <prompt> <max_tokens>"
            " [db_filename]\n",
            argv[0]);
        return 1;
    }

    db_filename = argc >= 5 ? argv[4] : NULL;
    max_tokens = atoi(argv[3]);
    if (thread_env != NULL && thread_env[0] != '\0') {
        num_threads = atoi(thread_env);
    }
    if (extension_path == NULL ||
        extension_path[0] == '\0') {
        extension_path = "./sqlite-llm/llm_ops.so";
    }

    /* Derive tokenizer extension path from llm_ops path
     * by replacing the filename. */
    if (tok_ext_env != NULL &&
        tok_ext_env[0] != '\0') {
        tok_ext_path = strdup(tok_ext_env);
    } else {
        const char *last_slash =
            strrchr(extension_path, '/');
        if (last_slash != NULL) {
            size_t dir_len =
                (size_t)(last_slash - extension_path + 1);
            tok_ext_path = (char *)malloc(
                dir_len + sizeof("llm_tokenizer.so"));
            if (tok_ext_path != NULL) {
                memcpy(tok_ext_path,
                       extension_path,
                       dir_len);
                memcpy(tok_ext_path + dir_len,
                       "llm_tokenizer.so",
                       sizeof("llm_tokenizer.so"));
            }
        } else {
            tok_ext_path = strdup("./llm_tokenizer.so");
        }
    }
    if (tok_ext_path == NULL) {
        fprintf(stderr,
                "failed to determine tokenizer "
                "extension path\n");
        return 1;
    }

    /* Resolve db and build paths */
    resolved_db = resolve_db(argv[1], db_filename);
    db_path = join_path(argv[1], resolved_db);
    json_path = join_path(argv[1], "tokenizer.json");
    if (db_path == NULL || json_path == NULL) {
        fprintf(stderr, "out of memory\n");
        free(db_path);
        free(json_path);
        free(tok_ext_path);
        return 1;
    }

    /* Open db + load extensions for tokenization */
    tok_db = open_model_db(
        db_path, extension_path, tok_ext_path);
    if (tok_db == NULL) {
        fprintf(stderr,
                "failed to open model db %s\n",
                db_path);
        free(db_path);
        free(json_path);
        free(tok_ext_path);
        return 1;
    }

    /* Encode prompt */
    if (!sql_encode(tok_db,
                    argv[2],
                    json_path,
                    &prompt_ids,
                    &prompt_len) ||
        prompt_len <= 0) {
        fprintf(stderr,
                "failed to encode prompt: %s\n",
                argv[2]);
        sqlite3_close(tok_db);
        free(db_path);
        free(json_path);
        free(tok_ext_path);
        return 1;
    }

    /* Run inference */
    if (!llmsql_native_generate_tokens(
            argv[1],
            db_filename,
            NULL,
            NULL,
            extension_path,
            num_threads,
            prompt_ids,
            prompt_len,
            max_tokens,
            &generation,
            error,
            sizeof(error))) {
        fprintf(
            stderr,
            "%s\n",
            error[0] != '\0'
                ? error
                : "native graph runtime failed");
        free(prompt_ids);
        sqlite3_close(tok_db);
        free(db_path);
        free(json_path);
        free(tok_ext_path);
        return 1;
    }

    /* Decode output */
    {
        char *output_text = sql_decode(
            tok_db,
            generation.token_ids,
            generation.token_count);
        if (output_text != NULL) {
            printf("%s\n", output_text);
            free(output_text);
        } else {
            /* fallback: print raw token ids */
            for (int i = 0;
                 i < generation.token_count;
                 ++i) {
                if (i > 0) {
                    printf(",");
                }
                printf("%d", generation.token_ids[i]);
            }
            printf("\n");
        }
    }

    llmsql_native_free_generation(&generation);
    free(prompt_ids);
    sqlite3_close(tok_db);
    free(db_path);
    free(json_path);
    free(tok_ext_path);
    return 0;
}
