/*
 * test_llm_tokenizer.c — unit tests for the llm_tokenizer SQLite extension.
 *
 * Creates an in-memory database with a small vocab table and
 * tokenizer.json, then verifies:
 *   1. llm_tokenize()   – text → BLOB of int32 token IDs
 *   2. llm_detokenize()  – BLOB → text
 *   3. llm_token_count() – count of IDs in a BLOB
 *   4. llm_token_at()    – individual token ID access
 *   5. Round-trip: detokenize(tokenize(text)) == text
 *
 * Build:
 *   gcc -O2 -o test_llm_tokenizer test_llm_tokenizer.c -lsqlite3
 *
 * Run:
 *   ./test_llm_tokenizer
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sqlite3.h>

/* ── helpers ─────────────────────────────────────────────────── */

static int total_tests = 0;
static int passed_tests = 0;

#define CHECK(cond, msg)                                 \
    do {                                                 \
        ++total_tests;                                   \
        if (!(cond)) {                                   \
            fprintf(stderr,                              \
                    "FAIL [%s:%d]: %s\n",                \
                    __FILE__,                             \
                    __LINE__,                             \
                    (msg));                               \
        } else {                                         \
            ++passed_tests;                              \
        }                                                \
    } while (0)

/* Path to the tokenizer extension (.so / .dylib / .dll). */
static const char *ext_path(void) {
    const char *p = getenv("LLM_TOKENIZER_PATH");
    if (p != NULL && p[0] != '\0') {
        return p;
    }
    return "./llm_tokenizer";
}

/* Write a minimal tokenizer.json to a temp file and return
 * its path (static buffer). */
static const char *write_tokenizer_json(void) {
    static char path[] = "/tmp/test_llm_tok.json";
    FILE *f = fopen(path, "w");
    if (f == NULL) {
        return NULL;
    }
    fprintf(f,
            "{\n"
            "  \"model\": {\n"
            "    \"type\": \"BPE\",\n"
            "    \"vocab\": {\n"
            "      \"h\": 1, \"e\": 2, \"l\": 3,"
            " \"o\": 4,\n"
            "      \"he\": 5, \"ll\": 6,"
            " \"hello\": 7,\n"
            "      \"<|endoftext|>\": 0\n"
            "    },\n"
            "    \"merges\": [\n"
            "      \"h e\",\n"
            "      \"l l\",\n"
            "      \"he ll\",\n"
            "      \"hell o\"\n"
            "    ]\n"
            "  }\n"
            "}\n");
    fclose(f);
    return path;
}

/* Create an in-memory database with a small vocab table. */
static sqlite3 *create_test_db(void) {
    sqlite3 *db = NULL;
    int rc;

    rc = sqlite3_open(":memory:", &db);
    if (rc != SQLITE_OK) {
        fprintf(stderr,
                "sqlite3_open: %s\n",
                sqlite3_errmsg(db));
        return NULL;
    }

    /* Enable extension loading */
    sqlite3_db_config(
        db,
        SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION,
        1,
        NULL);

    /* Load the tokenizer extension */
    {
        char *err = NULL;
        rc = sqlite3_load_extension(
            db, ext_path(), NULL, &err);
        if (rc != SQLITE_OK) {
            fprintf(stderr,
                    "load_extension(%s): %s\n",
                    ext_path(),
                    err ? err : "unknown error");
            sqlite3_free(err);
            sqlite3_close(db);
            return NULL;
        }
    }

    /* Create vocab table */
    rc = sqlite3_exec(
        db,
        "CREATE TABLE vocab ("
        "  token_id INTEGER PRIMARY KEY,"
        "  token TEXT NOT NULL,"
        "  is_special INTEGER NOT NULL DEFAULT 0"
        ");"
        "INSERT INTO vocab VALUES(0,"
        " '<|endoftext|>', 1);"
        "INSERT INTO vocab VALUES(1, 'h', 0);"
        "INSERT INTO vocab VALUES(2, 'e', 0);"
        "INSERT INTO vocab VALUES(3, 'l', 0);"
        "INSERT INTO vocab VALUES(4, 'o', 0);"
        "INSERT INTO vocab VALUES(5, 'he', 0);"
        "INSERT INTO vocab VALUES(6, 'll', 0);"
        "INSERT INTO vocab VALUES(7, 'hello', 0);"
        "INSERT INTO vocab VALUES(8, 'Ġhello', 0);",
        NULL,
        NULL,
        NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr,
                "create vocab: %s\n",
                sqlite3_errmsg(db));
        sqlite3_close(db);
        return NULL;
    }

    return db;
}

/* ── test: llm_tokenize ──────────────────────────────────────── */

static void test_tokenize(sqlite3 *db) {
    const char *json_path = write_tokenizer_json();
    sqlite3_stmt *stmt = NULL;
    char sql[512];
    int rc;

    CHECK(json_path != NULL, "write tokenizer.json");

    snprintf(sql,
             sizeof(sql),
             "SELECT llm_tokenize('hello', '%s')",
             json_path);

    rc = sqlite3_prepare_v2(
        db, sql, -1, &stmt, NULL);
    CHECK(rc == SQLITE_OK, "prepare llm_tokenize");

    rc = sqlite3_step(stmt);
    CHECK(rc == SQLITE_ROW, "step llm_tokenize");

    /* Should be BLOB type */
    CHECK(sqlite3_column_type(stmt, 0) == SQLITE_BLOB,
          "result is BLOB");

    {
        int nbytes = sqlite3_column_bytes(stmt, 0);
        const int32_t *ids =
            (const int32_t *)sqlite3_column_blob(
                stmt, 0);
        int count = nbytes / (int)sizeof(int32_t);

        /* "hello" should merge to a single token 7 */
        CHECK(count == 1,
              "tokenize 'hello' -> 1 token");
        if (count >= 1) {
            CHECK(ids[0] == 7,
                  "tokenize 'hello' -> id 7");
        }
    }
    sqlite3_finalize(stmt);

    /* Test "he" -> token 5 */
    snprintf(sql,
             sizeof(sql),
             "SELECT llm_tokenize('he', '%s')",
             json_path);
    rc = sqlite3_prepare_v2(
        db, sql, -1, &stmt, NULL);
    CHECK(rc == SQLITE_OK, "prepare llm_tokenize 'he'");
    rc = sqlite3_step(stmt);
    CHECK(rc == SQLITE_ROW, "step llm_tokenize 'he'");
    {
        int nbytes = sqlite3_column_bytes(stmt, 0);
        const int32_t *ids =
            (const int32_t *)sqlite3_column_blob(
                stmt, 0);
        int count = nbytes / (int)sizeof(int32_t);
        CHECK(count == 1,
              "tokenize 'he' -> 1 token");
        if (count >= 1) {
            CHECK(ids[0] == 5,
                  "tokenize 'he' -> id 5");
        }
    }
    sqlite3_finalize(stmt);
}

/* ── test: llm_detokenize ────────────────────────────────────── */

static void test_detokenize(sqlite3 *db) {
    sqlite3_stmt *stmt = NULL;
    int rc;

    /* Construct a BLOB with token id 7 (= "hello") */
    /* X'07000000' is int32 little-endian 7 */
    rc = sqlite3_prepare_v2(
        db,
        "SELECT llm_detokenize(X'07000000')",
        -1,
        &stmt,
        NULL);
    CHECK(rc == SQLITE_OK,
          "prepare llm_detokenize");
    rc = sqlite3_step(stmt);
    CHECK(rc == SQLITE_ROW,
          "step llm_detokenize");
    CHECK(sqlite3_column_type(stmt, 0) == SQLITE_TEXT,
          "result is TEXT");
    {
        const char *text =
            (const char *)sqlite3_column_text(
                stmt, 0);
        CHECK(text != NULL &&
                  strcmp(text, "hello") == 0,
              "detokenize id=7 -> 'hello'");
    }
    sqlite3_finalize(stmt);

    /* Detokenize ids [5, 6, 4] = "he" + "ll" + "o" =
     * "hello" */
    /* 5=0x05000000, 6=0x06000000, 4=0x04000000 */
    rc = sqlite3_prepare_v2(
        db,
        "SELECT llm_detokenize("
        "X'050000000600000004000000')",
        -1,
        &stmt,
        NULL);
    CHECK(rc == SQLITE_OK,
          "prepare detokenize [5,6,4]");
    rc = sqlite3_step(stmt);
    CHECK(rc == SQLITE_ROW,
          "step detokenize [5,6,4]");
    {
        const char *text =
            (const char *)sqlite3_column_text(
                stmt, 0);
        CHECK(text != NULL &&
                  strcmp(text, "hello") == 0,
              "detokenize [5,6,4] -> 'hello'");
    }
    sqlite3_finalize(stmt);

    /* Byte-BPE token Ġhello should detokenize to leading-space text. */
    rc = sqlite3_prepare_v2(
        db,
        "SELECT llm_detokenize(X'08000000')",
        -1,
        &stmt,
        NULL);
    CHECK(rc == SQLITE_OK,
          "prepare detokenize byte-bpe");
    rc = sqlite3_step(stmt);
    CHECK(rc == SQLITE_ROW,
          "step detokenize byte-bpe");
    {
        const char *text =
            (const char *)sqlite3_column_text(
                stmt, 0);
        CHECK(text != NULL &&
                  strcmp(text, " hello") == 0,
              "detokenize Ġhello -> ' hello'");
    }
    sqlite3_finalize(stmt);

    /* Special tokens should be skipped */
    /* [0, 7] where 0 = <|endoftext|> (special) */
    rc = sqlite3_prepare_v2(
        db,
        "SELECT llm_detokenize("
        "X'0000000007000000')",
        -1,
        &stmt,
        NULL);
    CHECK(rc == SQLITE_OK,
          "prepare detokenize [0,7]");
    rc = sqlite3_step(stmt);
    CHECK(rc == SQLITE_ROW,
          "step detokenize [0,7]");
    {
        const char *text =
            (const char *)sqlite3_column_text(
                stmt, 0);
        CHECK(text != NULL &&
                  strcmp(text, "hello") == 0,
              "detokenize [0,7] -> 'hello' "
              "(special skipped)");
    }
    sqlite3_finalize(stmt);
}

/* ── test: llm_token_count ───────────────────────────────────── */

static void test_token_count(sqlite3 *db) {
    sqlite3_stmt *stmt = NULL;
    int rc;

    /* 3 int32 = 12 bytes → 3 tokens */
    rc = sqlite3_prepare_v2(
        db,
        "SELECT llm_token_count("
        "X'050000000600000004000000')",
        -1,
        &stmt,
        NULL);
    CHECK(rc == SQLITE_OK,
          "prepare llm_token_count");
    rc = sqlite3_step(stmt);
    CHECK(rc == SQLITE_ROW,
          "step llm_token_count");
    CHECK(sqlite3_column_int(stmt, 0) == 3,
          "token_count([5,6,4]) == 3");
    sqlite3_finalize(stmt);
}

/* ── test: llm_token_at ──────────────────────────────────────── */

static void test_token_at(sqlite3 *db) {
    sqlite3_stmt *stmt = NULL;
    int rc;

    /* Index 0 of [5, 6, 4] → 5 */
    rc = sqlite3_prepare_v2(
        db,
        "SELECT llm_token_at("
        "X'050000000600000004000000', 0)",
        -1,
        &stmt,
        NULL);
    CHECK(rc == SQLITE_OK,
          "prepare llm_token_at idx 0");
    rc = sqlite3_step(stmt);
    CHECK(rc == SQLITE_ROW,
          "step llm_token_at idx 0");
    CHECK(sqlite3_column_int(stmt, 0) == 5,
          "token_at([5,6,4], 0) == 5");
    sqlite3_finalize(stmt);

    /* Index 2 of [5, 6, 4] → 4 */
    rc = sqlite3_prepare_v2(
        db,
        "SELECT llm_token_at("
        "X'050000000600000004000000', 2)",
        -1,
        &stmt,
        NULL);
    CHECK(rc == SQLITE_OK,
          "prepare llm_token_at idx 2");
    rc = sqlite3_step(stmt);
    CHECK(rc == SQLITE_ROW,
          "step llm_token_at idx 2");
    CHECK(sqlite3_column_int(stmt, 0) == 4,
          "token_at([5,6,4], 2) == 4");
    sqlite3_finalize(stmt);

    /* Out-of-bounds → NULL */
    rc = sqlite3_prepare_v2(
        db,
        "SELECT llm_token_at("
        "X'050000000600000004000000', 10)",
        -1,
        &stmt,
        NULL);
    CHECK(rc == SQLITE_OK,
          "prepare llm_token_at out-of-bounds");
    rc = sqlite3_step(stmt);
    CHECK(rc == SQLITE_ROW,
          "step llm_token_at out-of-bounds");
    CHECK(sqlite3_column_type(stmt, 0) == SQLITE_NULL,
          "token_at out-of-bounds -> NULL");
    sqlite3_finalize(stmt);
}

/* ── test: round-trip ────────────────────────────────────────── */

static void test_round_trip(sqlite3 *db) {
    const char *json_path = write_tokenizer_json();
    sqlite3_stmt *stmt = NULL;
    char sql[512];
    int rc;

    snprintf(
        sql,
        sizeof(sql),
        "SELECT llm_detokenize("
        "  llm_tokenize('hello', '%s')"
        ")",
        json_path);

    rc = sqlite3_prepare_v2(
        db, sql, -1, &stmt, NULL);
    CHECK(rc == SQLITE_OK,
          "prepare round-trip");
    rc = sqlite3_step(stmt);
    CHECK(rc == SQLITE_ROW,
          "step round-trip");
    {
        const char *text =
            (const char *)sqlite3_column_text(
                stmt, 0);
        CHECK(text != NULL &&
                  strcmp(text, "hello") == 0,
              "round-trip 'hello' -> 'hello'");
    }
    sqlite3_finalize(stmt);
}

/* ── main ────────────────────────────────────────────────────── */

int main(void) {
    sqlite3 *db = create_test_db();
    if (db == NULL) {
        fprintf(stderr, "failed to create test db\n");
        return 1;
    }

    test_tokenize(db);
    test_detokenize(db);
    test_token_count(db);
    test_token_at(db);
    test_round_trip(db);

    sqlite3_close(db);

    printf("%d / %d tests passed\n",
           passed_tests,
           total_tests);

    return passed_tests == total_tests ? 0 : 1;
}
