/*
 * C++ demo: run Qwen inference with string prompt input/output.
 *
 * Usage:
 *   cpp_qwen_graph_string <model_dir> <prompt> <max_tokens> [db]
 *
 * Tokenization is performed via the llm_tokenizer SQLite extension
 * (loaded alongside llm_ops), so no C tokenizer library is linked.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "native_graph_runtime.h"

#include <sqlite3.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

/* ── path helpers ──────────────────────────────────────────── */

std::string joinPath(
    const std::string &dir,
    const std::string &name
) {
    if (dir.empty()) {
        return name;
    }
    if (dir.back() == '/') {
        return dir + name;
    }
    return dir + "/" + name;
}

std::string resolveDb(
    const std::string &modelDir,
    const char *explicitDb
) {
    if (explicitDb != nullptr && explicitDb[0] != '\0') {
        return explicitDb;
    }
    struct stat st {};
    std::string p = joinPath(modelDir, "model.db");
    if (stat(p.c_str(), &st) == 0) {
        return "model.db";
    }
    p = joinPath(modelDir, "model_int8.db");
    if (stat(p.c_str(), &st) == 0) {
        return "model_int8.db";
    }
    return "model.db";
}

/* ── RAII wrappers ─────────────────────────────────────────── */

struct SqliteDeleter {
    void operator()(sqlite3 *db) const {
        sqlite3_close(db);
    }
};
using SqlitePtr =
    std::unique_ptr<sqlite3, SqliteDeleter>;

struct GenerationGuard {
    llmsql_generation gen{};
    ~GenerationGuard() {
        llmsql_native_free_generation(&gen);
    }
};

/* ── open db + load extensions ─────────────────────────────── */

SqlitePtr openModelDb(
    const std::string &dbPath,
    const std::string &extPath,
    const std::string &tokExtPath
) {
    sqlite3 *raw = nullptr;
    if (sqlite3_open_v2(
            dbPath.c_str(),
            &raw,
            SQLITE_OPEN_READONLY,
            nullptr) != SQLITE_OK) {
        if (raw != nullptr) {
            sqlite3_close(raw);
        }
        return nullptr;
    }
    SqlitePtr db(raw);
    sqlite3_enable_load_extension(db.get(), 1);

    char *err = nullptr;
    if (sqlite3_load_extension(
            db.get(),
            extPath.c_str(),
            "sqlite3_llm_ops_init",
            &err) != SQLITE_OK) {
        std::cerr << "load llm_ops: "
                  << (err ? err : "unknown") << "\n";
        sqlite3_free(err);
        return nullptr;
    }
    if (sqlite3_load_extension(
            db.get(),
            tokExtPath.c_str(),
            "sqlite3_llm_tokenizer_init",
            &err) != SQLITE_OK) {
        std::cerr << "load llm_tokenizer: "
                  << (err ? err : "unknown") << "\n";
        sqlite3_free(err);
        return nullptr;
    }
    return db;
}

/* ── tokenize/detokenize via SQL ───────────────────────────── */

bool sqlEncode(
    sqlite3 *db,
    const char *text,
    const std::string &jsonPath,
    std::vector<int> &out
) {
    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(
            db,
            "SELECT llm_tokenize(?1, ?2)",
            -1,
            &stmt,
            nullptr) != SQLITE_OK) {
        return false;
    }
    sqlite3_bind_text(
        stmt, 1, text, -1, SQLITE_STATIC);
    sqlite3_bind_text(
        stmt,
        2,
        jsonPath.c_str(),
        -1,
        SQLITE_STATIC);

    bool ok = false;
    if (sqlite3_step(stmt) == SQLITE_ROW &&
        sqlite3_column_type(stmt, 0) == SQLITE_BLOB) {
        int nbytes = sqlite3_column_bytes(stmt, 0);
        int count =
            nbytes / static_cast<int>(sizeof(int32_t));
        const auto *blob =
            static_cast<const int32_t *>(
                sqlite3_column_blob(stmt, 0));
        out.resize(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            out[static_cast<size_t>(i)] =
                static_cast<int>(blob[i]);
        }
        ok = true;
    }
    sqlite3_finalize(stmt);
    return ok;
}

std::string sqlDecode(
    sqlite3 *db,
    const int *ids,
    int count
) {
    std::vector<int32_t> blob(
        static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        blob[static_cast<size_t>(i)] =
            static_cast<int32_t>(ids[i]);
    }

    sqlite3_stmt *stmt = nullptr;
    if (sqlite3_prepare_v2(
            db,
            "SELECT llm_detokenize(?1)",
            -1,
            &stmt,
            nullptr) != SQLITE_OK) {
        return {};
    }
    sqlite3_bind_blob(
        stmt,
        1,
        blob.data(),
        count *
            static_cast<int>(sizeof(int32_t)),
        SQLITE_STATIC);

    std::string result;
    if (sqlite3_step(stmt) == SQLITE_ROW &&
        sqlite3_column_type(stmt, 0) == SQLITE_TEXT) {
        const char *text =
            reinterpret_cast<const char *>(
                sqlite3_column_text(stmt, 0));
        if (text != nullptr) {
            result = text;
        }
    }
    sqlite3_finalize(stmt);
    return result;
}

void printTokenIds(
    const int *ids,
    int count
) {
    std::cout << "token_ids: ";
    for (int i = 0; i < count; ++i) {
        if (i > 0) {
            std::cout << ',';
        }
        std::cout << ids[i];
    }
    std::cout << '\n';
}

} // namespace

/* ── main ──────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    std::vector<const char *> positionalArgs;
    bool profileEnabled = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--profile") {
            profileEnabled = true;
            continue;
        }
        positionalArgs.push_back(argv[i]);
    }

    if (positionalArgs.size() < 3 || positionalArgs.size() > 4) {
        std::cerr
            << "usage: " << argv[0]
            << " [--profile] <model_dir> <prompt> <max_tokens>"
               " [db_filename]\n";
        return 1;
    }

    const char *extensionPath =
        std::getenv("LLM_SQL_EXTENSION_PATH");
    const char *tokExtEnv =
        std::getenv("LLM_TOKENIZER_EXTENSION_PATH");
    const char *threadEnv =
        std::getenv("LLM_SQL_THREADS");
    const char *dbFilename =
        positionalArgs.size() >= 4 ? positionalArgs[3] : nullptr;
    int numThreads = 4;

    if (threadEnv != nullptr && threadEnv[0] != '\0') {
        numThreads = std::atoi(threadEnv);
    }
    if (extensionPath == nullptr ||
        extensionPath[0] == '\0') {
        extensionPath = "./sqlite-llm/llm_ops.so";
    }

    /* Derive tokenizer extension path */
    std::string tokExtPath;
    if (tokExtEnv != nullptr &&
        tokExtEnv[0] != '\0') {
        tokExtPath = tokExtEnv;
    } else {
        std::string ep(extensionPath);
        auto pos = ep.rfind('/');
        if (pos != std::string::npos) {
            tokExtPath =
                ep.substr(0, pos + 1) +
                "llm_tokenizer.so";
        } else {
            tokExtPath = "./llm_tokenizer.so";
        }
    }

    std::string modelDir = positionalArgs[0];
    std::string resolvedDb =
        resolveDb(modelDir, dbFilename);
    std::string dbPath =
        joinPath(modelDir, resolvedDb);
    std::string jsonPath =
        joinPath(modelDir, "tokenizer.json");

    /* Open db + load extensions */
    SqlitePtr tokDb = openModelDb(
        dbPath,
        extensionPath,
        tokExtPath);
    if (!tokDb) {
        std::cerr << "failed to open model db: "
                  << dbPath << "\n";
        return 1;
    }

    /* Encode prompt */
    std::vector<int> promptIds;
    if (!sqlEncode(tokDb.get(),
                   positionalArgs[1],
                   jsonPath,
                   promptIds) ||
        promptIds.empty()) {
        std::cerr << "failed to encode prompt: "
                  << positionalArgs[1] << "\n";
        return 1;
    }

    /* Run inference */
    GenerationGuard genGuard;
    llmsql_profile profile{};
    llmsql_profile *profilePtr = profileEnabled ? &profile : nullptr;
    char error[512] = {0};

    if (!llmsql_native_generate_tokens_profiled(
            positionalArgs[0],
            dbFilename,
            nullptr,
            nullptr,
            extensionPath,
            numThreads,
            promptIds.data(),
            static_cast<int>(promptIds.size()),
            std::atoi(positionalArgs[2]),
            &genGuard.gen,
            profilePtr,
            error,
            sizeof(error))) {
        std::cerr
            << (error[0] != '\0'
                    ? error
                    : "native graph runtime failed")
            << "\n";
        llmsql_native_free_profile(&profile);
        return 1;
    }

    /* Decode output */
    std::string output = sqlDecode(
        tokDb.get(),
        genGuard.gen.token_ids,
        genGuard.gen.token_count);
    printTokenIds(
        genGuard.gen.token_ids,
        genGuard.gen.token_count);
    std::cout << "decoded_text: " << output << "\n";

    if (profileEnabled) {
        llmsql_native_print_profile(stdout, &profile);
        llmsql_native_free_profile(&profile);
    }

    return 0;
}
