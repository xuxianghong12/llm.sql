/*
 * C++ demo: run Qwen inference with string prompt input/output.
 *
 * Usage:
 *   cpp_qwen_graph_string <model_dir> <prompt> <max_tokens> [db]
 *
 * Unlike cpp_qwen_graph_tokens (which requires comma-separated token
 * IDs), this demo accepts a plain-text prompt and prints the
 * decoded output string directly.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "bpe_tokenizer.h"
#include "native_graph_runtime.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

namespace {

struct TokenizerDeleter {
    void operator()(llmsql_tokenizer *t) const {
        llmsql_tokenizer_free(t);
    }
};

using TokenizerPtr =
    std::unique_ptr<llmsql_tokenizer, TokenizerDeleter>;

struct GenerationGuard {
    llmsql_generation gen{};
    ~GenerationGuard() {
        llmsql_native_free_generation(&gen);
    }
};

} // namespace

int main(int argc, char **argv) {
    if (argc < 4 || argc > 5) {
        std::cerr << "usage: " << argv[0]
                  << " <model_dir> <prompt> <max_tokens>"
                     " [db_filename]\n";
        return 1;
    }

    const char *extensionPath =
        std::getenv("LLM_SQL_EXTENSION_PATH");
    const char *threadEnv = std::getenv("LLM_SQL_THREADS");
    const char *dbFilename =
        argc >= 5 ? argv[4] : nullptr;
    int numThreads = 4;

    if (threadEnv != nullptr && threadEnv[0] != '\0') {
        numThreads = std::atoi(threadEnv);
    }
    if (extensionPath == nullptr || extensionPath[0] == '\0') {
        extensionPath = "./sqlite-llm/llm_ops.so";
    }

    /* ---- tokenizer: encode prompt ---- */
    TokenizerPtr tokenizer(
        llmsql_tokenizer_create(argv[1], dbFilename));
    if (!tokenizer) {
        std::cerr << "failed to create tokenizer from "
                  << argv[1] << "\n";
        return 1;
    }

    int *promptIds = nullptr;
    int promptLen = 0;
    if (!llmsql_tokenizer_encode(
            tokenizer.get(), argv[2], &promptIds,
            &promptLen) ||
        promptLen <= 0) {
        std::cerr << "failed to encode prompt: " << argv[2]
                  << "\n";
        return 1;
    }
    std::unique_ptr<int, decltype(&std::free)> promptGuard(
        promptIds, std::free);

    /* ---- run inference ---- */
    GenerationGuard genGuard;
    char error[512] = {0};

    if (!llmsql_native_generate_tokens(
            argv[1],
            dbFilename,
            nullptr,
            nullptr,
            extensionPath,
            numThreads,
            promptIds,
            promptLen,
            std::atoi(argv[3]),
            &genGuard.gen,
            error,
            sizeof(error))) {
        std::cerr << (error[0] != '\0'
                          ? error
                          : "native graph runtime failed")
                  << "\n";
        return 1;
    }

    /* ---- tokenizer: decode output ---- */
    char *outputText = nullptr;
    if (llmsql_tokenizer_decode(
            tokenizer.get(),
            genGuard.gen.token_ids,
            genGuard.gen.token_count,
            &outputText)) {
        std::cout << outputText << "\n";
        std::free(outputText);
    } else {
        /* fallback: print raw token ids */
        for (int i = 0; i < genGuard.gen.token_count; ++i) {
            if (i > 0) {
                std::cout << ',';
            }
            std::cout << genGuard.gen.token_ids[i];
        }
        std::cout << '\n';
    }

    return 0;
}
