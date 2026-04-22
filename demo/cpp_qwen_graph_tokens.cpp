/*
 * @SPDX-License-Identifier: Apache-2.0
 * @Author: xuxianghong12
 * @Date: 2026-04-22 07:56:41
 * @LastEditTime: 2026-04-23 00:08:05
 * @LastEditors: xuxianghong12
 */
#include "native_graph_runtime.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

static bool parseCsvIds(const std::string &csv, std::vector<int> *out) {
    std::size_t start = 0;
    while (start < csv.size()) {
        std::size_t end = csv.find(',', start);
        std::string token = csv.substr(start, end == std::string::npos ? std::string::npos : end - start);
        char *parse_end = nullptr;
        long value = std::strtol(token.c_str(), &parse_end, 10);
        if (parse_end == token.c_str() || *parse_end != '\0') {
            return false;
        }
        out->push_back(static_cast<int>(value));
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return !out->empty();
}

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
        std::cerr << "usage: " << argv[0] << " [--profile] <model_dir> <token_csv> <max_tokens> [db_filename]\n";
        return 1;
    }

    std::vector<int> promptIds;
    if (!parseCsvIds(positionalArgs[1], &promptIds)) {
        std::cerr << "failed to parse token csv: " << positionalArgs[1] << "\n";
        return 1;
    }

    const char *extensionPath = std::getenv("LLM_SQL_EXTENSION_PATH");
    const char *threadEnv = std::getenv("LLM_SQL_THREADS");
    int numThreads = 1;
    if (threadEnv != nullptr && threadEnv[0] != '\0') {
        numThreads = std::atoi(threadEnv);
    }
    if (extensionPath == nullptr || extensionPath[0] == '\0') {
        extensionPath = "./sqlite-llm/llm_ops.so";
    }

    llmsql_generation generation{};
    llmsql_profile profile{};
    llmsql_profile *profilePtr = profileEnabled ? &profile : nullptr;
    char error[512] = {0};
    if (!llmsql_native_generate_tokens_profiled(
            positionalArgs[0],
            positionalArgs.size() >= 4 ? positionalArgs[3] : nullptr,
            nullptr,
            nullptr,
            extensionPath,
            numThreads,
            promptIds.data(),
            static_cast<int>(promptIds.size()),
            std::atoi(positionalArgs[2]),
            &generation,
            profilePtr,
            error,
            sizeof(error))) {
        std::cerr << (error[0] != '\0' ? error : "native graph runtime failed") << "\n";
        llmsql_native_free_profile(&profile);
        return 1;
    }

    for (int i = 0; i < generation.token_count; ++i) {
        if (i > 0) {
            std::cout << ',';
        }
        std::cout << generation.token_ids[i];
    }
    std::cout << '\n';

    if (profileEnabled) {
        llmsql_native_print_profile(stdout, &profile);
        llmsql_native_free_profile(&profile);
    }

    llmsql_native_free_generation(&generation);
    return 0;
}
