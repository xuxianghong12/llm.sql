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
    if (argc < 4 || argc > 5) {
        std::cerr << "usage: " << argv[0] << " <model_dir> <token_csv> <max_tokens> [db_filename]\n";
        return 1;
    }

    std::vector<int> promptIds;
    if (!parseCsvIds(argv[2], &promptIds)) {
        std::cerr << "failed to parse token csv: " << argv[2] << "\n";
        return 1;
    }

    const char *extensionPath = std::getenv("LLM_SQL_EXTENSION_PATH");
    const char *threadEnv = std::getenv("LLM_SQL_THREADS");
    int numThreads = 4;
    if (threadEnv != nullptr && threadEnv[0] != '\0') {
        numThreads = std::atoi(threadEnv);
    }
    if (extensionPath == nullptr || extensionPath[0] == '\0') {
        extensionPath = "./sqlite-llm/llm_ops.so";
    }

    llmsql_generation generation{};
    char error[512] = {0};
    if (!llmsql_native_generate_tokens(
            argv[1],
            argc >= 5 ? argv[4] : nullptr,
            nullptr,
            nullptr,
            extensionPath,
            numThreads,
            promptIds.data(),
            static_cast<int>(promptIds.size()),
            std::atoi(argv[3]),
            &generation,
            error,
            sizeof(error))) {
        std::cerr << (error[0] != '\0' ? error : "native graph runtime failed") << "\n";
        return 1;
    }

    for (int i = 0; i < generation.token_count; ++i) {
        if (i > 0) {
            std::cout << ',';
        }
        std::cout << generation.token_ids[i];
    }
    std::cout << '\n';

    llmsql_native_free_generation(&generation);
    return 0;
}