/*
 * C demo: run Qwen inference with string prompt input/output.
 *
 * Usage:
 *   c_qwen_graph_string <model_dir> <prompt> <max_tokens> [db]
 *
 * Unlike c_qwen_graph_tokens (which requires comma-separated token
 * IDs), this demo accepts a plain-text prompt and prints the
 * decoded output string directly.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "bpe_tokenizer.h"
#include "native_graph_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *extension_path = getenv("LLM_SQL_EXTENSION_PATH");
    const char *thread_env = getenv("LLM_SQL_THREADS");
    const char *db_filename;
    llmsql_tokenizer *tokenizer = NULL;
    llmsql_generation generation;
    char error[512] = {0};
    int *prompt_ids = NULL;
    int prompt_len = 0;
    int max_tokens;
    int num_threads = 4;
    char *output_text = NULL;

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
    if (extension_path == NULL || extension_path[0] == '\0') {
        extension_path = "./sqlite-llm/llm_ops.so";
    }

    /* ---- tokenizer: encode prompt ---- */
    tokenizer = llmsql_tokenizer_create(argv[1], db_filename);
    if (tokenizer == NULL) {
        fprintf(stderr,
                "failed to create tokenizer from %s\n",
                argv[1]);
        return 1;
    }

    if (!llmsql_tokenizer_encode(
            tokenizer, argv[2], &prompt_ids, &prompt_len) ||
        prompt_len <= 0) {
        fprintf(stderr,
                "failed to encode prompt: %s\n", argv[2]);
        llmsql_tokenizer_free(tokenizer);
        return 1;
    }

    /* ---- run inference ---- */
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
            stderr, "%s\n",
            error[0] != '\0' ? error
                              : "native graph runtime failed");
        free(prompt_ids);
        llmsql_tokenizer_free(tokenizer);
        return 1;
    }

    /* ---- tokenizer: decode output ---- */
    if (llmsql_tokenizer_decode(
            tokenizer,
            generation.token_ids,
            generation.token_count,
            &output_text)) {
        printf("%s\n", output_text);
        free(output_text);
    } else {
        /* fallback: print raw token ids */
        for (int i = 0; i < generation.token_count; ++i) {
            if (i > 0) {
                printf(",");
            }
            printf("%d", generation.token_ids[i]);
        }
        printf("\n");
    }

    llmsql_native_free_generation(&generation);
    free(prompt_ids);
    llmsql_tokenizer_free(tokenizer);
    return 0;
}
