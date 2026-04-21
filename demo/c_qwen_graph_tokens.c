#include "native_graph_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_csv_ids(const char *csv, int **out_ids, int *out_count) {
    int *ids = NULL;
    int count = 0;
    const char *cursor = csv;
    while (*cursor != '\0') {
        char *end = NULL;
        int *next;
        long value = strtol(cursor, &end, 10);
        if (end == cursor) {
            free(ids);
            return 0;
        }
        next = (int *)realloc(ids, (size_t)(count + 1) * sizeof(int));
        if (next == NULL) {
            free(ids);
            return 0;
        }
        ids = next;
        ids[count++] = (int)value;
        cursor = end;
        if (*cursor == ',') {
            cursor += 1;
        }
    }
    *out_ids = ids;
    *out_count = count;
    return count > 0;
}

int main(int argc, char **argv) {
    const char *extension_path = getenv("LLM_SQL_EXTENSION_PATH");
    const char *thread_env = getenv("LLM_SQL_THREADS");
    const char *db_filename;
    llmsql_generation generation;
    char error[512] = {0};
    int *prompt_ids = NULL;
    int prompt_len = 0;
    int max_tokens;
    int num_threads = 4;

    if (argc < 4 || argc > 5) {
        fprintf(stderr, "usage: %s <model_dir> <token_csv> <max_tokens> [db_filename]\n", argv[0]);
        return 1;
    }

    if (!parse_csv_ids(argv[2], &prompt_ids, &prompt_len)) {
        fprintf(stderr, "failed to parse token csv: %s\n", argv[2]);
        return 1;
    }
    max_tokens = atoi(argv[3]);
    db_filename = argc >= 5 ? argv[4] : NULL;
    if (thread_env != NULL && thread_env[0] != '\0') {
        num_threads = atoi(thread_env);
    }
    if (extension_path == NULL || extension_path[0] == '\0') {
        extension_path = "./sqlite-llm/llm_ops.so";
    }

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
        fprintf(stderr, "%s\n", error[0] != '\0' ? error : "native graph runtime failed");
        free(prompt_ids);
        return 1;
    }

    for (int i = 0; i < generation.token_count; ++i) {
        if (i > 0) {
            printf(",");
        }
        printf("%d", generation.token_ids[i]);
    }
    printf("\n");

    llmsql_native_free_generation(&generation);
    free(prompt_ids);
    return 0;
}
