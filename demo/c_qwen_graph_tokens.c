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
    const char *positionals[4] = {0};
    const char *db_filename;
    llmsql_generation generation;
    llmsql_profile profile;
    llmsql_profile *profile_ptr = NULL;
    char error[512] = {0};
    int *prompt_ids = NULL;
    int prompt_len = 0;
    int max_tokens;
    int num_threads = 1;
    int positional_count = 0;
    int profile_enabled = 0;

    memset(&generation, 0, sizeof(generation));
    memset(&profile, 0, sizeof(profile));
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--profile") == 0) {
            profile_enabled = 1;
            continue;
        }
        if (positional_count >= 4) {
            positional_count = 5;
            break;
        }
        positionals[positional_count++] = argv[i];
    }

    if (positional_count < 3 || positional_count > 4) {
        fprintf(stderr,
                "usage: %s [--profile] <model_dir> <token_csv> <max_tokens> "
                "[db_filename]\n",
                argv[0]);
        return 1;
    }

    if (!parse_csv_ids(positionals[1], &prompt_ids, &prompt_len)) {
        fprintf(stderr, "failed to parse token csv: %s\n", positionals[1]);
        return 1;
    }
    max_tokens = atoi(positionals[2]);
    db_filename = positional_count >= 4 ? positionals[3] : NULL;
    if (thread_env != NULL && thread_env[0] != '\0') {
        num_threads = atoi(thread_env);
    }
    if (extension_path == NULL || extension_path[0] == '\0') {
        extension_path = "./sqlite-llm/llm_ops.so";
    }
    if (profile_enabled) {
        profile_ptr = &profile;
    }

    if (!llmsql_native_generate_tokens_profiled(positionals[0],
                                                db_filename,
                                                NULL,
                                                NULL,
                                                extension_path,
                                                num_threads,
                                                prompt_ids,
                                                prompt_len,
                                                max_tokens,
                                                &generation,
                                                profile_ptr,
                                                error,
                                                sizeof(error))) {
        fprintf(stderr,
                "%s\n",
                error[0] != '\0' ? error : "native graph runtime failed");
        llmsql_native_free_profile(&profile);
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

    if (profile_enabled) {
        llmsql_native_print_profile(stdout, &profile);
        llmsql_native_free_profile(&profile);
    }

    llmsql_native_free_generation(&generation);
    free(prompt_ids);
    return 0;
}
