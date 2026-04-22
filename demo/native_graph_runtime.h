#ifndef LLM_SQL_DEMO_NATIVE_GRAPH_RUNTIME_H
#define LLM_SQL_DEMO_NATIVE_GRAPH_RUNTIME_H

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int *token_ids;
    int token_count;
} llmsql_generation;

typedef struct {
    char name[128];
    double total_us;
} llmsql_profile_op_timing;

typedef struct {
    char mode[32];
    char db_filename[256];
    int prompt_tokens;
    int generated_tokens;
    double prefill_ms;
    int decode_steps;
    double decode_total_ms;
    double decode_avg_ms;
    double *decode_step_ms;
    int decode_step_count;
    double tokens_per_sec;
    double peak_rss_mb;
    double start_rss_mb;
    double after_prefill_rss_mb;
    double end_rss_mb;
    int sql_call_count;
    llmsql_profile_op_timing *op_timings;
    int op_timing_count;
    char memory_summary[512];
} llmsql_profile;

int llmsql_native_generate_tokens_profiled(
    const char *model_dir,
    const char *db_filename,
    const char *prefill_graph_path,
    const char *decode_graph_path,
    const char *extension_path,
    int num_threads,
    const int *prompt_ids,
    int prompt_len,
    int max_tokens,
    llmsql_generation *out,
    llmsql_profile *profile,
    char *error_buf,
    size_t error_buf_size
);

int llmsql_native_generate_tokens(
    const char *model_dir,
    const char *db_filename,
    const char *prefill_graph_path,
    const char *decode_graph_path,
    const char *extension_path,
    int num_threads,
    const int *prompt_ids,
    int prompt_len,
    int max_tokens,
    llmsql_generation *out,
    char *error_buf,
    size_t error_buf_size
);

void llmsql_native_free_generation(llmsql_generation *generation);
void llmsql_native_free_profile(llmsql_profile *profile);
void llmsql_native_print_profile(FILE *stream, const llmsql_profile *profile);

#ifdef __cplusplus
}
#endif

#endif