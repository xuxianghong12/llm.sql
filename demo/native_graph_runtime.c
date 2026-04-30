#define _POSIX_C_SOURCE 200809L

#include "native_graph_runtime.h"

#include <sqlite3.h>

#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define ND_MAGIC 0x80000000u

static char *llmsql_runtime_strdup(const char *src) {
    size_t len;
    char *copy;
    if (src == NULL) {
        return NULL;
    }
    len = strlen(src);
    copy = (char *)malloc(len + 1);
    if (copy == NULL) {
        return NULL;
    }
    memcpy(copy, src, len + 1);
    return copy;
}

typedef struct {
    void *data;
    int size;
} Blob;

typedef enum {
    VALUE_EMPTY = 0,
    VALUE_NULL = 1,
    VALUE_BLOB = 2,
    VALUE_INT = 3,
    VALUE_FLOAT = 4,
    VALUE_TEXT = 5,
} ValueType;

typedef struct {
    ValueType type;
    int owned;
    union {
        Blob blob;
        int64_t int_value;
        double float_value;
        char *text_value;
    } as;
} Value;

typedef enum {
    ARG_REF = 0,
    ARG_TEXT = 1,
    ARG_INT = 2,
    ARG_FLOAT = 3,
    ARG_BOOL = 4,
    ARG_NULL = 5,
} ArgKind;

typedef struct {
    ArgKind kind;
    int slot;
    int64_t int_value;
    double float_value;
    int bool_value;
    char *text_value;
} NativeArg;

typedef enum {
    INSTR_PLACEHOLDER = 0,
    INSTR_SQL = 1,
    INSTR_OUTPUT = 2,
} InstrKind;

typedef struct {
    InstrKind kind;
    int slot;
    int layer;
    char *name;
    char *target;
    char *sql;
    NativeArg *args;
    int arg_count;
    int *output_slots;
    int output_count;
    sqlite3_stmt *stmt;
} NativeInstr;

typedef struct {
    NativeInstr *instrs;
    int instr_count;
    int slot_count;
} NativeGraph;

typedef struct {
    Value *values;
    int count;
} GraphOutputs;

typedef struct {
    const char *cur;
    const char *end;
    char error[512];
} JsonParser;

typedef struct {
    llmsql_profile *profile;
} ExecProfileContext;

static void
set_error(char *error_buf, size_t error_buf_size, const char *fmt, ...);

static char *join_path(const char *dir, const char *file) {
    size_t len = strlen(dir) + strlen(file) + 2;
    char *path = (char *)malloc(len);
    if (path == NULL) {
        return NULL;
    }
    snprintf(path, len, "%s/%s", dir, file);
    return path;
}

static double monotonic_time_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0.0;
    }
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static double current_rss_mb(void) {
#if defined(__linux__)
    FILE *fp = fopen("/proc/self/statm", "r");
    unsigned long total_pages = 0;
    unsigned long resident_pages = 0;
    long page_size;
    double rss_mb = 0.0;
    if (fp == NULL) {
        return 0.0;
    }
    if (fscanf(fp, "%lu %lu", &total_pages, &resident_pages) == 2) {
        page_size = sysconf(_SC_PAGESIZE);
        if (page_size > 0) {
            rss_mb = ((double)resident_pages * (double)page_size) /
                     (1024.0 * 1024.0);
        }
    }
    fclose(fp);
    return rss_mb;
#else
    return 0.0;
#endif
}

static double peak_rss_mb(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0.0;
    }
#if defined(__APPLE__)
    return (double)usage.ru_maxrss / (1024.0 * 1024.0);
#else
    return (double)usage.ru_maxrss / 1024.0;
#endif
}

static void update_peak_rss(llmsql_profile *profile) {
    double peak_rss;
    if (profile == NULL) {
        return;
    }
    peak_rss = peak_rss_mb();
    if (peak_rss > profile->peak_rss_mb) {
        profile->peak_rss_mb = peak_rss;
    }
}

static void record_rss_snapshot(llmsql_profile *profile, double *slot) {
    if (profile == NULL) {
        return;
    }
    if (slot != NULL) {
        *slot = current_rss_mb();
    }
    update_peak_rss(profile);
}

static int compare_profile_op_timings_desc(const void *lhs, const void *rhs) {
    const llmsql_profile_op_timing *a = (const llmsql_profile_op_timing *)lhs;
    const llmsql_profile_op_timing *b = (const llmsql_profile_op_timing *)rhs;
    if (a->total_us < b->total_us) {
        return 1;
    }
    if (a->total_us > b->total_us) {
        return -1;
    }
    return strcmp(a->name, b->name);
}

static void sort_profile_op_timings(llmsql_profile *profile) {
    if (profile == NULL || profile->op_timing_count <= 1 ||
        profile->op_timings == NULL) {
        return;
    }
    qsort(profile->op_timings,
          (size_t)profile->op_timing_count,
          sizeof(profile->op_timings[0]),
          compare_profile_op_timings_desc);
}

static const char *profile_instr_name(const NativeInstr *instr) {
    if (instr == NULL) {
        return "sql";
    }
    if (instr->target != NULL && instr->target[0] != '\0') {
        return instr->target;
    }
    if (instr->name != NULL && instr->name[0] != '\0') {
        return instr->name;
    }
    return "sql";
}

static int profile_add_op_timing(llmsql_profile *profile,
                                 const NativeInstr *instr,
                                 double elapsed_us,
                                 char *error_buf,
                                 size_t error_buf_size) {
    const char *name;
    llmsql_profile_op_timing *next_entries;
    int i;
    if (profile == NULL) {
        return 1;
    }
    name = profile_instr_name(instr);
    for (i = 0; i < profile->op_timing_count; ++i) {
        if (strcmp(profile->op_timings[i].name, name) == 0) {
            profile->op_timings[i].total_us += elapsed_us;
            return 1;
        }
    }
    next_entries = (llmsql_profile_op_timing *)realloc(
        profile->op_timings,
        (size_t)(profile->op_timing_count + 1) * sizeof(*profile->op_timings));
    if (next_entries == NULL) {
        set_error(
            error_buf, error_buf_size, "out of memory recording op timings");
        return 0;
    }
    profile->op_timings = next_entries;
    memset(&profile->op_timings[profile->op_timing_count],
           0,
           sizeof(profile->op_timings[profile->op_timing_count]));
    snprintf(profile->op_timings[profile->op_timing_count].name,
             sizeof(profile->op_timings[profile->op_timing_count].name),
             "%s",
             name);
    profile->op_timings[profile->op_timing_count].total_us = elapsed_us;
    profile->op_timing_count += 1;
    return 1;
}

static void build_memory_summary(llmsql_profile *profile) {
    if (profile == NULL) {
        return;
    }
    snprintf(profile->memory_summary,
             sizeof(profile->memory_summary),
             "start RSS: %.1f MB\nafter prefill RSS: %.1f MB\nend RSS: %.1f "
             "MB\npeak RSS: %.1f MB",
             profile->start_rss_mb,
             profile->after_prefill_rss_mb,
             profile->end_rss_mb,
             profile->peak_rss_mb);
}

void llmsql_native_free_profile(llmsql_profile *profile) {
    if (profile == NULL) {
        return;
    }
    free(profile->decode_step_ms);
    free(profile->op_timings);
    memset(profile, 0, sizeof(*profile));
}

void llmsql_native_print_profile(FILE *stream, const llmsql_profile *profile) {
    FILE *out = stream != NULL ? stream : stdout;
    int i;
    int top_n;
    if (profile == NULL) {
        return;
    }
    fprintf(out,
            "\n============================================================\n");
    fprintf(out, "Profile\n");
    fprintf(out,
            "============================================================\n");
    fprintf(out,
            "Mode: %s\n",
            profile->mode[0] != '\0' ? profile->mode : "native_graph");
    fprintf(out,
            "Database: %s\n",
            profile->db_filename[0] != '\0' ? profile->db_filename
                                            : "model.db");
    fprintf(out, "Prompt tokens: %d\n", profile->prompt_tokens);
    fprintf(out, "Generated tokens: %d\n", profile->generated_tokens);
    fprintf(out, "Prefill latency: %.2f ms\n", profile->prefill_ms);
    fprintf(out, "Decode steps: %d\n", profile->decode_steps);
    fprintf(out, "Decode total: %.2f ms\n", profile->decode_total_ms);
    fprintf(out, "Decode avg/step: %.2f ms\n", profile->decode_avg_ms);
    fprintf(out, "Throughput: %.2f tokens/sec\n", profile->tokens_per_sec);
    fprintf(out, "Peak RSS: %.1f MB\n", profile->peak_rss_mb);
    // NOTE: the per-step latency details can be very verbose, so we omit them
    // by default. You can enable them by uncommenting the relevant code in
    // llmsql_native_print_profile and recompile if you want to see them. if
    // (profile->decode_step_count > 0 && profile->decode_step_ms != NULL) {
    //     fprintf(out, "Decode step latency (ms):\n");
    //     for (i = 0; i < profile->decode_step_count; ++i) {
    //         fprintf(out, "  step %-3d %.2f\n", i + 1,
    //         profile->decode_step_ms[i]);
    //     }
    // }
    if (profile->sql_call_count > 0) {
        fprintf(out, "SQL call count: %d\n", profile->sql_call_count);
    }
    if (profile->op_timing_count > 0 && profile->op_timings != NULL) {
        top_n = profile->op_timing_count < 5 ? profile->op_timing_count : 5;
        fprintf(out, "Top op timings (us):\n");
        for (i = 0; i < top_n; ++i) {
            fprintf(out,
                    "  %-30s %10.2f\n",
                    profile->op_timings[i].name,
                    profile->op_timings[i].total_us);
        }
    }
    if (profile->memory_summary[0] != '\0') {
        fprintf(out, "\nMemory summary:\n%s\n", profile->memory_summary);
    }
}

static char *read_text_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    char *buffer;
    long size;
    if (fp == NULL) {
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    size = ftell(fp);
    if (size < 0) {
        fclose(fp);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return NULL;
    }
    buffer = (char *)malloc((size_t)size + 1);
    if (buffer == NULL) {
        fclose(fp);
        return NULL;
    }
    if (fread(buffer, 1, (size_t)size, fp) != (size_t)size) {
        free(buffer);
        fclose(fp);
        return NULL;
    }
    buffer[size] = '\0';
    fclose(fp);
    return buffer;
}

static void
set_error(char *error_buf, size_t error_buf_size, const char *fmt, ...) {
    va_list args;
    if (error_buf == NULL || error_buf_size == 0) {
        return;
    }
    va_start(args, fmt);
    vsnprintf(error_buf, error_buf_size, fmt, args);
    va_end(args);
}

static int parser_fail(JsonParser *parser, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(parser->error, sizeof(parser->error), fmt, args);
    va_end(args);
    return 0;
}

static void parser_skip_ws(JsonParser *parser) {
    while (parser->cur < parser->end && isspace((unsigned char)*parser->cur)) {
        parser->cur += 1;
    }
}

static int parser_expect(JsonParser *parser, char expected) {
    parser_skip_ws(parser);
    if (parser->cur >= parser->end || *parser->cur != expected) {
        return parser_fail(parser, "expected '%c'", expected);
    }
    parser->cur += 1;
    return 1;
}

static int parser_match(JsonParser *parser, char expected) {
    parser_skip_ws(parser);
    if (parser->cur < parser->end && *parser->cur == expected) {
        parser->cur += 1;
        return 1;
    }
    return 0;
}

static int hex_value(int ch) {
    if (ch >= '0' && ch <= '9') {
        return ch - '0';
    }
    if (ch >= 'a' && ch <= 'f') {
        return 10 + ch - 'a';
    }
    if (ch >= 'A' && ch <= 'F') {
        return 10 + ch - 'A';
    }
    return -1;
}

static int parser_parse_string(JsonParser *parser, char **out) {
    size_t cap = 32;
    size_t len = 0;
    char *buffer;
    parser_skip_ws(parser);
    if (parser->cur >= parser->end || *parser->cur != '"') {
        return parser_fail(parser, "expected string");
    }
    parser->cur += 1;
    buffer = (char *)malloc(cap);
    if (buffer == NULL) {
        return parser_fail(parser, "out of memory parsing string");
    }
    while (parser->cur < parser->end) {
        char ch = *parser->cur++;
        if (ch == '"') {
            buffer[len] = '\0';
            *out = buffer;
            return 1;
        }
        if (ch == '\\') {
            int code;
            if (parser->cur >= parser->end) {
                free(buffer);
                return parser_fail(parser, "unterminated escape sequence");
            }
            ch = *parser->cur++;
            switch (ch) {
            case '"':
            case '\\':
            case '/':
                break;
            case 'b':
                ch = '\b';
                break;
            case 'f':
                ch = '\f';
                break;
            case 'n':
                ch = '\n';
                break;
            case 'r':
                ch = '\r';
                break;
            case 't':
                ch = '\t';
                break;
            case 'u':
                if (parser->end - parser->cur < 4) {
                    free(buffer);
                    return parser_fail(parser, "short unicode escape");
                }
                code = 0;
                for (int i = 0; i < 4; ++i) {
                    int digit = hex_value((unsigned char)parser->cur[i]);
                    if (digit < 0) {
                        free(buffer);
                        return parser_fail(parser, "invalid unicode escape");
                    }
                    code = (code << 4) | digit;
                }
                parser->cur += 4;
                ch = (char)(code >= 0 && code <= 0x7f ? code : '?');
                break;
            default:
                free(buffer);
                return parser_fail(parser, "unsupported escape sequence");
            }
        }
        if (len + 2 > cap) {
            char *next;
            cap *= 2;
            next = (char *)realloc(buffer, cap);
            if (next == NULL) {
                free(buffer);
                return parser_fail(parser, "out of memory growing string");
            }
            buffer = next;
        }
        buffer[len++] = ch;
    }
    free(buffer);
    return parser_fail(parser, "unterminated string");
}

static int parser_parse_int64(JsonParser *parser, int64_t *out) {
    char *end_ptr;
    long long value;
    parser_skip_ws(parser);
    errno = 0;
    value = strtoll(parser->cur, &end_ptr, 10);
    if (end_ptr == parser->cur || errno != 0) {
        return parser_fail(parser, "expected integer");
    }
    parser->cur = end_ptr;
    *out = (int64_t)value;
    return 1;
}

static int parser_parse_double(JsonParser *parser, double *out) {
    char *end_ptr;
    double value;
    parser_skip_ws(parser);
    errno = 0;
    value = strtod(parser->cur, &end_ptr);
    if (end_ptr == parser->cur || errno != 0) {
        return parser_fail(parser, "expected number");
    }
    parser->cur = end_ptr;
    *out = value;
    return 1;
}

static int parser_parse_literal(JsonParser *parser, const char *literal) {
    size_t len = strlen(literal);
    parser_skip_ws(parser);
    if ((size_t)(parser->end - parser->cur) < len) {
        return 0;
    }
    if (strncmp(parser->cur, literal, len) != 0) {
        return 0;
    }
    parser->cur += len;
    return 1;
}

static int parser_skip_value(JsonParser *parser);

static int parser_skip_array(JsonParser *parser) {
    if (!parser_expect(parser, '[')) {
        return 0;
    }
    parser_skip_ws(parser);
    if (parser_match(parser, ']')) {
        return 1;
    }
    while (1) {
        if (!parser_skip_value(parser)) {
            return 0;
        }
        parser_skip_ws(parser);
        if (parser_match(parser, ']')) {
            return 1;
        }
        if (!parser_expect(parser, ',')) {
            return 0;
        }
    }
}

static int parser_skip_object(JsonParser *parser) {
    char *key = NULL;
    if (!parser_expect(parser, '{')) {
        return 0;
    }
    parser_skip_ws(parser);
    if (parser_match(parser, '}')) {
        return 1;
    }
    while (1) {
        if (!parser_parse_string(parser, &key)) {
            return 0;
        }
        free(key);
        key = NULL;
        if (!parser_expect(parser, ':')) {
            return 0;
        }
        if (!parser_skip_value(parser)) {
            return 0;
        }
        parser_skip_ws(parser);
        if (parser_match(parser, '}')) {
            return 1;
        }
        if (!parser_expect(parser, ',')) {
            return 0;
        }
    }
}

static int parser_skip_value(JsonParser *parser) {
    parser_skip_ws(parser);
    if (parser->cur >= parser->end) {
        return parser_fail(parser, "unexpected end of json");
    }
    switch (*parser->cur) {
    case '{':
        return parser_skip_object(parser);
    case '[':
        return parser_skip_array(parser);
    case '"': {
        char *tmp = NULL;
        int ok = parser_parse_string(parser, &tmp);
        free(tmp);
        return ok;
    }
    case 't':
        return parser_parse_literal(parser, "true")
                   ? 1
                   : parser_fail(parser, "invalid literal");
    case 'f':
        return parser_parse_literal(parser, "false")
                   ? 1
                   : parser_fail(parser, "invalid literal");
    case 'n':
        return parser_parse_literal(parser, "null")
                   ? 1
                   : parser_fail(parser, "invalid literal");
    default: {
        double ignored;
        return parser_parse_double(parser, &ignored);
    }
    }
}

static void free_native_arg(NativeArg *arg) {
    if (arg->kind == ARG_TEXT && arg->text_value != NULL) {
        free(arg->text_value);
        arg->text_value = NULL;
    }
}

static void free_native_instr(NativeInstr *instr) {
    int i;
    if (instr->stmt != NULL) {
        sqlite3_finalize(instr->stmt);
        instr->stmt = NULL;
    }
    free(instr->name);
    free(instr->target);
    free(instr->sql);
    for (i = 0; i < instr->arg_count; ++i) {
        free_native_arg(&instr->args[i]);
    }
    free(instr->args);
    free(instr->output_slots);
    memset(instr, 0, sizeof(*instr));
}

static void free_native_graph(NativeGraph *graph) {
    int i;
    for (i = 0; i < graph->instr_count; ++i) {
        free_native_instr(&graph->instrs[i]);
    }
    free(graph->instrs);
    memset(graph, 0, sizeof(*graph));
}

static int parser_parse_arg(JsonParser *parser, NativeArg *arg) {
    char *key = NULL;
    char *kind = NULL;
    memset(arg, 0, sizeof(*arg));
    if (!parser_expect(parser, '{')) {
        return 0;
    }
    parser_skip_ws(parser);
    if (parser_match(parser, '}')) {
        return parser_fail(parser, "empty arg object");
    }
    while (1) {
        if (!parser_parse_string(parser, &key)) {
            return 0;
        }
        if (!parser_expect(parser, ':')) {
            free(key);
            return 0;
        }
        if (strcmp(key, "kind") == 0) {
            if (!parser_parse_string(parser, &kind)) {
                free(key);
                return 0;
            }
        } else if (strcmp(key, "slot") == 0) {
            int64_t slot_value;
            if (!parser_parse_int64(parser, &slot_value)) {
                free(key);
                free(kind);
                return 0;
            }
            arg->slot = (int)slot_value;
        } else if (strcmp(key, "value") == 0) {
            if (kind == NULL) {
                free(key);
                free(kind);
                return parser_fail(parser, "arg kind must appear before value");
            }
            if (strcmp(kind, "text") == 0) {
                if (!parser_parse_string(parser, &arg->text_value)) {
                    free(key);
                    free(kind);
                    return 0;
                }
            } else if (strcmp(kind, "int") == 0) {
                if (!parser_parse_int64(parser, &arg->int_value)) {
                    free(key);
                    free(kind);
                    return 0;
                }
            } else if (strcmp(kind, "float") == 0) {
                if (!parser_parse_double(parser, &arg->float_value)) {
                    free(key);
                    free(kind);
                    return 0;
                }
            } else if (strcmp(kind, "bool") == 0) {
                if (parser_parse_literal(parser, "true")) {
                    arg->bool_value = 1;
                } else if (parser_parse_literal(parser, "false")) {
                    arg->bool_value = 0;
                } else {
                    free(key);
                    free(kind);
                    return parser_fail(parser, "expected boolean value");
                }
            } else if (strcmp(kind, "ref") == 0) {
                int64_t ref_slot;
                if (!parser_parse_int64(parser, &ref_slot)) {
                    free(key);
                    free(kind);
                    return 0;
                }
                arg->slot = (int)ref_slot;
            } else if (strcmp(kind, "null") == 0) {
                if (!parser_parse_literal(parser, "null")) {
                    free(key);
                    free(kind);
                    return parser_fail(parser, "expected null value");
                }
            } else {
                free(key);
                free(kind);
                return parser_fail(parser, "unsupported arg kind: %s", kind);
            }
        } else {
            if (!parser_skip_value(parser)) {
                free(key);
                free(kind);
                return 0;
            }
        }
        free(key);
        key = NULL;
        parser_skip_ws(parser);
        if (parser_match(parser, '}')) {
            break;
        }
        if (!parser_expect(parser, ',')) {
            free(kind);
            return 0;
        }
    }
    if (kind == NULL) {
        return parser_fail(parser, "arg missing kind");
    }
    if (strcmp(kind, "ref") == 0) {
        arg->kind = ARG_REF;
    } else if (strcmp(kind, "text") == 0) {
        arg->kind = ARG_TEXT;
    } else if (strcmp(kind, "int") == 0) {
        arg->kind = ARG_INT;
    } else if (strcmp(kind, "float") == 0) {
        arg->kind = ARG_FLOAT;
    } else if (strcmp(kind, "bool") == 0) {
        arg->kind = ARG_BOOL;
    } else if (strcmp(kind, "null") == 0) {
        arg->kind = ARG_NULL;
    } else {
        free(kind);
        return parser_fail(parser, "unsupported arg kind: %s", kind);
    }
    free(kind);
    return 1;
}

static int
parser_parse_args(JsonParser *parser, NativeArg **out_args, int *out_count) {
    NativeArg *args = NULL;
    int count = 0;
    if (!parser_expect(parser, '[')) {
        return 0;
    }
    parser_skip_ws(parser);
    if (parser_match(parser, ']')) {
        *out_args = NULL;
        *out_count = 0;
        return 1;
    }
    while (1) {
        NativeArg arg;
        NativeArg *next;
        if (!parser_parse_arg(parser, &arg)) {
            for (int i = 0; i < count; ++i) {
                free_native_arg(&args[i]);
            }
            free(args);
            return 0;
        }
        next =
            (NativeArg *)realloc(args, (size_t)(count + 1) * sizeof(NativeArg));
        if (next == NULL) {
            free_native_arg(&arg);
            for (int i = 0; i < count; ++i) {
                free_native_arg(&args[i]);
            }
            free(args);
            return parser_fail(parser, "out of memory growing args array");
        }
        args = next;
        args[count++] = arg;
        parser_skip_ws(parser);
        if (parser_match(parser, ']')) {
            break;
        }
        if (!parser_expect(parser, ',')) {
            for (int i = 0; i < count; ++i) {
                free_native_arg(&args[i]);
            }
            free(args);
            return 0;
        }
    }
    *out_args = args;
    *out_count = count;
    return 1;
}

static int
parser_parse_int_array(JsonParser *parser, int **out_values, int *out_count) {
    int *values = NULL;
    int count = 0;
    if (!parser_expect(parser, '[')) {
        return 0;
    }
    parser_skip_ws(parser);
    if (parser_match(parser, ']')) {
        *out_values = NULL;
        *out_count = 0;
        return 1;
    }
    while (1) {
        int64_t value;
        int *next;
        if (!parser_parse_int64(parser, &value)) {
            free(values);
            return 0;
        }
        next = (int *)realloc(values, (size_t)(count + 1) * sizeof(int));
        if (next == NULL) {
            free(values);
            return parser_fail(parser, "out of memory growing int array");
        }
        values = next;
        values[count++] = (int)value;
        parser_skip_ws(parser);
        if (parser_match(parser, ']')) {
            break;
        }
        if (!parser_expect(parser, ',')) {
            free(values);
            return 0;
        }
    }
    *out_values = values;
    *out_count = count;
    return 1;
}

static int parser_parse_instr(JsonParser *parser, NativeInstr *instr) {
    char *key = NULL;
    char *kind = NULL;
    memset(instr, 0, sizeof(*instr));
    if (!parser_expect(parser, '{')) {
        return 0;
    }
    parser_skip_ws(parser);
    if (parser_match(parser, '}')) {
        return parser_fail(parser, "empty instruction object");
    }
    while (1) {
        if (!parser_parse_string(parser, &key)) {
            return 0;
        }
        if (!parser_expect(parser, ':')) {
            free(key);
            return 0;
        }
        if (strcmp(key, "kind") == 0) {
            if (!parser_parse_string(parser, &kind)) {
                free(key);
                return 0;
            }
        } else if (strcmp(key, "slot") == 0) {
            int64_t slot_value;
            if (!parser_parse_int64(parser, &slot_value)) {
                free(key);
                free(kind);
                return 0;
            }
            instr->slot = (int)slot_value;
        } else if (strcmp(key, "layer") == 0) {
            int64_t layer_value;
            if (!parser_parse_int64(parser, &layer_value)) {
                free(key);
                free(kind);
                return 0;
            }
            instr->layer = (int)layer_value;
        } else if (strcmp(key, "name") == 0) {
            if (!parser_parse_string(parser, &instr->name)) {
                free(key);
                free(kind);
                return 0;
            }
        } else if (strcmp(key, "target") == 0) {
            if (!parser_parse_string(parser, &instr->target)) {
                free(key);
                free(kind);
                return 0;
            }
        } else if (strcmp(key, "sql") == 0) {
            if (!parser_parse_string(parser, &instr->sql)) {
                free(key);
                free(kind);
                return 0;
            }
        } else if (strcmp(key, "args") == 0) {
            if (!parser_parse_args(parser, &instr->args, &instr->arg_count)) {
                free(key);
                free(kind);
                return 0;
            }
        } else if (strcmp(key, "slots") == 0) {
            if (!parser_parse_int_array(
                    parser, &instr->output_slots, &instr->output_count)) {
                free(key);
                free(kind);
                return 0;
            }
        } else {
            if (!parser_skip_value(parser)) {
                free(key);
                free(kind);
                return 0;
            }
        }
        free(key);
        key = NULL;
        parser_skip_ws(parser);
        if (parser_match(parser, '}')) {
            break;
        }
        if (!parser_expect(parser, ',')) {
            free(kind);
            return 0;
        }
    }
    if (kind == NULL) {
        return parser_fail(parser, "instruction missing kind");
    }
    if (strcmp(kind, "placeholder") == 0) {
        instr->kind = INSTR_PLACEHOLDER;
    } else if (strcmp(kind, "sql") == 0) {
        instr->kind = INSTR_SQL;
    } else if (strcmp(kind, "output") == 0) {
        instr->kind = INSTR_OUTPUT;
    } else {
        free(kind);
        return parser_fail(parser, "unsupported instruction kind: %s", kind);
    }
    free(kind);
    return 1;
}

static int parser_parse_instrs(JsonParser *parser, NativeGraph *graph) {
    NativeInstr *instrs = NULL;
    int count = 0;
    if (!parser_expect(parser, '[')) {
        return 0;
    }
    parser_skip_ws(parser);
    if (parser_match(parser, ']')) {
        graph->instrs = NULL;
        graph->instr_count = 0;
        return 1;
    }
    while (1) {
        NativeInstr instr;
        NativeInstr *next;
        if (!parser_parse_instr(parser, &instr)) {
            for (int i = 0; i < count; ++i) {
                free_native_instr(&instrs[i]);
            }
            free(instrs);
            return 0;
        }
        next = (NativeInstr *)realloc(
            instrs, (size_t)(count + 1) * sizeof(NativeInstr));
        if (next == NULL) {
            free_native_instr(&instr);
            for (int i = 0; i < count; ++i) {
                free_native_instr(&instrs[i]);
            }
            free(instrs);
            return parser_fail(parser,
                               "out of memory growing instruction array");
        }
        instrs = next;
        instrs[count++] = instr;
        parser_skip_ws(parser);
        if (parser_match(parser, ']')) {
            break;
        }
        if (!parser_expect(parser, ',')) {
            for (int i = 0; i < count; ++i) {
                free_native_instr(&instrs[i]);
            }
            free(instrs);
            return 0;
        }
    }
    graph->instrs = instrs;
    graph->instr_count = count;
    return 1;
}

static int load_native_graph(const char *path,
                             NativeGraph *graph,
                             char *error_buf,
                             size_t error_buf_size) {
    char *text = read_text_file(path);
    JsonParser parser;
    char *key = NULL;
    memset(graph, 0, sizeof(*graph));
    if (text == NULL) {
        set_error(
            error_buf, error_buf_size, "failed to read graph file: %s", path);
        return 0;
    }
    parser.cur = text;
    parser.end = text + strlen(text);
    parser.error[0] = '\0';
    if (!parser_expect(&parser, '{')) {
        set_error(error_buf, error_buf_size, "%s", parser.error);
        free(text);
        return 0;
    }
    while (1) {
        parser_skip_ws(&parser);
        if (parser_match(&parser, '}')) {
            break;
        }
        if (!parser_parse_string(&parser, &key)) {
            set_error(error_buf, error_buf_size, "%s", parser.error);
            free(text);
            return 0;
        }
        if (!parser_expect(&parser, ':')) {
            set_error(error_buf, error_buf_size, "%s", parser.error);
            free(key);
            free(text);
            return 0;
        }
        if (strcmp(key, "format") == 0 || strcmp(key, "graph") == 0) {
            char *ignored = NULL;
            if (!parser_parse_string(&parser, &ignored)) {
                set_error(error_buf, error_buf_size, "%s", parser.error);
                free(key);
                free(text);
                return 0;
            }
            free(ignored);
        } else if (strcmp(key, "slot_count") == 0) {
            int64_t slot_count;
            if (!parser_parse_int64(&parser, &slot_count)) {
                set_error(error_buf, error_buf_size, "%s", parser.error);
                free(key);
                free(text);
                return 0;
            }
            graph->slot_count = (int)slot_count;
        } else if (strcmp(key, "instructions") == 0) {
            if (!parser_parse_instrs(&parser, graph)) {
                set_error(error_buf, error_buf_size, "%s", parser.error);
                free(key);
                free(text);
                return 0;
            }
        } else {
            if (!parser_skip_value(&parser)) {
                set_error(error_buf, error_buf_size, "%s", parser.error);
                free(key);
                free(text);
                return 0;
            }
        }
        free(key);
        key = NULL;
        parser_skip_ws(&parser);
        if (parser_match(&parser, '}')) {
            break;
        }
        if (!parser_expect(&parser, ',')) {
            set_error(error_buf, error_buf_size, "%s", parser.error);
            free(text);
            return 0;
        }
    }
    free(text);
    if (graph->slot_count <= 0 || graph->instr_count <= 0) {
        free_native_graph(graph);
        set_error(
            error_buf, error_buf_size, "graph %s is empty or invalid", path);
        return 0;
    }
    return 1;
}

static void value_reset(Value *value) {
    if (value->owned) {
        if (value->type == VALUE_BLOB && value->as.blob.data != NULL) {
            free(value->as.blob.data);
        } else if (value->type == VALUE_TEXT && value->as.text_value != NULL) {
            free(value->as.text_value);
        }
    }
    memset(value, 0, sizeof(*value));
}

static int value_clone(const Value *src,
                       Value *dst,
                       char *error_buf,
                       size_t error_buf_size) {
    memset(dst, 0, sizeof(*dst));
    dst->type = src->type;
    switch (src->type) {
    case VALUE_EMPTY:
    case VALUE_NULL:
        return 1;
    case VALUE_INT:
        dst->as.int_value = src->as.int_value;
        return 1;
    case VALUE_FLOAT:
        dst->as.float_value = src->as.float_value;
        return 1;
    case VALUE_TEXT: {
        size_t len = strlen(src->as.text_value);
        dst->as.text_value = (char *)malloc(len + 1);
        if (dst->as.text_value == NULL) {
            set_error(
                error_buf, error_buf_size, "out of memory cloning text result");
            return 0;
        }
        memcpy(dst->as.text_value, src->as.text_value, len + 1);
        dst->owned = 1;
        return 1;
    }
    case VALUE_BLOB:
        dst->as.blob.data = malloc((size_t)src->as.blob.size);
        if (dst->as.blob.data == NULL) {
            set_error(
                error_buf, error_buf_size, "out of memory cloning blob result");
            return 0;
        }
        memcpy(dst->as.blob.data, src->as.blob.data, (size_t)src->as.blob.size);
        dst->as.blob.size = src->as.blob.size;
        dst->owned = 1;
        return 1;
    }
    set_error(
        error_buf, error_buf_size, "unsupported value type %d", (int)src->type);
    return 0;
}

static int bind_value(sqlite3_stmt *stmt,
                      int index,
                      const Value *value,
                      char *error_buf,
                      size_t error_buf_size) {
    int rc;
    switch (value->type) {
    case VALUE_EMPTY:
    case VALUE_NULL:
        rc = sqlite3_bind_null(stmt, index);
        break;
    case VALUE_BLOB:
        rc = sqlite3_bind_blob(stmt,
                               index,
                               value->as.blob.data,
                               value->as.blob.size,
                               SQLITE_STATIC);
        break;
    case VALUE_INT:
        rc = sqlite3_bind_int64(stmt, index, value->as.int_value);
        break;
    case VALUE_FLOAT:
        rc = sqlite3_bind_double(stmt, index, value->as.float_value);
        break;
    case VALUE_TEXT:
        rc = sqlite3_bind_text(
            stmt, index, value->as.text_value, -1, SQLITE_STATIC);
        break;
    default:
        set_error(error_buf,
                  error_buf_size,
                  "unsupported bind value type %d",
                  (int)value->type);
        return 0;
    }
    if (rc != SQLITE_OK) {
        set_error(error_buf,
                  error_buf_size,
                  "sqlite bind failed: %s",
                  sqlite3_errstr(rc));
        return 0;
    }
    return 1;
}

static int bind_arg(sqlite3_stmt *stmt,
                    int index,
                    const NativeArg *arg,
                    const Value *env,
                    char *error_buf,
                    size_t error_buf_size) {
    Value temp;
    memset(&temp, 0, sizeof(temp));
    switch (arg->kind) {
    case ARG_REF:
        return bind_value(
            stmt, index, &env[arg->slot], error_buf, error_buf_size);
    case ARG_TEXT:
        temp.type = VALUE_TEXT;
        temp.as.text_value = arg->text_value;
        return bind_value(stmt, index, &temp, error_buf, error_buf_size);
    case ARG_INT:
        temp.type = VALUE_INT;
        temp.as.int_value = arg->int_value;
        return bind_value(stmt, index, &temp, error_buf, error_buf_size);
    case ARG_FLOAT:
        temp.type = VALUE_FLOAT;
        temp.as.float_value = arg->float_value;
        return bind_value(stmt, index, &temp, error_buf, error_buf_size);
    case ARG_BOOL:
        temp.type = VALUE_INT;
        temp.as.int_value = arg->bool_value ? 1 : 0;
        return bind_value(stmt, index, &temp, error_buf, error_buf_size);
    case ARG_NULL:
        temp.type = VALUE_NULL;
        return bind_value(stmt, index, &temp, error_buf, error_buf_size);
    }
    set_error(
        error_buf, error_buf_size, "unsupported arg kind %d", (int)arg->kind);
    return 0;
}

static int extract_value(sqlite3_stmt *stmt,
                         int column,
                         Value *out,
                         char *error_buf,
                         size_t error_buf_size) {
    int type = sqlite3_column_type(stmt, column);
    memset(out, 0, sizeof(*out));
    switch (type) {
    case SQLITE_NULL:
        out->type = VALUE_NULL;
        return 1;
    case SQLITE_INTEGER:
        out->type = VALUE_INT;
        out->as.int_value = sqlite3_column_int64(stmt, column);
        return 1;
    case SQLITE_FLOAT:
        out->type = VALUE_FLOAT;
        out->as.float_value = sqlite3_column_double(stmt, column);
        return 1;
    case SQLITE_TEXT: {
        const unsigned char *text = sqlite3_column_text(stmt, column);
        size_t len = strlen((const char *)text);
        out->as.text_value = (char *)malloc(len + 1);
        if (out->as.text_value == NULL) {
            set_error(
                error_buf, error_buf_size, "out of memory copying text result");
            return 0;
        }
        memcpy(out->as.text_value, text, len + 1);
        out->type = VALUE_TEXT;
        out->owned = 1;
        return 1;
    }
    case SQLITE_BLOB: {
        const void *blob = sqlite3_column_blob(stmt, column);
        int size = sqlite3_column_bytes(stmt, column);
        out->as.blob.data = malloc((size_t)size);
        if (out->as.blob.data == NULL) {
            set_error(
                error_buf, error_buf_size, "out of memory copying blob result");
            return 0;
        }
        memcpy(out->as.blob.data, blob, (size_t)size);
        out->as.blob.size = size;
        out->type = VALUE_BLOB;
        out->owned = 1;
        return 1;
    }
    }
    set_error(
        error_buf, error_buf_size, "unsupported sqlite column type %d", type);
    return 0;
}

static void graph_outputs_free(GraphOutputs *outputs) {
    int i;
    for (i = 0; i < outputs->count; ++i) {
        value_reset(&outputs->values[i]);
    }
    free(outputs->values);
    memset(outputs, 0, sizeof(*outputs));
}

static int
parse_placeholder_name(const char *name, int *layer_idx, int *value_idx) {
    return sscanf(name, "past_key_values_%d_%d", layer_idx, value_idx) == 2;
}

static int execute_graph(sqlite3 *db,
                         const NativeGraph *graph,
                         const Blob *input_blob,
                         const Blob *past_kv,
                         int past_kv_count,
                         GraphOutputs *outputs,
                         ExecProfileContext *profile_ctx,
                         char *error_buf,
                         size_t error_buf_size) {
    Value *env = (Value *)calloc((size_t)graph->slot_count, sizeof(Value));
    int *moved = NULL;
    int rc = 0;
    int i;
    memset(outputs, 0, sizeof(*outputs));
    if (env == NULL) {
        set_error(error_buf,
                  error_buf_size,
                  "out of memory allocating execution env");
        return 0;
    }
    rc = 1;
    for (i = 0; i < graph->instr_count; ++i) {
        const NativeInstr *instr = &graph->instrs[i];
        if (instr->kind == INSTR_PLACEHOLDER) {
            int layer_idx = 0;
            int value_idx = 0;
            value_reset(&env[instr->slot]);
            if (strcmp(instr->name, "input_ids") == 0) {
                env[instr->slot].type = VALUE_BLOB;
                env[instr->slot].owned = 0;
                env[instr->slot].as.blob = *input_blob;
                continue;
            }
            if (!parse_placeholder_name(instr->name, &layer_idx, &value_idx)) {
                set_error(error_buf,
                          error_buf_size,
                          "unsupported placeholder: %s",
                          instr->name);
                rc = 0;
                break;
            }
            if (past_kv == NULL) {
                set_error(error_buf,
                          error_buf_size,
                          "missing KV cache for placeholder %s",
                          instr->name);
                rc = 0;
                break;
            }
            if (layer_idx * 2 + value_idx >= past_kv_count) {
                set_error(error_buf,
                          error_buf_size,
                          "KV cache index out of range for %s",
                          instr->name);
                rc = 0;
                break;
            }
            env[instr->slot].type = VALUE_BLOB;
            env[instr->slot].owned = 0;
            env[instr->slot].as.blob = past_kv[layer_idx * 2 + value_idx];
            continue;
        }
        if (instr->kind == INSTR_SQL) {
            sqlite3_stmt *stmt = instr->stmt;
            int bind_index;
            double step_start_ms = 0.0;
            if (profile_ctx != NULL && profile_ctx->profile != NULL) {
                step_start_ms = monotonic_time_ms();
            }
            if (sqlite3_reset(stmt) != SQLITE_OK ||
                sqlite3_clear_bindings(stmt) != SQLITE_OK) {
                set_error(error_buf,
                          error_buf_size,
                          "failed to reset statement for %s",
                          instr->name != NULL ? instr->name : "sql");
                rc = 0;
                break;
            }
            for (bind_index = 0; bind_index < instr->arg_count; ++bind_index) {
                if (!bind_arg(stmt,
                              bind_index + 1,
                              &instr->args[bind_index],
                              env,
                              error_buf,
                              error_buf_size)) {
                    rc = 0;
                    break;
                }
            }
            if (!rc) {
                break;
            }
            if (sqlite3_step(stmt) != SQLITE_ROW) {
                set_error(error_buf,
                          error_buf_size,
                          "statement failed for %s: %s",
                          instr->name != NULL ? instr->name : "sql",
                          sqlite3_errmsg(db));
                rc = 0;
                break;
            }
            value_reset(&env[instr->slot]);
            if (!extract_value(
                    stmt, 0, &env[instr->slot], error_buf, error_buf_size)) {
                rc = 0;
                break;
            }
            if (profile_ctx != NULL && profile_ctx->profile != NULL) {
                profile_ctx->profile->sql_call_count += 1;
                if (!profile_add_op_timing(
                        profile_ctx->profile,
                        instr,
                        (monotonic_time_ms() - step_start_ms) * 1000.0,
                        error_buf,
                        error_buf_size)) {
                    rc = 0;
                    break;
                }
            }
            continue;
        }
        if (instr->kind == INSTR_OUTPUT) {
            outputs->values =
                (Value *)calloc((size_t)instr->output_count, sizeof(Value));
            outputs->count = instr->output_count;
            moved = (int *)calloc((size_t)graph->slot_count, sizeof(int));
            if (outputs->values == NULL || moved == NULL) {
                set_error(error_buf,
                          error_buf_size,
                          "out of memory collecting graph outputs");
                rc = 0;
                break;
            }
            for (int out_index = 0; out_index < instr->output_count;
                 ++out_index) {
                int slot = instr->output_slots[out_index];
                if (slot < 0 || slot >= graph->slot_count) {
                    set_error(error_buf,
                              error_buf_size,
                              "output slot out of range: %d",
                              slot);
                    rc = 0;
                    break;
                }
                if (env[slot].owned && !moved[slot]) {
                    outputs->values[out_index] = env[slot];
                    moved[slot] = 1;
                    memset(&env[slot], 0, sizeof(env[slot]));
                } else if (!value_clone(&env[slot],
                                        &outputs->values[out_index],
                                        error_buf,
                                        error_buf_size)) {
                    rc = 0;
                    break;
                }
            }
            break;
        }
    }
    free(moved);
    for (i = 0; i < graph->slot_count; ++i) {
        value_reset(&env[i]);
    }
    free(env);
    if (!rc) {
        graph_outputs_free(outputs);
    }
    return rc;
}

static Blob make_input_blob(const int *token_ids, int count) {
    Blob blob = {NULL, 0};
    int header_size = 8 + 2 * 4;
    int payload_size = count * 4;
    uint32_t *u32;
    int32_t *i32;
    float *fp32;
    blob.size = header_size + payload_size;
    blob.data = malloc((size_t)blob.size);
    if (blob.data == NULL) {
        blob.size = 0;
        return blob;
    }
    u32 = (uint32_t *)blob.data;
    i32 = (int32_t *)blob.data;
    fp32 = (float *)((char *)blob.data + header_size);
    u32[0] = ND_MAGIC;
    i32[1] = 2;
    i32[2] = 1;
    i32[3] = count;
    for (int i = 0; i < count; ++i) {
        fp32[i] = (float)token_ids[i];
    }
    return blob;
}

static int argmax_last_row(const Blob *blob, int vocab_size) {
    const float *tail =
        (const float *)((const char *)blob->data + blob->size - vocab_size * 4);
    int best_index = 0;
    float best_value = tail[0];
    for (int i = 1; i < vocab_size; ++i) {
        if (tail[i] > best_value) {
            best_value = tail[i];
            best_index = i;
        }
    }
    return best_index;
}

static void free_blob_array(Blob *blobs, int count) {
    if (blobs == NULL) {
        return;
    }
    for (int i = 0; i < count; ++i) {
        free(blobs[i].data);
    }
    free(blobs);
}

static int
resolve_int_metadata(sqlite3 *db, const char *key, int default_value) {
    sqlite3_stmt *stmt = NULL;
    int result = default_value;
    if (sqlite3_prepare_v2(
            db, "SELECT value FROM metadata WHERE key = ?", -1, &stmt, NULL) !=
        SQLITE_OK) {
        return default_value;
    }
    sqlite3_bind_text(stmt, 1, key, -1, SQLITE_STATIC);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const unsigned char *value = sqlite3_column_text(stmt, 0);
        if (value != NULL) {
            if (((const char *)value)[0] == '[') {
                result = atoi((const char *)value + 1);
            } else {
                result = atoi((const char *)value);
            }
        }
    }
    sqlite3_finalize(stmt);
    return result;
}

static int prepare_graph_statements(sqlite3 *db,
                                    NativeGraph *graph,
                                    char *error_buf,
                                    size_t error_buf_size) {
    int i;
    for (i = 0; i < graph->instr_count; ++i) {
        NativeInstr *instr = &graph->instrs[i];
        if (instr->kind != INSTR_SQL) {
            continue;
        }
        if (sqlite3_prepare_v2(db, instr->sql, -1, &instr->stmt, NULL) !=
            SQLITE_OK) {
            set_error(error_buf,
                      error_buf_size,
                      "failed to prepare %s (%s): %s",
                      instr->name != NULL ? instr->name : "sql",
                      instr->target != NULL ? instr->target : "unknown",
                      sqlite3_errmsg(db));
            return 0;
        }
    }
    return 1;
}

static int open_runtime_db(const char *db_path,
                           const char *extension_path,
                           int num_threads,
                           sqlite3 **out_db,
                           char *error_buf,
                           size_t error_buf_size) {
    sqlite3 *db = NULL;
    char *load_error = NULL;
    char thread_sql[64];
    if (sqlite3_open(db_path, &db) != SQLITE_OK) {
        set_error(error_buf, error_buf_size, "failed to open %s", db_path);
        if (db != NULL) {
            sqlite3_close(db);
        }
        return 0;
    }
    if (sqlite3_enable_load_extension(db, 1) != SQLITE_OK) {
        set_error(error_buf,
                  error_buf_size,
                  "failed to enable extension loading: %s",
                  sqlite3_errmsg(db));
        sqlite3_close(db);
        return 0;
    }
    if (sqlite3_load_extension(
            db, extension_path, "sqlite3_llm_ops_init", &load_error) !=
        SQLITE_OK) {
        set_error(error_buf,
                  error_buf_size,
                  "failed to load extension: %s",
                  load_error != NULL ? load_error : sqlite3_errmsg(db));
        sqlite3_free(load_error);
        sqlite3_close(db);
        return 0;
    }
    sqlite3_exec(db, "PRAGMA journal_mode=OFF", NULL, NULL, NULL);
    sqlite3_exec(db, "PRAGMA synchronous=OFF", NULL, NULL, NULL);
    sqlite3_exec(db, "PRAGMA temp_store=MEMORY", NULL, NULL, NULL);
    snprintf(thread_sql,
             sizeof(thread_sql),
             "SELECT llm_set_threads(%d)",
             num_threads > 0 ? num_threads : 1);
    sqlite3_exec(db, thread_sql, NULL, NULL, NULL);
    *out_db = db;
    return 1;
}

static const char *resolve_default_db_name(const char *model_dir,
                                           const char *explicit_db) {
    char *path = NULL;
    struct stat st;
    static char selected[32];
    if (explicit_db != NULL && explicit_db[0] != '\0') {
        return explicit_db;
    }
    path = join_path(model_dir, "model.db");
    if (path != NULL && stat(path, &st) == 0) {
        free(path);
        strcpy(selected, "model.db");
        return selected;
    }
    free(path);
    path = join_path(model_dir, "model_int8.db");
    if (path != NULL && stat(path, &st) == 0) {
        free(path);
        strcpy(selected, "model_int8.db");
        return selected;
    }
    free(path);
    strcpy(selected, "model.db");
    return selected;
}

int llmsql_native_generate_tokens_profiled(const char *model_dir,
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
                                           size_t error_buf_size) {
    const char *resolved_db = resolve_default_db_name(model_dir, db_filename);
    char *db_path = NULL;
    char *prefill_path = NULL;
    char *decode_path = NULL;
    sqlite3 *db = NULL;
    NativeGraph prefill_graph;
    NativeGraph decode_graph;
    Blob prompt_blob = {NULL, 0};
    Blob *kv_cache = NULL;
    int kv_count = 0;
    int vocab_size = 0;
    int eos_token_id = 0;
    int *tokens = NULL;
    int token_count = 0;
    int decode_steps = 0;
    int rc = 0;
    memset(&prefill_graph, 0, sizeof(prefill_graph));
    memset(&decode_graph, 0, sizeof(decode_graph));
    memset(out, 0, sizeof(*out));

    if (prompt_ids == NULL || prompt_len <= 0) {
        set_error(error_buf, error_buf_size, "prompt token ids are required");
        return 0;
    }
    if (max_tokens <= 0) {
        set_error(error_buf, error_buf_size, "max_tokens must be positive");
        return 0;
    }

    if (profile != NULL) {
        llmsql_native_free_profile(profile);
        snprintf(profile->mode, sizeof(profile->mode), "%s", "native_graph");
        snprintf(profile->db_filename,
                 sizeof(profile->db_filename),
                 "%s",
                 resolved_db != NULL ? resolved_db : "");
        profile->prompt_tokens = prompt_len;
        if (max_tokens > 1) {
            profile->decode_step_ms = (double *)calloc(
                (size_t)(max_tokens - 1), sizeof(*profile->decode_step_ms));
            if (profile->decode_step_ms == NULL) {
                llmsql_native_free_profile(profile);
                set_error(error_buf,
                          error_buf_size,
                          "out of memory allocating decode profile buffer");
                return 0;
            }
        }
        record_rss_snapshot(profile, &profile->start_rss_mb);
    }

    db_path = join_path(model_dir, resolved_db);
    prefill_path = prefill_graph_path != NULL
                       ? llmsql_runtime_strdup(prefill_graph_path)
                       : join_path(model_dir, "prefill.native.json");
    decode_path = decode_graph_path != NULL
                      ? llmsql_runtime_strdup(decode_graph_path)
                      : join_path(model_dir, "decode.native.json");
    if (db_path == NULL || prefill_path == NULL || decode_path == NULL) {
        set_error(
            error_buf, error_buf_size, "out of memory resolving runtime paths");
        goto cleanup;
    }
    if (!load_native_graph(
            prefill_path, &prefill_graph, error_buf, error_buf_size)) {
        goto cleanup;
    }
    if (!load_native_graph(
            decode_path, &decode_graph, error_buf, error_buf_size)) {
        goto cleanup;
    }
    if (!open_runtime_db(db_path,
                         extension_path,
                         num_threads,
                         &db,
                         error_buf,
                         error_buf_size)) {
        goto cleanup;
    }
    if (!prepare_graph_statements(
            db, &prefill_graph, error_buf, error_buf_size)) {
        goto cleanup;
    }
    if (!prepare_graph_statements(
            db, &decode_graph, error_buf, error_buf_size)) {
        goto cleanup;
    }

    vocab_size = resolve_int_metadata(db, "vocab_size", 0);
    eos_token_id = resolve_int_metadata(db, "eos_token_id", 0);
    if (vocab_size <= 0) {
        set_error(error_buf,
                  error_buf_size,
                  "failed to resolve vocab_size from metadata");
        goto cleanup;
    }

    prompt_blob = make_input_blob(prompt_ids, prompt_len);
    if (prompt_blob.data == NULL) {
        set_error(
            error_buf, error_buf_size, "out of memory building input blob");
        goto cleanup;
    }

    tokens = (int *)malloc((size_t)max_tokens * sizeof(int));
    if (tokens == NULL) {
        set_error(
            error_buf, error_buf_size, "out of memory allocating token buffer");
        goto cleanup;
    }

    {
        GraphOutputs prefill_outputs;
        Blob logits = {NULL, 0};
        double prefill_start_ms = 0.0;
        memset(&prefill_outputs, 0, sizeof(prefill_outputs));
        if (profile != NULL) {
            prefill_start_ms = monotonic_time_ms();
        }
        if (!execute_graph(db,
                           &prefill_graph,
                           &prompt_blob,
                           NULL,
                           0,
                           &prefill_outputs,
                           NULL,
                           error_buf,
                           error_buf_size)) {
            goto cleanup;
        }
        if (profile != NULL) {
            profile->prefill_ms = monotonic_time_ms() - prefill_start_ms;
            record_rss_snapshot(profile, &profile->after_prefill_rss_mb);
        }
        if (prefill_outputs.count < 1 ||
            prefill_outputs.values[0].type != VALUE_BLOB) {
            graph_outputs_free(&prefill_outputs);
            set_error(error_buf,
                      error_buf_size,
                      "prefill graph returned invalid logits");
            goto cleanup;
        }
        logits = prefill_outputs.values[0].as.blob;
        prefill_outputs.values[0].type = VALUE_EMPTY;
        token_count = 1;
        tokens[0] = argmax_last_row(&logits, vocab_size);
        kv_count = prefill_outputs.count - 1;
        if (kv_count > 0) {
            kv_cache = (Blob *)calloc((size_t)kv_count, sizeof(Blob));
            if (kv_cache == NULL) {
                free(logits.data);
                graph_outputs_free(&prefill_outputs);
                set_error(error_buf,
                          error_buf_size,
                          "out of memory allocating KV cache");
                goto cleanup;
            }
            for (int i = 0; i < kv_count; ++i) {
                if (prefill_outputs.values[i + 1].type != VALUE_BLOB) {
                    free(logits.data);
                    graph_outputs_free(&prefill_outputs);
                    set_error(error_buf,
                              error_buf_size,
                              "prefill output %d is not a blob",
                              i + 1);
                    goto cleanup;
                }
                kv_cache[i] = prefill_outputs.values[i + 1].as.blob;
                prefill_outputs.values[i + 1].type = VALUE_EMPTY;
            }
        }
        graph_outputs_free(&prefill_outputs);
        free(logits.data);
    }

    while (token_count < max_tokens) {
        GraphOutputs decode_outputs;
        Blob logits = {NULL, 0};
        Blob *next_kv = NULL;
        Blob step_blob = make_input_blob(&tokens[token_count - 1], 1);
        double decode_start_ms = 0.0;
        memset(&decode_outputs, 0, sizeof(decode_outputs));
        if (step_blob.data == NULL) {
            set_error(error_buf,
                      error_buf_size,
                      "out of memory building decode input blob");
            goto cleanup;
        }
        if (profile != NULL) {
            decode_start_ms = monotonic_time_ms();
        }
        if (!execute_graph(db,
                           &decode_graph,
                           &step_blob,
                           kv_cache,
                           kv_count,
                           &decode_outputs,
                           NULL,
                           error_buf,
                           error_buf_size)) {
            free(step_blob.data);
            goto cleanup;
        }
        free(step_blob.data);
        if (decode_outputs.count < 1 ||
            decode_outputs.values[0].type != VALUE_BLOB) {
            graph_outputs_free(&decode_outputs);
            set_error(error_buf,
                      error_buf_size,
                      "decode graph returned invalid logits");
            goto cleanup;
        }
        logits = decode_outputs.values[0].as.blob;
        decode_outputs.values[0].type = VALUE_EMPTY;
        next_kv =
            (Blob *)calloc((size_t)(decode_outputs.count - 1), sizeof(Blob));
        if (next_kv == NULL) {
            free(logits.data);
            graph_outputs_free(&decode_outputs);
            set_error(error_buf,
                      error_buf_size,
                      "out of memory allocating decode KV cache");
            goto cleanup;
        }
        for (int i = 1; i < decode_outputs.count; ++i) {
            if (decode_outputs.values[i].type != VALUE_BLOB) {
                free(logits.data);
                free_blob_array(next_kv, i - 1);
                graph_outputs_free(&decode_outputs);
                set_error(error_buf,
                          error_buf_size,
                          "decode output %d is not a blob",
                          i);
                goto cleanup;
            }
            next_kv[i - 1] = decode_outputs.values[i].as.blob;
            decode_outputs.values[i].type = VALUE_EMPTY;
        }
        graph_outputs_free(&decode_outputs);
        free_blob_array(kv_cache, kv_count);
        kv_cache = next_kv;
        kv_count =
            decode_graph.instrs[decode_graph.instr_count - 1].output_count - 1;
        tokens[token_count] = argmax_last_row(&logits, vocab_size);
        free(logits.data);
        if (profile != NULL) {
            double decode_ms = monotonic_time_ms() - decode_start_ms;
            profile->decode_step_ms[decode_steps] = decode_ms;
            profile->decode_total_ms += decode_ms;
            update_peak_rss(profile);
        }
        decode_steps += 1;
        if (tokens[token_count] == eos_token_id) {
            token_count += 1;
            break;
        }
        token_count += 1;
    }

    out->token_ids = tokens;
    out->token_count = token_count;
    tokens = NULL;
    if (profile != NULL) {
        profile->generated_tokens = token_count;
        profile->decode_steps = decode_steps;
        profile->decode_step_count = decode_steps;
        profile->decode_avg_ms =
            decode_steps > 0 ? profile->decode_total_ms / (double)decode_steps
                             : 0.0;
        profile->tokens_per_sec =
            profile->decode_total_ms > 0.0
                ? ((double)decode_steps * 1000.0) / profile->decode_total_ms
                : 0.0;
        record_rss_snapshot(profile, &profile->end_rss_mb);
        if (decode_steps > 0 && kv_cache != NULL) {
            Blob replay_blob =
                make_input_blob(&out->token_ids[out->token_count - 1], 1);
            GraphOutputs replay_outputs;
            ExecProfileContext replay_ctx;
            char replay_error[512] = {0};
            memset(&replay_outputs, 0, sizeof(replay_outputs));
            memset(&replay_ctx, 0, sizeof(replay_ctx));
            replay_ctx.profile = profile;
            free(profile->op_timings);
            profile->op_timings = NULL;
            profile->op_timing_count = 0;
            profile->sql_call_count = 0;
            if (replay_blob.data != NULL) {
                if (execute_graph(db,
                                  &decode_graph,
                                  &replay_blob,
                                  kv_cache,
                                  kv_count,
                                  &replay_outputs,
                                  &replay_ctx,
                                  replay_error,
                                  sizeof(replay_error))) {
                    graph_outputs_free(&replay_outputs);
                    sort_profile_op_timings(profile);
                } else {
                    graph_outputs_free(&replay_outputs);
                    free(profile->op_timings);
                    profile->op_timings = NULL;
                    profile->op_timing_count = 0;
                    profile->sql_call_count = 0;
                }
                free(replay_blob.data);
            }
        }
        build_memory_summary(profile);
    }
    rc = 1;

cleanup:
    if (!rc && profile != NULL) {
        llmsql_native_free_profile(profile);
    }
    free(tokens);
    free_blob_array(kv_cache, kv_count);
    free(prompt_blob.data);
    if (db != NULL) {
        sqlite3_close(db);
    }
    free_native_graph(&prefill_graph);
    free_native_graph(&decode_graph);
    free(db_path);
    free(prefill_path);
    free(decode_path);
    return rc;
}

int llmsql_native_generate_tokens(const char *model_dir,
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
                                  size_t error_buf_size) {
    return llmsql_native_generate_tokens_profiled(model_dir,
                                                  db_filename,
                                                  prefill_graph_path,
                                                  decode_graph_path,
                                                  extension_path,
                                                  num_threads,
                                                  prompt_ids,
                                                  prompt_len,
                                                  max_tokens,
                                                  out,
                                                  NULL,
                                                  error_buf,
                                                  error_buf_size);
}

void llmsql_native_free_generation(llmsql_generation *generation) {
    if (generation == NULL) {
        return;
    }
    free(generation->token_ids);
    generation->token_ids = NULL;
    generation->token_count = 0;
}
