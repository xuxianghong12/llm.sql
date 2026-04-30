/*
 * llm_tokenizer.c — SQLite extension: BPE tokenizer for LLM inference.
 *
 * Provides SQL scalar functions so that any language with SQLite
 * bindings can tokenize/detokenize without linking a C library:
 *
 *   llm_tokenize(text, tokenizer_json_path)
 *       → BLOB of int32 token IDs
 *
 *   llm_detokenize(token_ids_blob)
 *       → TEXT (decoded UTF-8 string)
 *
 * The vocab table is read from the *same* database connection
 * (no external db_path needed for detokenize) while BPE merge
 * rules come from the HuggingFace ``tokenizer.json`` file.
 *
 * Build:
 *   gcc -O2 -Wall -Wextra -Wpedantic -fPIC -shared \
 *       -o llm_tokenizer.so llm_tokenizer.c -lsqlite3
 *
 * Load:
 *   .load ./llm_tokenizer
 *   SELECT hex(llm_tokenize('hello', '/path/to/tokenizer.json'));
 *   SELECT llm_detokenize(X'....');
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *llmsql_strdup(const char *src) {
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

/* ================================================================== */
/*  GPT-2 style byte <-> unicode mapping                              */
/* ================================================================== */

static int g_byte_to_unicode[256];
static int g_unicode_to_byte[512];
static int g_mapping_ready;

static void init_byte_unicode_mapping(void) {
    int bs[256], cs[256];
    int n = 0, total = 0, i;

    if (g_mapping_ready) {
        return;
    }
    memset(g_unicode_to_byte, -1, sizeof(g_unicode_to_byte));

    for (i = 0x21; i <= 0x7E; ++i) {
        bs[total] = i;
        cs[total] = i;
        ++total;
    }
    for (i = 0xA1; i <= 0xAC; ++i) {
        bs[total] = i;
        cs[total] = i;
        ++total;
    }
    for (i = 0xAE; i <= 0xFF; ++i) {
        bs[total] = i;
        cs[total] = i;
        ++total;
    }
    for (i = 0; i < 256; ++i) {
        int j, found = 0;
        for (j = 0; j < total; ++j) {
            if (bs[j] == i) {
                found = 1;
                break;
            }
        }
        if (!found) {
            bs[total] = i;
            cs[total] = 0x0100 + n;
            ++n;
            ++total;
        }
    }
    for (i = 0; i < total; ++i) {
        g_byte_to_unicode[bs[i]] = cs[i];
        if (cs[i] <
            (int)(sizeof(g_unicode_to_byte) / sizeof(g_unicode_to_byte[0]))) {
            g_unicode_to_byte[cs[i]] = bs[i];
        }
    }
    g_mapping_ready = 1;
}

/* ================================================================== */
/*  UTF-8 helpers                                                     */
/* ================================================================== */

static int encode_utf8(int cp, char *out) {
    if (cp < 0) {
        return 0;
    }
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

static int decode_utf8(const char **p) {
    const unsigned char *s = (const unsigned char *)*p;
    int cp, n;
    if (s[0] < 0x80) {
        *p += 1;
        return s[0];
    }
    if ((s[0] & 0xE0) == 0xC0) {
        cp = s[0] & 0x1F;
        n = 2;
    } else if ((s[0] & 0xF0) == 0xE0) {
        cp = s[0] & 0x0F;
        n = 3;
    } else if ((s[0] & 0xF8) == 0xF0) {
        cp = s[0] & 0x07;
        n = 4;
    } else {
        *p += 1;
        return -1;
    }
    for (int i = 1; i < n; ++i) {
        if ((s[i] & 0xC0) != 0x80) {
            *p += 1;
            return -1;
        }
        cp = (cp << 6) | (s[i] & 0x3F);
    }
    *p += n;
    return cp;
}

/* ================================================================== */
/*  Dynamic byte buffer                                               */
/* ================================================================== */

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} DynBuf;

static void dynbuf_init(DynBuf *b) {
    b->data = NULL;
    b->len = 0;
    b->cap = 0;
}

static int dynbuf_push(DynBuf *b, const char *src, size_t n) {
    if (b->len + n > b->cap) {
        size_t newcap = b->cap == 0 ? 256 : b->cap;
        char *tmp;
        while (newcap < b->len + n) {
            newcap *= 2;
        }
        tmp = (char *)realloc(b->data, newcap);
        if (tmp == NULL) {
            return 0;
        }
        b->data = tmp;
        b->cap = newcap;
    }
    memcpy(b->data + b->len, src, n);
    b->len += n;
    return 1;
}

static int dynbuf_push_byte(DynBuf *b, char c) {
    return dynbuf_push(b, &c, 1);
}

static void dynbuf_free(DynBuf *b) {
    free(b->data);
    b->data = NULL;
    b->len = 0;
    b->cap = 0;
}

/* ================================================================== */
/*  Hash table: string -> int                                         */
/* ================================================================== */

typedef struct {
    char *key;
    int value;
} HTEntry;

typedef struct {
    HTEntry *entries;
    int cap;
    int count;
} HashTable;

static uint32_t ht_hash(const char *s) {
    uint32_t h = 5381;
    while (*s) {
        h = ((h << 5) + h) + (unsigned char)*s;
        ++s;
    }
    return h;
}

static int ht_init(HashTable *ht, int cap) {
    ht->entries = (HTEntry *)calloc((size_t)cap, sizeof(HTEntry));
    if (ht->entries == NULL) {
        return 0;
    }
    ht->cap = cap;
    ht->count = 0;
    return 1;
}

static void ht_free(HashTable *ht) {
    int i;
    for (i = 0; i < ht->cap; ++i) {
        free(ht->entries[i].key);
    }
    free(ht->entries);
    ht->entries = NULL;
    ht->cap = 0;
    ht->count = 0;
}

static int ht_put(HashTable *ht, const char *key, int value) {
    uint32_t idx;
    if (ht->count * 4 >= ht->cap * 3) {
        int newcap = ht->cap * 2;
        HashTable newht;
        int i;
        if (!ht_init(&newht, newcap)) {
            return 0;
        }
        for (i = 0; i < ht->cap; ++i) {
            if (ht->entries[i].key != NULL) {
                if (!ht_put(&newht, ht->entries[i].key, ht->entries[i].value)) {
                    ht_free(&newht);
                    return 0;
                }
            }
        }
        ht_free(ht);
        *ht = newht;
    }
    idx = ht_hash(key) % (uint32_t)ht->cap;
    while (ht->entries[idx].key != NULL) {
        if (strcmp(ht->entries[idx].key, key) == 0) {
            ht->entries[idx].value = value;
            return 1;
        }
        idx = (idx + 1) % (uint32_t)ht->cap;
    }
    ht->entries[idx].key = llmsql_strdup(key);
    if (ht->entries[idx].key == NULL) {
        return 0;
    }
    ht->entries[idx].value = value;
    ++ht->count;
    return 1;
}

static int ht_get(const HashTable *ht, const char *key, int *out) {
    uint32_t idx;
    if (ht->cap == 0) {
        return 0;
    }
    idx = ht_hash(key) % (uint32_t)ht->cap;
    while (ht->entries[idx].key != NULL) {
        if (strcmp(ht->entries[idx].key, key) == 0) {
            *out = ht->entries[idx].value;
            return 1;
        }
        idx = (idx + 1) % (uint32_t)ht->cap;
    }
    return 0;
}

/* ================================================================== */
/*  Tokenizer internal state                                          */
/* ================================================================== */

typedef struct {
    /* id -> token text (UTF-8, byte-level encoded) */
    char **id_to_token;
    int *is_special;
    int vocab_size;

    /* token text -> id */
    HashTable token_to_id;

    /* BPE merge rank: "tokenA\ttokenB" -> rank */
    HashTable merge_rank;
    int merge_count;
} Tokenizer;

/* ================================================================== */
/*  Read file helper                                                  */
/* ================================================================== */

static char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    long sz;
    char *buf;
    if (f == NULL) {
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    sz = ftell(f);
    if (sz < 0) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    buf = (char *)malloc((size_t)sz + 1);
    if (buf == NULL) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    buf[sz] = '\0';
    if (out_len != NULL) {
        *out_len = (size_t)sz;
    }
    return buf;
}

/* ================================================================== */
/*  Load vocab from the SAME db connection                            */
/* ================================================================== */

static int load_vocab_from_conn(sqlite3 *db, Tokenizer *tok) {
    sqlite3_stmt *stmt = NULL;
    int max_id = -1;
    int rc = 0;

    if (sqlite3_prepare_v2(
            db, "SELECT MAX(token_id) FROM vocab", -1, &stmt, NULL) !=
        SQLITE_OK) {
        return 0;
    }
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        max_id = sqlite3_column_int(stmt, 0);
    }
    sqlite3_finalize(stmt);
    stmt = NULL;

    if (max_id < 0) {
        return 0;
    }

    tok->vocab_size = max_id + 1;
    tok->id_to_token = (char **)calloc((size_t)tok->vocab_size, sizeof(char *));
    tok->is_special = (int *)calloc((size_t)tok->vocab_size, sizeof(int));
    if (tok->id_to_token == NULL || tok->is_special == NULL) {
        return 0;
    }
    if (!ht_init(&tok->token_to_id, tok->vocab_size * 2)) {
        return 0;
    }

    if (sqlite3_prepare_v2(db,
                           "SELECT token_id, token, is_special FROM vocab",
                           -1,
                           &stmt,
                           NULL) != SQLITE_OK) {
        return 0;
    }
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int tid = sqlite3_column_int(stmt, 0);
        const char *text = (const char *)sqlite3_column_text(stmt, 1);
        int is_sp = sqlite3_column_int(stmt, 2);
        if (tid < 0 || tid >= tok->vocab_size || text == NULL) {
            continue;
        }
        tok->id_to_token[tid] = llmsql_strdup(text);
        tok->is_special[tid] = is_sp;
        ht_put(&tok->token_to_id, text, tid);
    }
    sqlite3_finalize(stmt);
    rc = 1;
    return rc;
}

/* ================================================================== */
/*  Minimal JSON parser for tokenizer.json                            */
/* ================================================================== */

typedef struct {
    const char *data;
    size_t pos;
    size_t len;
} JsonCur;

static void json_skip_ws(JsonCur *c) {
    while (c->pos < c->len &&
           (c->data[c->pos] == ' ' || c->data[c->pos] == '\t' ||
            c->data[c->pos] == '\n' || c->data[c->pos] == '\r')) {
        ++c->pos;
    }
}

static int json_peek(JsonCur *c) {
    json_skip_ws(c);
    if (c->pos >= c->len) {
        return -1;
    }
    return (unsigned char)c->data[c->pos];
}

static int json_consume(JsonCur *c, char ch) {
    if (json_peek(c) != ch) {
        return 0;
    }
    ++c->pos;
    return 1;
}

static char *json_parse_string(JsonCur *c) {
    DynBuf buf;
    dynbuf_init(&buf);
    if (!json_consume(c, '"')) {
        return NULL;
    }
    while (c->pos < c->len && c->data[c->pos] != '"') {
        if (c->data[c->pos] == '\\') {
            ++c->pos;
            if (c->pos >= c->len) {
                goto fail;
            }
            switch (c->data[c->pos]) {
            case '"':
                dynbuf_push_byte(&buf, '"');
                break;
            case '\\':
                dynbuf_push_byte(&buf, '\\');
                break;
            case '/':
                dynbuf_push_byte(&buf, '/');
                break;
            case 'b':
                dynbuf_push_byte(&buf, '\b');
                break;
            case 'f':
                dynbuf_push_byte(&buf, '\f');
                break;
            case 'n':
                dynbuf_push_byte(&buf, '\n');
                break;
            case 'r':
                dynbuf_push_byte(&buf, '\r');
                break;
            case 't':
                dynbuf_push_byte(&buf, '\t');
                break;
            case 'u': {
                unsigned int cp = 0;
                int i;
                ++c->pos;
                for (i = 0; i < 4; ++i) {
                    unsigned int d;
                    if (c->pos >= c->len) {
                        goto fail;
                    }
                    d = (unsigned char)c->data[c->pos];
                    if (d >= '0' && d <= '9') {
                        d -= '0';
                    } else if (d >= 'a' && d <= 'f') {
                        d = d - 'a' + 10;
                    } else if (d >= 'A' && d <= 'F') {
                        d = d - 'A' + 10;
                    } else {
                        goto fail;
                    }
                    cp = (cp << 4) | d;
                    ++c->pos;
                }
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    unsigned int lo = 0;
                    if (c->pos + 1 < c->len && c->data[c->pos] == '\\' &&
                        c->data[c->pos + 1] == 'u') {
                        c->pos += 2;
                        for (i = 0; i < 4; ++i) {
                            unsigned int d;
                            if (c->pos >= c->len) {
                                goto fail;
                            }
                            d = (unsigned char)c->data[c->pos];
                            if (d >= '0' && d <= '9') {
                                d -= '0';
                            } else if (d >= 'a' && d <= 'f') {
                                d = d - 'a' + 10;
                            } else if (d >= 'A' && d <= 'F') {
                                d = d - 'A' + 10;
                            } else {
                                goto fail;
                            }
                            lo = (lo << 4) | d;
                            ++c->pos;
                        }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                }
                {
                    char u8[4];
                    int nn = encode_utf8((int)cp, u8);
                    dynbuf_push(&buf, u8, (size_t)nn);
                }
                continue;
            }
            default:
                dynbuf_push_byte(&buf, c->data[c->pos]);
                break;
            }
            ++c->pos;
        } else {
            dynbuf_push_byte(&buf, c->data[c->pos]);
            ++c->pos;
        }
    }
    if (!json_consume(c, '"')) {
        goto fail;
    }
    dynbuf_push_byte(&buf, '\0');
    return buf.data;
fail:
    dynbuf_free(&buf);
    return NULL;
}

static int json_skip_value(JsonCur *c) {
    int ch = json_peek(c);
    if (ch == '"') {
        char *s = json_parse_string(c);
        free(s);
        return s != NULL;
    }
    if (ch == '{') {
        ++c->pos;
        if (json_peek(c) == '}') {
            ++c->pos;
            return 1;
        }
        for (;;) {
            char *key = json_parse_string(c);
            if (key == NULL) {
                return 0;
            }
            free(key);
            if (!json_consume(c, ':')) {
                return 0;
            }
            if (!json_skip_value(c)) {
                return 0;
            }
            if (json_peek(c) == '}') {
                ++c->pos;
                return 1;
            }
            if (!json_consume(c, ',')) {
                return 0;
            }
        }
    }
    if (ch == '[') {
        ++c->pos;
        if (json_peek(c) == ']') {
            ++c->pos;
            return 1;
        }
        for (;;) {
            if (!json_skip_value(c)) {
                return 0;
            }
            if (json_peek(c) == ']') {
                ++c->pos;
                return 1;
            }
            if (!json_consume(c, ',')) {
                return 0;
            }
        }
    }
    while (c->pos < c->len) {
        char cc = c->data[c->pos];
        if (cc == ',' || cc == '}' || cc == ']' || cc == ' ' || cc == '\t' ||
            cc == '\n' || cc == '\r') {
            break;
        }
        ++c->pos;
    }
    return 1;
}

static int json_find_key(JsonCur *c, const char *key) {
    if (!json_consume(c, '{')) {
        return 0;
    }
    if (json_peek(c) == '}') {
        return 0;
    }
    for (;;) {
        char *k = json_parse_string(c);
        if (k == NULL) {
            return 0;
        }
        if (!json_consume(c, ':')) {
            free(k);
            return 0;
        }
        if (strcmp(k, key) == 0) {
            free(k);
            return 1;
        }
        free(k);
        if (!json_skip_value(c)) {
            return 0;
        }
        if (json_peek(c) == '}') {
            return 0;
        }
        if (!json_consume(c, ',')) {
            return 0;
        }
    }
}

/* ================================================================== */
/*  Parse vocab + merges from tokenizer.json                          */
/* ================================================================== */

static int parse_vocab_object(JsonCur *c, HashTable *token_to_id) {
    if (!json_consume(c, '{')) {
        return 0;
    }
    if (json_peek(c) == '}') {
        ++c->pos;
        return 1;
    }
    for (;;) {
        char *key = json_parse_string(c);
        int value;
        if (key == NULL) {
            return 0;
        }
        if (!json_consume(c, ':')) {
            free(key);
            return 0;
        }
        json_skip_ws(c);
        value = 0;
        {
            int neg = 0;
            if (c->pos < c->len && c->data[c->pos] == '-') {
                neg = 1;
                ++c->pos;
            }
            while (c->pos < c->len && c->data[c->pos] >= '0' &&
                   c->data[c->pos] <= '9') {
                value = value * 10 + (c->data[c->pos] - '0');
                ++c->pos;
            }
            if (neg) {
                value = -value;
            }
        }
        ht_put(token_to_id, key, value);
        free(key);
        if (json_peek(c) == '}') {
            ++c->pos;
            return 1;
        }
        if (!json_consume(c, ',')) {
            return 0;
        }
    }
}

static int
parse_merges_array(JsonCur *c, HashTable *merge_rank, int *out_count) {
    int rank = 0;
    if (!json_consume(c, '[')) {
        return 0;
    }
    if (json_peek(c) == ']') {
        ++c->pos;
        *out_count = 0;
        return 1;
    }
    for (;;) {
        char *merge_str = json_parse_string(c);
        char *space;
        if (merge_str == NULL) {
            return 0;
        }
        /* merge_str is e.g. "Ġ t" — split on first space
         * to build the pair key "tokenA\ttokenB" */
        space = strchr(merge_str, ' ');
        if (space != NULL) {
            *space = '\0';
            {
                size_t alen = strlen(merge_str);
                size_t blen = strlen(space + 1);
                char *pair_key = (char *)malloc(alen + 1 + blen + 1);
                if (pair_key != NULL) {
                    memcpy(pair_key, merge_str, alen);
                    pair_key[alen] = '\t';
                    memcpy(pair_key + alen + 1, space + 1, blen + 1);
                    ht_put(merge_rank, pair_key, rank);
                    free(pair_key);
                }
            }
        }
        free(merge_str);
        ++rank;
        if (json_peek(c) == ']') {
            ++c->pos;
            break;
        }
        if (!json_consume(c, ',')) {
            return 0;
        }
    }
    *out_count = rank;
    return 1;
}

static int load_merges_from_json(const char *json_path, Tokenizer *tok) {
    size_t flen = 0;
    char *data = read_file(json_path, &flen);
    JsonCur cursor;
    int rc = 0;

    if (data == NULL) {
        return 0;
    }
    cursor.data = data;
    cursor.pos = 0;
    cursor.len = flen;

    if (!ht_init(&tok->merge_rank, 1024)) {
        free(data);
        return 0;
    }

    /* Navigate: root -> "model" -> "vocab" */
    {
        JsonCur vc = cursor;
        if (json_find_key(&vc, "model")) {
            JsonCur mc = vc;
            if (json_find_key(&mc, "vocab")) {
                HashTable json_vocab;
                if (ht_init(&json_vocab, 1024)) {
                    parse_vocab_object(&mc, &json_vocab);
                    for (int i = 0; i < json_vocab.cap; ++i) {
                        if (json_vocab.entries[i].key != NULL) {
                            int dummy;
                            if (!ht_get(&tok->token_to_id,
                                        json_vocab.entries[i].key,
                                        &dummy)) {
                                int tid = json_vocab.entries[i].value;
                                ht_put(&tok->token_to_id,
                                       json_vocab.entries[i].key,
                                       tid);
                                if (tid >= 0 && tid < tok->vocab_size &&
                                    tok->id_to_token[tid] == NULL) {
                                    tok->id_to_token[tid] = llmsql_strdup(
                                        json_vocab.entries[i].key);
                                }
                            }
                        }
                    }
                    ht_free(&json_vocab);
                }
            }
        }
    }

    /* Navigate: root -> "model" -> "merges" */
    {
        JsonCur mc = cursor;
        if (json_find_key(&mc, "model")) {
            if (json_find_key(&mc, "merges")) {
                rc = parse_merges_array(
                    &mc, &tok->merge_rank, &tok->merge_count);
            }
        }
    }

    free(data);
    return rc;
}

/* ================================================================== */
/*  Tokenizer lifecycle                                               */
/* ================================================================== */

static Tokenizer *tokenizer_create(void) {
    Tokenizer *tok = (Tokenizer *)calloc(1, sizeof(Tokenizer));
    return tok;
}

static void tokenizer_free(Tokenizer *tok) {
    int i;
    if (tok == NULL) {
        return;
    }
    if (tok->id_to_token != NULL) {
        for (i = 0; i < tok->vocab_size; ++i) {
            free(tok->id_to_token[i]);
        }
        free(tok->id_to_token);
    }
    free(tok->is_special);
    ht_free(&tok->token_to_id);
    ht_free(&tok->merge_rank);
    free(tok);
}

/* ================================================================== */
/*  Pre-tokenization                                                  */
/* ================================================================== */

static int is_letter(const char *p, int *bytes) {
    unsigned char c = (unsigned char)*p;
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
        *bytes = 1;
        return 1;
    }
    if (c >= 0xC0) {
        int n = 1;
        if ((c & 0xE0) == 0xC0) {
            n = 2;
        } else if ((c & 0xF0) == 0xE0) {
            n = 3;
        } else if ((c & 0xF8) == 0xF0) {
            n = 4;
        }
        *bytes = n;
        return 1;
    }
    return 0;
}

static int is_digit_char(char c) {
    return c >= '0' && c <= '9';
}

static int is_space_char(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
           c == '\v';
}

typedef struct {
    const char *start;
    int len;
} Span;

typedef struct {
    Span *items;
    int count;
    int cap;
} SpanList;

static int span_push(SpanList *sl, const char *start, int len) {
    if (sl->count >= sl->cap) {
        int newcap = sl->cap == 0 ? 64 : sl->cap * 2;
        Span *tmp = (Span *)realloc(sl->items, (size_t)newcap * sizeof(Span));
        if (tmp == NULL) {
            return 0;
        }
        sl->items = tmp;
        sl->cap = newcap;
    }
    sl->items[sl->count].start = start;
    sl->items[sl->count].len = len;
    ++sl->count;
    return 1;
}

static void pretokenize(const char *text, SpanList *out) {
    const char *p = text;
    out->items = NULL;
    out->count = 0;
    out->cap = 0;

    while (*p != '\0') {
        int bytes = 0;

        {
            const char *start = p;
            int lb = 0;

            if (*p != '\0' && !is_digit_char(*p) && !is_letter(p, &lb) &&
                *p != '\r' && *p != '\n') {
                const char *next = p;
                if ((unsigned char)*next >= 0xC0) {
                    int n = 1;
                    unsigned char c = (unsigned char)*next;
                    if ((c & 0xE0) == 0xC0) {
                        n = 2;
                    } else if ((c & 0xF0) == 0xE0) {
                        n = 3;
                    } else if ((c & 0xF8) == 0xF0) {
                        n = 4;
                    }
                    next += n;
                } else {
                    next += 1;
                }
                if (is_letter(next, &lb)) {
                    p = next;
                }
            }

            if (is_letter(p, &lb)) {
                p += lb;
                while (*p != '\0' && is_letter(p, &lb)) {
                    p += lb;
                }
                span_push(out, start, (int)(p - start));
                continue;
            }
            p = start;
        }

        if (is_digit_char(*p)) {
            const char *start = p;
            int cnt = 0;
            while (*p != '\0' && is_digit_char(*p) && cnt < 3) {
                ++p;
                ++cnt;
            }
            span_push(out, start, (int)(p - start));
            continue;
        }

        if (is_space_char(*p)) {
            const char *start = p;
            while (*p != '\0' && is_space_char(*p)) {
                ++p;
            }
            span_push(out, start, (int)(p - start));
            continue;
        }

        if (!is_letter(p, &bytes) && !is_digit_char(*p) && !is_space_char(*p)) {
            const char *start = p;
            while (*p != '\0' && !is_letter(p, &bytes) && !is_digit_char(*p) &&
                   !is_space_char(*p)) {
                ++p;
            }
            span_push(out, start, (int)(p - start));
            continue;
        }

        span_push(out, p, 1);
        ++p;
    }
}

/* ================================================================== */
/*  BPE encoding                                                      */
/* ================================================================== */

typedef struct {
    char **tokens;
    int count;
    int cap;
} TokenList;

static void toklist_init(TokenList *tl) {
    tl->tokens = NULL;
    tl->count = 0;
    tl->cap = 0;
}

static int toklist_push(TokenList *tl, const char *s, int len) {
    if (tl->count >= tl->cap) {
        int newcap = tl->cap == 0 ? 32 : tl->cap * 2;
        char **tmp =
            (char **)realloc(tl->tokens, (size_t)newcap * sizeof(char *));
        if (tmp == NULL) {
            return 0;
        }
        tl->tokens = tmp;
        tl->cap = newcap;
    }
    tl->tokens[tl->count] = (char *)malloc((size_t)len + 1);
    if (tl->tokens[tl->count] == NULL) {
        return 0;
    }
    memcpy(tl->tokens[tl->count], s, (size_t)len);
    tl->tokens[tl->count][len] = '\0';
    ++tl->count;
    return 1;
}

static void toklist_free(TokenList *tl) {
    int i;
    for (i = 0; i < tl->count; ++i) {
        free(tl->tokens[i]);
    }
    free(tl->tokens);
    tl->tokens = NULL;
    tl->count = 0;
    tl->cap = 0;
}

static int
bytes_to_initial_tokens(const char *raw, int rawlen, TokenList *out) {
    int i;
    toklist_init(out);
    for (i = 0; i < rawlen; ++i) {
        unsigned char byte = (unsigned char)raw[i];
        int cp = g_byte_to_unicode[byte];
        char u8[4];
        int n = encode_utf8(cp, u8);
        if (!toklist_push(out, u8, n)) {
            toklist_free(out);
            return 0;
        }
    }
    return 1;
}

static int
get_merge_rank(const HashTable *merge_rank, const char *a, const char *b) {
    size_t alen = strlen(a);
    size_t blen = strlen(b);
    char *key = (char *)malloc(alen + 1 + blen + 1);
    int rank = -1;
    if (key == NULL) {
        return -1;
    }
    memcpy(key, a, alen);
    key[alen] = '\t';
    memcpy(key + alen + 1, b, blen + 1);
    if (!ht_get(merge_rank, key, &rank)) {
        rank = -1;
    }
    free(key);
    return rank;
}

static void apply_bpe_merges(TokenList *tl, const HashTable *merge_rank) {
    for (;;) {
        int best_rank = -1;
        int best_idx = -1;
        int i;

        if (tl->count < 2) {
            break;
        }
        for (i = 0; i < tl->count - 1; ++i) {
            int r =
                get_merge_rank(merge_rank, tl->tokens[i], tl->tokens[i + 1]);
            if (r >= 0 && (best_rank < 0 || r < best_rank)) {
                best_rank = r;
                best_idx = i;
            }
        }
        if (best_idx < 0) {
            break;
        }
        {
            size_t alen = strlen(tl->tokens[best_idx]);
            size_t blen = strlen(tl->tokens[best_idx + 1]);
            char *merged = (char *)malloc(alen + blen + 1);
            if (merged == NULL) {
                break;
            }
            memcpy(merged, tl->tokens[best_idx], alen);
            memcpy(merged + alen, tl->tokens[best_idx + 1], blen + 1);
            free(tl->tokens[best_idx]);
            free(tl->tokens[best_idx + 1]);
            tl->tokens[best_idx] = merged;
            for (i = best_idx + 1; i < tl->count - 1; ++i) {
                tl->tokens[i] = tl->tokens[i + 1];
            }
            --tl->count;
        }
    }
}

/* ================================================================== */
/*  Encode: text -> BLOB of int32 token IDs                           */
/* ================================================================== */

static int tokenizer_encode(const Tokenizer *tok,
                            const char *text,
                            int32_t **out_ids,
                            int *out_count) {
    SpanList spans;
    int32_t *ids = NULL;
    int id_count = 0;
    int id_cap = 0;
    int si;

    if (tok == NULL || text == NULL) {
        return 0;
    }
    pretokenize(text, &spans);

    for (si = 0; si < spans.count; ++si) {
        TokenList tl;
        int ti;

        if (!bytes_to_initial_tokens(
                spans.items[si].start, spans.items[si].len, &tl)) {
            free(spans.items);
            free(ids);
            return 0;
        }
        apply_bpe_merges(&tl, &tok->merge_rank);

        for (ti = 0; ti < tl.count; ++ti) {
            int tid;
            if (ht_get(&tok->token_to_id, tl.tokens[ti], &tid)) {
                if (id_count >= id_cap) {
                    int newcap = id_cap == 0 ? 64 : id_cap * 2;
                    int32_t *tmp = (int32_t *)realloc(
                        ids, (size_t)newcap * sizeof(int32_t));
                    if (tmp == NULL) {
                        toklist_free(&tl);
                        free(spans.items);
                        free(ids);
                        return 0;
                    }
                    ids = tmp;
                    id_cap = newcap;
                }
                ids[id_count++] = (int32_t)tid;
            }
        }
        toklist_free(&tl);
    }
    free(spans.items);
    *out_ids = ids;
    *out_count = id_count;
    return 1;
}

/* ================================================================== */
/*  Decode: token IDs -> UTF-8 text                                   */
/* ================================================================== */

static int token_text_to_bytes(const char *text, DynBuf *out) {
    const char *p = text;
    while (*p != '\0') {
        int cp = decode_utf8(&p);
        if (cp < 0) {
            continue;
        }
        if (cp < (int)(sizeof(g_unicode_to_byte) /
                       sizeof(g_unicode_to_byte[0])) &&
            g_unicode_to_byte[cp] >= 0) {
            dynbuf_push_byte(out, (char)g_unicode_to_byte[cp]);
        } else {
            char buf[4];
            int n = encode_utf8(cp, buf);
            dynbuf_push(out, buf, (size_t)n);
        }
    }
    return 1;
}

static int tokenizer_decode(const Tokenizer *tok,
                            const int32_t *ids,
                            int count,
                            char **out_text) {
    DynBuf buf;
    int i;

    if (tok == NULL || ids == NULL) {
        return 0;
    }
    dynbuf_init(&buf);

    for (i = 0; i < count; ++i) {
        int tid = (int)ids[i];
        const char *text;
        if (tid < 0 || tid >= tok->vocab_size) {
            continue;
        }
        if (tok->is_special[tid]) {
            continue;
        }
        text = tok->id_to_token[tid];
        if (text == NULL) {
            continue;
        }
        token_text_to_bytes(text, &buf);
    }
    dynbuf_push_byte(&buf, '\0');
    *out_text = buf.data;
    return 1;
}

/* ================================================================== */
/*  SQL function: llm_tokenize(text, tokenizer_json_path)             */
/*                                                                    */
/*  Returns BLOB of little-endian int32 token IDs.                    */
/* ================================================================== */

static void
sql_llm_tokenize(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    const char *text;
    const char *json_path;
    sqlite3 *db;
    Tokenizer *tok = NULL;
    int32_t *ids = NULL;
    int count = 0;

    if (argc != 2) {
        sqlite3_result_error(ctx,
                             "llm_tokenize requires 2 arguments: "
                             "text, tokenizer_json_path",
                             -1);
        return;
    }

    text = (const char *)sqlite3_value_text(argv[0]);
    json_path = (const char *)sqlite3_value_text(argv[1]);

    if (text == NULL || json_path == NULL) {
        sqlite3_result_error(ctx, "NULL argument", -1);
        return;
    }

    init_byte_unicode_mapping();

    db = sqlite3_context_db_handle(ctx);
    tok = tokenizer_create();
    if (tok == NULL) {
        sqlite3_result_error_nomem(ctx);
        return;
    }

    if (!load_vocab_from_conn(db, tok)) {
        tokenizer_free(tok);
        sqlite3_result_error(ctx, "failed to load vocab table", -1);
        return;
    }

    load_merges_from_json(json_path, tok);

    if (!tokenizer_encode(tok, text, &ids, &count)) {
        tokenizer_free(tok);
        sqlite3_result_error(ctx, "tokenizer encode failed", -1);
        return;
    }

    sqlite3_result_blob(
        ctx, ids, count * (int)sizeof(int32_t), SQLITE_TRANSIENT);
    free(ids);
    tokenizer_free(tok);
}

/* ================================================================== */
/*  SQL function: llm_detokenize(token_ids_blob)                      */
/*                                                                    */
/*  Reads vocab from the current db connection.                       */
/*  token_ids_blob is a BLOB of little-endian int32 token IDs.        */
/*  Returns decoded UTF-8 TEXT.                                       */
/* ================================================================== */

static void
sql_llm_detokenize(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    const void *blob;
    int blob_bytes;
    int count;
    sqlite3 *db;
    Tokenizer *tok = NULL;
    char *out_text = NULL;

    if (argc != 1) {
        sqlite3_result_error(ctx,
                             "llm_detokenize requires 1 argument: "
                             "token_ids_blob",
                             -1);
        return;
    }

    if (sqlite3_value_type(argv[0]) != SQLITE_BLOB) {
        sqlite3_result_error(ctx, "llm_detokenize: argument must be BLOB", -1);
        return;
    }

    blob = sqlite3_value_blob(argv[0]);
    blob_bytes = sqlite3_value_bytes(argv[0]);

    if (blob_bytes % (int)sizeof(int32_t) != 0) {
        sqlite3_result_error(ctx,
                             "llm_detokenize: BLOB size must be "
                             "multiple of 4",
                             -1);
        return;
    }

    count = blob_bytes / (int)sizeof(int32_t);

    init_byte_unicode_mapping();

    db = sqlite3_context_db_handle(ctx);
    tok = tokenizer_create();
    if (tok == NULL) {
        sqlite3_result_error_nomem(ctx);
        return;
    }

    if (!load_vocab_from_conn(db, tok)) {
        tokenizer_free(tok);
        sqlite3_result_error(ctx, "failed to load vocab table", -1);
        return;
    }

    if (!tokenizer_decode(tok, (const int32_t *)blob, count, &out_text)) {
        tokenizer_free(tok);
        sqlite3_result_error(ctx, "tokenizer decode failed", -1);
        return;
    }

    sqlite3_result_text(ctx, out_text, -1, SQLITE_TRANSIENT);
    free(out_text);
    tokenizer_free(tok);
}

/* ================================================================== */
/*  SQL function: llm_token_count(token_ids_blob)                     */
/*                                                                    */
/*  Returns the number of tokens in a BLOB.                           */
/* ================================================================== */

static void
sql_llm_token_count(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    int blob_bytes;

    if (argc != 1) {
        sqlite3_result_error(ctx, "llm_token_count requires 1 argument", -1);
        return;
    }

    if (sqlite3_value_type(argv[0]) != SQLITE_BLOB) {
        sqlite3_result_error(ctx, "llm_token_count: argument must be BLOB", -1);
        return;
    }

    blob_bytes = sqlite3_value_bytes(argv[0]);
    sqlite3_result_int(ctx, blob_bytes / (int)sizeof(int32_t));
}

/* ================================================================== */
/*  SQL function: llm_token_at(token_ids_blob, index)                 */
/*                                                                    */
/*  Returns the token ID at position *index* (0-based).               */
/* ================================================================== */

static void
sql_llm_token_at(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    const int32_t *ids;
    int blob_bytes, count, idx;

    if (argc != 2) {
        sqlite3_result_error(ctx,
                             "llm_token_at requires 2 arguments: "
                             "blob, index",
                             -1);
        return;
    }

    if (sqlite3_value_type(argv[0]) != SQLITE_BLOB) {
        sqlite3_result_error(ctx,
                             "llm_token_at: first argument must be "
                             "BLOB",
                             -1);
        return;
    }

    blob_bytes = sqlite3_value_bytes(argv[0]);
    count = blob_bytes / (int)sizeof(int32_t);
    idx = sqlite3_value_int(argv[1]);

    if (idx < 0 || idx >= count) {
        sqlite3_result_null(ctx);
        return;
    }

    ids = (const int32_t *)sqlite3_value_blob(argv[0]);
    sqlite3_result_int(ctx, (int)ids[idx]);
}

/* ================================================================== */
/*  Extension entry point                                             */
/* ================================================================== */

#ifdef _WIN32
__declspec(dllexport)
#elif defined(__GNUC__)
__attribute__((visibility("default")))
#endif
    int sqlite3_llm_tokenizer_init(sqlite3 *db,
                                   char **pzErrMsg,
                                   const sqlite3_api_routines *pApi) {
    int rc;
    (void)pzErrMsg;
    SQLITE_EXTENSION_INIT2(pApi);

    rc = sqlite3_create_function(db,
                                 "llm_tokenize",
                                 2,
                                 SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                                 NULL,
                                 sql_llm_tokenize,
                                 NULL,
                                 NULL);
    if (rc != SQLITE_OK) {
        return rc;
    }

    rc = sqlite3_create_function(db,
                                 "llm_detokenize",
                                 1,
                                 SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                                 NULL,
                                 sql_llm_detokenize,
                                 NULL,
                                 NULL);
    if (rc != SQLITE_OK) {
        return rc;
    }

    rc = sqlite3_create_function(db,
                                 "llm_token_count",
                                 1,
                                 SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                                 NULL,
                                 sql_llm_token_count,
                                 NULL,
                                 NULL);
    if (rc != SQLITE_OK) {
        return rc;
    }

    rc = sqlite3_create_function(db,
                                 "llm_token_at",
                                 2,
                                 SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                                 NULL,
                                 sql_llm_token_at,
                                 NULL,
                                 NULL);
    if (rc != SQLITE_OK) {
        return rc;
    }

    return SQLITE_OK;
}

/*
 * Alias: sqlite3_llmtokenizer_init
 *
 * SQLite derives the entry-point name from the filename by
 * stripping underscores:
 *   llm_tokenizer.so  →  sqlite3_llmtokenizer_init
 */
#ifdef _WIN32
__declspec(dllexport)
#elif defined(__GNUC__)
__attribute__((visibility("default")))
#endif
    int sqlite3_llmtokenizer_init(sqlite3 *db,
                                  char **pzErrMsg,
                                  const sqlite3_api_routines *pApi) {
    return sqlite3_llm_tokenizer_init(db, pzErrMsg, pApi);
}
