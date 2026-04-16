/*
 * llm_param_mgr.c - Layer-aware Parameter Manager Implementation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llm_param_mgr.h"
#include "llm_pcache.h"
#include <sqlite3.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <regex.h>

/* ── Data Structures ────────────────────────────────────────────────────── */

/* Parameter cache entry */
typedef struct {
    char *name;    /* Parameter name (owned) */
    void *data;    /* Parameter data (owned or borrowed from pcache) */
    size_t size;   /* Data size in bytes */
    int layer_id;  /* Layer index (-1 for non-layer params) */
    int ref_count; /* Reference count (0 = can evict) */
    int is_loaded; /* 1 if data is loaded, 0 otherwise */
} param_entry_t;

/* Parameter manager */
struct param_manager {
    sqlite3 *db;       /* Database connection */
    int num_layers;    /* Number of transformer layers */
    int current_layer; /* Currently active layer */

    /* Parameter cache (hash table) */
    param_entry_t *entries; /* Array of entries */
    int num_entries;        /* Number of entries */
    int capacity;           /* Capacity of entries array */

    /* Layer index (for fast layer-based queries) */
    int *layer_start; /* layer_start[i] = first entry index for layer i */
    int *layer_count; /* layer_count[i] = number of entries for layer i */

    /* Memory budget */
    size_t budget_bytes; /* Memory budget in bytes (0 = unlimited) */
    size_t used_bytes;   /* Currently used bytes */

    /* Regex for layer extraction */
    regex_t layer_regex; /* Compiled regex: layers[._](\d+)[._] */
};

/* ── Helper Functions ───────────────────────────────────────────────────── */

/* Extract layer ID from parameter name */
static int extract_layer_id(param_manager_t *mgr, const char *name) {
    regmatch_t matches[2];
    if (regexec(&mgr->layer_regex, name, 2, matches, 0) == 0) {
        /* Extract the captured group (layer number) */
        int start = matches[1].rm_so;
        int end = matches[1].rm_eo;
        char buf[16];
        int len = end - start;
        if (len > 0 && len < 16) {
            memcpy(buf, name + start, len);
            buf[len] = '\0';
            return atoi(buf);
        }
    }
    return -1; /* Non-layer parameter */
}

/* Hash function for parameter names */
static unsigned int hash_string(const char *str) {
    unsigned int hash = 5381;
    int c;
    while ((c = *str++))
        hash = ((hash << 5) + hash) + c;
    return hash;
}

/* Find entry by name (linear probe hash table) */
static param_entry_t *find_entry(param_manager_t *mgr, const char *name) {
    if (mgr->num_entries == 0)
        return NULL;

    unsigned int hash = hash_string(name);
    int idx = hash % mgr->capacity;
    int probe = 0;

    while (probe < mgr->capacity) {
        param_entry_t *entry = &mgr->entries[idx];
        if (entry->name == NULL) {
            return NULL; /* Empty slot, not found */
        }
        if (strcmp(entry->name, name) == 0) {
            return entry; /* Found */
        }
        idx = (idx + 1) % mgr->capacity;
        probe++;
    }
    return NULL; /* Not found */
}

/* Insert entry into hash table */
static param_entry_t *insert_entry(param_manager_t *mgr, const char *name) {
    /* Resize if load factor > 0.7 */
    if (mgr->num_entries * 10 > mgr->capacity * 7) {
        int new_capacity = mgr->capacity * 2;
        param_entry_t *new_entries =
            calloc(new_capacity, sizeof(param_entry_t));
        if (!new_entries)
            return NULL;

        /* Rehash */
        for (int i = 0; i < mgr->capacity; i++) {
            if (mgr->entries[i].name != NULL) {
                unsigned int hash = hash_string(mgr->entries[i].name);
                int idx = hash % new_capacity;
                while (new_entries[idx].name != NULL) {
                    idx = (idx + 1) % new_capacity;
                }
                new_entries[idx] = mgr->entries[i];
            }
        }
        free(mgr->entries);
        mgr->entries = new_entries;
        mgr->capacity = new_capacity;
    }

    /* Find empty slot */
    unsigned int hash = hash_string(name);
    int idx = hash % mgr->capacity;
    while (mgr->entries[idx].name != NULL) {
        idx = (idx + 1) % mgr->capacity;
    }

    /* Initialize entry */
    mgr->entries[idx].name = strdup(name);
    mgr->entries[idx].data = NULL;
    mgr->entries[idx].size = 0;
    mgr->entries[idx].layer_id = extract_layer_id(mgr, name);
    mgr->entries[idx].ref_count = 0;
    mgr->entries[idx].is_loaded = 0;
    mgr->num_entries++;

    return &mgr->entries[idx];
}

/* Evict least recently used entries to free memory */
static void evict_lru_entries(param_manager_t *mgr, size_t bytes_needed) {
    /* Simple strategy: evict all entries with ref_count == 0 */
    for (int i = 0; i < mgr->capacity; i++) {
        if (mgr->entries[i].name != NULL && mgr->entries[i].ref_count == 0 &&
            mgr->entries[i].is_loaded) {

            free(mgr->entries[i].data);
            mgr->entries[i].data = NULL;
            mgr->entries[i].is_loaded = 0;
            mgr->used_bytes -= mgr->entries[i].size;

            if (mgr->used_bytes + bytes_needed <= mgr->budget_bytes) {
                return; /* Freed enough */
            }
        }
    }
}

/* ── Public API ─────────────────────────────────────────────────────────── */

param_manager_t *
param_manager_create(const char *db_path, int num_layers, size_t budget_mb) {
    param_manager_t *mgr = calloc(1, sizeof(param_manager_t));
    if (!mgr)
        return NULL;

    /* Open database */
    int rc = sqlite3_open(db_path, &mgr->db);
    if (rc != SQLITE_OK) {
        fprintf(
            stderr, "Failed to open database: %s\n", sqlite3_errmsg(mgr->db));
        free(mgr);
        return NULL;
    }

    /* Set pragmas for read-only inference */
    sqlite3_exec(mgr->db, "PRAGMA journal_mode=OFF", NULL, NULL, NULL);
    sqlite3_exec(mgr->db, "PRAGMA synchronous=OFF", NULL, NULL, NULL);
    sqlite3_exec(mgr->db, "PRAGMA locking_mode=EXCLUSIVE", NULL, NULL, NULL);
    sqlite3_exec(mgr->db, "PRAGMA temp_store=MEMORY", NULL, NULL, NULL);
    sqlite3_exec(
        mgr->db, "PRAGMA mmap_size=0", NULL, NULL, NULL); /* Force pcache */

    mgr->num_layers = num_layers;
    mgr->current_layer = -1;
    mgr->budget_bytes = budget_mb * 1024 * 1024;
    mgr->used_bytes = 0;

    /* Initialize hash table */
    mgr->capacity = 1024; /* Initial capacity */
    mgr->entries = calloc(mgr->capacity, sizeof(param_entry_t));
    mgr->num_entries = 0;

    /* Initialize layer index */
    mgr->layer_start =
        calloc(num_layers + 1, sizeof(int)); /* +1 for layer -1 */
    mgr->layer_count = calloc(num_layers + 1, sizeof(int));

    /* Compile regex for layer extraction */
    if (regcomp(&mgr->layer_regex, "layers[._]([0-9]+)[._]", REG_EXTENDED) !=
        0) {
        fprintf(stderr, "Failed to compile regex\n");
        param_manager_destroy(mgr);
        return NULL;
    }

    /* Build parameter index */
    sqlite3_stmt *stmt;
    rc = sqlite3_prepare_v2(mgr->db,
                            "SELECT name FROM model_params ORDER BY name",
                            -1,
                            &stmt,
                            NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr,
                "Failed to prepare statement: %s\n",
                sqlite3_errmsg(mgr->db));
        param_manager_destroy(mgr);
        return NULL;
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *name = (const char *)sqlite3_column_text(stmt, 0);
        param_entry_t *entry = insert_entry(mgr, name);
        if (!entry) {
            fprintf(stderr, "Failed to insert entry: %s\n", name);
            sqlite3_finalize(stmt);
            param_manager_destroy(mgr);
            return NULL;
        }
    }
    sqlite3_finalize(stmt);

    /* Build layer index */
    for (int i = 0; i < mgr->capacity; i++) {
        if (mgr->entries[i].name != NULL) {
            int lid = mgr->entries[i].layer_id + 1; /* +1 to handle -1 */
            mgr->layer_count[lid]++;
        }
    }

    /* Compute layer_start from layer_count */
    int offset = 0;
    for (int i = 0; i <= num_layers; i++) {
        mgr->layer_start[i] = offset;
        offset += mgr->layer_count[i];
    }

    return mgr;
}

void param_manager_destroy(param_manager_t *mgr) {
    if (!mgr)
        return;

    /* Free entries */
    if (mgr->entries) {
        for (int i = 0; i < mgr->capacity; i++) {
            if (mgr->entries[i].name) {
                free(mgr->entries[i].name);
                if (mgr->entries[i].data) {
                    free(mgr->entries[i].data);
                }
            }
        }
        free(mgr->entries);
    }

    free(mgr->layer_start);
    free(mgr->layer_count);
    regfree(&mgr->layer_regex);

    if (mgr->db) {
        sqlite3_close(mgr->db);
    }

    free(mgr);
}

const void *
param_manager_get(param_manager_t *mgr, const char *name, size_t *out_size) {
    param_entry_t *entry = find_entry(mgr, name);
    if (!entry) {
        *out_size = 0;
        return NULL;
    }

    /* If not loaded, load it now */
    if (!entry->is_loaded) {
        sqlite3_stmt *stmt;
        int rc =
            sqlite3_prepare_v2(mgr->db,
                               "SELECT data FROM model_params WHERE name = ?",
                               -1,
                               &stmt,
                               NULL);
        if (rc != SQLITE_OK) {
            *out_size = 0;
            return NULL;
        }

        sqlite3_bind_text(stmt, 1, name, -1, SQLITE_STATIC);

        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const void *blob = sqlite3_column_blob(stmt, 0);
            int size = sqlite3_column_bytes(stmt, 0);

            /* Check budget */
            if (mgr->budget_bytes > 0 &&
                mgr->used_bytes + size > mgr->budget_bytes) {
                evict_lru_entries(mgr, size);
            }

            /* Copy data (TODO: zero-copy via sqlite3_blob_open) */
            entry->data = malloc(size);
            if (entry->data) {
                memcpy(entry->data, blob, size);
                entry->size = size;
                entry->is_loaded = 1;
                entry->ref_count = 1;
                mgr->used_bytes += size;
            }
        }

        sqlite3_finalize(stmt);
    } else {
        /* Already loaded, increment ref count */
        entry->ref_count++;
    }

    *out_size = entry->size;
    return entry->data;
}

int param_manager_load_layer(param_manager_t *mgr, int layer_id) {
    /* Note: pcache functions are optional and may not be available
     * We skip calling them to avoid potential issues */

    /* Load all parameters for this layer */
    char pattern[64];
    if (layer_id >= 0) {
        snprintf(pattern, sizeof(pattern), "layers.%d.%%", layer_id);
    } else {
        /* Non-layer params: load embed, norm, lm_head */
        pattern[0] = '\0';
    }

    sqlite3_stmt *stmt;
    const char *sql =
        (layer_id >= 0)
            ? "SELECT name, data FROM model_params WHERE name LIKE ?"
            : "SELECT name, data FROM model_params WHERE name NOT LIKE "
              "'layers.%'";

    int rc = sqlite3_prepare_v2(mgr->db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        return -1;
    }

    if (layer_id >= 0) {
        sqlite3_bind_text(stmt, 1, pattern, -1, SQLITE_STATIC);
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char *name = (const char *)sqlite3_column_text(stmt, 0);
        const void *blob = sqlite3_column_blob(stmt, 1);
        int size = sqlite3_column_bytes(stmt, 1);

        param_entry_t *entry = find_entry(mgr, name);
        if (entry && !entry->is_loaded) {
            /* Check budget */
            if (mgr->budget_bytes > 0 &&
                mgr->used_bytes + size > mgr->budget_bytes) {
                evict_lru_entries(mgr, size);
            }

            /* Copy data */
            entry->data = malloc(size);
            if (entry->data) {
                memcpy(entry->data, blob, size);
                entry->size = size;
                entry->is_loaded = 1;
                entry->ref_count = 1;
                mgr->used_bytes += size;
            }
        }
    }

    sqlite3_finalize(stmt);
    mgr->current_layer = layer_id;

    return 0;
}

void param_manager_evict_layer(param_manager_t *mgr, int layer_id) {
    /* Note: pcache functions are optional, skip calling them */

    /* Mark all entries for this layer as evictable (ref_count = 0) */
    for (int i = 0; i < mgr->capacity; i++) {
        if (mgr->entries[i].name != NULL &&
            mgr->entries[i].layer_id == layer_id) {
            mgr->entries[i].ref_count = 0;
        }
    }

    /* Only evict if over budget */
    if (mgr->budget_bytes > 0 && mgr->used_bytes > mgr->budget_bytes) {
        evict_lru_entries(mgr, 0);
    }
}

void param_manager_reset(param_manager_t *mgr) {
    /* Note: pcache functions are optional, skip calling them */

    /* Reset ref counts but keep loaded data */
    for (int i = 0; i < mgr->capacity; i++) {
        if (mgr->entries[i].name != NULL) {
            mgr->entries[i].ref_count = 0;
        }
    }

    mgr->current_layer = -1;
}

void param_manager_get_stats(param_manager_t *mgr,
                             double *out_used_mb,
                             double *out_budget_mb) {
    *out_used_mb = mgr->used_bytes / (1024.0 * 1024.0);
    *out_budget_mb = mgr->budget_bytes / (1024.0 * 1024.0);
}

void param_manager_set_active_window(param_manager_t *mgr,
                                     int current_layer,
                                     int lookahead) {
    /* Mark window layers as protected */
    for (int i = 0; i < mgr->capacity; i++) {
        if (mgr->entries[i].name != NULL) {
            int lid = mgr->entries[i].layer_id;
            if (lid >= current_layer && lid < current_layer + lookahead) {
                mgr->entries[i].ref_count = 1; /* Protected */
            } else {
                mgr->entries[i].ref_count = 0; /* Evictable */
            }
        }
    }

    /* Evict if over budget */
    if (mgr->budget_bytes > 0 && mgr->used_bytes > mgr->budget_bytes) {
        evict_lru_entries(mgr, 0);
    }
}
