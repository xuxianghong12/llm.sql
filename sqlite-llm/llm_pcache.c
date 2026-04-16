/*
 * llm_pcache.c — Layer-aware SQLite page cache for LLM inference
 *
 * Implements sqlite3_pcache_methods2 with:
 *   - Layer-priority eviction (done-layer pages evicted first)
 *   - Hard memory budget (pages refused when budget exceeded)
 *   - Statistics collection for profiling
 *
 * The library is loaded as a .so and provides a C API (llm_pcache.h)
 * callable from Python via ctypes.  It uses sqlite3_config() to
 * register itself as SQLite's global page cache — no source
 * modifications to SQLite required.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <sqlite3.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <pthread.h>
#include <stdatomic.h>

#include "llm_pcache.h"

/* ── Compile-time limits ────────────────────────────────────────────────── */
#define LLM_PCACHE_MAX_LAYERS 256
/* Initial hash-table capacity (per cache instance) */
#define LLM_PCACHE_INIT_BUCKETS 1024

/* ── Visibility ─────────────────────────────────────────────────────────── */
#if defined(__GNUC__) || defined(__clang__)
#define LLM_EXPORT __attribute__((visibility("default")))
#else
#define LLM_EXPORT
#endif

/* ====================================================================== */
/*  Internal data structures                                               */
/* ====================================================================== */

/*
 * Each cached page has a small header (PageEntry) that lives alongside
 * the page data.  We store them in a hash table keyed by page number.
 */
typedef struct PageEntry PageEntry;
struct PageEntry {
    unsigned key;             /* page number (≥ 1)                    */
    int pinned;               /* 1 = pinned by SQLite                 */
    int layer;                /* layer that last touched this page    */
    sqlite3_pcache_page page; /* .pBuf → page data, .pExtra → extra   */
    PageEntry *hash_next;     /* hash chain                            */
    /* LRU doubly-linked list */
    PageEntry *lru_prev;
    PageEntry *lru_next;
};

/*
 * Per-cache-instance state (one per open DB file).
 */
typedef struct LlmPcache LlmPcache;
struct LlmPcache {
    int szPage;     /* page size (bytes)                         */
    int szExtra;    /* extra bytes per page entry                */
    int bPurgeable; /* cache is purgeable                        */
    int nMax;       /* max pages (from xCachesize)               */

    /* Hash table */
    PageEntry **buckets;
    int nBuckets;
    int nPage;   /* total pages in cache                      */
    int nPinned; /* pinned pages                              */

    /* LRU list (unpinned pages only).  lru_head = oldest (evict first). */
    PageEntry *lru_head;
    PageEntry *lru_tail;

    pthread_mutex_t lock;
};

/* ── Global state ───────────────────────────────────────────────────────── */

/*
 * Thread-local: when a thread is inside begin_layer_io / end_layer_io,
 * this holds the layer index for precise miss-only tagging.
 * -1 means "not in a layer I/O region" (fall back to global active_layer).
 */
static _Thread_local int tls_layer_io = -1;

static struct {
    /* Configuration (set once at install time) */
    int64_t budget_bytes;
    int num_layers;
    int policy;
    int prefetch_window;

    /* Layer state: 0 = unknown, 1 = active, 2 = done */
    int layer_state[LLM_PCACHE_MAX_LAYERS];
    int layer_prefetch_rank[LLM_PCACHE_MAX_LAYERS];
    int active_layer; /* -1 = none */

    /* Statistics (atomic for lock-free reads) */
    _Atomic int64_t hits;
    _Atomic int64_t misses;
    _Atomic int64_t evictions;
    _Atomic int64_t evictions_layer;
    _Atomic int64_t evictions_other;
    _Atomic int64_t mem_used; /* approximate current usage */

    /* Per-layer page counts (how many pages are currently in cache for
     * each layer).  Updated on miss (alloc) and eviction. */
    _Atomic int64_t pages_per_layer[LLM_PCACHE_MAX_LAYERS];

    /* Saved default methods for uninstall */
    sqlite3_pcache_methods2 default_methods;
    int installed;

    pthread_mutex_t global_lock;
} g_llm = {
    .budget_bytes = 0,
    .num_layers = 0,
    .policy = LLM_PCACHE_POLICY_LRU,
    .prefetch_window = 2,
    .active_layer = -1,
    .installed = 0,
    .global_lock = PTHREAD_MUTEX_INITIALIZER,
};

/* ====================================================================== */
/*  Hash-table helpers                                                     */
/* ====================================================================== */

static inline int hash_bucket(unsigned key, int nBuckets) {
    return (int)(key % (unsigned)nBuckets);
}

static PageEntry *ht_find(LlmPcache *c, unsigned key) {
    int b = hash_bucket(key, c->nBuckets);
    for (PageEntry *p = c->buckets[b]; p; p = p->hash_next) {
        if (p->key == key)
            return p;
    }
    return NULL;
}

static void ht_insert(LlmPcache *c, PageEntry *entry) {
    int b = hash_bucket(entry->key, c->nBuckets);
    entry->hash_next = c->buckets[b];
    c->buckets[b] = entry;
}

static void ht_remove(LlmPcache *c, PageEntry *entry) {
    int b = hash_bucket(entry->key, c->nBuckets);
    PageEntry **pp = &c->buckets[b];
    while (*pp) {
        if (*pp == entry) {
            *pp = entry->hash_next;
            entry->hash_next = NULL;
            return;
        }
        pp = &(*pp)->hash_next;
    }
}

/* ====================================================================== */
/*  LRU helpers                                                            */
/* ====================================================================== */

/* Remove an entry from the LRU list (must be unpinned). */
static void lru_remove(LlmPcache *c, PageEntry *entry) {
    if (entry->lru_prev)
        entry->lru_prev->lru_next = entry->lru_next;
    else
        c->lru_head = entry->lru_next;
    if (entry->lru_next)
        entry->lru_next->lru_prev = entry->lru_prev;
    else
        c->lru_tail = entry->lru_prev;
    entry->lru_prev = entry->lru_next = NULL;
}

/* Append to LRU tail (most recently unpinned). */
static void lru_push_back(LlmPcache *c, PageEntry *entry) {
    entry->lru_prev = c->lru_tail;
    entry->lru_next = NULL;
    if (c->lru_tail)
        c->lru_tail->lru_next = entry;
    else
        c->lru_head = entry;
    c->lru_tail = entry;
}

/* ====================================================================== */
/*  Eviction logic                                                         */
/* ====================================================================== */

/*
 * Return the layer index to assign to newly loaded (miss) pages.
 * Prefers the thread-local layer I/O index (set by begin_layer_io)
 * over the global active_layer.
 */
static inline int miss_layer_tag(void) {
    return (tls_layer_io >= 0) ? tls_layer_io : g_llm.active_layer;
}

/*
 * Increment the per-layer page counter for layer_tag.
 * Called when a new page is assigned to a layer.
 */
static inline void track_new_page(int layer_tag) {
    if (layer_tag >= 0 && layer_tag < LLM_PCACHE_MAX_LAYERS)
        atomic_fetch_add(&g_llm.pages_per_layer[layer_tag], 1);
}

static inline int normalize_policy(int policy) {
    return policy == LLM_PCACHE_POLICY_OPT ? LLM_PCACHE_POLICY_OPT
                                           : LLM_PCACHE_POLICY_LRU;
}

static inline int normalize_prefetch_window(int window) {
    if (window < 0)
        return 0;
    if (window > LLM_PCACHE_MAX_LAYERS)
        return LLM_PCACHE_MAX_LAYERS;
    return window;
}

/*
 * LRU baseline priority (lower = evict sooner):
 *   0 — done-layer page (already processed, safe to evict)
 *   1 — unknown-layer page (global params, embeddings)
 *   2 — active/future-layer page (needed soon, evict last)
 */
static int lru_eviction_priority(PageEntry *entry) {
    int layer = entry->layer;
    if (layer < 0 || layer >= g_llm.num_layers)
        return 1;
    int state = g_llm.layer_state[layer];
    if (state == 2)
        return 0; /* done */
    if (state == 1)
        return 2; /* active */
    /* For unknown layers, check if they are ahead of active */
    if (g_llm.active_layer >= 0 && layer > g_llm.active_layer)
        return 2;
    return 1;
}

/*
 * OPT bucket ordering for deterministic transformer layer access:
 *   0 — past/done layer page (no future reuse in this forward pass)
 *   1 — future layer page (evict farthest next-use first)
 *   2 — global/unknown page (fallback bucket; keep if possible)
 *   3 — current layer page (needed immediately, evict last)
 */
static int opt_eviction_bucket(PageEntry *entry) {
    int layer = entry->layer;
    if (layer < 0 || layer >= g_llm.num_layers)
        return 2;
    if (g_llm.active_layer < 0)
        return 2;
    if (g_llm.layer_state[layer] == 2 || layer < g_llm.active_layer)
        return 0;
    if (layer == g_llm.active_layer)
        return 3;
    if (g_llm.layer_prefetch_rank[layer] > 0)
        return 3;
    return 1;
}

static int opt_future_distance(PageEntry *entry) {
    if (g_llm.active_layer < 0)
        return -1;
    if (entry->layer < 0 || entry->layer >= g_llm.num_layers)
        return -1;
    if (entry->layer <= g_llm.active_layer)
        return -1;
    return entry->layer - g_llm.active_layer;
}

static int opt_prefetch_rank(PageEntry *entry) {
    if (entry->layer < 0 || entry->layer >= g_llm.num_layers)
        return -1;
    return g_llm.layer_prefetch_rank[entry->layer];
}

static PageEntry *choose_lru_candidate(LlmPcache *c, int *out_priority) {
    PageEntry *best = NULL;
    int best_prio = 999;

    for (PageEntry *p = c->lru_head; p; p = p->lru_next) {
        if (p->pinned)
            continue;
        int prio = lru_eviction_priority(p);
        if (prio < best_prio) {
            best = p;
            best_prio = prio;
            if (prio == 0)
                break;
        }
    }

    if (out_priority)
        *out_priority = best_prio;
    return best;
}

static PageEntry *choose_opt_candidate(LlmPcache *c, int *out_bucket) {
    PageEntry *best = NULL;
    int best_bucket = 999;
    int best_distance = -1;
    int best_prefetch_rank = -1;

    for (PageEntry *p = c->lru_head; p; p = p->lru_next) {
        if (p->pinned)
            continue;

        int bucket = opt_eviction_bucket(p);
        if (bucket < best_bucket) {
            best = p;
            best_bucket = bucket;
            best_distance = bucket == 1 ? opt_future_distance(p) : -1;
            best_prefetch_rank = bucket == 3 ? opt_prefetch_rank(p) : -1;
            if (bucket == 0)
                break;
            continue;
        }

        if (bucket != best_bucket)
            continue;

        if (bucket == 1) {
            int distance = opt_future_distance(p);
            if (distance > best_distance) {
                best = p;
                best_distance = distance;
            }
        } else if (bucket == 3) {
            int rank = opt_prefetch_rank(p);
            if (rank > best_prefetch_rank) {
                best = p;
                best_prefetch_rank = rank;
            }
        }
    }

    if (out_bucket)
        *out_bucket = best_bucket;
    return best;
}

/*
 * Find and evict one unpinned page according to the configured policy.
 *
 * Returns the freed PageEntry (ready to reuse) or NULL if nothing
 * can be evicted.
 */
static PageEntry *evict_one(LlmPcache *c) {
    PageEntry *best = NULL;
    int best_bucket = 999;

    if (g_llm.policy == LLM_PCACHE_POLICY_OPT)
        best = choose_opt_candidate(c, &best_bucket);
    else
        best = choose_lru_candidate(c, &best_bucket);

    if (!best)
        return NULL;

    /* Remove from LRU and hash table */
    lru_remove(c, best);
    ht_remove(c, best);
    c->nPage--;

    int64_t page_mem =
        (int64_t)(c->szPage + c->szExtra + (int)sizeof(PageEntry));
    atomic_fetch_sub(&g_llm.mem_used, page_mem);
    atomic_fetch_add(&g_llm.evictions, 1);
    if (best_bucket == 0)
        atomic_fetch_add(&g_llm.evictions_layer, 1);
    else
        atomic_fetch_add(&g_llm.evictions_other, 1);

    /* Decrement per-layer page count */
    int evicted_layer = best->layer;
    if (evicted_layer >= 0 && evicted_layer < LLM_PCACHE_MAX_LAYERS)
        atomic_fetch_sub(&g_llm.pages_per_layer[evicted_layer], 1);

    return best;
}

/* ====================================================================== */
/*  pcache_methods2 implementation                                         */
/* ====================================================================== */

static int llm_xInit(void *pArg) {
    (void)pArg;
    return SQLITE_OK;
}

static void llm_xShutdown(void *pArg) {
    (void)pArg;
}

static sqlite3_pcache *llm_xCreate(int szPage, int szExtra, int bPurgeable) {
    LlmPcache *c = (LlmPcache *)calloc(1, sizeof(LlmPcache));
    if (!c)
        return NULL;

    c->szPage = szPage;
    c->szExtra = szExtra;
    c->bPurgeable = bPurgeable;
    c->nMax = 1000; /* default; xCachesize will update */
    c->nBuckets = LLM_PCACHE_INIT_BUCKETS;
    c->buckets = (PageEntry **)calloc((size_t)c->nBuckets, sizeof(PageEntry *));
    if (!c->buckets) {
        free(c);
        return NULL;
    }
    pthread_mutex_init(&c->lock, NULL);
    return (sqlite3_pcache *)c;
}

static void llm_xCachesize(sqlite3_pcache *pCache, int nCachesize) {
    LlmPcache *c = (LlmPcache *)pCache;
    pthread_mutex_lock(&c->lock);
    c->nMax = nCachesize;
    pthread_mutex_unlock(&c->lock);
}

static int llm_xPagecount(sqlite3_pcache *pCache) {
    LlmPcache *c = (LlmPcache *)pCache;
    return c->nPage;
}

static sqlite3_pcache_page *
llm_xFetch(sqlite3_pcache *pCache, unsigned key, int createFlag) {
    LlmPcache *c = (LlmPcache *)pCache;
    pthread_mutex_lock(&c->lock);

    /* Try to find existing page */
    PageEntry *entry = ht_find(c, key);
    if (entry) {
        /* Cache hit */
        atomic_fetch_add(&g_llm.hits, 1);
        if (!entry->pinned) {
            lru_remove(c, entry);
            entry->pinned = 1;
            c->nPinned++;
        }
        /* Update layer tag — but NOT during precise layer I/O regions.
         * When tls_layer_io >= 0, a thread is loading a specific layer's
         * params; hits on shared pages (B-tree internals, index nodes)
         * should keep their original layer tag to avoid tag drift.
         */
        if (tls_layer_io < 0) {
            entry->layer = g_llm.active_layer;
        }
        pthread_mutex_unlock(&c->lock);
        return &entry->page;
    }

    /* Cache miss */
    atomic_fetch_add(&g_llm.misses, 1);

    if (createFlag == 0) {
        pthread_mutex_unlock(&c->lock);
        return NULL;
    }

    /* Check memory budget */
    int64_t page_mem =
        (int64_t)(c->szPage + c->szExtra + (int)sizeof(PageEntry));
    int64_t current = atomic_load(&g_llm.mem_used);
    if (g_llm.budget_bytes > 0 && current + page_mem > g_llm.budget_bytes) {
        /* Try to evict to make room */
        PageEntry *evicted = evict_one(c);
        if (evicted) {
            /* Reuse the evicted entry's memory */
            memset(evicted->page.pBuf, 0, (size_t)c->szPage);
            memset(evicted->page.pExtra, 0, (size_t)c->szExtra);
            evicted->key = key;
            evicted->pinned = 1;
            evicted->layer = miss_layer_tag();
            track_new_page(evicted->layer);
            evicted->hash_next = NULL;
            evicted->lru_prev = evicted->lru_next = NULL;
            ht_insert(c, evicted);
            c->nPage++;
            c->nPinned++;
            atomic_fetch_add(&g_llm.mem_used, page_mem);
            pthread_mutex_unlock(&c->lock);
            return &evicted->page;
        }
        if (createFlag == 1) {
            /* createFlag=1: return NULL if allocation is not easy */
            pthread_mutex_unlock(&c->lock);
            return NULL;
        }
        /* createFlag=2: must try harder — allow over-budget */
    }

    /* Check if we're over nMax and need to evict */
    if (c->bPurgeable && c->nPage >= c->nMax) {
        PageEntry *evicted = evict_one(c);
        if (evicted) {
            memset(evicted->page.pBuf, 0, (size_t)c->szPage);
            memset(evicted->page.pExtra, 0, (size_t)c->szExtra);
            evicted->key = key;
            evicted->pinned = 1;
            evicted->layer = miss_layer_tag();
            track_new_page(evicted->layer);
            evicted->hash_next = NULL;
            evicted->lru_prev = evicted->lru_next = NULL;
            ht_insert(c, evicted);
            c->nPage++;
            c->nPinned++;
            atomic_fetch_add(&g_llm.mem_used, page_mem);
            pthread_mutex_unlock(&c->lock);
            return &evicted->page;
        }
        if (createFlag == 1) {
            pthread_mutex_unlock(&c->lock);
            return NULL;
        }
    }

    /* Allocate new page */
    size_t alloc_sz =
        sizeof(PageEntry) + (size_t)c->szPage + (size_t)c->szExtra;
    entry = (PageEntry *)calloc(1, alloc_sz);
    if (!entry) {
        pthread_mutex_unlock(&c->lock);
        return NULL;
    }

    /* Layout: [PageEntry][page_data][extra_data] */
    entry->page.pBuf = (char *)entry + sizeof(PageEntry);
    entry->page.pExtra = (char *)entry + sizeof(PageEntry) + c->szPage;
    entry->key = key;
    entry->pinned = 1;
    entry->layer = miss_layer_tag();
    track_new_page(entry->layer);

    ht_insert(c, entry);
    c->nPage++;
    c->nPinned++;
    atomic_fetch_add(&g_llm.mem_used, page_mem);

    pthread_mutex_unlock(&c->lock);
    return &entry->page;
}

static void
llm_xUnpin(sqlite3_pcache *pCache, sqlite3_pcache_page *pPage, int discard) {
    LlmPcache *c = (LlmPcache *)pCache;
    /* pPage is embedded in PageEntry; recover the container */
    PageEntry *entry = (PageEntry *)((char *)pPage - offsetof(PageEntry, page));

    pthread_mutex_lock(&c->lock);

    if (discard) {
        /* Remove and free the page */
        if (!entry->pinned) {
            lru_remove(c, entry);
        }
        ht_remove(c, entry);
        c->nPage--;
        if (entry->pinned)
            c->nPinned--;
        int64_t page_mem =
            (int64_t)(c->szPage + c->szExtra + (int)sizeof(PageEntry));
        atomic_fetch_sub(&g_llm.mem_used, page_mem);
        int layer = entry->layer;
        if (layer >= 0 && layer < LLM_PCACHE_MAX_LAYERS)
            atomic_fetch_sub(&g_llm.pages_per_layer[layer], 1);
        free(entry);
    } else {
        /* Unpin: add to LRU tail */
        if (entry->pinned) {
            entry->pinned = 0;
            c->nPinned--;
            lru_push_back(c, entry);
        }
    }

    pthread_mutex_unlock(&c->lock);
}

static void llm_xRekey(sqlite3_pcache *pCache,
                       sqlite3_pcache_page *pPage,
                       unsigned oldKey,
                       unsigned newKey) {
    LlmPcache *c = (LlmPcache *)pCache;
    PageEntry *entry = (PageEntry *)((char *)pPage - offsetof(PageEntry, page));

    pthread_mutex_lock(&c->lock);
    if (entry->key == oldKey) {
        ht_remove(c, entry);
        entry->key = newKey;
        ht_insert(c, entry);
    }
    pthread_mutex_unlock(&c->lock);
}

static void llm_xTruncate(sqlite3_pcache *pCache, unsigned iLimit) {
    LlmPcache *c = (LlmPcache *)pCache;

    pthread_mutex_lock(&c->lock);

    /* Scan all buckets and remove pages with key >= iLimit */
    for (int b = 0; b < c->nBuckets; b++) {
        PageEntry **pp = &c->buckets[b];
        while (*pp) {
            PageEntry *entry = *pp;
            if (entry->key >= iLimit) {
                *pp = entry->hash_next;
                if (!entry->pinned) {
                    lru_remove(c, entry);
                }
                int64_t page_mem =
                    (int64_t)(c->szPage + c->szExtra + (int)sizeof(PageEntry));
                atomic_fetch_sub(&g_llm.mem_used, page_mem);
                if (entry->pinned)
                    c->nPinned--;
                c->nPage--;
                int layer = entry->layer;
                if (layer >= 0 && layer < LLM_PCACHE_MAX_LAYERS)
                    atomic_fetch_sub(&g_llm.pages_per_layer[layer], 1);
                free(entry);
            } else {
                pp = &entry->hash_next;
            }
        }
    }

    pthread_mutex_unlock(&c->lock);
}

static void llm_xDestroy(sqlite3_pcache *pCache) {
    LlmPcache *c = (LlmPcache *)pCache;

    pthread_mutex_lock(&c->lock);

    /* Free all pages */
    for (int b = 0; b < c->nBuckets; b++) {
        PageEntry *entry = c->buckets[b];
        while (entry) {
            PageEntry *next = entry->hash_next;
            int64_t page_mem =
                (int64_t)(c->szPage + c->szExtra + (int)sizeof(PageEntry));
            atomic_fetch_sub(&g_llm.mem_used, page_mem);
            int layer = entry->layer;
            if (layer >= 0 && layer < LLM_PCACHE_MAX_LAYERS)
                atomic_fetch_sub(&g_llm.pages_per_layer[layer], 1);
            free(entry);
            entry = next;
        }
    }

    pthread_mutex_unlock(&c->lock);
    pthread_mutex_destroy(&c->lock);
    free(c->buckets);
    free(c);
}

static void llm_xShrink(sqlite3_pcache *pCache) {
    LlmPcache *c = (LlmPcache *)pCache;

    pthread_mutex_lock(&c->lock);

    /* Evict all unpinned done-layer pages first, then any unpinned */
    while (c->lru_head) {
        PageEntry *entry = c->lru_head;
        if (entry->pinned)
            break;
        lru_remove(c, entry);
        ht_remove(c, entry);
        int64_t page_mem =
            (int64_t)(c->szPage + c->szExtra + (int)sizeof(PageEntry));
        atomic_fetch_sub(&g_llm.mem_used, page_mem);
        atomic_fetch_add(&g_llm.evictions, 1);
        int layer = entry->layer;
        if (layer >= 0 && layer < LLM_PCACHE_MAX_LAYERS)
            atomic_fetch_sub(&g_llm.pages_per_layer[layer], 1);
        c->nPage--;
        free(entry);
    }

    pthread_mutex_unlock(&c->lock);
}

/* ====================================================================== */
/*  Public API                                                             */
/* ====================================================================== */

LLM_EXPORT int llm_pcache_set_policy(int policy) {
    pthread_mutex_lock(&g_llm.global_lock);
    g_llm.policy = normalize_policy(policy);
    int normalized = g_llm.policy;
    pthread_mutex_unlock(&g_llm.global_lock);
    return normalized;
}

LLM_EXPORT int llm_pcache_get_policy(void) {
    return g_llm.policy;
}

LLM_EXPORT int llm_pcache_set_prefetch_window(int window) {
    pthread_mutex_lock(&g_llm.global_lock);
    g_llm.prefetch_window = normalize_prefetch_window(window);
    int normalized = g_llm.prefetch_window;
    pthread_mutex_unlock(&g_llm.global_lock);
    return normalized;
}

LLM_EXPORT int llm_pcache_get_prefetch_window(void) {
    return g_llm.prefetch_window;
}

LLM_EXPORT int llm_pcache_install(int64_t budget_bytes, int num_layers) {
    pthread_mutex_lock(&g_llm.global_lock);

    if (g_llm.installed) {
        pthread_mutex_unlock(&g_llm.global_lock);
        return -1; /* already installed */
    }

    g_llm.budget_bytes = budget_bytes;
    g_llm.num_layers = (num_layers > LLM_PCACHE_MAX_LAYERS)
                           ? LLM_PCACHE_MAX_LAYERS
                           : num_layers;
    g_llm.active_layer = -1;
    memset(g_llm.layer_state, 0, sizeof(g_llm.layer_state));
    memset(g_llm.layer_prefetch_rank, 0, sizeof(g_llm.layer_prefetch_rank));

    /* Reset stats */
    atomic_store(&g_llm.hits, 0);
    atomic_store(&g_llm.misses, 0);
    atomic_store(&g_llm.evictions, 0);
    atomic_store(&g_llm.evictions_layer, 0);
    atomic_store(&g_llm.evictions_other, 0);
    atomic_store(&g_llm.mem_used, 0);

    memset((void *)g_llm.pages_per_layer, 0, sizeof(g_llm.pages_per_layer));
    int rc = sqlite3_config(SQLITE_CONFIG_GETPCACHE2, &g_llm.default_methods);
    if (rc != SQLITE_OK) {
        /* SQLite may already be initialized — need to shut down first */
        sqlite3_shutdown();
        rc = sqlite3_config(SQLITE_CONFIG_GETPCACHE2, &g_llm.default_methods);
        if (rc != SQLITE_OK) {
            pthread_mutex_unlock(&g_llm.global_lock);
            return rc;
        }
    }

    /* Register our custom pcache */
    sqlite3_pcache_methods2 methods;
    memset(&methods, 0, sizeof(methods));
    methods.iVersion = 1;
    methods.pArg = NULL;
    methods.xInit = llm_xInit;
    methods.xShutdown = llm_xShutdown;
    methods.xCreate = llm_xCreate;
    methods.xCachesize = llm_xCachesize;
    methods.xPagecount = llm_xPagecount;
    methods.xFetch = llm_xFetch;
    methods.xUnpin = llm_xUnpin;
    methods.xRekey = llm_xRekey;
    methods.xTruncate = llm_xTruncate;
    methods.xDestroy = llm_xDestroy;
    methods.xShrink = llm_xShrink;

    sqlite3_shutdown();
    rc = sqlite3_config(SQLITE_CONFIG_PCACHE2, &methods);
    if (rc != SQLITE_OK) {
        pthread_mutex_unlock(&g_llm.global_lock);
        return rc;
    }

    sqlite3_initialize();
    g_llm.installed = 1;
    pthread_mutex_unlock(&g_llm.global_lock);
    return 0;
}

LLM_EXPORT int llm_pcache_uninstall(void) {
    pthread_mutex_lock(&g_llm.global_lock);
    if (!g_llm.installed) {
        pthread_mutex_unlock(&g_llm.global_lock);
        return 0;
    }

    sqlite3_shutdown();
    int rc = sqlite3_config(SQLITE_CONFIG_PCACHE2, &g_llm.default_methods);
    sqlite3_initialize();

    g_llm.installed = 0;
    pthread_mutex_unlock(&g_llm.global_lock);
    return rc;
}

LLM_EXPORT void llm_pcache_set_active_layer(int layer_idx) {
    if (layer_idx >= 0 && layer_idx < g_llm.num_layers) {
        g_llm.layer_state[layer_idx] = 1; /* active */
    }
    memset(g_llm.layer_prefetch_rank, 0, sizeof(g_llm.layer_prefetch_rank));
    g_llm.active_layer = layer_idx;
}

LLM_EXPORT void llm_pcache_mark_layer_done(int layer_idx) {
    if (layer_idx >= 0 && layer_idx < g_llm.num_layers) {
        g_llm.layer_state[layer_idx] = 2; /* done */
        g_llm.layer_prefetch_rank[layer_idx] = 0;
    }
}

LLM_EXPORT void llm_pcache_prefetch_layer(int layer_idx) {
    /* Mark a future layer as "active" so its pages get high priority
     * during eviction, even before set_active_layer() is called.
     * This is a hint that the layer will be needed soon.
     */
    if (layer_idx >= 0 && layer_idx < g_llm.num_layers) {
        if (g_llm.layer_state[layer_idx] == 0) {
            g_llm.layer_state[layer_idx] = 1; /* active/upcoming */
        }
        if (g_llm.active_layer >= 0 && layer_idx > g_llm.active_layer) {
            int distance = layer_idx - g_llm.active_layer;
            if (distance <= g_llm.prefetch_window) {
                g_llm.layer_prefetch_rank[layer_idx] = distance;
            }
        }
    }
}

LLM_EXPORT void llm_pcache_reset_layers(void) {
    memset(g_llm.layer_state, 0, sizeof(g_llm.layer_state));
    memset(g_llm.layer_prefetch_rank, 0, sizeof(g_llm.layer_prefetch_rank));
    g_llm.active_layer = -1;
}

LLM_EXPORT void llm_pcache_begin_layer_io(int layer_idx) {
    /* Enter a precise layer I/O region.  New pages (misses) will be
     * tagged with layer_idx regardless of the global active_layer.
     * Hit pages keep their existing tag to avoid drift on shared
     * B-tree internal / index pages.
     */
    tls_layer_io = layer_idx;
}

LLM_EXPORT void llm_pcache_end_layer_io(void) {
    /* Leave the precise layer I/O region.  Subsequent misses revert
     * to using the global active_layer for tagging.
     */
    tls_layer_io = -1;
}

LLM_EXPORT void llm_pcache_get_stats(llm_pcache_stats_t *stats) {
    if (!stats)
        return;
    stats->cache_hits = atomic_load(&g_llm.hits);
    stats->cache_misses = atomic_load(&g_llm.misses);
    stats->evictions = atomic_load(&g_llm.evictions);
    stats->evictions_layer = atomic_load(&g_llm.evictions_layer);
    stats->evictions_other = atomic_load(&g_llm.evictions_other);
    stats->memory_used = atomic_load(&g_llm.mem_used);
    stats->memory_budget = g_llm.budget_bytes;
    stats->active_layer = g_llm.active_layer;
    stats->num_layers = g_llm.num_layers;
    stats->policy = g_llm.policy;
    stats->prefetch_window = g_llm.prefetch_window;
    /* pages_current and pages_pinned are per-instance — report global mem */
    stats->pages_current = 0;
    stats->pages_pinned = 0;
}

LLM_EXPORT void llm_pcache_reset_stats(void) {
    atomic_store(&g_llm.hits, 0);
    atomic_store(&g_llm.misses, 0);
    atomic_store(&g_llm.evictions, 0);
    atomic_store(&g_llm.evictions_layer, 0);
    atomic_store(&g_llm.evictions_other, 0);
}

LLM_EXPORT int64_t llm_pcache_layer_pages(int layer_idx) {
    /* Return the number of pages currently in cache for layer_idx. */
    if (layer_idx < 0 || layer_idx >= LLM_PCACHE_MAX_LAYERS)
        return 0;
    return atomic_load(&g_llm.pages_per_layer[layer_idx]);
}
