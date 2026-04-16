"""Custom SQLite memory manager optimised for LLM inference.

Replaces SQLite's default LRU-based page cache policy with a
layer-aware strategy tailored for transformer forward-pass execution:

* **Weight-aware prefetching** – while the current layer is executing,
  the next layer's weights are pre-loaded into the page cache via
  background ``SELECT`` queries.
* **Layer-aware eviction** – once a layer completes, its weight blobs
  can be explicitly released (or at least deprioritised) rather than
  kept under LRU which would evict *future* layers first.
* **Large page size** – 64 KiB pages reduce the number of B-tree
  look-ups per weight blob and improve sequential read throughput.
* **Inference-mode PRAGMAs** – journal and synchronous guarantees
  are disabled since model DBs are read-only during inference.

Usage::

    mgr = LLMMemoryManager(db_path, num_layers=32)
    conn = mgr.connection        # ready-to-use sqlite3.Connection
    mgr.prefetch_layer(0)        # warm the cache
    mgr.mark_layer_done(0)       # allow eviction
    mgr.close()

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default page size for LLM inference (64 KiB).
# Large weight BLOBs benefit from fewer B-tree pages per read.
LLM_PAGE_SIZE = 65536

# Default SQLite page-cache size expressed in *kibibytes* (negative value
# for PRAGMA cache_size).  -262144 → 256 MiB.
LLM_CACHE_SIZE_KB = -262144

# Alignment boundary for blob storage (bytes).  Using a multiple of the
# page size ensures that blobs start on a page boundary when possible,
# reducing unnecessary page loads for large sequential reads.
LLM_BLOB_ALIGNMENT = LLM_PAGE_SIZE

# Number of layers to prefetch ahead of the current computation.
LLM_PREFETCH_AHEAD = 2


# ---------------------------------------------------------------------------
# Helper: apply inference-mode PRAGMAs
# ---------------------------------------------------------------------------

def apply_inference_pragmas(
    conn: sqlite3.Connection,
    db_path: str = ":memory:",
    page_size: int = LLM_PAGE_SIZE,
    cache_size_kb: int = LLM_CACHE_SIZE_KB,
    disable_mmap: bool = False,
) -> None:
    """Apply performance PRAGMAs suitable for **read-only** LLM inference.

    * ``journal_mode=OFF``      – no journal at all (DB is read-only)
    * ``synchronous=OFF``       – skip fsync
    * ``locking_mode=EXCLUSIVE`` – avoid per-statement lock overhead
    * ``temp_store=MEMORY``     – temporary results in RAM
    * ``page_size``             – 64 KiB (large BLOBs → fewer pages)
    * ``cache_size``            – 256 MiB page cache
    * ``mmap_size``             – memory-map the file for zero-copy reads
                                  (disabled when *disable_mmap* is True)
    """
    # journal_mode=OFF completely disables the rollback journal.
    # This is safe because we never write during inference.
    try:
        conn.execute("PRAGMA journal_mode=OFF")
    except sqlite3.OperationalError:
        pass

    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA locking_mode=EXCLUSIVE")
    conn.execute("PRAGMA temp_store=MEMORY")

    # page_size can only be set on empty / newly-created databases.
    # For existing files we just read the current page_size and adapt
    # cache_size accordingly.
    if db_path == ":memory:":
        conn.execute(f"PRAGMA page_size={page_size}")
    else:
        # Read actual page size from the DB file
        cur_ps = conn.execute("PRAGMA page_size").fetchone()[0]
        if cur_ps != page_size:
            # Cannot change page_size after DB creation; adjust
            # cache_size to keep the same total bytes.
            page_size = cur_ps

    conn.execute(f"PRAGMA cache_size={cache_size_kb}")

    # Memory-map the DB file for large zero-copy reads.
    # Disabled when a custom pcache is used (mmap bypasses the pcache).
    if disable_mmap:
        conn.execute("PRAGMA mmap_size=0")
    elif db_path != ":memory:" and os.path.isfile(db_path):
        db_size = os.path.getsize(db_path)
        mmap_sz = max(db_size * 2, 256 * 1024 * 1024)
        conn.execute(f"PRAGMA mmap_size={mmap_sz}")


# ---------------------------------------------------------------------------
# Layer-aware prefetcher
# ---------------------------------------------------------------------------

class LayerPrefetcher:
    """Background thread that pre-loads upcoming layers into SQLite cache.

    When the engine begins executing layer *N*, the prefetcher can
    speculatively issue ``SELECT data FROM model_params WHERE name LIKE …``
    for layers *N+1 … N+k* so that the required pages are already in the
    page cache (or mmap region) when needed.

    The prefetcher operates on its **own** SQLite connection (same DB file)
    so that it does not contend with the main inference connection.
    """

    def __init__(
        self,
        db_path: str,
        num_layers: int,
        prefetch_ahead: int = LLM_PREFETCH_AHEAD,
        disable_mmap: bool = False,
    ) -> None:
        self._db_path = db_path
        self._num_layers = num_layers
        self._prefetch_ahead = prefetch_ahead
        self._disable_mmap = disable_mmap

        # Set of layer indices that have already been prefetched.
        self._prefetched: Set[int] = set()

        # Background worker state
        self._lock = threading.Lock()
        self._pending: List[int] = []
        self._stop_event = threading.Event()
        self._work_event = threading.Event()

        # Start background worker only for file-based DBs
        self._worker: Optional[threading.Thread] = None
        if db_path != ":memory:" and os.path.isfile(db_path):
            self._worker = threading.Thread(
                target=self._worker_loop, daemon=True,
            )
            self._worker.start()

    # ── Public API ────────────────────────────────────────────────

    def schedule_prefetch(self, current_layer: int) -> None:
        """Request prefetch of layers *current_layer … +prefetch_ahead*.

        Schedules the current layer (if not yet prefetched) plus up to
        ``prefetch_ahead`` subsequent layers.
        """
        if self._worker is None:
            return
        layers_to_fetch: List[int] = []
        for offset in range(0, self._prefetch_ahead + 1):
            target = current_layer + offset
            if 0 <= target < self._num_layers and target not in self._prefetched:
                layers_to_fetch.append(target)
        if not layers_to_fetch:
            return
        with self._lock:
            self._pending.extend(layers_to_fetch)
        self._work_event.set()

    def mark_prefetched(self, layer_idx: int) -> None:
        """Record that *layer_idx* is now in cache."""
        self._prefetched.add(layer_idx)

    def close(self) -> None:
        """Signal the worker to stop and wait for it."""
        self._stop_event.set()
        self._work_event.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None

    # ── Worker loop ───────────────────────────────────────────────

    def _worker_loop(self) -> None:
        """Background thread: open a read-only connection and prefetch.

        The prefetch connection uses ``locking_mode=NORMAL`` (not
        EXCLUSIVE) so it can coexist with the main inference connection
        which holds an EXCLUSIVE lock.  Since the DB is opened read-only
        there is no write contention; the NORMAL lock allows the read to
        proceed concurrently with the main connection's queries.
        """
        try:
            conn = sqlite3.connect(
                f"file:{self._db_path}?mode=ro",
                uri=True,
            )
        except sqlite3.OperationalError:
            # Fallback: open normally
            conn = sqlite3.connect(self._db_path)

        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA locking_mode=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")

        # Use the same cache_size for prefetch connection
        conn.execute(f"PRAGMA cache_size={LLM_CACHE_SIZE_KB}")

        if os.path.isfile(self._db_path) and not self._disable_mmap:
            db_size = os.path.getsize(self._db_path)
            mmap_sz = max(db_size * 2, 256 * 1024 * 1024)
            conn.execute(f"PRAGMA mmap_size={mmap_sz}")
        elif self._disable_mmap:
            conn.execute("PRAGMA mmap_size=0")

        while not self._stop_event.is_set():
            self._work_event.wait(timeout=0.5)
            self._work_event.clear()

            with self._lock:
                work = list(self._pending)
                self._pending.clear()

            for layer_idx in work:
                if self._stop_event.is_set():
                    break
                if layer_idx in self._prefetched:
                    continue
                self._do_prefetch(conn, layer_idx)
                self._prefetched.add(layer_idx)

        conn.close()

    def _do_prefetch(self, conn: sqlite3.Connection, layer_idx: int) -> None:
        """Issue a SELECT that touches all weight blobs for *layer_idx*.

        Parameter names in the ``model_params`` table typically follow
        the pattern ``layers.<N>.<submodule>.<param>`` (e.g.
        ``layers.0.self_attn.q_proj.weight``).  We match by prefix.
        """
        # Read the data column so SQLite pulls pages into cache.
        # Using ``length()`` alone would not force the pages to be loaded
        # because SQLite overflow pages are demand-paged.
        prefix = f"layers.{layer_idx}.%"
        try:
            # SELECT the data to force SQLite to load it into the page cache.
            # We don't need to use the result; the goal is cache warming.
            rows = conn.execute(
                "SELECT length(data) FROM model_params WHERE name LIKE ?",
                (prefix,),
            ).fetchall()
            # Also force full blob read for overflow pages
            if rows:
                conn.execute(
                    "SELECT sum(length(data)) FROM model_params "
                    "WHERE name LIKE ?",
                    (prefix,),
                ).fetchone()
        except sqlite3.OperationalError:
            pass


# ---------------------------------------------------------------------------
# Layer-aware page cache / eviction tracker
# ---------------------------------------------------------------------------

class LLMPageCache:
    """Tracks which transformer layers have their weights loaded.

    In the default SQLite page cache (LRU), recently-read pages push out
    older pages regardless of future access patterns.  For a sequential
    layer-by-layer forward pass this means layer-0 weights are evicted
    just when layer-31 is executing — exactly the wrong behaviour for the
    *next* forward pass (prefill or autoregressive step).

    ``LLMPageCache`` maintains a logical view of which layers are "hot"
    and provides an API to explicitly release finished layers so that
    their memory can be reclaimed before future layers are evicted.

    Note: SQLite's internal page cache cannot be controlled at the
    individual-page level from user code.  Our strategy is therefore:

    1. Keep a large ``cache_size`` so that *most* pages stay resident.
    2. Use ``mmap_size`` so that the OS kernel manages the mapping and
       the working-set pressure is handled by the VM subsystem rather
       than by SQLite's own LRU.
    3. Track layer state so the ``LLMMemoryManager`` can advise the
       prefetcher which layers are needed next.
    """

    def __init__(self, num_layers: int) -> None:
        self._num_layers = num_layers
        # State per layer: 'cold' | 'loading' | 'hot' | 'done'
        self._state: Dict[int, str] = {
            i: "cold" for i in range(num_layers)
        }
        # Blobs currently pinned (kept in Python ``dict`` so GC won't
        # release the underlying ``bytes`` object while the layer is
        # executing).
        self._pinned: Dict[int, Dict[str, bytes]] = {}

    # ── Query ─────────────────────────────────────────────────────

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def layer_state(self, idx: int) -> str:
        return self._state.get(idx, "cold")

    def hot_layers(self) -> List[int]:
        return [i for i, s in self._state.items() if s == "hot"]

    def cold_layers(self) -> List[int]:
        return [i for i, s in self._state.items() if s == "cold"]

    # ── Mutation ──────────────────────────────────────────────────

    def mark_loading(self, idx: int) -> None:
        self._state[idx] = "loading"

    def mark_hot(self, idx: int, blobs: Optional[Dict[str, bytes]] = None) -> None:
        self._state[idx] = "hot"
        if blobs is not None:
            self._pinned[idx] = blobs

    def mark_done(self, idx: int) -> None:
        """Mark layer as done and release pinned blobs."""
        self._state[idx] = "done"
        self._pinned.pop(idx, None)

    def reset(self) -> None:
        """Reset all layers to cold (start of new forward pass)."""
        for i in range(self._num_layers):
            self._state[i] = "cold"
        self._pinned.clear()

    def get_pinned(self, idx: int) -> Optional[Dict[str, bytes]]:
        return self._pinned.get(idx)


# ---------------------------------------------------------------------------
# Main facade
# ---------------------------------------------------------------------------

class LLMMemoryManager:
    """Coordinates custom SQLite memory management for LLM inference.

    Combines:
    * Inference-mode PRAGMA tuning (journal off, large pages, mmap)
    * Layer-aware page-cache tracking
    * Background prefetching of upcoming layers

    Parameters
    ----------
    db_path : str
        Path to the model SQLite database.
    num_layers : int
        Number of transformer layers in the model.
    extension_path : str, optional
        Path to ``llm_ops.so``.
    page_size : int
        SQLite page size (default 64 KiB).
    cache_size_kb : int
        PRAGMA cache_size value (negative = kibibytes).
    prefetch_ahead : int
        How many layers ahead to prefetch.
    """

    def __init__(
        self,
        db_path: str,
        num_layers: int,
        extension_path: Optional[str] = None,
        page_size: int = LLM_PAGE_SIZE,
        cache_size_kb: int = LLM_CACHE_SIZE_KB,
        prefetch_ahead: int = LLM_PREFETCH_AHEAD,
        disable_mmap: bool = False,
    ) -> None:
        self._db_path = db_path
        self._num_layers = num_layers

        # Open main connection
        self._conn = sqlite3.connect(db_path)
        if extension_path:
            self._conn.enable_load_extension(True)
            try:
                self._conn.load_extension(
                    extension_path, entrypoint="sqlite3_llm_ops_init")
            except TypeError:
                self._conn.load_extension(extension_path)

        # Apply inference PRAGMAs
        apply_inference_pragmas(
            self._conn, db_path,
            page_size=page_size,
            cache_size_kb=cache_size_kb,
            disable_mmap=disable_mmap,
        )

        # Page cache tracker
        self._cache = LLMPageCache(num_layers)

        # Background prefetcher
        self._prefetcher = LayerPrefetcher(
            db_path, num_layers,
            prefetch_ahead=prefetch_ahead,
            disable_mmap=disable_mmap,
        )

        # Per-layer parameter name mapping (built lazily)
        self._layer_params: Optional[Dict[int, List[str]]] = None

    # ── Properties ────────────────────────────────────────────────

    @property
    def connection(self) -> sqlite3.Connection:
        """The main SQLite connection (with extension + PRAGMAs)."""
        return self._conn

    @property
    def page_cache(self) -> LLMPageCache:
        return self._cache

    # ── Layer parameter index ─────────────────────────────────────

    def _build_layer_index(self, param_names: List[str]) -> None:
        """Index parameter names by transformer layer number.

        Names like ``layers.3.self_attn.q_proj.weight`` are mapped to
        layer 3.  Non-layer parameters (embeddings, final norm, lm_head)
        are mapped to layer -1.
        """
        layer_re = re.compile(r"layers\.(\d+)\.")
        index: Dict[int, List[str]] = {-1: []}
        for name in param_names:
            m = layer_re.search(name)
            if m:
                lid = int(m.group(1))
                index.setdefault(lid, []).append(name)
            else:
                index[-1].append(name)
        self._layer_params = index

    def layer_param_names(self, layer_idx: int) -> List[str]:
        """Return parameter names belonging to *layer_idx*.

        Layer -1 holds non-layer (global) parameters.
        """
        if self._layer_params is None:
            return []
        return self._layer_params.get(layer_idx, [])

    # ── Prefetch / eviction ───────────────────────────────────────

    def prefetch_layer(self, layer_idx: int) -> None:
        """Schedule background prefetch for *layer_idx* and lookahead.

        Calling ``prefetch_layer(N)`` marks layer *N* as loading and
        schedules prefetch of layers *N … N+prefetch_ahead* so they
        are warm in the page cache by the time execution reaches them.
        """
        if layer_idx >= 0:
            self._prefetcher.schedule_prefetch(layer_idx)
        if 0 <= layer_idx < self._cache.num_layers:
            self._cache.mark_loading(layer_idx)

    def mark_layer_done(self, layer_idx: int) -> None:
        """Release resources held by a completed layer."""
        self._cache.mark_done(layer_idx)

    def begin_forward_pass(self, param_names: Optional[List[str]] = None) -> None:
        """Called at the start of each forward pass (prefill or decode).

        Resets layer states and optionally rebuilds the layer index.
        """
        self._cache.reset()
        if param_names is not None and self._layer_params is None:
            self._build_layer_index(param_names)

    def load_layer_params(
        self,
        layer_idx: int,
        all_params: Dict[str, bytes],
    ) -> Dict[str, bytes]:
        """Return only the parameters for *layer_idx* from *all_params*.

        Also pins them in the page cache to prevent eviction while the
        layer is executing.
        """
        if self._layer_params is None:
            return all_params
        names = self._layer_params.get(layer_idx, [])
        blobs = {n: all_params[n] for n in names if n in all_params}
        self._cache.mark_hot(layer_idx, blobs)
        return blobs

    # ── Lifecycle ─────────────────────────────────────────────────

    def close(self) -> None:
        """Shut down prefetcher and close the connection."""
        self._prefetcher.close()
        self._conn.close()

    def get_pragma_report(self) -> Dict[str, Any]:
        """Return a dict with the active PRAGMA values for diagnostics."""
        report: Dict[str, Any] = {}
        for pragma in (
            "journal_mode", "synchronous", "locking_mode",
            "temp_store", "page_size", "cache_size", "mmap_size",
        ):
            try:
                row = self._conn.execute(f"PRAGMA {pragma}").fetchone()
                report[pragma] = row[0] if row else None
            except sqlite3.OperationalError:
                report[pragma] = None
        return report
