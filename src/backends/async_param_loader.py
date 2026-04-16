"""Async parameter loader for SQLite-based LLM inference.

Runs a background thread that pre-fetches transformer layer parameters
from SQLite while the main thread computes the previous layer.  This
overlaps I/O with computation and hides SQL query latency.

Architecture::

    Main thread:     [get layer 0] [compute 0] [get layer 1] [compute 1] ...
    Loader thread:             [load layer 1]            [load layer 2] ...
                                      ↑ overlap ↑

The two threads share a single global pcache (installed via
``sqlite3_config(SQLITE_CONFIG_PCACHE2)``).  Pages loaded by the
background thread are immediately visible to the main thread's connection,
so ``get_layer_params()`` typically returns after a short wait or
immediately (cache warm).

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import os
import queue
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional


class AsyncParamLoader:
    """Background-thread parameter loader for per-layer streaming inference.

    Parameters
    ----------
    db_path : str
        Path to the model SQLite database.
    extension_path : str, optional
        Path to ``llm_ops.so`` (needed for the background connection to
        apply the same PRAGMAs as the main connection).
    layer_param_names : dict[int, list[str]]
        Mapping from layer index to list of parameter names for that layer.
        Layer -1 holds non-layer (global) parameters.
    layer_param_sql : dict[int, str]
        Pre-built ``SELECT name, data FROM model_params WHERE name IN (…)``
        query for each layer index.
    pcache : LLMPcache, optional
        Custom page-cache wrapper.  When provided, ``begin_layer_io`` /
        ``end_layer_io`` are called around each SQL fetch so new pages
        receive precise layer tags.
    lookahead : int
        Number of layers to pre-fetch ahead of the current layer.
        Defaults to 1 (load layer N+1 while computing layer N).
    """

    def __init__(
        self,
        db_path: str,
        extension_path: Optional[str],
        layer_param_names: Dict[int, List[str]],
        layer_param_sql: Dict[int, str],
        pcache: Optional[Any] = None,
        lookahead: int = 1,
    ) -> None:
        self._db_path = db_path
        self._extension_path = extension_path
        self._layer_param_names = layer_param_names
        self._layer_param_sql = layer_param_sql
        self._pcache = pcache
        self._lookahead = max(1, lookahead)

        # Request queue: layer indices the loader should fetch.
        self._req_queue: queue.Queue[Optional[int]] = queue.Queue()

        # Completed results: layer_idx → dict[name, data_bytes]
        self._results: Dict[int, Dict[str, bytes]] = {}

        # Per-layer Event: set when that layer's data is ready.
        self._ready: Dict[int, threading.Event] = {}

        # Timing statistics (seconds) for adaptive lookahead.
        self._load_times: List[float] = []   # last N layer load times
        self._compute_times: List[float] = []  # last N layer compute times
        self._stat_window = 4  # rolling window size

        # Set of layers already requested (avoids duplicate queuing).
        self._requested: set = set()
        self._lock = threading.Lock()

        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="async-param-loader"
        )

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background loader thread."""
        self._thread.start()

    def stop(self) -> None:
        """Signal the worker to stop and wait for it."""
        self._stop_event.set()
        self._req_queue.put(None)  # unblock the worker
        self._thread.join(timeout=5.0)

    # ── Main-thread API ───────────────────────────────────────────

    def request_layer(self, layer_idx: int) -> None:
        """Request background loading of *layer_idx* (non-blocking).

        Idempotent: a layer that has already been requested or loaded
        will not be queued again.
        """
        with self._lock:
            if layer_idx in self._requested:
                return
            self._requested.add(layer_idx)
            # Ensure an Event exists for this layer.
            if layer_idx not in self._ready:
                self._ready[layer_idx] = threading.Event()
        self._req_queue.put(layer_idx)

    def get_layer_params(self, layer_idx: int) -> Dict[str, bytes]:
        """Return parameters for *layer_idx*, blocking until they are ready.

        If the layer was pre-fetched, this returns immediately.
        Otherwise it waits for the background thread to finish loading.

        Returns
        -------
        dict[str, bytes]
            Mapping from parameter name to BLOB bytes.
        """
        # Ensure the layer was requested (fallback if caller forgot).
        self.request_layer(layer_idx)

        event = self._ready.get(layer_idx)
        if event is not None:
            event.wait()

        return self._results.pop(layer_idx, {})

    def release_layer(self, layer_idx: int) -> None:
        """Free result buffer for *layer_idx* (idempotent)."""
        self._results.pop(layer_idx, None)

    def reset(self) -> None:
        """Reset state for a new forward pass.

        Clears the requested-layer set, all result buffers, and all
        per-layer Events so the next forward pass starts fresh.
        Pending queue items are drained but the worker thread keeps running.
        """
        # Drain the request queue to avoid carrying over stale requests.
        while True:
            try:
                self._req_queue.get_nowait()
            except queue.Empty:
                break

        with self._lock:
            self._requested.clear()
            self._results.clear()
            self._ready.clear()

    def record_compute_time(self, elapsed: float) -> None:
        """Record the compute time for the most recently executed layer.

        Used by the adaptive lookahead heuristic.
        """
        self._compute_times.append(elapsed)
        if len(self._compute_times) > self._stat_window:
            self._compute_times.pop(0)

    def set_decode_mode(self) -> None:
        """Switch to decode (single-token) mode.

        Decode steps are short (~4-5 ms per layer), so I/O typically
        exceeds compute.  Start with lookahead=2 to hide latency.
        """
        self._lookahead = 2

    def set_prefill_mode(self) -> None:
        """Switch to prefill (full-prompt) mode.

        Prefill compute time grows with sequence length and usually
        exceeds per-layer I/O.  lookahead=1 is sufficient.
        """
        self._lookahead = 1

    def suggested_lookahead(self) -> int:
        """Return an adaptive lookahead based on recent load/compute times.

        If average load time exceeds average compute time, increase
        lookahead so more layers are pre-fetched.  Cap at 3.
        """
        if not self._load_times or not self._compute_times:
            return self._lookahead
        avg_load = sum(self._load_times) / len(self._load_times)
        avg_compute = sum(self._compute_times) / len(self._compute_times)
        if avg_compute <= 0:
            return self._lookahead
        ratio = avg_load / avg_compute
        if ratio <= 1.0:
            return 1
        return min(3, int(ratio) + 1)

    # ── Worker loop ───────────────────────────────────────────────

    def _open_connection(self) -> sqlite3.Connection:
        """Open a read-only SQLite connection for the background thread.

        Uses the same pcache (globally installed) but a separate
        connection object, which is required because Python's ``sqlite3``
        module is not thread-safe across connections.

        mmap is disabled (``PRAGMA mmap_size=0``) so all reads go
        through the custom pcache rather than bypassing it.
        """
        try:
            conn = sqlite3.connect(
                f"file:{self._db_path}?mode=ro", uri=True
            )
        except sqlite3.OperationalError:
            conn = sqlite3.connect(self._db_path)

        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA locking_mode=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=0")   # force all I/O through pcache

        if self._extension_path:
            conn.enable_load_extension(True)
            try:
                conn.load_extension(
                    self._extension_path, entrypoint="sqlite3_llm_ops_init"
                )
            except TypeError:
                conn.load_extension(self._extension_path)

        return conn

    def _worker_loop(self) -> None:
        """Background thread: open a connection and process load requests."""
        try:
            conn = self._open_connection()
        except Exception:
            # If we can't open the DB, signal all pending events as done
            # (with empty results) so the main thread doesn't hang.
            with self._lock:
                for ev in self._ready.values():
                    ev.set()
            return

        while not self._stop_event.is_set():
            try:
                layer_idx = self._req_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if layer_idx is None:
                break  # stop sentinel

            self._load_layer(conn, layer_idx)

        conn.close()

    def _load_layer(self, conn: sqlite3.Connection, layer_idx: int) -> None:
        """Fetch all parameters for *layer_idx* from SQLite.

        Uses ``begin_layer_io`` / ``end_layer_io`` to ensure new cache
        pages are tagged with the correct layer index.
        """
        names = self._layer_param_names.get(layer_idx)
        if not names:
            # No params for this layer — signal immediately with empty dict.
            self._results[layer_idx] = {}
            ev = self._ready.get(layer_idx)
            if ev:
                ev.set()
            return

        sql = self._layer_param_sql.get(layer_idx)
        if not sql:
            self._results[layer_idx] = {}
            ev = self._ready.get(layer_idx)
            if ev:
                ev.set()
            return

        pcache = self._pcache
        t0 = time.perf_counter()

        if pcache is not None:
            pcache.begin_layer_io(layer_idx)

        try:
            rows = conn.execute(sql, names).fetchall()
        except sqlite3.OperationalError:
            rows = []
        finally:
            if pcache is not None:
                pcache.end_layer_io()

        elapsed = time.perf_counter() - t0
        self._load_times.append(elapsed)
        if len(self._load_times) > self._stat_window:
            self._load_times.pop(0)

        blob_map: Dict[str, bytes] = {name: data for name, data in rows}
        self._results[layer_idx] = blob_map

        ev = self._ready.get(layer_idx)
        if ev:
            ev.set()
