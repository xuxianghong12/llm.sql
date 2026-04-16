"""Python wrapper for the custom LLM page cache shared library.

Provides a Python-friendly interface to ``llm_pcache.so`` — a custom
SQLite page cache implementation (``sqlite3_pcache_methods2``) optimised
for transformer inference with layer-aware eviction and memory budgets.

Usage::

    from src.backends.llm_pcache import LLMPcache

    pcache = LLMPcache(budget_mb=50, num_layers=32)
    pcache.install()

    # ... open SQLite connections and run inference ...
    pcache.set_active_layer(0)
    # ... layer 0 computation ...
    pcache.mark_layer_done(0)

    stats = pcache.get_stats()
    print(f"Hit rate: {stats['hit_rate']:.1%}")

    pcache.uninstall()

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import ctypes
import os
from typing import Any, Dict, Optional


LLM_PCACHE_POLICY_LRU = "lru"
LLM_PCACHE_POLICY_OPT = "opt"
_POLICY_TO_ENUM = {
    LLM_PCACHE_POLICY_LRU: 0,
    LLM_PCACHE_POLICY_OPT: 1,
}


def _normalize_policy(policy: str) -> str:
    value = str(policy).strip().lower()
    if value not in _POLICY_TO_ENUM:
        raise ValueError(
            f"Unsupported pcache policy {policy!r}; expected one of "
            f"{sorted(_POLICY_TO_ENUM)}"
        )
    return value


# ---------------------------------------------------------------------------
# Default page-cache budget (bytes)
# ---------------------------------------------------------------------------

LLM_PCACHE_DEFAULT_BUDGET_MB: float = 100
"""Default page-cache memory budget in MB.  100 MB is sufficient to hold
~1.5 transformer layers for a typical H=1024 model and creates meaningful
eviction pressure for layer-aware caching."""


def auto_budget(
    layer_param_bytes: int,
    lookahead: int = 2,
    overhead_factor: float = 1.2,
) -> int:
    """Compute a recommended pcache budget in bytes.

    The cache must hold at least ``lookahead + 1`` layers simultaneously
    (current layer + pre-fetched layers) plus 20% overhead for B-tree
    internal nodes and index pages.

    Parameters
    ----------
    layer_param_bytes : int
        Total bytes of parameters for one transformer layer.
    lookahead : int
        Number of layers to pre-fetch ahead of the current one.
        Defaults to 2 (matches ``AsyncParamLoader`` decode mode).
    overhead_factor : float
        Multiplier for B-tree overhead.  Default 1.2 = 20% headroom.

    Returns
    -------
    int
        Recommended budget in bytes.
    """
    return int((lookahead + 1) * layer_param_bytes * overhead_factor)

# ---------------------------------------------------------------------------
# Locate the shared library
# ---------------------------------------------------------------------------

def _find_pcache_so() -> str:
    """Search for ``llm_pcache.so`` in standard locations."""
    candidates = [
        # Next to this Python file
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "sqlite-llm", "llm_pcache.so"),
        # In the sqlite-llm build directory
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "llm_pcache.so"),
        # Current working directory
        "llm_pcache.so",
        # sqlite-llm subdirectory
        os.path.join("sqlite-llm", "llm_pcache.so"),
    ]
    for path in candidates:
        full = os.path.abspath(path)
        if os.path.isfile(full):
            return full
    raise FileNotFoundError(
        "llm_pcache.so not found.  Build it with: cd sqlite-llm && make"
    )


# ---------------------------------------------------------------------------
# ctypes stats structure
# ---------------------------------------------------------------------------

class _PcacheStats(ctypes.Structure):
    """Mirrors ``llm_pcache_stats_t`` from llm_pcache.h."""
    _fields_ = [
        ("cache_hits",      ctypes.c_int64),
        ("cache_misses",    ctypes.c_int64),
        ("pages_current",   ctypes.c_int64),
        ("pages_pinned",    ctypes.c_int64),
        ("evictions",       ctypes.c_int64),
        ("evictions_layer", ctypes.c_int64),
        ("evictions_other", ctypes.c_int64),
        ("memory_used",     ctypes.c_int64),
        ("memory_budget",   ctypes.c_int64),
        ("active_layer",    ctypes.c_int32),
        ("num_layers",      ctypes.c_int32),
        ("policy",          ctypes.c_int32),
        ("prefetch_window", ctypes.c_int32),
    ]


# ---------------------------------------------------------------------------
# Main wrapper class
# ---------------------------------------------------------------------------

class LLMPcache:
    """Manage the custom LLM page cache lifecycle.

    Parameters
    ----------
    budget_mb : float
        Maximum cache memory in megabytes.  Default is **100 MB**, which
        is sufficient to hold ~1.5 transformer layers for a typical
        H=1024 model and creates meaningful eviction pressure for
        layer-aware caching.  Pass 0 for unlimited.
    num_layers : int
        Number of transformer layers in the model.
    policy : {"lru", "opt"}
        Eviction policy. ``opt`` is the default because transformer layer
        access is deterministic under streaming inference.
    prefetch_window : int
        Number of future layers that receive explicit protection in the OPT
        victim score. Default is 2.
    so_path : str, optional
        Explicit path to ``llm_pcache.so``.
    """

    def __init__(
        self,
        budget_mb: float = LLM_PCACHE_DEFAULT_BUDGET_MB,
        num_layers: int = 0,
        policy: str = LLM_PCACHE_POLICY_OPT,
        prefetch_window: int = 2,
        so_path: Optional[str] = None,
    ) -> None:
        self._budget_bytes = int(budget_mb * 1024 * 1024)
        self._num_layers = num_layers
        self._policy = _normalize_policy(policy)
        self._prefetch_window = max(0, int(prefetch_window))
        self._so_path = so_path or _find_pcache_so()
        self._lib: Optional[ctypes.CDLL] = None
        self._installed = False

    def install(self) -> None:
        """Load the .so and register the custom page cache with SQLite.

        Must be called **before** opening any SQLite connections.
        """
        if self._installed:
            return

        self._lib = ctypes.CDLL(self._so_path)

        # Set up function signatures
        self._lib.llm_pcache_install.argtypes = [ctypes.c_int64, ctypes.c_int]
        self._lib.llm_pcache_install.restype = ctypes.c_int

        if hasattr(self._lib, "llm_pcache_set_policy"):
            self._lib.llm_pcache_set_policy.argtypes = [ctypes.c_int]
            self._lib.llm_pcache_set_policy.restype = ctypes.c_int

        if hasattr(self._lib, "llm_pcache_get_policy"):
            self._lib.llm_pcache_get_policy.argtypes = []
            self._lib.llm_pcache_get_policy.restype = ctypes.c_int

        if hasattr(self._lib, "llm_pcache_set_prefetch_window"):
            self._lib.llm_pcache_set_prefetch_window.argtypes = [ctypes.c_int]
            self._lib.llm_pcache_set_prefetch_window.restype = ctypes.c_int

        if hasattr(self._lib, "llm_pcache_get_prefetch_window"):
            self._lib.llm_pcache_get_prefetch_window.argtypes = []
            self._lib.llm_pcache_get_prefetch_window.restype = ctypes.c_int

        self._lib.llm_pcache_uninstall.argtypes = []
        self._lib.llm_pcache_uninstall.restype = ctypes.c_int

        self._lib.llm_pcache_set_active_layer.argtypes = [ctypes.c_int]
        self._lib.llm_pcache_set_active_layer.restype = None

        self._lib.llm_pcache_mark_layer_done.argtypes = [ctypes.c_int]
        self._lib.llm_pcache_mark_layer_done.restype = None

        self._lib.llm_pcache_prefetch_layer.argtypes = [ctypes.c_int]
        self._lib.llm_pcache_prefetch_layer.restype = None

        self._lib.llm_pcache_reset_layers.argtypes = []
        self._lib.llm_pcache_reset_layers.restype = None

        self._lib.llm_pcache_begin_layer_io.argtypes = [ctypes.c_int]
        self._lib.llm_pcache_begin_layer_io.restype = None

        self._lib.llm_pcache_end_layer_io.argtypes = []
        self._lib.llm_pcache_end_layer_io.restype = None

        self._lib.llm_pcache_layer_pages.argtypes = [ctypes.c_int]
        self._lib.llm_pcache_layer_pages.restype = ctypes.c_int64

        self._lib.llm_pcache_get_stats.argtypes = [
            ctypes.POINTER(_PcacheStats)]
        self._lib.llm_pcache_get_stats.restype = None

        self._lib.llm_pcache_reset_stats.argtypes = []
        self._lib.llm_pcache_reset_stats.restype = None

        if hasattr(self._lib, "llm_pcache_set_policy"):
            self._lib.llm_pcache_set_policy(_POLICY_TO_ENUM[self._policy])
        elif self._policy != LLM_PCACHE_POLICY_LRU:
            raise RuntimeError(
                "llm_pcache.so does not support selectable policies; rebuild sqlite-llm"
            )

        if hasattr(self._lib, "llm_pcache_set_prefetch_window"):
            self._lib.llm_pcache_set_prefetch_window(self._prefetch_window)

        rc = self._lib.llm_pcache_install(
            self._budget_bytes, self._num_layers)
        if rc != 0:
            raise RuntimeError(
                f"llm_pcache_install failed with code {rc}")
        self._installed = True

    def uninstall(self) -> None:
        """Restore SQLite's default page cache."""
        if not self._installed or self._lib is None:
            return
        self._lib.llm_pcache_uninstall()
        self._installed = False

    @property
    def installed(self) -> bool:
        return self._installed

    @property
    def policy(self) -> str:
        return self._policy

    @property
    def prefetch_window(self) -> int:
        return self._prefetch_window

    def set_policy(self, policy: str) -> None:
        normalized = _normalize_policy(policy)
        if self._lib and hasattr(self._lib, "llm_pcache_set_policy"):
            self._lib.llm_pcache_set_policy(_POLICY_TO_ENUM[normalized])
        elif normalized != LLM_PCACHE_POLICY_LRU:
            raise RuntimeError(
                "llm_pcache.so does not support selectable policies; rebuild sqlite-llm"
            )
        self._policy = normalized

    def set_prefetch_window(self, window: int) -> None:
        normalized = max(0, int(window))
        if self._lib and hasattr(self._lib, "llm_pcache_set_prefetch_window"):
            self._lib.llm_pcache_set_prefetch_window(normalized)
        self._prefetch_window = normalized

    # ── Layer hints ──────────────────────────────────────────────

    def set_active_layer(self, layer_idx: int) -> None:
        """Notify the cache that *layer_idx* is now being processed."""
        if self._lib and self._installed:
            self._lib.llm_pcache_set_active_layer(layer_idx)

    def mark_layer_done(self, layer_idx: int) -> None:
        """Notify the cache that *layer_idx* is complete."""
        if self._lib and self._installed:
            self._lib.llm_pcache_mark_layer_done(layer_idx)

    def prefetch_layer(self, layer_idx: int) -> None:
        """Hint that *layer_idx* will be needed soon (protect from eviction)."""
        if self._lib and self._installed:
            self._lib.llm_pcache_prefetch_layer(layer_idx)

    def reset_layers(self) -> None:
        """Reset all layer states for a new forward pass."""
        if self._lib and self._installed:
            self._lib.llm_pcache_reset_layers()

    def begin_layer_io(self, layer_idx: int) -> None:
        """Begin a precise layer I/O region.

        While active, cache misses are tagged with *layer_idx* exactly
        (not the global active_layer).  Cache hits keep their existing
        tag to avoid drift on shared B-tree internal/index pages.

        Call :meth:`end_layer_io` when the SQL query completes.
        """
        if self._lib and self._installed:
            self._lib.llm_pcache_begin_layer_io(layer_idx)

    def end_layer_io(self) -> None:
        """End the current layer I/O region started by :meth:`begin_layer_io`."""
        if self._lib and self._installed:
            self._lib.llm_pcache_end_layer_io()

    def layer_pages(self, layer_idx: int) -> int:
        """Return the number of pages currently cached for *layer_idx*."""
        if self._lib and self._installed:
            return int(self._lib.llm_pcache_layer_pages(layer_idx))
        return 0

    # ── Statistics ───────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return a dictionary of cache statistics."""
        if not self._lib or not self._installed:
            return {}
        stats = _PcacheStats()
        self._lib.llm_pcache_get_stats(ctypes.byref(stats))
        total = stats.cache_hits + stats.cache_misses
        hit_rate = stats.cache_hits / total if total > 0 else 0.0
        return {
            "policy": self._policy,
            "cache_hits": stats.cache_hits,
            "cache_misses": stats.cache_misses,
            "hit_rate": hit_rate,
            "pages_current": stats.pages_current,
            "pages_pinned": stats.pages_pinned,
            "evictions": stats.evictions,
            "evictions_layer": stats.evictions_layer,
            "evictions_other": stats.evictions_other,
            "memory_used_mb": stats.memory_used / (1024 * 1024),
            "memory_budget_mb": stats.memory_budget / (1024 * 1024),
            "active_layer": stats.active_layer,
            "num_layers": stats.num_layers,
            "prefetch_window": stats.prefetch_window,
        }

    def reset_stats(self) -> None:
        """Reset all statistics counters."""
        if self._lib and self._installed:
            self._lib.llm_pcache_reset_stats()

    # ── Context manager ──────────────────────────────────────────

    def __enter__(self) -> "LLMPcache":
        self.install()
        return self

    def __exit__(self, *args: Any) -> None:
        self.uninstall()

    def __del__(self) -> None:
        try:
            self.uninstall()
        except Exception:
            pass
