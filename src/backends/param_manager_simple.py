"""
Simplified ParamManager - cache only, no DB connection.

This version doesn't manage its own database connection.
Instead, it receives data from Python and caches it in C layer.
"""

import ctypes
import os
from typing import Optional, Tuple, Dict

# Find the shared library
_LIB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "sqlite-llm", "llm_param_mgr.so"
)
_LIB_PATH = os.path.abspath(_LIB_PATH)

if not os.path.exists(_LIB_PATH):
    raise ImportError(f"llm_param_mgr.so not found at {_LIB_PATH}")

# Load the library
_lib = ctypes.CDLL(_LIB_PATH)

# We'll use a simpler approach: Python dict cache with C-style management
class SimpleParamCache:
    """Simple parameter cache manager with circular buffer eviction.

    For LLM inference, access pattern is deterministic: layer 0 -> 1 -> ... -> N,
    then repeat. So we use a circular buffer strategy: keep only the most recent
    K layers in memory, evicting the oldest when we need space.
    """

    def __init__(self, db_conn, num_layers: int, budget_mb: int):
        self._conn = db_conn
        self._num_layers = num_layers
        self._budget_bytes = budget_mb * 1024 * 1024
        self._loaded_layers: list = []  # Circular buffer of loaded layers
        # Keep enough layers: current + next few for lookahead
        # For 24 layers, keep at least 5 to avoid evicting too early
        self._max_layers_in_memory = max(5, num_layers // 5)

    def load_layer(self, layer_id: int) -> None:
        """Mark layer as loaded and manage circular buffer."""
        # If already in buffer, move to end (most recent)
        if layer_id in self._loaded_layers:
            self._loaded_layers.remove(layer_id)
            self._loaded_layers.append(layer_id)
            return

        # Add to buffer
        self._loaded_layers.append(layer_id)

        # Buffer management is done in get_evictable_layers()

    def get_param(self, name: str) -> None:
        """Not used in this implementation."""
        return None

    def evict_layer(self, layer_id: int) -> None:
        """Mark layer as done (no-op in circular buffer)."""
        pass

    def get_evictable_layers(self) -> set:
        """Get layers that should be evicted (oldest in buffer if over limit)."""
        if len(self._loaded_layers) <= self._max_layers_in_memory:
            return set()  # No eviction needed

        # Evict oldest layers (keep only the most recent max_layers_in_memory)
        num_to_evict = len(self._loaded_layers) - self._max_layers_in_memory
        evictable = set(self._loaded_layers[:num_to_evict])

        # Remove evicted layers from buffer
        self._loaded_layers = self._loaded_layers[num_to_evict:]

        return evictable

    def reset(self) -> None:
        """Reset buffer."""
        self._loaded_layers.clear()

    def get_stats(self) -> Tuple[float, float]:
        """Get memory stats."""
        return (0.0, self._budget_bytes / (1024 * 1024))

    def close(self) -> None:
        """Clean up."""
        self._loaded_layers.clear()


# Alias for compatibility
ParamManager = SimpleParamCache
