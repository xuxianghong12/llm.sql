"""
Model wrappers for exporting Qwen2 with KV Cache support.

These wrappers convert between PyTorch's DynamicCache format and tuple format
that torch.export can handle, enabling proper KV cache export for both prefill
and decoding stages.
"""
import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache
from typing import Tuple


def _cache_to_tuple(cache) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """Normalize transformers cache outputs across API variants."""
    if cache is None:
        return tuple()
    if hasattr(cache, "layers"):
        return tuple((layer.keys, layer.values) for layer in cache.layers)
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return tuple(zip(cache.key_cache, cache.value_cache))
    if isinstance(cache, (tuple, list)):
        return tuple((layer[0], layer[1]) for layer in cache)
    raise TypeError(f"Unsupported cache type: {type(cache)!r}")


class Qwen2PrefillWrapper(nn.Module):
    """
    Wrapper for prefill stage export.

    Takes input_ids, outputs (logits, kv_cache_tuple).
    The kv_cache_tuple is a tuple of (key, value) pairs for each layer,
    which torch.export can properly handle.
    """

    def __init__(self, model: Qwen2ForCausalLM):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass for prefill stage.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs, shape (batch, seqlen)

        Returns
        -------
        logits : torch.Tensor
            Output logits, shape (batch, seqlen, vocab_size)
        kv_cache : Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
            KV cache as tuple of (key, value) pairs for each layer
        """
        output = self.model(input_ids, use_cache=True, return_dict=False)
        logits = output[0]
        cache = output[1]
        kv_tuple = _cache_to_tuple(cache)
        return logits, kv_tuple


class Qwen2DecodeWrapper(nn.Module):
    """
    Wrapper for decode stage export.

    Takes (input_ids, past_key_values_tuple), outputs (logits, updated_kv_tuple).
    Converts tuple format to DynamicCache for the model, then extracts
    updated cache back to tuple format.
    """

    def __init__(self, model: Qwen2ForCausalLM):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass for decode stage.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs, shape (batch, 1) - single token
        past_key_values : Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
            Past KV cache as tuple of (key, value) pairs for each layer

        Returns
        -------
        logits : torch.Tensor
            Output logits, shape (batch, 1, vocab_size)
        kv_cache : Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
            Updated KV cache as tuple of (key, value) pairs for each layer
        """
        # Convert tuple to DynamicCache
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(past_key_values):
            cache.update(k, v, layer_idx=layer_idx)

        # Forward pass with cache
        output = self.model(
            input_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=False
        )
        logits = output[0]
        new_cache = output[1]
        kv_tuple = _cache_to_tuple(new_cache)
        return logits, kv_tuple


def create_prefill_model(base_model: Qwen2ForCausalLM) -> Qwen2PrefillWrapper:
    """Create a wrapper for prefill stage export."""
    return Qwen2PrefillWrapper(base_model)


def create_decode_model(base_model: Qwen2ForCausalLM) -> Qwen2DecodeWrapper:
    """Create a wrapper for decode stage export."""
    return Qwen2DecodeWrapper(base_model)
