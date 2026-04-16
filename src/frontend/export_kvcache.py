"""
Export utilities for creating prefill and decode graphs with KV Cache.

This module provides functions to export Qwen2 models with proper KV cache
handling for both prefill and decoding stages.

Requires torch>=2.11.0 and transformers>=5.4.0 where DynamicCache is
registered as a pytree node, enabling proper ``torch.export`` support.
"""
import torch
from torch.export import Dim
from transformers import Qwen2ForCausalLM
from src.models.qwen2_wrappers import (
    Qwen2PrefillWrapper,
    Qwen2DecodeWrapper,
    _cache_to_tuple,
)
from typing import Tuple


def export_prefill_graph(
    model: Qwen2ForCausalLM,
    example_seq_len: int = 5
) -> torch.export.ExportedProgram:
    """
    Export the prefill graph for initial prompt processing.

    Parameters
    ----------
    model : Qwen2ForCausalLM
        The base Qwen2 model with use_cache=True
    example_seq_len : int
        Example sequence length for tracing (default: 5)

    Returns
    -------
    ExportedProgram
        Exported prefill graph that outputs (logits, k0, v0, k1, v1, …)
    """
    model.eval()
    wrapper = Qwen2PrefillWrapper(model)

    example_ids = torch.randint(0, model.config.vocab_size, (1, example_seq_len), dtype=torch.long)
    example_inputs = (example_ids,)

    # Use Dim.AUTO so that the sequence-length dimension is dynamic
    dynamic_shapes = {"input_ids": {0: Dim.AUTO, 1: Dim.AUTO}}

    with torch.no_grad():
        exported = torch.export.export(
            wrapper,
            example_inputs,
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )

    return exported


def export_decode_graph(
    model: Qwen2ForCausalLM,
    example_cache_seq_len: int = 5
) -> torch.export.ExportedProgram:
    """
    Export the decode graph for single-token generation with KV cache.

    The cache sequence-length dimension is exported as dynamic so that
    the same graph works for any number of cached tokens.

    Parameters
    ----------
    model : Qwen2ForCausalLM
        The base Qwen2 model with use_cache=True
    example_cache_seq_len : int
        Example cache sequence length for tracing (default: 5)

    Returns
    -------
    ExportedProgram
        Exported decode graph that takes (input_ids, past_kv) and
        outputs (logits, updated_k0, updated_v0, …)
    """
    model.eval()
    wrapper = Qwen2DecodeWrapper(model)

    with torch.no_grad():
        example_ids = torch.randint(
            0, model.config.vocab_size,
            (1, example_cache_seq_len),
            dtype=torch.long
        )
        prefill_output = model(example_ids, use_cache=True, return_dict=False)
        example_cache = prefill_output[1]
        past_kv_tuple = _cache_to_tuple(example_cache)

        new_token = torch.randint(0, model.config.vocab_size, (1, 1), dtype=torch.long)
        example_inputs = (new_token, past_kv_tuple)

        # Mark all dims Dim.AUTO so that the cache seq-length is dynamic
        num_layers = len(past_kv_tuple)
        past_kv_dynamic = tuple(
            (
                {0: Dim.AUTO, 1: Dim.AUTO, 2: Dim.AUTO, 3: Dim.AUTO},
                {0: Dim.AUTO, 1: Dim.AUTO, 2: Dim.AUTO, 3: Dim.AUTO},
            )
            for _ in range(num_layers)
        )

        dynamic_shapes = {
            "input_ids": {0: Dim.AUTO, 1: Dim.AUTO},
            "past_key_values": past_kv_dynamic,
        }

        exported = torch.export.export(
            wrapper,
            example_inputs,
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )

    return exported


def export_model_for_kvcache(
    model: Qwen2ForCausalLM,
    example_seq_len: int = 5
) -> Tuple[torch.export.ExportedProgram, torch.export.ExportedProgram]:
    """
    Export both prefill and decode graphs for KV cache inference.

    Parameters
    ----------
    model : Qwen2ForCausalLM
        The base Qwen2 model with use_cache=True
    example_seq_len : int
        Example sequence length for tracing (default: 5)

    Returns
    -------
    ep_prefill : ExportedProgram
        Exported prefill graph
    ep_decode : ExportedProgram
        Exported decode graph
    """
    ep_prefill = export_prefill_graph(model, example_seq_len)
    ep_decode = export_decode_graph(model, example_seq_len)
    return ep_prefill, ep_decode
