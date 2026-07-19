from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec
from typing import Any


def install_flash_attention_import_fallback(torch: Any) -> bool:
    """Satisfy old LingBot-VA's eager flash-attn import with PyTorch SDPA.

    The official training path used by this repository selects ``attn_mode=flex``.
    This fallback is therefore normally import-only, but remains numerically valid if
    an upstream code path calls it with the standard ``[B, S, H, D]`` tensors.
    """

    try:
        __import__("flash_attn_interface")
        return False
    except ModuleNotFoundError:
        pass
    try:
        __import__("flash_attn")
        return False
    except ModuleNotFoundError:
        pass

    def flash_attn_func(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        dropout = float(kwargs.get("dropout_p", 0.0))
        causal = bool(kwargs.get("causal", False))
        scale = kwargs.get("softmax_scale")
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=dropout,
            is_causal=causal,
            scale=scale,
        )
        return output.transpose(1, 2).contiguous()

    module = types.ModuleType("flash_attn")
    module.__spec__ = ModuleSpec("flash_attn", loader=None)
    module.flash_attn_func = flash_attn_func
    sys.modules["flash_attn"] = module
    return True
