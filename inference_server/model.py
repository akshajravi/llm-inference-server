"""Model + tokenizer loading. Loaded exactly once per process, cached here.

Every engine (P0 through P3) shares this loader so that a benchmark comparing them
is comparing schedulers, not model setup.
"""

from __future__ import annotations

import functools

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from inference_server.config import CONFIG, Config, seed_everything


@functools.lru_cache(maxsize=1)
def load(config: Config | None = None) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model and tokenizer once. Subsequent calls hit the cache."""
    cfg = config or CONFIG
    seed_everything()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Static batching (P1) left-pads so the newest token is always at index -1.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=cfg.torch_dtype)
    model.to(cfg.device)
    model.eval()
    torch.set_grad_enabled(False)
    return model, tokenizer
