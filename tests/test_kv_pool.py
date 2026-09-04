"""P3 (Days 6-9) — the physical KV pool: shape, size, and the arithmetic behind it.

Cheap tests on purpose. The pool is preallocated once at startup, so a shape error here
does not surface as a wrong answer — it surfaces as an OOM at load time on the rented
GPU, which is the worst possible place to find it.
"""

from __future__ import annotations

import pytest
import torch

from inference_server.config import CONFIG, DEFAULT_NUM_BLOCKS
from inference_server.core.kv_pool import ModelDims, PagedKVPool

# gpt2's real numbers, written out rather than read from the model so these run with no
# download and no device, and so a silent change in how dims are derived is caught.
GPT2 = ModelDims(num_layers=12, num_kv_heads=12, head_dim=64, dtype=torch.float32)


@pytest.mark.phase("P3")
def test_dims_are_read_off_the_model(model_and_tokenizer):
    model, _ = model_and_tokenizer
    assert ModelDims.from_model(model, dtype=torch.float32) == GPT2


@pytest.mark.phase("P3")
def test_bytes_per_token_is_the_textbook_formula():
    """2 (K and V) x layers x kv_heads x head_dim x itemsize."""
    assert GPT2.bytes_per_token() == 2 * 12 * 12 * 64 * 4
    assert GPT2.bytes_per_block(block_size=16) == GPT2.bytes_per_token() * 16


@pytest.mark.phase("P3")
def test_budget_sizing_never_overcommits():
    """blocks_for_budget must round *down*. Rounding up is an OOM at startup."""
    per_block = GPT2.bytes_per_block(16)
    assert GPT2.blocks_for_budget(per_block * 10, 16) == 10
    assert GPT2.blocks_for_budget(per_block * 10 - 1, 16) == 9
    assert GPT2.blocks_for_budget(per_block - 1, 16) == 0
    assert GPT2.blocks_for_budget(0, 16) == 0


@pytest.mark.phase("P3")
def test_pool_shape_is_block_major():
    """[num_blocks, block_size, kv_heads, head_dim] — one block is contiguous, which is
    what the Day 8 gather indexes with a single index_select over the block table."""
    pool = PagedKVPool(GPT2, num_blocks=8, block_size=16, device="cpu")
    assert len(pool.k) == len(pool.v) == 12
    for layer in range(12):
        assert pool.k[layer].shape == (8, 16, 12, 64)
        assert pool.v[layer].shape == (8, 16, 12, 64)


@pytest.mark.phase("P3")
def test_reported_size_matches_the_tensors_actually_allocated():
    """`describe()` goes into the writeup, so it must not drift from reality."""
    pool = PagedKVPool(GPT2, num_blocks=8, block_size=16, device="cpu")
    real = sum(t.numel() * t.element_size() for t in pool.k + pool.v)
    assert pool.nbytes == real
    assert pool.num_slots == 8 * 16


@pytest.mark.phase("P3")
def test_default_pool_holds_exactly_the_contiguous_worst_case():
    """The config default is a derivation, not a guess: P3 must be measured holding the
    same worst case P2 reserves, or the waste comparison is against a smaller pool."""
    # The *default*, not CONFIG: NUM_BLOCKS may legitimately shrink the pool for a local
    # test run, and that must not read as the published reservation having changed.
    assert DEFAULT_NUM_BLOCKS * CONFIG.block_size == CONFIG.max_running * CONFIG.max_seq_len
