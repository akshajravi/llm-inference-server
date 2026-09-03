"""S3 (Day 14) — the Triton paged-attention kernel, and its PyTorch mirror, against
the gather path.

Two tiers, because the author's machine cannot run Triton (Apple Silicon; no wheels):

  1. `paged_attention_reference` — the pure-PyTorch transliteration of the kernel's
     block loop and online softmax — runs HERE on CPU against `paged_attention_gather`.
     This is the only verification of the kernel's algorithm available on the Mac.
     It covers the indexing (block table walk, causal bound, tail mask, GQA mapping,
     power-of-two padding) but not the Triton API.

  2. The kernel itself runs only where `triton` imports AND CUDA exists, and is
     otherwise skipped with a reason. On the GPU box `ATTENTION=triton make test` runs
     these against the gather path on random inputs — the guide's precondition for
     letting the kernel near M1 — and then the M1 goldens through the paged engine.

The random layouts come from the same builder the gather tests use (blocks drawn in
shuffled order, so a contiguity assumption is caught), generalised over block_size,
head_dim and kv_heads so the padding-and-masking of non-power-of-two sizes is hit.
"""

from __future__ import annotations

import pytest
import torch

from inference_server.config import CONFIG
from inference_server.core import attention_triton
from inference_server.core.attention import paged_attention, paged_attention_gather, paged_attention_triton
from inference_server.core.attention_triton import paged_attention_reference, triton_available

H = 3            # query heads
NUM_BLOCKS = 64

needs_cuda_triton = pytest.mark.skipif(
    not triton_available(),
    reason=(
        "Triton kernel tests need `triton` and a CUDA device"
        + ("" if attention_triton.triton is not None else " (triton not importable: no macOS wheels)")
        + ("" if torch.cuda.is_available() else " (no CUDA device)")
        + " — they run on the GPU box via `ATTENTION=triton make test`"
    ),
)


def _build(lengths, *, bs=4, d=8, heads=H, kv_heads=H, seed=0, dtype=torch.float32, device="cpu"):
    """Random K/V for each (context_len, query_len) scattered into a shuffled block
    layout. Returns everything the kernel signature takes, on `device`."""
    g = torch.Generator().manual_seed(seed)
    k_pool = torch.randn(NUM_BLOCKS, bs, kv_heads, d, generator=g)
    v_pool = torch.randn(NUM_BLOCKS, bs, kv_heads, d, generator=g)
    perm = torch.randperm(NUM_BLOCKS, generator=g).tolist()

    queries, tables = [], []
    for ctx, q_len in lengths:
        nblocks = -(-ctx // bs)
        blocks, perm = perm[:nblocks], perm[nblocks:]
        queries.append(torch.randn(q_len, heads, d, generator=g))
        tables.append(blocks)
    width = max(len(t) for t in tables)
    block_tables = torch.tensor([t + [0] * (width - len(t)) for t in tables])
    to = dict(dtype=dtype, device=device)
    return (
        torch.cat(queries).to(**to),
        k_pool.to(**to),
        v_pool.to(**to),
        block_tables.to(device),
        torch.tensor([c for c, _ in lengths], device=device),
        torch.tensor([q for _, q in lengths], device=device),
    )


# The shapes the scheduler produces, plus the ones the guide warns about.
CASES = {
    "prefill_on_boundary": [(4, 4), (8, 8), (16, 16)],
    "prefill_spilling":    [(5, 5), (9, 9), (17, 17)],
    "prefill_partial":     [(1, 1), (3, 3), (15, 15)],
    "prefill_ragged":      [(7, 7), (4, 4), (13, 13), (1, 1), (16, 16)],
    "decode_singles":      [(1, 1), (4, 1), (5, 1), (8, 1), (9, 1), (16, 1), (17, 1)],
    "decode_ragged":       [(9, 1), (4, 1), (17, 1), (1, 1), (12, 1), (5, 1)],
    "mixed":               [(6, 1), (10, 10), (4, 1), (9, 5), (16, 16), (13, 1)],
    "partial_query":       [(11, 3), (8, 4), (20, 7)],
    "padding_hits_block0": [(3, 3), (19, 1), (4, 1)],
}


def _compare(fn, lengths, atol=1e-5, rtol=1e-5, **kw):
    q, k_pool, v_pool, tables, ctx, ql = _build(lengths, **kw)
    scale = q.shape[-1] ** -0.5
    ref = paged_attention_gather(q, k_pool, v_pool, tables, ctx, ql, scale)
    out = fn(q, k_pool, v_pool, tables, ctx, ql, scale)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


# ==================================================================== tier 1: mirror
@pytest.mark.phase("S3")
@pytest.mark.parametrize("case", list(CASES))
def test_mirror_matches_gather(case):
    """The kernel's block loop + online softmax, in PyTorch, on every batch shape."""
    _compare(paged_attention_reference, CASES[case])


@pytest.mark.phase("S3")
def test_mirror_block_table_padding_is_never_read():
    """0-padded table entries point at block 0, which some other sequence owns. The
    causal bound must keep the walk inside `ceil(n_keys / block_size)` blocks."""
    for seed in range(3):
        _compare(paged_attention_reference, CASES["padding_hits_block0"], seed=seed)


@pytest.mark.phase("S3")
def test_mirror_grouped_query_heads():
    """head -> kv_head = head // group. kv_heads=1 makes every head share one tile."""
    _compare(paged_attention_reference, [(9, 9), (5, 1), (12, 3)], kv_heads=1)


@pytest.mark.phase("S3")
def test_mirror_production_block_size():
    """CONFIG.block_size (16) with gpt2's head_dim (64): the shapes M1 actually runs."""
    _compare(paged_attention_reference, [(33, 1), (16, 16), (40, 9)], bs=CONFIG.block_size, d=64)


@pytest.mark.phase("S3")
def test_mirror_non_power_of_two_sizes():
    """block_size 6 and head_dim 12 pad to 8 and 16 inside the tile; the surplus lanes
    must be masked out of both the scores and the accumulator."""
    _compare(paged_attention_reference, [(13, 13), (7, 1), (18, 4)], bs=6, d=12)


@pytest.mark.phase("S3")
def test_mirror_empty_batch():
    q = torch.empty(0, H, 8)
    pool = torch.zeros(4, 4, H, 8)
    zero = torch.zeros(0, dtype=torch.long)
    out = paged_attention_reference(q, pool, pool, torch.zeros(0, 1, dtype=torch.long), zero, zero, 0.5)
    assert out.shape == (0, H, 8)


@pytest.mark.phase("S3")
def test_token_map_is_the_last_query_lens_positions():
    """The per-token tables the launcher hands the kernel encode attention.py's rule:
    sequence i's tokens sit at ctx_i - q_i .. ctx_i - 1 of its own context."""
    seq_idx, q_pos, cu = attention_triton._token_map(torch.tensor([6, 10, 4]), torch.tensor([1, 3, 4]))
    assert seq_idx.tolist() == [0, 1, 1, 1, 2, 2, 2, 2]
    assert q_pos.tolist() == [5, 7, 8, 9, 0, 1, 2, 3]
    assert cu.tolist() == [0, 1, 4, 8]


# ================================================================ failing loudly
@pytest.mark.phase("S3")
@pytest.mark.skipif(triton_available(), reason="this box can run the kernel; the loud-failure path is for boxes that cannot")
def test_triton_path_fails_loudly_without_triton_or_cuda(monkeypatch):
    """ATTENTION=triton on a machine that cannot run it must raise with the reason,
    never fall back to gather — a silent fallback would publish the wrong numbers."""
    q, k_pool, v_pool, tables, ctx, ql = _build([(5, 5)])
    with pytest.raises(NotImplementedError, match="triton"):
        paged_attention_triton(q, k_pool, v_pool, tables, ctx, ql, 0.5)
    monkeypatch.setattr(CONFIG, "attention", "triton")
    with pytest.raises(NotImplementedError, match="ATTENTION=triton"):
        paged_attention(q, k_pool, v_pool, tables, ctx, ql, 0.5)


# =================================================================== tier 2: kernel
@needs_cuda_triton
@pytest.mark.phase("S3")
@pytest.mark.parametrize("case", list(CASES))
def test_kernel_matches_gather_fp32(case):
    _compare(paged_attention_triton, CASES[case], atol=1e-4, rtol=1e-4, device="cuda")


@needs_cuda_triton
@pytest.mark.phase("S3")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kernel_matches_gather_half_precision(dtype):
    """The kernel accumulates in fp32 and casts once at the store; the gather path does
    its `probs @ v` in the pool dtype. Tolerance covers that difference, not more."""
    _compare(paged_attention_triton, CASES["mixed"], atol=2e-2, rtol=2e-2, device="cuda", dtype=dtype)


@needs_cuda_triton
@pytest.mark.phase("S3")
def test_kernel_production_shapes():
    """block_size 16, head_dim 64, and a context long enough for many blocks."""
    _compare(paged_attention_triton, [(300, 1), (16, 16), (129, 1), (64, 40)],
             atol=1e-4, rtol=1e-4, bs=CONFIG.block_size, d=64, device="cuda")


@needs_cuda_triton
@pytest.mark.phase("S3")
def test_kernel_grouped_query_heads():
    _compare(paged_attention_triton, [(9, 9), (5, 1), (12, 3)], atol=1e-4, rtol=1e-4, kv_heads=1, device="cuda")


@needs_cuda_triton
@pytest.mark.phase("S3")
def test_kernel_non_power_of_two_sizes():
    _compare(paged_attention_triton, [(13, 13), (7, 1), (18, 4)], atol=1e-4, rtol=1e-4, bs=6, d=12, device="cuda")


@needs_cuda_triton
@pytest.mark.phase("S3")
def test_kernel_block_table_padding_is_never_read():
    for seed in range(3):
        _compare(paged_attention_triton, CASES["padding_hits_block0"], atol=1e-4, rtol=1e-4,
                 seed=seed, device="cuda")


@needs_cuda_triton
@pytest.mark.phase("S3")
def test_kernel_accepts_non_contiguous_query():
    """The executor hands over `query[0].transpose(0, 1)` — strided, not contiguous.
    The launcher passes strides through rather than copying."""
    q, k_pool, v_pool, tables, ctx, ql = _build(CASES["mixed"], device="cuda")
    scale = q.shape[-1] ** -0.5
    strided = q.transpose(0, 1).contiguous().transpose(0, 1)      # same values, [heads, T, d] memory
    assert not strided.is_contiguous()
    ref = paged_attention_gather(q, k_pool, v_pool, tables, ctx, ql, scale)
    out = paged_attention_triton(strided, k_pool, v_pool, tables, ctx, ql, scale)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@needs_cuda_triton
@pytest.mark.phase("S3")
def test_kernel_empty_batch():
    q = torch.empty(0, H, 8, device="cuda")
    pool = torch.zeros(4, 4, H, 8, device="cuda")
    zero = torch.zeros(0, dtype=torch.long, device="cuda")
    out = paged_attention_triton(q, pool, pool, torch.zeros(0, 1, dtype=torch.long, device="cuda"), zero, zero, 0.5)
    assert out.shape == (0, H, 8)
