"""P3 (Day 8) — the gather kernel against dense attention, with no model in sight.

Random q/k/v on CPU, block_size 4, so a block boundary is every four tokens and the
cases the guide warns about — a sequence ending exactly on a boundary, one ending
mid-block, a partially filled last block — are all reachable by hand.

The reference is deliberately naive: for each sequence, copy its K/V out of the pool
into one contiguous tensor and run textbook causal attention on it. If the paged path
and that agree to float tolerance on every shape the scheduler can produce, the
remaining risk is in *what gets written to the pool*, which is the executor's problem
and M1's job to catch.
"""

from __future__ import annotations

import pytest
import torch

from inference_server.config import CONFIG
from inference_server.core import attention
from inference_server.core.attention import paged_attention, paged_attention_gather

BS = 4          # block size
H = 3           # heads
D = 8           # head_dim
SCALE = D ** -0.5
NUM_BLOCKS = 64


def _dense_causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q: [q_len, H, D] — the LAST q_len positions of a context k/v: [ctx, H, D]."""
    ctx, q_len = k.shape[0], q.shape[0]
    scores = torch.einsum("qhd,khd->hqk", q, k) * SCALE
    q_pos = torch.arange(ctx - q_len, ctx)
    k_pos = torch.arange(ctx)
    scores = scores.masked_fill(~(k_pos[None, :] <= q_pos[:, None])[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)


def _build(lengths: list[tuple[int, int]], kv_heads: int = H, seed: int = 0):
    """Scatter random K/V for each (context_len, query_len) into a random block layout.

    Blocks are drawn without replacement across sequences and deliberately NOT in
    ascending order, so a kernel that assumed a sequence's blocks are contiguous or
    sorted would be caught here rather than by a wrong token at step 200.
    """
    g = torch.Generator().manual_seed(seed)
    k_pool = torch.randn(NUM_BLOCKS, BS, kv_heads, D, generator=g)
    v_pool = torch.randn(NUM_BLOCKS, BS, kv_heads, D, generator=g)
    perm = torch.randperm(NUM_BLOCKS, generator=g).tolist()

    queries, refs, tables, ctx_lens, q_lens = [], [], [], [], []
    for ctx, q_len in lengths:
        nblocks = -(-ctx // BS)
        blocks, perm = perm[:nblocks], perm[nblocks:]
        flat = torch.tensor([b * BS + p % BS for p, b in ((p, blocks[p // BS]) for p in range(ctx))])
        k = k_pool.view(-1, kv_heads, D)[flat]
        v = v_pool.view(-1, kv_heads, D)[flat]
        q = torch.randn(q_len, H, D, generator=g)
        groups = H // kv_heads
        refs.append(_dense_causal(q, k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)))
        queries.append(q)
        tables.append(blocks)
        ctx_lens.append(ctx)
        q_lens.append(q_len)

    width = max(len(t) for t in tables)
    block_tables = torch.tensor([t + [0] * (width - len(t)) for t in tables])
    return (
        torch.cat(queries),
        k_pool,
        v_pool,
        block_tables,
        torch.tensor(ctx_lens),
        torch.tensor(q_lens),
        torch.cat(refs),
    )


def _run(lengths, **kw):
    q, k_pool, v_pool, tables, ctx, ql, ref = _build(lengths, **kw)
    out = paged_attention_gather(q, k_pool, v_pool, tables, ctx, ql, SCALE)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------------ prefill
@pytest.mark.phase("P3")
@pytest.mark.parametrize("ctx", [1, 3, 4, 5, 8, 9, 15, 16, 17])
def test_prefill_single_sequence_across_block_boundaries(ctx):
    """query_len == context_len, causal over the whole prompt. 4, 8 and 16 sit exactly
    on a boundary; 5, 9 and 17 spill one token into a fresh block; 3 and 15 leave the
    last block partially filled."""
    _run([(ctx, ctx)])


@pytest.mark.phase("P3")
def test_prefill_ragged_batch():
    """Several prompts of different lengths in one packed row. Each must attend only
    within its own span — a kernel that leaked across the boundary would produce output
    that is fluent and wrong, which is why the reference is per-sequence."""
    _run([(7, 7), (4, 4), (13, 13), (1, 1), (16, 16)])


# ------------------------------------------------------------------------- decode
@pytest.mark.phase("P3")
@pytest.mark.parametrize("ctx", [1, 4, 5, 8, 9, 16, 17])
def test_decode_single_sequence(ctx):
    """query_len == 1: the new token is the last position of the context and sees
    everything, including itself, which the executor has already written."""
    _run([(ctx, 1)])


@pytest.mark.phase("P3")
def test_decode_ragged_batch_takes_the_batched_path():
    """All-decode batches go through the vectorised gather. Mixed context lengths mean
    the padding beyond each sequence's context_len must be masked, not just the pad in
    the block table — the tail of a real block is garbage too."""
    _run([(9, 1), (4, 1), (17, 1), (1, 1), (12, 1), (5, 1)])


# -------------------------------------------------------------------------- mixed
@pytest.mark.phase("P3")
def test_mixed_prefill_and_decode_in_one_call():
    """The scheduler never produces this today, but the kernel's contract says nothing
    about homogeneity and chunked prefill (future work) will need it. Also the shape a
    P4 recompute produces: a query spanning prompt+output next to plain decodes."""
    _run([(6, 1), (10, 10), (4, 1), (9, 5), (16, 16), (13, 1)])


@pytest.mark.phase("P3")
def test_partial_query_over_a_longer_context():
    """query_len strictly between 1 and context_len — a chunk of a prompt whose earlier
    chunk is already cached. Positions must be counted from the END of the context."""
    _run([(11, 3), (8, 4), (20, 7)])


# ----------------------------------------------------------------------- layouts
@pytest.mark.phase("P3")
def test_block_table_padding_is_never_read():
    """Padding the block table with 0 points every short row at block 0, which some
    other sequence may own. The kernel must bound reads by context_lens so that a
    padded entry never contributes a key. Two seeds so block 0 is actually in use."""
    for seed in range(3):
        _run([(3, 3), (19, 1), (4, 1)], seed=seed)


@pytest.mark.phase("P3")
def test_grouped_query_heads_are_expanded():
    """A pool sized on num_key_value_heads must still serve num_attention_heads.
    gpt2 is 1:1 so M1 cannot see this; the random test can."""
    _run([(9, 9), (5, 1)], kv_heads=1)


@pytest.mark.phase("P3")
def test_empty_batch_returns_empty():
    q = torch.empty(0, H, D)
    pool = torch.zeros(4, BS, H, D)
    out = paged_attention_gather(q, pool, pool, torch.zeros(0, 1, dtype=torch.long),
                                 torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), SCALE)
    assert out.shape == (0, H, D)


# ----------------------------------------------------------------------- contract
@pytest.mark.phase("P3")
def test_query_must_be_counted_in_context():
    """context_lens is the count AFTER this pass. A caller that passed the pre-pass
    count would have the new token attend to nothing including itself; that is a
    contract violation, not a numerical edge case, so it raises."""
    q, k_pool, v_pool, tables, ctx, ql, _ = _build([(4, 4)])
    with pytest.raises(ValueError, match="query_len exceeds context_len"):
        paged_attention_gather(q, k_pool, v_pool, tables, ctx - 1, ql, SCALE)


@pytest.mark.phase("P3")
def test_query_lens_must_sum_to_num_tokens():
    q, k_pool, v_pool, tables, ctx, ql, _ = _build([(4, 4), (2, 2)])
    with pytest.raises(ValueError, match="query_lens sum"):
        paged_attention_gather(q[:-1], k_pool, v_pool, tables, ctx, ql, SCALE)


@pytest.mark.phase("P3")
def test_dispatcher_defaults_to_gather_and_triton_is_named_stretch(monkeypatch):
    q, k_pool, v_pool, tables, ctx, ql, ref = _build([(5, 5)])
    monkeypatch.setattr(CONFIG, "attention", "gather")
    torch.testing.assert_close(paged_attention(q, k_pool, v_pool, tables, ctx, ql, SCALE), ref, atol=1e-5, rtol=1e-5)

    monkeypatch.setattr(CONFIG, "attention", "triton")
    with pytest.raises(NotImplementedError, match="S3"):
        paged_attention(q, k_pool, v_pool, tables, ctx, ql, SCALE)

    monkeypatch.setattr(CONFIG, "attention", "cuda-graphs")
    with pytest.raises(ValueError, match="ATTENTION"):
        paged_attention(q, k_pool, v_pool, tables, ctx, ql, SCALE)


@pytest.mark.phase("P3")
def test_triton_shares_the_gather_signature():
    """S3 fills in the body; the executor must not have to change. Pin the arity."""
    import inspect

    gather = inspect.signature(attention.paged_attention_gather)
    triton = inspect.signature(attention.paged_attention_triton)
    assert list(gather.parameters) == list(triton.parameters) == list(inspect.signature(paged_attention).parameters)
