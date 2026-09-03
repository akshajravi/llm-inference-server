"""Paged attention — P3 (Days 8-9). Attention that reads K/V through a block table.

Under paging a sequence's keys and values are no longer one contiguous slab; they are
scattered across the pool in `block_size`-token pieces, and only the sequence's block
table knows where. Attention therefore has one extra job: gather the pieces back into
logical order before doing the usual `softmax(q k^T * scale) v`. Everything else — the
math, the causal rule, the float32 accumulation — is exactly HuggingFace's eager
attention, which is what the M1 goldens were produced with.

Two implementations, deliberately ordered:

  1. paged_attention_gather()  — PyTorch, correct and slow. THIS is the shipping path.
     It satisfies M3 and unblocks P4, which is what the sprint is graded on.
  2. paged_attention_triton()  — S3, stretch (Day 14 only). Same signature; must match
     the gather path on random inputs before it is allowed near M1.

`paged_attention()` dispatches between them on CONFIG.attention.

THE SIGNATURE. Both kernels take and return exactly this (vLLM's convention, no padding
on the token axis so a ragged batch costs nothing to pack)::

    paged_attention(
        query:        [num_tokens, num_heads, head_dim]      new tokens of every sequence,
                                                             concatenated in batch order
        k_pool:       [num_blocks, block_size, num_kv_heads, head_dim]   PagedKVPool.k[layer]
        v_pool:       same shape                                          PagedKVPool.v[layer]
        block_tables: LongTensor [batch, max_blocks]         row i = seq i's physical blocks
                                                             in logical order, right-padded
                                                             with 0. Zero rather than -1 so
                                                             that a kernel which reads a pad
                                                             entry lands on a real block
                                                             instead of out of bounds; the
                                                             pad is never *used* because
                                                             context_lens bounds every read.
        context_lens: LongTensor [batch]                     tokens in the cache for seq i
                                                             INCLUDING the ones written on
                                                             this pass — i.e. num_cached
                                                             after the pass, not before
        query_lens:   LongTensor [batch]                     how many of `num_tokens` belong
                                                             to seq i; sum == num_tokens
        scale:        float                                  usually head_dim ** -0.5
    ) -> [num_tokens, num_heads, head_dim]

Sequence i's `query_lens[i]` tokens are its LAST `query_lens[i]` positions: they occupy
logical positions `context_lens[i] - query_lens[i] .. context_lens[i] - 1`. That single
rule covers every case the scheduler produces —

    prefill:  query_len == context_len     causal over the whole prompt
    decode:   query_len == 1               one token attending to everything before it
    recompute (P4): query_len == context_len again, over prompt + output so far

— and mixed lengths within one batch, because each sequence's positions are derived from
its own two numbers. Nothing here knows whether the batch is homogeneous.

PRECONDITION the caller owns: the K/V for the query tokens must already be in the pool
when this is called. The executor writes them in `Cache.update` before the attention
hook runs, so the gather sees a complete context. Reading before writing would be
attending to whatever the previous tenant of that block left behind.

No sequences, no scheduler, no model here. Tensors in, tensor out — so the unit tests
can hit block boundaries with random inputs on CPU in milliseconds.

See IMPLEMENTATION_GUIDE.md "Days 6-9" and "Day 14".
"""

from __future__ import annotations

import torch

from inference_server.config import CONFIG


def paged_attention(
    query: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    query_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Dispatch on CONFIG.attention. The executor calls this and nothing else, so
    switching kernels is an environment variable rather than an edit."""
    if CONFIG.attention == "gather":
        return paged_attention_gather(query, k_pool, v_pool, block_tables, context_lens, query_lens, scale)
    if CONFIG.attention == "triton":
        return paged_attention_triton(query, k_pool, v_pool, block_tables, context_lens, query_lens, scale)
    raise ValueError(f"unknown ATTENTION={CONFIG.attention!r}; expected 'gather' or 'triton'")


# ----------------------------------------------------------------------- gather path
def paged_attention_gather(
    query: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    query_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """The shipping path: index_select the blocks, then dense masked attention.

    Two shapes of batch get two code paths, chosen by memory rather than speed:

    - **All decode** (every query_len == 1 — the overwhelmingly common step) is done as
      one batched op. Every sequence's blocks are gathered at once into a
      [batch, max_blocks * block_size, ...] tensor and the padding beyond each
      context_len is masked. Scores are [batch, heads, 1, max_ctx]: tiny.

    - **Anything else** (prefill, recompute, or a mixed batch) loops over sequences.
      A batched prefill would need scores of [batch, heads, max_q, max_ctx], which for
      32 prompts at max_seq_len is 1.6 GiB of float32 on top of a 2.4 GiB pool — the
      loop bounds that to one sequence at a time. Prefill happens once per sequence,
      so the per-op overhead is paid once, not per token.

    The causal rule in both paths is the same one: query token t of sequence i sits at
    logical position `ctx_i - q_i + t` and may attend to keys at positions `<= that`.
    Masked entries get -inf, which after softmax is exactly the 0 that HF's finfo.min
    additive mask also produces — no numerical difference on the M1 goldens.
    """
    _check(query, k_pool, block_tables, context_lens, query_lens)
    if query.shape[0] == 0:
        return query.new_empty((0, query.shape[1], query.shape[2]))

    if bool((query_lens == 1).all()):
        return _decode_batched(query, k_pool, v_pool, block_tables, context_lens, scale)
    return _ragged_loop(query, k_pool, v_pool, block_tables, context_lens, query_lens, scale)


def _decode_batched(query, k_pool, v_pool, block_tables, context_lens, scale) -> torch.Tensor:
    """query is [batch, heads, dim] here (one token per sequence)."""
    batch, num_heads, head_dim = query.shape
    block_size = k_pool.shape[1]
    max_ctx = int(context_lens.max())
    max_blocks = -(-max_ctx // block_size)
    tables = block_tables[:, :max_blocks]

    # [batch, max_blocks, block_size, kv_heads, dim] -> [batch, max_blocks*block_size, kv_heads, dim]
    k = k_pool[tables].flatten(1, 2)[:, :max_ctx]
    v = v_pool[tables].flatten(1, 2)[:, :max_ctx]
    k = _expand_kv_heads(k, num_heads).transpose(1, 2)     # [batch, heads, ctx, dim]
    v = _expand_kv_heads(v, num_heads).transpose(1, 2)

    q = query.unsqueeze(2)                                  # [batch, heads, 1, dim]
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale   # [batch, heads, 1, ctx]

    # Key j is real for sequence i iff j < context_lens[i]. Beyond that is either the
    # unused tail of the last block or another sequence's padding — never attended.
    valid = torch.arange(max_ctx, device=query.device)[None, :] < context_lens[:, None]
    scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))

    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
    return torch.matmul(probs, v).squeeze(2)                # [batch, heads, dim]


def _ragged_loop(query, k_pool, v_pool, block_tables, context_lens, query_lens, scale) -> torch.Tensor:
    num_heads = query.shape[1]
    block_size = k_pool.shape[1]
    out = torch.empty_like(query)
    start = 0
    for i in range(block_tables.shape[0]):
        q_len = int(query_lens[i])
        ctx = int(context_lens[i])
        if q_len == 0:
            continue
        nblocks = -(-ctx // block_size)
        blocks = block_tables[i, :nblocks]

        # index_select over the block table, then drop the unused tail of the last block.
        k = k_pool[blocks].flatten(0, 1)[:ctx]              # [ctx, kv_heads, dim]
        v = v_pool[blocks].flatten(0, 1)[:ctx]
        k = _expand_kv_heads(k, num_heads).transpose(0, 1)  # [heads, ctx, dim]
        v = _expand_kv_heads(v, num_heads).transpose(0, 1)

        q = query[start:start + q_len].transpose(0, 1)      # [heads, q_len, dim]
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale   # [heads, q_len, ctx]

        # Query token t is at logical position ctx - q_len + t; it sees keys 0..that.
        q_pos = torch.arange(ctx - q_len, ctx, device=query.device)
        k_pos = torch.arange(ctx, device=query.device)
        causal = k_pos[None, :] <= q_pos[:, None]           # [q_len, ctx]
        scores = scores.masked_fill(~causal[None], float("-inf"))

        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
        out[start:start + q_len] = torch.matmul(probs, v).transpose(0, 1)
        start += q_len
    return out


def _expand_kv_heads(kv: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Grouped-query models cache fewer heads than they attend with; repeat each KV
    head for its group. gpt2 has num_kv_heads == num_heads and this is a no-op, but the
    pool is sized off `num_key_value_heads` so the attention must honour it too."""
    kv_heads = kv.shape[-2]
    if kv_heads == num_heads:
        return kv
    if num_heads % kv_heads:
        raise ValueError(f"{num_heads} query heads not divisible by {kv_heads} kv heads")
    return kv.repeat_interleave(num_heads // kv_heads, dim=-2)


def _check(query, k_pool, block_tables, context_lens, query_lens) -> None:
    """Shape assertions, because every one of these fails silently otherwise: a wrong
    context_len reads a neighbour's block, a wrong query_len samples the wrong logit."""
    if query.dim() != 3:
        raise ValueError(f"query must be [num_tokens, heads, dim], got {tuple(query.shape)}")
    if k_pool.dim() != 4:
        raise ValueError(f"k_pool must be [blocks, block_size, kv_heads, dim], got {tuple(k_pool.shape)}")
    batch = block_tables.shape[0]
    if context_lens.shape != (batch,) or query_lens.shape != (batch,):
        raise ValueError("context_lens and query_lens must be [batch] to match block_tables")
    if int(query_lens.sum()) != query.shape[0]:
        raise ValueError(f"query_lens sum to {int(query_lens.sum())} but query has {query.shape[0]} tokens")
    if bool((query_lens > context_lens).any()):
        raise ValueError("query_len exceeds context_len — the query tokens must be counted in the context")
    block_size = k_pool.shape[1]
    need = -(-context_lens // block_size)
    if bool((need > block_tables.shape[1]).any()):
        raise ValueError("a sequence's context needs more blocks than its block_tables row holds")


# ----------------------------------------------------------------------- triton path
def paged_attention_triton(
    query: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    query_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """S3 (Day 14): one Triton program per (query token, head), walking the block
    table with an online softmax. Lives in attention_triton.py; this is the seam the
    dispatcher sees, with the same signature as the gather path so the executor did
    not change when it landed.

    Requires triton and a CUDA device; anywhere else (this repo's dev Mac included) it
    raises NotImplementedError naming the reason rather than falling back, so
    ATTENTION=triton can never quietly publish the gather path's numbers. The kernel
    is verified against `paged_attention_gather` on random inputs by
    tests/test_paged_attention_triton.py on CUDA; on machines without triton the same
    file verifies a PyTorch mirror of the kernel's algorithm instead. See the
    attention_triton module docstring for what is and is not verified where.
    """
    # Imported here, not at module top: attention_triton imports `_check` from this
    # module, and it must stay importable on a machine with no triton at all.
    from inference_server.core.attention_triton import paged_attention_triton_launch

    return paged_attention_triton_launch(query, k_pool, v_pool, block_tables, context_lens, query_lens, scale)
