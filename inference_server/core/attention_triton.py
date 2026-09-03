"""Paged attention in Triton — S3 (Day 14). One program per (query token, head).

Same contract as `paged_attention_gather` (see attention.py's module docstring for the
full signature): a packed `[num_tokens, heads, dim]` query, block-major K/V pools, a
block table, `context_lens`, `query_lens`, `scale` -> `[num_tokens, heads, dim]`. The
gather path materialises every sequence's context as a dense tensor and runs a
textbook softmax over it; this kernel never materialises anything. Each program walks
its sequence's block table one block at a time and folds each `[block_size, head_dim]`
tile of K and V into a running (max, denominator, numerator) — the "online softmax"
of flash attention — so the working set per program is one tile plus one `head_dim`
accumulator, whatever the context length.

WHAT ONE PROGRAM DOES.  Program `(t, h)` owns query token `t` of the packed row and
attention head `h`.

    seq      = seq_idx[t]                      which sequence the token belongs to
    q_pos    = q_pos[t]                        its logical position in that sequence
    n_keys   = q_pos + 1                       causal: it sees keys 0..q_pos inclusive
    kv_head  = h // (num_heads // num_kv_heads)   GQA: several query heads share a KV head

    m = -inf, l = 0, acc = 0
    for blk in 0 .. ceil(n_keys / block_size) - 1:
        phys  = block_tables[seq, blk]
        K, V  = pool[phys, :, kv_head, :]      one [block_size, head_dim] tile each
        s     = K @ q                          [block_size] scores, -inf beyond n_keys
        m'    = max(m, max(s))
        alpha = exp(m - m')                    rescale what was accumulated under the old max
        p     = exp(s - m')
        l     = alpha * l + sum(p)
        acc   = alpha * acc + p @ V
        m     = m'
    out[t, h] = acc / l

Every case the scheduler produces is the same loop. Decode (`query_len == 1`) is the
hot path and is exactly one program per (sequence, head) walking the whole context.
Prefill and P4 recompute (`query_len == context_len`) are `query_len` programs per
head, each with its own causal bound `n_keys`; the kernel does not distinguish them,
and no effort has gone into making prefill fast — it happens once per sequence and
the gather path was never the prefill bottleneck either. Mixed batches work because
the per-token `seq_idx` / `q_pos` tables are derived from each sequence's own two
numbers, the same rule attention.py states.

WHY THE LAUNCHER PRECOMPUTES `seq_idx` AND `q_pos`.  The kernel could binary-search
`cu_query_lens` for its token, but a per-token int32 lookup table is a few hundred
bytes, costs two vectorised ops on the host, and removes a data-dependent loop from
the kernel that an interviewer would otherwise have to verify. `cu_query_lens` is
still computed — it is what the two tables are derived from.

NUMERICS.  Scores, softmax and the accumulator are float32 whatever the pool dtype
(`tl.load(...).to(tl.float32)`), and the result is cast to the query's dtype at the
store. That is the same precision policy as the gather path's
`softmax(..., dtype=float32)`, so an fp32 pool should agree to ~1e-5 and a bf16 pool
to bf16 rounding of the final cast plus the gather path's own bf16 `probs @ v`.

CONSERVATIVE BY DESIGN.  Plain `tl.load` / `tl.store` with masks, explicit
multiply-and-reduce instead of `tl.dot` (a single query row cannot feed a tensor
core anyway, and `tl.dot` has minimum tile sizes), fixed `BLOCK_*` constexprs
derived from `block_size` and `head_dim`, no autotuning, no `tl.dot` layouts, no
TMA, no software pipelining hints. Both `block_size` and `head_dim` are padded up to
a power of two for `tl.arange` and masked back down, so gpt2's 64-dim heads and a
test's 4-token blocks take the same code path as Llama's 128 and the production 16.

=====================================================================================
VERIFICATION STATUS — READ THIS.  The Triton kernel in this file is UNVERIFIED on the
author's development machine. That machine is Apple Silicon (macOS / MPS); Triton
publishes no macOS wheels, so the kernel can be neither compiled nor executed here,
not even under `TRITON_INTERPRET=1`. What HAS been verified here, on CPU:

    `paged_attention_reference` — a pure-PyTorch function that mirrors the kernel's
    algorithm step for step: same per-token / per-head program structure, same block
    loop over the block table, same padded-and-masked tiles, same running max /
    denominator / accumulator updates in the same order. It is tested against
    `paged_attention_gather` in tests/test_paged_attention_triton.py on prefill,
    decode, mixed, block-boundary, non-power-of-two and GQA cases at 1e-5 tolerance.

That catches every *algorithmic* mistake — indexing, the causal bound, tail masking,
GQA head mapping, the rescale order — but it cannot catch a Triton-API mistake (a
wrong broadcast, a dtype the compiler rejects, a loop-carried type mismatch). Those
are caught by the CUDA-only tests in the same file, which are skipped here with a
reason and run the first time the GPU box runs `make test`.

HOW TO VERIFY ON THE GPU BOX (RunPod / Vast, 4090 or A10):

    pip install -r requirements-gpu.txt        # pins torch+cu124 and triton
    ATTENTION=triton make test                 # kernel vs gather on random inputs,
                                               #   then M1 goldens through the kernel
    ATTENTION=triton make bench                # the S3 number for the writeup

If the kernel tests fail on the box, the gather path is untouched: unset ATTENTION
and everything downstream (M1-M5) is exactly what it was before this file existed.
=====================================================================================

See IMPLEMENTATION_GUIDE.md "Day 14" item 4 and PRD FR4 / S3.
"""

from __future__ import annotations

import torch

from inference_server.core.attention import _check

# Triton is imported lazily and guarded: this module must be importable on the dev Mac
# (so the reference and its tests run) even though triton itself cannot be installed
# there. `_TRITON_IMPORT_ERROR` keeps the real reason for the launcher's error message.
try:
    import triton
    import triton.language as tl

    _TRITON_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # ModuleNotFoundError on macOS; ImportError on a broken install
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = exc


def triton_available() -> bool:
    """True iff the kernel can actually run here: triton imports AND a CUDA device
    exists. Triton on a CPU-only Linux box imports fine and then fails at launch, so
    both halves are checked."""
    return triton is not None and torch.cuda.is_available()


# ------------------------------------------------------------------ shared helpers
def _next_pow2(n: int) -> int:
    """`tl.arange` needs a power-of-two extent; the surplus lanes are masked off."""
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _token_map(context_lens: torch.Tensor, query_lens: torch.Tensor):
    """Per-token lookup tables from the per-sequence lengths.

    Returns `(seq_idx, q_pos, cu_query_lens)`, all LongTensors on the input device:

        cu_query_lens[i]  = sum(query_lens[:i])           token t of seq i is packed at
                                                         cu_query_lens[i] + t
        seq_idx[tok]      = the sequence packed token `tok` belongs to
        q_pos[tok]        = its logical position:  ctx_i - q_i + (tok - cu_query_lens[i])

    This is the "last `query_lens[i]` positions of the context" rule from attention.py
    written out per token. The reference and the launcher both use it, so the mirror
    exercises the very same mapping the kernel is handed.
    """
    batch = query_lens.shape[0]
    device = query_lens.device
    cu = torch.zeros(batch + 1, dtype=torch.long, device=device)
    cu[1:] = torch.cumsum(query_lens, 0)
    seq_idx = torch.repeat_interleave(torch.arange(batch, device=device), query_lens)
    tok = torch.arange(int(cu[-1]), device=device)
    q_pos = context_lens[seq_idx] - query_lens[seq_idx] + (tok - cu[seq_idx])
    return seq_idx, q_pos, cu


# ------------------------------------------------------------------ the kernel
if triton is not None:

    @triton.jit
    def _paged_attention_kernel(
        # pointers
        q_ptr, k_ptr, v_ptr, o_ptr,
        block_tables_ptr, seq_idx_ptr, q_pos_ptr,
        # scalars
        scale,
        # strides (elements). q/o: [tokens, heads, dim]; pools: [block, slot, kv_head, dim]
        stride_qt, stride_qh, stride_qd,
        stride_ot, stride_oh, stride_od,
        stride_kb, stride_ks, stride_kh, stride_kd,
        stride_vb, stride_vs, stride_vh, stride_vd,
        stride_bt,                          # block_tables row stride
        # compile-time constants
        GROUP: tl.constexpr,                # num_heads // num_kv_heads
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,           # tokens per pool block (the real one)
        BLOCK_B: tl.constexpr,              # next_pow2(BLOCK_SIZE): tile rows
        BLOCK_D: tl.constexpr,              # next_pow2(HEAD_DIM):   tile cols
    ):
        # ---- which (token, head) this program owns ---------------------------------
        tok = tl.program_id(0)
        head = tl.program_id(1)
        seq = tl.load(seq_idx_ptr + tok)
        q_pos = tl.load(q_pos_ptr + tok)
        kv_head = head // GROUP

        # Causal bound: keys 0 .. q_pos inclusive. The kernel never reads past this, so
        # the tail of the last block and the 0-padding of the block table are unreachable.
        n_keys = q_pos + 1
        num_blocks = (n_keys + BLOCK_SIZE - 1) // BLOCK_SIZE

        # ---- the query row, once, pre-scaled, in float32 --------------------------
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < HEAD_DIM
        q = tl.load(
            q_ptr + tok * stride_qt + head * stride_qh + offs_d * stride_qd,
            mask=mask_d, other=0.0,
        ).to(tl.float32) * scale                                        # [BLOCK_D]

        # ---- online-softmax state ---------------------------------------------------
        # m/l are shape-[1] rather than 0-d so the loop-carried types are unambiguous.
        m_i = tl.zeros([1], dtype=tl.float32) - float("inf")           # running max
        l_i = tl.zeros([1], dtype=tl.float32)                           # running denominator
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)                     # running numerator

        offs_b = tl.arange(0, BLOCK_B)
        for blk in range(0, num_blocks):
            # Logical -> physical through the block table; int64 before it scales a
            # pool stride so a multi-GiB pool cannot overflow int32 offsets.
            phys = tl.load(block_tables_ptr + seq * stride_bt + blk).to(tl.int64)
            kv_pos = blk * BLOCK_SIZE + offs_b
            mask_kv = (offs_b < BLOCK_SIZE) & (kv_pos < n_keys)         # [BLOCK_B]
            mask_tile = mask_kv[:, None] & mask_d[None, :]              # [BLOCK_B, BLOCK_D]

            tile_off = (offs_b[:, None] * stride_ks + kv_head * stride_kh + offs_d[None, :] * stride_kd)
            k = tl.load(k_ptr + phys * stride_kb + tile_off, mask=mask_tile, other=0.0).to(tl.float32)

            # scores for this block: one dot product per key row, masked to -inf.
            s = tl.sum(k * q[None, :], axis=1)                          # [BLOCK_B]
            s = tl.where(mask_kv, s, float("-inf"))

            # Online softmax: fold this block into the running (max, denom, numer).
            m_new = tl.maximum(m_i, tl.max(s, axis=0))                 # [1]
            alpha = tl.exp(m_i - m_new)                                 # [1], 0 on first block
            p = tl.exp(s - m_new)                                       # [BLOCK_B], 0 where masked

            tile_off_v = (offs_b[:, None] * stride_vs + kv_head * stride_vh + offs_d[None, :] * stride_vd)
            v = tl.load(v_ptr + phys * stride_vb + tile_off_v, mask=mask_tile, other=0.0).to(tl.float32)

            l_i = alpha * l_i + tl.sum(p, axis=0)
            acc = alpha * acc + tl.sum(p[:, None] * v, axis=0)          # [BLOCK_D]
            m_i = m_new

        out = acc / l_i                                                 # [BLOCK_D]
        tl.store(
            o_ptr + tok * stride_ot + head * stride_oh + offs_d * stride_od,
            out.to(o_ptr.dtype.element_ty),
            mask=mask_d,
        )


# ------------------------------------------------------------------ the launcher
def paged_attention_triton_launch(
    query: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    query_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Validate, build the per-token tables, launch one program per (token, head).

    Fails loudly — never falls back to the gather path — when the kernel cannot run:
    the whole point of ATTENTION=triton is to measure this kernel, and a silent
    fallback would publish the gather path's numbers under the kernel's name.
    """
    if triton is None:
        raise NotImplementedError(
            "ATTENTION=triton: the S3 kernel cannot run here — triton is not installed "
            f"({_TRITON_IMPORT_ERROR!r}). Triton has no macOS wheels; run this on the CUDA box "
            "with `pip install -r requirements-gpu.txt`, or use ATTENTION=gather."
        )
    if query.device.type != "cuda":
        raise NotImplementedError(
            f"ATTENTION=triton: the S3 kernel needs a CUDA device, got {query.device}. "
            "Use ATTENTION=gather on cpu/mps."
        )
    _check(query, k_pool, block_tables, context_lens, query_lens)
    if k_pool.shape != v_pool.shape:
        raise ValueError(f"k_pool {tuple(k_pool.shape)} and v_pool {tuple(v_pool.shape)} differ")
    for name, t in (("k_pool", k_pool), ("v_pool", v_pool), ("block_tables", block_tables)):
        if t.device != query.device:
            raise ValueError(f"{name} is on {t.device}, query is on {query.device}")

    num_tokens, num_heads, head_dim = query.shape
    _, block_size, num_kv_heads, pool_dim = k_pool.shape
    if pool_dim != head_dim:
        raise ValueError(f"query head_dim {head_dim} != pool head_dim {pool_dim}")
    if num_heads % num_kv_heads:
        raise ValueError(f"{num_heads} query heads not divisible by {num_kv_heads} kv heads")

    out = torch.empty((num_tokens, num_heads, head_dim), dtype=query.dtype, device=query.device)
    if num_tokens == 0:
        return out

    seq_idx, q_pos, _cu_query_lens = _token_map(context_lens.to(query.device), query_lens.to(query.device))
    seq_idx = seq_idx.to(torch.int32).contiguous()
    q_pos = q_pos.to(torch.int32).contiguous()
    tables = block_tables.to(device=query.device, dtype=torch.int32).contiguous()

    grid = (num_tokens, num_heads)
    _paged_attention_kernel[grid](
        query, k_pool, v_pool, out,
        tables, seq_idx, q_pos,
        float(scale),
        query.stride(0), query.stride(1), query.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
        v_pool.stride(0), v_pool.stride(1), v_pool.stride(2), v_pool.stride(3),
        tables.stride(0),
        GROUP=num_heads // num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        BLOCK_B=_next_pow2(block_size),
        BLOCK_D=_next_pow2(head_dim),
        num_warps=4,
    )
    return out


# ------------------------------------------------------------------ the mirror
def paged_attention_reference(
    query: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    query_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """The kernel's algorithm in PyTorch, step for step. Runs anywhere, slowly.

    This is NOT a second reference implementation of attention — that is
    `paged_attention_gather`. It is a transliteration of `_paged_attention_kernel`:
    the same per-(token, head) program, the same padded `[BLOCK_B, BLOCK_D]` tiles with
    the same masks, the same `-inf` scores, the same `m / l / acc` updates in the same
    order. Every line here corresponds to a line in the kernel. Its job is to let the
    kernel's indexing and online-softmax logic be tested on a machine that cannot run
    Triton; a bug in the block loop or the causal bound shows up here first.
    """
    _check(query, k_pool, block_tables, context_lens, query_lens)
    num_tokens, num_heads, head_dim = query.shape
    _, block_size, num_kv_heads, _ = k_pool.shape
    group = num_heads // num_kv_heads
    BLOCK_B, BLOCK_D = _next_pow2(block_size), _next_pow2(head_dim)

    out = torch.empty_like(query)
    if num_tokens == 0:
        return out
    seq_idx, q_pos, _ = _token_map(context_lens, query_lens)

    offs_b = torch.arange(BLOCK_B)
    offs_d = torch.arange(BLOCK_D)
    mask_d = offs_d < head_dim
    neg_inf = torch.tensor(float("-inf"))

    for tok in range(num_tokens):                       # program_id(0)
        seq, pos = int(seq_idx[tok]), int(q_pos[tok])
        n_keys = pos + 1
        num_blocks = (n_keys + block_size - 1) // block_size
        for head in range(num_heads):                   # program_id(1)
            kv_head = head // group

            q = torch.zeros(BLOCK_D)                    # masked load, other=0
            q[mask_d] = query[tok, head].float() * scale

            m_i = torch.full((1,), float("-inf"))
            l_i = torch.zeros(1)
            acc = torch.zeros(BLOCK_D)

            for blk in range(num_blocks):
                phys = int(block_tables[seq, blk])
                kv_pos = blk * block_size + offs_b
                mask_kv = (offs_b < block_size) & (kv_pos < n_keys)
                mask_tile = mask_kv[:, None] & mask_d[None, :]

                k = torch.zeros(BLOCK_B, BLOCK_D)
                k[:block_size, :head_dim] = k_pool[phys, :, kv_head].float()
                k = torch.where(mask_tile, k, torch.zeros(()))          # masked load

                s = (k * q[None, :]).sum(1)
                s = torch.where(mask_kv, s, neg_inf)

                m_new = torch.maximum(m_i, s.max().reshape(1))
                alpha = torch.exp(m_i - m_new)
                p = torch.exp(s - m_new)

                v = torch.zeros(BLOCK_B, BLOCK_D)
                v[:block_size, :head_dim] = v_pool[phys, :, kv_head].float()
                v = torch.where(mask_tile, v, torch.zeros(()))

                l_i = alpha * l_i + p.sum()
                acc = alpha * acc + (p[:, None] * v).sum(0)
                m_i = m_new

            out[tok, head] = (acc / l_i)[mask_d].to(query.dtype)        # masked store
    return out
