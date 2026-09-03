"""Paged executor — P3 (Days 8-9). One forward pass whose KV lives in the block pool.

Same contract as executor.py — `execute(seqs) -> one greedy token per sequence`,
homogeneous prefill-or-decode batches, `num_cached` advanced here and nowhere else —
but the memory model underneath is the whole point of P3:

    P2:  KV is [batch, heads, L, dim], one row per sequence, left-padded to the longest.
         Membership changes copy; every row reserves the worst case.
    P3:  KV is PagedKVPool: [num_blocks, block_size, heads, dim] per layer, allocated
         once. A sequence is a list of block indices. Nothing is padded, nothing is
         copied on membership change, and the tensor never grows.

THE PACKING. There is no batch dimension in the forward pass at all. Every sequence's
new tokens are concatenated into ONE row — `input_ids [1, total_tokens]` — with a
per-token `position_ids` so that each sequence's tokens know their own logical
position. The model's attention is replaced (see below) with a kernel that knows where
sequence boundaries are, so tokens from different sequences never see each other even
though they share a row. Sampling reads the last logit of each sequence's span.

That is what makes prefill and decode the same code path here: a decode batch of 32 is
32 tokens in the row, one per sequence; a prefill of 3 prompts is their lengths summed.
The P2 executor needed two methods and a merge step; this one needs `_pack`.

HOW THE HUGGINGFACE MODEL IS HOOKED (rather than reimplemented). Two seams in
transformers make it possible to reuse the loaded GPT-2 weights untouched:

  1. `AttentionInterface.register(name, fn)` — GPT2Attention looks its attention
     function up by `config._attn_implementation`, and a registered name that is NOT
     in the mask registry makes `create_causal_mask` return no mask at all (it assumes
     a custom kernel brings its own). Ours does: the block tables and lengths ride in
     through `**kwargs` of `model(...)`, which GPT-2 forwards untouched down to the
     attention call.
  2. `Cache.update(key, value, layer_idx)` — called once per layer with the new tokens'
     K/V before attention runs. `_PagedCache.update` scatters them into the pool at
     the slots this pass was allocated and returns them unchanged; the attention hook
     ignores what it is handed and reads everything, new tokens included, back out of
     the pool through the block tables. Writing first and reading through the table
     means the kernel sees one consistent picture of the context.

The alternative — a hand-written GPT-2 forward over the module's weights — was
rejected because it duplicates ~60 lines of transformer arithmetic that would need
re-verifying against M1 for every dtype and device, and would silently drift from
whatever numerics `model.load()` gives the other engines.

THE CAVEAT that comes with the hook: `config._attn_implementation` is process-global
state on a model object shared by every engine in the process (`model.load()` is an
lru_cache). It is flipped for exactly the duration of one forward call and restored in
`finally`, so the contiguous engines — which the test suite builds in the same process —
still see their default kernel. Two engines stepping *concurrently* on one model would
race on it; nothing in this repo does that (the bench builds engines one at a time and
the suite shuts each down before the next), and it is documented here rather than
guarded with a lock because a lock across engines would hide the real fix, which is
one model per engine.

STATELESS BY DESIGN. Nothing survives between `execute` calls: no cached batch order, no
row mapping, no cache object. Every step rebuilds its inputs from each sequence's
`block_table`, `num_cached` and `next_input_ids`. That is a P4 requirement, not
tidiness: preemption may free a running sequence's blocks and reset `num_cached` to 0
(it comes back as a prefill over prompt + output) or swap it back in with a different
set of physical blocks holding the same contents. An executor that remembered anything
about the previous step would be wrong on the step after either of those.

See IMPLEMENTATION_GUIDE.md "Days 6-9".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AttentionInterface, Cache

from inference_server.config import CONFIG
from inference_server.core.attention import paged_attention
from inference_server.core.kv_pool import PagedKVPool
from inference_server.core.sequence import Sequence

#: The name GPT-2 looks up in AttentionInterface while a paged forward is running.
#: Prefixed so it cannot collide with transformers' own "paged|..." entries.
ATTN_IMPL = "inference_server_paged"


@dataclass(frozen=True)
class _StepContext:
    """Everything one forward pass needs to know about the batch, built fresh per step.

    Frozen because it is handed into the model as a kwarg and read from inside twelve
    attention calls; nothing may mutate it partway through a pass.
    """

    write_slots: torch.Tensor    # LongTensor [total_tokens]: flat pool slot per new token
    block_tables: torch.Tensor   # LongTensor [batch, max_blocks], 0-padded
    context_lens: torch.Tensor   # LongTensor [batch]: num_cached AFTER this pass
    query_lens: torch.Tensor     # LongTensor [batch]: new tokens per sequence
    pool: PagedKVPool


class _PagedCache(Cache):
    """A `Cache` whose only job is to intercept `update` and write into the pool.

    transformers calls `update(key, value, layer_idx)` from inside each attention layer
    with the new tokens' projections, shaped [1, heads, total_tokens, dim]. This writes
    them to `pool.k[layer_idx]` / `pool.v[layer_idx]` at the slots the scheduler
    allocated for this pass, and returns them unchanged — the attention hook does not
    use the return value, it reads the pool.

    It holds no layers and answers nothing else: `get_seq_length` is defined only
    because GPT-2 would call it to invent position IDs if we forgot to pass them, and
    a loud error there beats a silent off-by-`past_seen_tokens`.
    """

    def __init__(self, ctx: _StepContext) -> None:
        super().__init__(layers=[])
        self.ctx = ctx

    def update(self, key_states, value_states, layer_idx: int, *args, **kwargs):
        heads, dim = key_states.shape[1], key_states.shape[3]
        pool = self.ctx.pool
        # [1, heads, T, dim] -> [T, heads, dim], the pool's per-slot layout.
        k = key_states[0].transpose(0, 1)
        v = value_states[0].transpose(0, 1)
        # A view of the block-major pool as flat slots; index_copy_ writes through it.
        pool.k[layer_idx].view(-1, heads, dim).index_copy_(0, self.ctx.write_slots, k)
        pool.v[layer_idx].view(-1, heads, dim).index_copy_(0, self.ctx.write_slots, v)
        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        raise RuntimeError("PagedExecutor must pass explicit position_ids; the pool has no single length")

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0):
        raise RuntimeError("no mask is built for the paged kernel; create_causal_mask should have early-exited")


def _paged_attention_forward(module, query, key, value, attention_mask, scaling=None, **kwargs):
    """The function registered under ATTN_IMPL. transformers' attention-interface
    signature: [1, heads, T, dim] tensors in, ([1, T, heads, dim], weights) out.

    `key`/`value` are the return of `_PagedCache.update` and are ignored on purpose —
    reading the new tokens back out of the pool alongside the old ones is what proves
    the write landed where the block table says it did. If the scatter and the gather
    disagreed, M1 would fail here rather than in some later step.
    """
    ctx: _StepContext = kwargs["paged_ctx"]
    if scaling is None:
        scaling = query.shape[-1] ** -0.5
    q = query[0].transpose(0, 1)                                 # [T, heads, dim]
    out = paged_attention(
        q,
        ctx.pool.k[module.layer_idx],
        ctx.pool.v[module.layer_idx],
        ctx.block_tables,
        ctx.context_lens,
        ctx.query_lens,
        scaling,
    )
    return out.unsqueeze(0), None                                # [1, T, heads, dim]


AttentionInterface.register(ATTN_IMPL, _paged_attention_forward)


class PagedExecutor:
    """One forward pass over the pool, no policy. Drop-in for `Executor`."""

    def __init__(self, model, tokenizer, pool: PagedKVPool) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.pool = pool
        self.device = CONFIG.device

    # ---------------------------------------------------------------------- interface
    @torch.inference_mode()
    def execute(self, seqs: list[Sequence]) -> list[int]:
        """Advance every sequence by exactly one token. Returns the sampled token IDs.

        Homogeneity is asserted for parity with the P2 executor — the scheduler promises
        it — but this path would in fact run a mixed batch correctly, because the packed
        row and per-sequence query_lens never assumed otherwise. That is left as an
        observation rather than a feature: chunked prefill is named future work.
        """
        if not seqs:
            return []
        # `is_prefilling`, not `needs_prefill`: after a prefix-cache hit (S1) a sequence
        # has `num_cached > 0` and still feeds a multi-token remainder, and it belongs
        # with the prefills. This path would run a mixed batch correctly anyway.
        phase = seqs[0].is_prefilling
        assert all(s.is_prefilling is phase for s in seqs), (
            "mixed prefill/decode batch — the scheduler must split these"
        )
        # @torch.inference_mode() for the same reason as executor.py: grad mode is
        # thread-local, and the step loop runs on a worker thread.

        input_ids, position_ids, ctx, ends = self._pack(seqs)
        logits = self._forward(input_ids, position_ids, ctx)

        # The executor owns this counter because it is the only thing that knows the
        # pass actually ran; the scheduler sized the block tables off the same numbers.
        for s in seqs:
            s.num_cached += len(s.next_input_ids)

        # Each sequence's last token is at ends[i] - 1 of the packed row. Greedy; any
        # sampling params would land here and nowhere else (M1 asserts exact IDs).
        return logits[0, ends - 1, :].argmax(-1).tolist()

    def reset(self) -> None:
        """Nothing to drop. The pool is allocated for the process lifetime and blocks
        return to the allocator through the scheduler's eviction, not through here —
        which is exactly the "memory is flat by construction" property M4 wants."""

    # ------------------------------------------------------------------------ packing
    def _pack(self, seqs: list[Sequence]):
        """Flatten the batch into one row and describe it to the attention kernel.

        Every number here comes from the sequence itself — never from remembered state
        — so a sequence that was preempted, recomputed, or swapped since the last step
        is described correctly without this method knowing any of that happened.
        """
        ids: list[int] = []
        pos: list[int] = []
        slots: list[int] = []
        query_lens: list[int] = []
        context_lens: list[int] = []
        tables: list[list[int]] = []

        for s in seqs:
            new = s.next_input_ids
            table = s.block_table
            assert table is not None, f"{s.seq_id} reached the paged executor without a block table"
            after = s.num_cached + len(new)
            assert table.capacity >= after, (
                f"{s.seq_id}: block table holds {table.capacity} slots, pass needs {after} — "
                "the scheduler must call ensure_capacity before execute"
            )
            # Copy-on-write guard (S1, FR5): every block this pass writes into must be
            # ours alone. Under full-block-only prefix sharing it always already is —
            # the write starts on a block boundary in a freshly allocated block — so
            # this is one refcount read per written block, not a copy. It is here so
            # that a policy sharing partial blocks could never corrupt a neighbour.
            bs = table.block_size
            for logical in range(s.num_cached // bs, (after - 1) // bs + 1):
                table.ensure_private(logical, self._copy_block)
            ids.extend(new)
            pos.extend(range(s.num_cached, after))
            # The slots for the NEW tokens only: positions num_cached .. after-1.
            slots.extend(table.slots(after)[s.num_cached:])
            query_lens.append(len(new))
            context_lens.append(after)
            tables.append(list(table.blocks))

        width = max(len(t) for t in tables)
        padded = [t + [0] * (width - len(t)) for t in tables]     # 0-pad; see attention.py
        dev = self.device
        ctx = _StepContext(
            write_slots=torch.tensor(slots, dtype=torch.long, device=dev),
            block_tables=torch.tensor(padded, dtype=torch.long, device=dev),
            context_lens=torch.tensor(context_lens, dtype=torch.long, device=dev),
            query_lens=torch.tensor(query_lens, dtype=torch.long, device=dev),
            pool=self.pool,
        )
        ends = torch.tensor(query_lens, dtype=torch.long, device=dev).cumsum(0)
        input_ids = torch.tensor([ids], dtype=torch.long, device=dev)
        position_ids = torch.tensor([pos], dtype=torch.long, device=dev)
        return input_ids, position_ids, ctx, ends

    def _copy_block(self, src: int, dst: int) -> None:
        """Move one block's K/V, every layer, for `BlockTable.ensure_private`."""
        for layer in self.pool.k:
            layer[dst].copy_(layer[src])
        for layer in self.pool.v:
            layer[dst].copy_(layer[src])

    # ------------------------------------------------------------------------ forward
    def _forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor, ctx: _StepContext) -> torch.Tensor:
        """Run the shared model with attention swapped for the paged kernel.

        The swap is scoped to this call and undone in `finally`, because the model
        object is shared with every other engine in the process (see module docstring).
        No attention_mask is passed: the registered kernel is not in transformers' mask
        registry, so `create_causal_mask` early-exits with None, and causality plus
        sequence isolation are enforced inside `paged_attention` from `ctx` instead.
        """
        config = self.model.config
        previous = config._attn_implementation
        config._attn_implementation = ATTN_IMPL
        try:
            out = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=_PagedCache(ctx),
                use_cache=True,
                paged_ctx=ctx,
            )
        finally:
            config._attn_implementation = previous
        return out.logits
