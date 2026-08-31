"""Executor — P2 (Days 3-5). Given a set of sequences, run exactly one forward pass.

Knows nothing about scheduling — no admission, no eviction, no policy. That separation
is what lets P3 swap the attention path underneath without touching the scheduler.

THE RAGGED PROBLEM (Day 4, the sprint's largest correctness risk). Under static batching
every row started together and was the same length. Here they are not: one sequence is on
decode step 3 with 40 cached tokens while its neighbour is on step 200 with 700. They
still have to share one forward pass, which means one rectangular tensor.

The convention that makes it work, and that every method here assumes:

    The KV cache is [batch, heads, L, dim]. Row i holds its `num_cached` tokens in the
    LAST num_cached columns. The unused prefix is left-padding — real memory, garbage
    contents, masked out on every pass.

Left rather than right because it puts every row's newest token in the final column, so
`logits[:, -1]` is the right sample point for all rows at once no matter how ragged they
are. Right-padding would need a per-row gather and get the causal mask wrong.

Membership changes cost a copy; steps do not. Evicting rebuilds the batch dimension, and
merging a prefill into the decode set concatenates along it — both rare next to the
number of decode steps between them. P3 is what removes those copies, by making a
sequence's memory a list of block indices instead of a slice of one tensor.

See IMPLEMENTATION_GUIDE.md "Days 3-5".
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from inference_server.config import CONFIG
from inference_server.core.sequence import Sequence


class Executor:
    """One forward pass, no policy."""

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = CONFIG.device
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        #: Rows currently materialised in `_kv`, in cache-row order. The scheduler's
        #: running list is the truth; this is reconciled against it on every decode.
        self._batch: list[Sequence] = []
        self._kv: DynamicCache | None = None

    # ---------------------------------------------------------------------- interface
    @torch.inference_mode()
    def execute(self, seqs: list[Sequence]) -> list[int]:
        """Advance every sequence by exactly one token. Returns the sampled token IDs.

        The caller guarantees the batch is homogeneous — all prefill or all decode —
        because the two have different input widths and cannot share a pass. That is a
        scheduling decision, so it is asserted here rather than handled here.
        """
        if not seqs:
            return []

        phase = seqs[0].needs_prefill
        assert all(s.needs_prefill is phase for s in seqs), (
            "mixed prefill/decode batch — the scheduler must split these"
        )
        # @torch.inference_mode() is not decoration. torch.set_grad_enabled in model.py
        # is THREAD-LOCAL and the step loop runs in a worker thread, so the global switch
        # set at load time does not reach here. Without this the loop builds an autograd
        # graph across every decode step and retains each step's activations — measured
        # 4x slowdown and memory linear in output length back in P1.
        return self._prefill(seqs) if phase else self._decode(seqs)

    def reset(self) -> None:
        """Drop all cache state. Called when the pool goes idle so a finished burst does
        not hold its KV memory until the next arrival."""
        self._batch = []
        self._kv = None

    # ------------------------------------------------------------------------ prefill
    def _prefill(self, seqs: list[Sequence]) -> list[int]:
        """One padded pass over several whole prompts, then merge into the decode set.

        Prompts of different lengths pad against each other here, which is the same waste
        static batching had — but it is paid once per sequence rather than on every step,
        which is why it is tolerable and the decode-side version was not.
        """
        input_ids, attn = self._pad_left([s.prompt_token_ids for s in seqs])

        # Positions count real tokens, not columns. cumsum over the mask gives each real
        # token its rank among real tokens; pad slots clamp to 0 and are masked anyway.
        # Numbering by column would tell a padded row its first token is at position 3.
        position_ids = (attn.cumsum(-1) - 1).clamp(min=0)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attn,
            position_ids=position_ids,
            use_cache=True,
        )
        for s in seqs:
            # Its own prompt length, never the padded width — the pad columns exist in
            # the tensor but are not this sequence's tokens.
            s.num_cached = s.prompt_len

        self._absorb(seqs, out.past_key_values)
        return out.logits[:, -1, :].argmax(-1).tolist()

    # ------------------------------------------------------------------------- decode
    def _decode(self, seqs: list[Sequence]) -> list[int]:
        self._sync_rows(seqs)
        assert self._kv is not None

        cache_len = self._cache_len()
        n = len(seqs)

        input_ids = torch.tensor(
            [[s.output_token_ids[-1]] for s in seqs], device=self.device
        )
        # Each row's own position, taken from its own token count. Two rows in this pass
        # are at different positions; a shared counter would be right for at most one.
        position_ids = torch.tensor([[s.num_cached] for s in seqs], device=self.device)

        # Mask spans the whole cache plus the token being added. Row i's real tokens are
        # the last num_cached of the cache — everything before is left-pad it must not
        # attend to. Get this wrong and the row reads another sequence's garbage as
        # context: no crash, fluent output, silently different from running alone.
        attn = self._cache_mask(seqs, cache_len)
        attn = torch.cat([attn, torch.ones(n, 1, dtype=attn.dtype, device=self.device)], dim=1)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attn,
            position_ids=position_ids,
            past_key_values=self._kv,
            use_cache=True,
        )
        self._kv = out.past_key_values
        for s in seqs:
            # The executor owns this counter because it is the only thing that knows the
            # pass actually ran. A scheduler that advanced it optimistically would
            # desynchronise from the cache on the first failed step.
            s.num_cached += 1

        # Greedy. Sampling params (temperature, top-p) would land here and nowhere else;
        # keeping them out of the scheduler is why M1 can assert exact token IDs at all.
        return out.logits[:, -1, :].argmax(-1).tolist()

    # ------------------------------------------------------------------ tensor helpers
    def _pad_left(self, rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, row in enumerate(rows):
            ids[i, width - len(row):] = torch.tensor(row, dtype=torch.long)
            mask[i, width - len(row):] = 1
        return ids.to(self.device), mask.to(self.device)

    def _cache_mask(self, seqs: list[Sequence], cache_len: int) -> torch.Tensor:
        """1 for the last `num_cached` columns of each row, 0 for its left-pad."""
        cols = torch.arange(cache_len, device=self.device).unsqueeze(0)
        starts = torch.tensor(
            [cache_len - s.num_cached for s in seqs], device=self.device
        ).unsqueeze(1)
        return (cols >= starts).long()

    def _cache_len(self) -> int:
        assert self._kv is not None
        return self._kv.layers[0].keys.shape[-2]

    @staticmethod
    def _pad_cache_left(kv: DynamicCache, amount: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Grow every layer's sequence axis on the left, preserving the right-alignment
        invariant. Tensors are [batch, heads, seq, dim], so the pad tuple targets the
        second-to-last axis."""
        pad = (0, 0, amount, 0)
        return [
            (F.pad(layer.keys, pad), F.pad(layer.values, pad)) if amount else
            (layer.keys, layer.values)
            for layer in kv.layers
        ]

    def _absorb(self, new_seqs: list[Sequence], new_kv: DynamicCache) -> None:
        """Concatenate freshly prefilled rows onto the decode cache.

        The two sides rarely have the same length — a prompt of 200 tokens meets rows
        that have been decoding for 500 — so the shorter is left-padded up to the longer
        before they can share a tensor. This is the one place the batch dimension grows.
        """
        if self._kv is None or not self._batch:
            self._kv, self._batch = new_kv, list(new_seqs)
            return

        have, incoming = self._cache_len(), new_kv.layers[0].keys.shape[-2]
        target = max(have, incoming)
        left = self._pad_cache_left(self._kv, target - have)
        right = self._pad_cache_left(new_kv, target - incoming)

        self._kv = DynamicCache(
            [
                (torch.cat([lk, rk], dim=0), torch.cat([lv, rv], dim=0))
                for (lk, lv), (rk, rv) in zip(left, right)
            ]
        )
        self._batch = self._batch + list(new_seqs)

    def _sync_rows(self, seqs: list[Sequence]) -> None:
        """Make the cache's rows match the set the scheduler wants to run, in order.

        Sequences the scheduler evicted are dropped here; this is where their KV memory
        is actually reclaimed, one step after they finished. Everything requested must
        already be in the cache, because a sequence reaches decode only by prefilling.
        """
        want = [s.seq_id for s in seqs]
        have = [s.seq_id for s in self._batch]
        if want == have:
            return

        assert self._kv is not None
        position = {sid: i for i, sid in enumerate(have)}
        missing = [sid for sid in want if sid not in position]
        assert not missing, f"decode requested for sequences that never prefilled: {missing}"

        keep = [position[sid] for sid in want]
        self._kv.batch_select_indices(torch.tensor(keep, device=self.device))
        self._batch = [self._batch[i] for i in keep]
        self._trim_common_pad()

    def _trim_common_pad(self) -> None:
        """Drop left-pad columns no surviving row uses.

        Without this, evicting the longest sequence leaves every remaining row carrying
        its length forever: the tensor never shrinks, and each step attends over columns
        that are masked for all of them. Pure waste, and it compounds over a long run.
        """
        if not self._batch or self._kv is None:
            return
        dead = self._cache_len() - max(s.num_cached for s in self._batch)
        if dead <= 0:
            return
        self._kv = DynamicCache(
            [(layer.keys[:, :, dead:, :], layer.values[:, :, dead:, :]) for layer in self._kv.layers]
        )
