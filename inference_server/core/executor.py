"""Executor — P2 (Days 3-5). Given a set of sequences, run exactly one forward pass.

Knows nothing about scheduling — no admission, no eviction, no policy. That separation
is what lets P3 swap the attention path underneath without touching the scheduler.

The hard part is ragged batching: sequences in one batch have different lengths *and*
different KV cache lengths. Pad to the batch max, mask correctly. This is the single
largest correctness risk in the sprint; Day 4 is budgeted for it.

DAY 3 STATE — read this before trusting a benchmark from this file. `execute()` takes a
list and is called with a list, but internally it still runs one sequence per forward
pass. That is a stepping stone, deliberately: the step loop, the scheduler, and the
mutable batch are all being debugged today, and doing ragged masking at the same time
means a wrong token cannot be attributed to either. The interface is already
batch-shaped, so Day 4 rewrites the body of `_forward` and nothing above it.

Until that lands, `continuous` is correct but NOT fast — it has continuous batching's
scheduling with static batching's arithmetic. Do not report M2 off it.

See IMPLEMENTATION_GUIDE.md "Days 3-5".
"""

from __future__ import annotations

import torch

from inference_server.config import CONFIG
from inference_server.core.sequence import Sequence


class Executor:
    """One forward pass, no policy."""

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = CONFIG.device

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
        return [self._forward(s) for s in seqs]

    # ------------------------------------------------------------------------------
    # Day 4 replaces the body of this method with one batched, ragged-masked pass.
    # Everything above stays as it is — that is the point of the list-shaped interface.
    # ------------------------------------------------------------------------------
    def _forward(self, seq: Sequence) -> int:
        ids = torch.tensor([seq.next_input_ids], device=self.device)
        width = ids.shape[1]

        # Positions count this sequence's own real tokens, starting where its cache ends.
        # Prefill spans the whole prompt; decode is a single position. Neither is derived
        # from a batch column index, which is the assumption Day 4's padding will break
        # if it is ever allowed to creep in.
        position_ids = torch.arange(
            seq.next_position, seq.next_position + width, device=self.device
        ).unsqueeze(0)

        out = self.model(
            input_ids=ids,
            position_ids=position_ids,
            past_key_values=seq.kv,      # None on prefill; HF allocates it
            use_cache=True,
        )
        seq.kv = out.past_key_values
        # The executor owns this counter because it is the only thing that knows the pass
        # actually ran. A scheduler that advanced it optimistically would desynchronise
        # from the cache on the first failed step.
        seq.num_cached += width

        # Greedy. Sampling params (temperature, top-p) would land here and nowhere else;
        # keeping them out of the scheduler is why M1 can assert exact token IDs at all.
        return int(out.logits[:, -1, :].argmax(-1).item())
