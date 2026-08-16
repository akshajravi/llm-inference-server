"""Executor — P2 (Days 3-5). Given a set of sequences, run exactly one forward pass.

Knows nothing about scheduling — no admission, no eviction, no policy. That separation
is what lets P3 swap the attention path underneath without touching the scheduler.

The hard part is ragged batching: sequences in one batch have different lengths *and*
different KV cache lengths. Pad to the batch max, mask correctly. This is the single
largest correctness risk in the sprint; Day 4 is budgeted for it.

See IMPLEMENTATION_GUIDE.md "Days 3-5".
"""

from __future__ import annotations


class Executor:
    """Lands in P2. One forward pass, no policy."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Executor lands in P2 (Days 3-5)")
