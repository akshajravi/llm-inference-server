"""Scheduler — P2 (Days 3-5), extended in P4 (Days 10-11).

Owns the sequence pool and decides what runs each step. The whole project's thesis
lives in step():

    1. Evict sequences that hit EOS or max_tokens; free their resources
    2. Admit waiting sequences if there is capacity           (FIFO — defensible, no starvation)
    3. [P4] Preempt if a running sequence needs a block and none are free
    4. Executor runs one forward pass over the running set
    5. Sample one token per sequence; append; check termination

Prefill vs decode: prefill-priority with alternation (PRD 7). Chunked prefill is named
future work; the TTFT cost gets documented, not optimised away.

P4 adds: victim selection (most recently admitted), re-admission guarantee
(preempted != dropped, FR6), and the bounded queue that returns 503 (FR7).

Exit criteria: M2; no sequence's output depends on batch composition.
See IMPLEMENTATION_GUIDE.md "Days 3-5" and "Days 10-11".
"""

from __future__ import annotations


class Scheduler:
    """Lands in P2. See module docstring for the step loop it implements."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Scheduler lands in P2 (Days 3-5)")
