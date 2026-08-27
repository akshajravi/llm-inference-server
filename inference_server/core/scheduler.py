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

from collections import deque

from inference_server.config import CONFIG
from inference_server.core.executor import Executor
from inference_server.core.sequence import Sequence, Status


class Scheduler:
    """The mutable batch. Membership is decided fresh every single step.

    This is the entire difference from P1. Static batching froze membership when the
    batch launched, so a sequence that finished on step 2 held its slot until the
    slowest row finished — measured at 77.3% of compute on mixed traffic. Here, steps 1
    and 2 run *between every forward pass*, so a slot comes back the instant it is free.
    """

    def __init__(self, executor: Executor, eos_token_id: int) -> None:
        self.executor = executor
        self.eos_token_id = eos_token_id
        self.max_running = CONFIG.max_running

        # FIFO. Admitting the newest or the shortest request would raise throughput and
        # starve whoever arrived first; the queue is bounded in P4 (FR7), not here.
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    # ------------------------------------------------------------------- pool queries
    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    @property
    def queue_depth(self) -> int:
        """P4 sheds load (503) past CONFIG.max_queue_depth. P2 only reports it."""
        return len(self.waiting)

    # ------------------------------------------------------------------------ the step
    def step(self) -> list[Sequence]:
        """Run exactly one forward pass. Returns the sequences that finished on it.

        Returning the finished set rather than mutating a callback keeps the engine thin:
        it resolves those futures and asks for another step. The scheduler never learns
        that futures exist.
        """
        self._evict()
        self._admit()

        batch = self._select()
        if not batch:
            return []

        tokens = self.executor.execute(batch)
        for seq, token in zip(batch, tokens):
            seq.append_token(token, self.eos_token_id)

        # Collected after sampling, so a sequence that ends on this step is reported on
        # this step. Its slot is not reclaimed until the top of the next one, which is
        # the only place eviction is allowed to happen.
        return [s for s in batch if s.is_finished]

    # --------------------------------------------------------------------- the policy
    def _evict(self) -> None:
        """Drop finished sequences and free what they hold.

        P2 frees a whole per-sequence KV cache here. P3 frees block indices back to the
        allocator instead, and that is the only line in this file that changes.
        """
        if not any(s.is_finished for s in self.running):
            return
        for seq in self.running:
            if seq.is_finished:
                seq.kv = None
        self.running = [s for s in self.running if not s.is_finished]

    def _admit(self) -> None:
        """Fill free slots from the front of the queue.

        P4 inserts the memory check here: admit only if the blocks exist, and preempt a
        running sequence if they do not.
        """
        while self.waiting and len(self.running) < self.max_running:
            seq = self.waiting.popleft()
            seq.status = Status.RUNNING
            self.running.append(seq)

    def _select(self) -> list[Sequence]:
        """Choose this step's batch: prefill-priority with alternation.

        Prefill spans a whole prompt, decode spans one token — different input widths,
        so they cannot share a pass. Given that, one of the two has to go first, and
        prefill going first is what keeps TTFT from growing with queue depth.

        The cost is real and gets documented rather than optimised away: sequences
        already decoding stall for one pass whenever a newcomer prefills. Chunked
        prefill is the fix and is named future work, not sprint work.
        """
        prefills = [s for s in self.running if s.needs_prefill]
        return prefills if prefills else list(self.running)
