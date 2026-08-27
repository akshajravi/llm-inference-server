"""Sequence — P2 (Days 3-5). One request's state, and nothing else.

Holds: prompt IDs, generated IDs, KV cache (P2) or block table (P3), status, max_tokens,
arrival time. Knows how to answer "am I done?"; knows nothing about who else is running.

    WAITING --admit--> RUNNING --eos/max_tokens--> FINISHED
       ^                  |
       +---preempt (P4)---+          (SWAPPED is the S2 stretch path)

This file is deliberately the dullest of the three. The scheduler decides *who* runs and
the executor decides *how* a forward pass is shaped; a Sequence only knows what has
happened to it so far. Keeping policy out of here is what lets P4 add preemption by
editing one file — a Sequence that made its own decisions would have to be edited too.

See IMPLEMENTATION_GUIDE.md "Days 3-5".
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field


class Status(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"      # P4 stretch (S2): blocks parked in pinned host memory
    FINISHED = "finished"


@dataclass
class Sequence:
    """One request, tracked across many forward passes.

    The invariant every other file relies on: `num_cached` is exactly how many of this
    sequence's tokens have KV entries in the cache. Everything about how the next
    forward pass is shaped — prefill or decode, which position ID, how long the mask —
    is derived from it, so it must be advanced by the executor and never guessed at.
    """

    seq_id: str
    prompt_token_ids: list[int]
    max_tokens: int
    #: None = use the tokenizer default. Overridden per request because gpt2 greedy
    #: essentially never emits <|endoftext|>, so the EOS path would otherwise be dead code.
    eos_token_id: int | None = None

    status: Status = Status.WAITING
    output_token_ids: list[int] = field(default_factory=list)
    finish_reason: str = ""

    #: How many of this sequence's tokens are represented in the KV cache. Zero means
    #: prefill has not run yet. After prefill it equals the prompt length; each decode
    #: step adds one. This is *not* the same as len(prompt)+len(output): the most
    #: recently sampled token has been chosen but not yet fed through the model, which
    #: is precisely the off-by-one that ragged batching punishes on Day 4.
    num_cached: int = 0

    #: P2: this sequence's own KV cache object. P3 replaces it with a block table and
    #: the tensors move into one shared preallocated pool — the entire point of paging.
    #: Typed as object because nothing outside the executor may inspect it.
    kv: object | None = None

    # --- timing. Recorded here rather than in the engine because a sequence outlives
    # any single step, and TTFT must denote the same event it does in P0/P1. ---
    arrival_s: float = field(default_factory=time.perf_counter)
    first_token_s: float | None = None
    finished_s: float | None = None

    # ------------------------------------------------------------------ shape queries
    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_generated(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        """Tokens this sequence owns, cached or not. P3 sizes its block table off this."""
        return self.prompt_len + self.num_generated

    @property
    def needs_prefill(self) -> bool:
        """True until the prompt has been through the model once.

        The scheduler uses this to separate the batch into a prefill group and a decode
        group, because the two need different input shapes and cannot share a pass.
        """
        return self.num_cached == 0

    @property
    def next_input_ids(self) -> list[int]:
        """The tokens to feed on this sequence's next forward pass.

        Prefill hands over the whole prompt; decode hands over exactly the token sampled
        last step. Everything before it is already in the cache, which is the only reason
        decode is one token wide instead of `total_len` wide.
        """
        if self.needs_prefill:
            return self.prompt_token_ids
        return self.output_token_ids[-1:]

    @property
    def next_position(self) -> int:
        """Position ID of the first token in `next_input_ids`.

        Counts real tokens, never batch columns. Under ragged batching two sequences in
        the same pass sit at different positions and are padded to a common width, so a
        position derived from column index is wrong for every row but one.
        """
        return self.num_cached

    # -------------------------------------------------------------------- transitions
    def append_token(self, token_id: int, eos_id: int) -> None:
        """Record one sampled token and decide whether this sequence is done.

        Termination is decided here, once, so that the scheduler and the executor cannot
        disagree about who is still running. `eos_id` is passed in rather than read off
        the tokenizer because this file must not import the model.
        """
        if self.status is Status.FINISHED:
            # A finished sequence being fed again means the scheduler failed to evict it.
            # Under static batching that was normal and cost 77% of the compute; under
            # continuous batching it is a bug, so it fails loudly instead of silently.
            raise RuntimeError(f"{self.seq_id} is finished but was sampled again")

        now = time.perf_counter()
        if self.first_token_s is None:
            self.first_token_s = now

        self.output_token_ids.append(token_id)

        stop = self.eos_token_id if self.eos_token_id is not None else eos_id
        if token_id == stop:
            self._finish("eos", now)
        elif self.num_generated >= self.max_tokens:
            self._finish("length", now)

    def _finish(self, reason: str, now: float) -> None:
        self.status = Status.FINISHED
        self.finish_reason = reason
        self.finished_s = now

    @property
    def is_finished(self) -> bool:
        return self.status is Status.FINISHED

    # ------------------------------------------------------------------------ metrics
    @property
    def ttft_s(self) -> float:
        """Arrival to first sampled token. Measured from arrival, not from admission —
        time spent waiting in the queue is time the user spent waiting."""
        return (self.first_token_s - self.arrival_s) if self.first_token_s else 0.0

    @property
    def latency_s(self) -> float:
        return (self.finished_s - self.arrival_s) if self.finished_s else 0.0

    def __repr__(self) -> str:  # debugging a scheduler is unpleasant without this
        return (
            f"Sequence({self.seq_id} {self.status.value} "
            f"prompt={self.prompt_len} gen={self.num_generated}/{self.max_tokens} "
            f"cached={self.num_cached})"
        )
