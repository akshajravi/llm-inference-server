"""Sequence — P2 (Days 3-5). One request's state, and nothing else.

Holds: prompt IDs, generated IDs, KV cache (P2) or block table (P3), a host copy of the
blocks while swapped out (P4), status, max_tokens, arrival time. Knows how to answer "am I done?"; knows nothing about who else is running.

    WAITING --admit--> RUNNING --eos/max_tokens--> FINISHED
       ^                 |   ^
       |   preempt (P4)  |   |  admit (swap-in)
       +--- recompute ---+---+--> SWAPPED
              (num_cached -> 0)   (num_cached kept; KV parked on the host)

The two preemption edges differ in exactly one thing: what happens to `num_cached`.
Recompute zeroes it, so the sequence re-enters as a prefill over prompt + everything it
has generated (see `next_input_ids`). Swap keeps it, so the sequence re-enters as a
decode once its blocks are copied back — different physical block indices, same
contents, same `num_cached`. Neither edge touches `output_token_ids`, which is what
makes "preempted != dropped" (FR6) a structural property rather than a promise.

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
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # import-only-for-types: this file stays leaf-level
    from inference_server.core.block_table import BlockTable


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

    #: P3: this sequence's logical->physical block mapping, or None on the contiguous
    #: engines. The two are mutually exclusive by construction — an engine either owns
    #: a private cache object or holds blocks in the shared pool, never both — but they
    #: are separate fields so that P2 and P3 can be benchmarked in one process without
    #: either one reinterpreting the other's state.
    block_table: "BlockTable | None" = None

    #: P4 (S2): the host-side copy of this sequence's blocks while SWAPPED, else None.
    #: Opaque here for the same reason `kv` is — only kv_pool.py knows its layout.
    host_kv: object | None = None

    #: P4 bookkeeping. `preemptions` is how many times this sequence was a victim (the
    #: overload run charts the distribution); `preempted_step` is the scheduler step it
    #: last happened on, which is what the re-admission guard compares against.
    preemptions: int = 0
    preempted_step: int = -1

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
    def cached_after_next_pass(self) -> int:
        """What `num_cached` will be once the next forward pass completes.

        The block table must be grown to this *before* the pass runs, not after: the
        pass writes KV for these tokens, and writing into a block that has not been
        allocated yet is the paged equivalent of a segfault.
        """
        return self.num_cached + len(self.next_input_ids)

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

        Prefill hands over everything this sequence owns; decode hands over exactly the
        token sampled last step. Everything before it is already in the cache, which is
        the only reason decode is one token wide instead of `total_len` wide.

        On first admission "everything it owns" is just the prompt. After a recompute
        preemption (P4) it is prompt + every token generated so far: the cache was thrown
        away, so the whole history goes back through the model in one pass and the next
        token is sampled from its final logit exactly as if nothing had happened.

        One expression covers all of it, and the S1 case too: everything past
        `num_cached`. With nothing cached that is the whole history; in decode it is the
        one token sampled last step; after a prefix-cache hit it is the part of the
        prompt the cache did not have (at least one token — see PrefixCache.match).
        """
        return (self.prompt_token_ids + self.output_token_ids)[self.num_cached :]

    @property
    def is_prefilling(self) -> bool:
        """True when the next pass feeds more than one token — the shape the scheduler
        keys batch homogeneity on. `needs_prefill` (nothing cached) always implies it,
        so P2 sees no change. S1 adds the third case: a prefix-cache hit leaves
        `num_cached > 0` with a multi-token remainder still to feed, which must run
        with the prefills and not be mistaken for a decode."""
        return self.needs_prefill or len(self.next_input_ids) > 1

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

    def preempt(self, step: int, host_kv: object | None = None) -> None:
        """RUNNING -> WAITING (recompute) or RUNNING -> SWAPPED (swap). P4, FR6.

        The scheduler has already freed this sequence's blocks; this records what that
        means for the next forward pass. With no host copy the cache is simply gone, so
        `num_cached` drops to zero and the whole history is re-prefilled on re-admission.
        With one, the cache still exists — just not on the device — so `num_cached`
        stands and re-admission resumes as a one-token decode.

        The alternative, letting the scheduler poke `status` and `num_cached` directly,
        was how P2 admitted sequences. It is fine for one edge and wrong for four: the
        swap edge that forgot to keep `num_cached` would re-prefill *and* swap in, feeding
        the model a prompt it already has KV for, at positions that are already occupied.
        """
        if self.status is not Status.RUNNING:
            raise RuntimeError(f"{self.seq_id} preempted while {self.status.value}, not running")
        self.preemptions += 1
        self.preempted_step = step
        if host_kv is None:
            self.num_cached = 0
            self.status = Status.WAITING
        else:
            self.host_kv = host_kv
            self.status = Status.SWAPPED

    def admit(self) -> None:
        """WAITING or SWAPPED -> RUNNING. The swap-in itself is the scheduler's job (it
        owns the allocator and the pool); by the time this is called the blocks are
        back on the device and the host copy is no longer needed."""
        if self.status not in (Status.WAITING, Status.SWAPPED):
            raise RuntimeError(f"{self.seq_id} admitted while {self.status.value}")
        self.host_kv = None
        self.status = Status.RUNNING

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
