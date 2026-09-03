"""Scheduler — P2 (Days 3-5), extended in P4 (Days 10-11).

Owns the sequence pool and decides what runs each step. The whole project's thesis
lives in step():

    1. Evict sequences that hit EOS or max_tokens; free their resources
    2. Admit waiting sequences if there is capacity           (FIFO — defensible, no starvation)
       [P4] ...and only if the blocks exist; preempted sequences re-enter ahead of new ones
    3. [P4] Preempt if a running sequence needs a block and none are free
    4. Executor runs one forward pass over the running set
    5. Sample one token per sequence; append; check termination

Prefill vs decode: prefill-priority with alternation (PRD 7). Chunked prefill is named
future work; the TTFT cost gets documented, not optimised away.

P4 adds: victim selection (most recently admitted), re-admission guarantee
(preempted != dropped, FR6), and the bounded queue that raises QueueFull -> 503 (FR7).
Two strategies for the victim's KV, chosen by CONFIG.preemption (S2):

    recompute   free the blocks, num_cached -> 0, re-prefill the whole history later
    swap        copy the blocks to host memory, free them, copy back on re-admission

S1 adds prefix caching (core/prefix_cache.py): at admission a sequence adopts the
cached blocks of the longest prefix of its history already in the pool, and after every
pass its newly completed full blocks are published. Two rules keep it honest under P4:
a recompute victim frees (decrefs) everything and re-matches on re-admission; a swap
victim copies every block it references to the host — shared ones included — decrefs
them all, and comes back with private copies. Simplest correct rule; see `_preempt`.

Exit criteria: M2; M4 (30 min at 10x, nothing dropped without a 503); no sequence's
output depends on batch composition or on whether it was preempted.
See IMPLEMENTATION_GUIDE.md "Days 3-5" and "Days 10-11".
"""

from __future__ import annotations

from collections import deque

from inference_server.config import CONFIG
from inference_server.core.block_allocator import BlockAllocator
from inference_server.core.block_table import BlockTable
from inference_server.core.executor import Executor
from inference_server.core.prefix_cache import PrefixCache
from inference_server.core.sequence import Sequence, Status


class QueueFull(RuntimeError):
    """The waiting queue is at CONFIG.max_queue_depth (FR7). The server maps this to 503.

    An exception rather than a boolean from `add()` because the engine registers a
    future for every accepted request: a `False` a caller forgot to check would leave
    that future dangling forever, which is the "dropped-but-unacknowledged" outcome M4
    forbids. Carries the numbers the 503 body reports.
    """

    def __init__(self, depth: int, bound: int) -> None:
        self.depth, self.bound = depth, bound
        super().__init__(f"queue full: {depth} waiting, bound is {bound}")


class SequenceTooLong(ValueError):
    """This request could not run even with the whole pool to itself.

    Rejected at `add()` rather than discovered later, because the alternative is the
    preemption loop the guide warns about: the sequence is admitted, runs out of blocks,
    is itself the only victim, is re-admitted, and so on forever with every other
    request stuck behind it. Nothing that passes this check can loop — see `_grow`.
    """

    def __init__(self, seq: Sequence, needed: int, available: int) -> None:
        self.needed, self.available = needed, available
        super().__init__(
            f"{seq.seq_id} needs up to {needed} blocks "
            f"(prompt {seq.prompt_len} + max_tokens {seq.max_tokens}); pool has {available}"
        )


class Scheduler:
    """The mutable batch. Membership is decided fresh every single step.

    This is the entire difference from P1. Static batching froze membership when the
    batch launched, so a sequence that finished on step 2 held its slot until the
    slowest row finished — measured at 77.3% of compute on mixed traffic. Here, steps 1
    and 2 run *between every forward pass*, so a slot comes back the instant it is free.
    """

    def __init__(
        self,
        executor: Executor,
        eos_token_id: int,
        allocator: BlockAllocator | None = None,
        pool: "object | None" = None,
        prefix_cache: PrefixCache | None = None,
    ) -> None:
        self.executor = executor
        self.eos_token_id = eos_token_id
        self.max_running = CONFIG.max_running

        #: P3 only. None means the contiguous engines, which let HuggingFace grow a
        #: private cache per sequence. When present, every admitted sequence gets a
        #: block table and this scheduler becomes responsible for the memory — which is
        #: why `_grow` and the free in `_evict` are guarded on exactly this field, and
        #: why P2's behaviour is bit-for-bit unchanged when it is None.
        self.allocator = allocator
        #: P3/P4 only: the PagedKVPool the allocator's indices point into. The scheduler
        #: never reads or writes KV through it; it exists here so swap preemption (S2)
        #: can move a victim's blocks to host memory and back.
        self.pool = pool
        #: S1 only, and only meaningful with an allocator: the hash -> block index that
        #: `_admit` consults and `step` publishes into. None means every block is private.
        self.prefix_cache = prefix_cache if allocator is not None else None

        # FIFO. Admitting the newest or the shortest request would raise throughput and
        # starve whoever arrived first. Bounded (FR7): past CONFIG.max_queue_depth,
        # add() raises QueueFull and the caller gets a 503 instead of unbounded latency.
        self.waiting: deque[Sequence] = deque()
        #: P4. Victims wait here, not in `waiting`, and `_admit` drains this first. A
        #: preempted sequence already paid for its prompt (and its queue time) once; if
        #: it queued behind fresh arrivals under sustained overload it would be preempted
        #: again before ever reaching the front — that is starvation, and it is exactly
        #: what FR6's "re-admitted, never dropped" forbids. Unbounded, and exempt from
        #: the FR7 bound: it can never hold more than max_running sequences.
        self.preempted: deque[Sequence] = deque()
        self.running: list[Sequence] = []

        #: Step counter. Preempted sequences remember the step they were evicted on, and
        #: `_admit` refuses to bring one back on that same step (the guide's
        #: preemption-loop guard).
        self.step_count = 0
        #: Cumulative counters for /health and the overload run's charts. Only ever
        #: incremented; the gauges (depth, running, free blocks) are computed on demand.
        self.num_preemptions = 0
        self.num_swaps = 0
        self.num_completed = 0

    # ------------------------------------------------------------------- pool queries
    def add(self, seq: Sequence) -> None:
        """Enqueue a new request, or refuse it now. Nothing is ever refused later.

        Two refusals, both at the door: QueueFull when the waiting queue is at its bound
        (FR7), SequenceTooLong when the request could not fit the pool even alone. Every
        request past this line completes — that is the contract M4's overload run tests.
        """
        if len(self.waiting) >= CONFIG.max_queue_depth:
            raise QueueFull(len(self.waiting), CONFIG.max_queue_depth)
        if self.allocator is not None:
            # The last sampled token is never cached (it is returned, not fed), so the
            # most this sequence ever holds is the KV for prompt + max_tokens - 1 tokens.
            worst = self._blocks_for(seq.prompt_len + seq.max_tokens - 1)
            if worst > self.allocator.num_blocks:
                raise SequenceTooLong(seq, worst, self.allocator.num_blocks)
        self.waiting.append(seq)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.preempted or self.running)

    @property
    def queue_depth(self) -> int:
        """What the FR7 bound is compared against: new arrivals only. Preempted
        sequences are not in this number because they cannot be rejected."""
        return len(self.waiting)

    def stats(self) -> dict:
        """The single source of the /health counters. Names are the wire names."""
        swapped = sum(1 for s in self.preempted if s.status is Status.SWAPPED)
        prefix = self.prefix_cache.stats() if self.prefix_cache is not None else {}
        return {
            "queue_depth": len(self.waiting),
            "num_running": len(self.running),
            "num_waiting": len(self.waiting) + len(self.preempted) - swapped,
            "num_swapped": swapped,
            "free_blocks": self.allocator.num_free if self.allocator else 0,
            "num_blocks": self.allocator.num_blocks if self.allocator else 0,
            "preemptions": self.num_preemptions,
            "swaps": self.num_swaps,
            "completed": self.num_completed,
            # S1: present only when a cache is attached, so the P2/P3 counter set is
            # unchanged and /health on an engine without one reports nothing it lacks.
            **prefix,
        }

    # ------------------------------------------------------------------------ the step
    def step(self) -> list[Sequence]:
        """Run exactly one forward pass. Returns the sequences that finished on it.

        Returning the finished set rather than mutating a callback keeps the engine thin:
        it resolves those futures and asks for another step. The scheduler never learns
        that futures exist.
        """
        self.step_count += 1
        self._evict()
        self._admit()

        batch = self._select()
        if not batch:
            return []

        self._grow(batch)
        if not batch:
            # Everything selected was preempted to make room — possible when the only
            # prefill this step was also the youngest sequence. The decode set runs
            # next step; there is nothing to execute now.
            return []
        tokens = self.executor.execute(batch)
        for seq, token in zip(batch, tokens):
            seq.append_token(token, self.eos_token_id)
        self._publish(batch)

        # Collected after sampling, so a sequence that ends on this step is reported on
        # this step. Its slot is not reclaimed until the top of the next one, which is
        # the only place eviction is allowed to happen.
        finished = [s for s in batch if s.is_finished]
        self.num_completed += len(finished)
        return finished

    # ------------------------------------------------------------------------- memory
    def _grow(self, batch: list[Sequence]) -> None:
        """Give every sequence in this batch the blocks its pass is about to write,
        preempting until it can (FR6). Mutates `batch`: victims are removed from it.

        No-op without an allocator. With one, this is the single place in the system
        where "out of KV memory" is observable: `ensure_capacity` raises MemoryError.
        P3 let it propagate — a server that silently wrote past its pool would corrupt
        another sequence's KV rather than fail. P4 catches it here and evicts a victim,
        repeating until the growth succeeds. Nothing else in the file handles memory
        pressure, which is why a leak on the preemption path has one place to be.

        Victim: the most recently admitted running sequence (`running[-1]`; the list is
        in admission order). Preserves the most progress — the oldest sequence has the
        most cached work behind it and is closest to freeing everything — and cannot
        starve anyone: the oldest is only ever preempted when it is alone, and a
        sequence alone always fits (`add()` guarantees it). That last fact is also the
        termination argument: each preemption shrinks the running set, and a running
        set of one always grows. The victim may be the sequence that asked for the
        block; that is the case where it was the youngest and there is no one else to
        make room for it. The rejected alternative, evicting the *oldest* (LRU-style),
        throws away the most work per eviction and lets a stream of newcomers starve a
        long request indefinitely.

        Cheap on almost every step: a decode sequence crosses a block boundary once
        every `block_size` tokens, so 15 of 16 calls allocate nothing.
        """
        if self.allocator is None:
            return
        for seq in batch:
            if seq.block_table is None:
                seq.block_table = BlockTable(self.allocator, CONFIG.block_size)

        pending = list(batch)                   # not yet grown, in batch order
        while pending:
            seq = pending[0]
            try:
                seq.block_table.ensure_capacity(seq.cached_after_next_pass)
            except MemoryError:
                if not self.running:
                    raise                       # cannot happen past add(); fail loudly
                victim = self.running[-1]
                self._preempt(victim)
                if victim in pending:
                    pending.remove(victim)
                if victim in batch:
                    batch.remove(victim)
                continue                        # retry the same sequence
            pending.pop(0)

    def _preempt(self, victim: Sequence) -> None:
        """Take a running sequence off the device and put it at the front of the line.

        The free happens here, under either strategy, and before the status change, so
        that a sequence marked WAITING or SWAPPED never holds device blocks. The
        alternative — freeing lazily on re-admission — would make "no GPU block is held
        while SWAPPED" a lie and the pool's accounting unreadable during overload.
        """
        table = victim.block_table
        if table is None:                       # admitted, never grown: nothing to free
            table = victim.block_table = BlockTable(self.allocator, CONFIG.block_size)
        # S1: `table.blocks` may include blocks shared through the prefix cache. Swap
        # copies them out like any other and `free()` only decrefs them, so the other
        # referents keep theirs; the victim comes back with private copies (it loses
        # the sharing, nothing else). Recompute victims re-match on re-admission.
        host_kv = None
        if CONFIG.preemption == "swap" and self.pool is not None and table.blocks:
            host_kv = self.pool.swap_out(table.blocks)
            self.num_swaps += 1
        table.free()
        self.running.remove(victim)
        victim.preempt(self.step_count, host_kv)
        # Front of the queue, most recently preempted first: LIFO among victims means
        # the *oldest* running sequence displaced during one burst of preemptions is
        # the first one back, matching the victim policy from the other side.
        self.preempted.appendleft(victim)
        self.num_preemptions += 1

    def _swap_in(self, seq: Sequence) -> None:
        """Bring a SWAPPED sequence's blocks back. Caller has checked they exist."""
        table = seq.block_table
        table.ensure_capacity(seq.num_cached)   # exactly len(host_kv) blocks, by construction
        self.pool.swap_in(seq.host_kv, table.blocks)

    def _blocks_for(self, num_tokens: int) -> int:
        return -(-num_tokens // CONFIG.block_size)

    def _blocks_needed(self, seq: Sequence) -> int:
        """Blocks this sequence must hold after its next pass, less what it has.

        One formula for all three admission cases. A fresh sequence holds nothing and
        needs its whole prompt's worth (prefill writes it all at once). A recompute
        victim likewise, for prompt + generated. A swap victim needs `len(host_kv)` for
        the copy back plus, if its next decode crosses a block boundary, one more — and
        `cached_after_next_pass` already says so.
        """
        held = seq.block_table.num_blocks if seq.block_table is not None else 0
        return self._blocks_for(seq.cached_after_next_pass) - held

    def _reserved_for_running(self) -> int:
        """Blocks the current running set will ask for on its next pass.

        Admission checks against `num_free - this`, not `num_free`. Without the
        reservation a newcomer takes the last block, the next decode step finds the
        pool empty, and the newcomer — youngest, hence victim — is evicted before it
        has produced a token. That admit/evict cycle is the "preemption loop" the guide
        warns about: it terminates (see `_grow`), but every lap re-prefills a prompt for
        nothing. Reserving what the running set is about to need makes the newcomer
        wait one more step instead.
        """
        return sum(self._blocks_needed(s) for s in self.running)

    def _fits(self, seq: Sequence, shared: int = 0) -> bool:
        """`shared` (S1): blocks a prefix match would hand this sequence without
        allocating. They are already held by someone else, so they cost the free list
        nothing and come off what the sequence needs."""
        if self.allocator is None:
            return True
        return self._blocks_needed(seq) - shared <= self.allocator.num_free - self._reserved_for_running()

    # ------------------------------------------------------------- prefix cache (S1)
    def _prefix_lookup(self, seq: Sequence) -> tuple[list[int], list[bytes]]:
        """What the cache holds of this sequence's history, without taking it yet.

        Applies to a sequence with nothing cached: a fresh arrival, or a recompute
        victim (whose history is prompt + generated, and whose earlier blocks may well
        still be in the pool, held by whoever shared them). A swap victim brings its
        own KV back and needs nothing from the cache.
        """
        if self.prefix_cache is None or seq.num_cached != 0:
            return [], []
        return self.prefix_cache.match(seq.prompt_token_ids + seq.output_token_ids)

    def _prefix_claim(self, seq: Sequence, blocks: list[int], hashes: list[bytes]) -> None:
        """Adopt the matched blocks and advance `num_cached` past them, so the executor
        prefills only the remainder — never less than one token (the cap in
        `PrefixCache.match`), because the last prompt token must be fed to get a logit."""
        if seq.block_table is None:
            seq.block_table = BlockTable(self.allocator, CONFIG.block_size)
        history = seq.prompt_len + seq.num_generated
        full = (history - 1) // CONFIG.block_size
        seq.num_cached = self.prefix_cache.claim(seq.block_table, blocks, hashes, full)

    def _publish(self, batch: list[Sequence]) -> None:
        """After a pass: register every full block the batch completed. O(1) per
        decode step per sequence in the common case — nothing crossed a boundary."""
        if self.prefix_cache is None:
            return
        for seq in batch:
            if seq.block_table is not None:
                self.prefix_cache.register(
                    seq.block_table, seq.prompt_token_ids + seq.output_token_ids, seq.num_cached
                )

    # --------------------------------------------------------------------- the policy
    def _evict(self) -> None:
        """Drop finished sequences and free what they hold.

        P2 frees a whole per-sequence KV cache here; P3 also returns block indices to
        the allocator. The free must be explicit — nothing is garbage-collected out of a
        preallocated tensor, so a sequence dropped without this call leaks its blocks
        permanently. That is the failure the free-list test is aimed at.
        """
        if not any(s.is_finished for s in self.running):
            return
        for seq in self.running:
            if seq.is_finished:
                seq.kv = None
                if seq.block_table is not None:
                    seq.block_table.free()
        self.running = [s for s in self.running if not s.is_finished]

    def _admit(self) -> None:
        """Fill free slots: preempted sequences first, then the queue, in order.

        P4 adds the memory check: a sequence is admitted only if the blocks its next
        pass needs are free after the running set's own growth is reserved. Prefill
        needs every block at once (`allocate_many` is all-or-nothing), so admitting
        without the check would just move the MemoryError one call later, into `_grow`,
        where the newcomer would be the first victim.

        Head-of-line blocking is deliberate: if the front sequence does not fit, nothing
        behind it is admitted either. Skipping ahead to a smaller request is the
        shortest-job-first policy the P2 docstring rejected — it raises throughput and
        starves the long request forever.

        The loop guard: a sequence is never re-admitted on the step it was preempted.
        Structurally that is already true (`_grow` runs after `_admit`), but the check
        is explicit so the property survives a reordering of `step()`.
        """
        while len(self.running) < self.max_running:
            queue = self.preempted if self.preempted else self.waiting
            if not queue:
                return
            seq = queue[0]
            if seq.preempted_step == self.step_count:
                return
            blocks, hashes = self._prefix_lookup(seq)
            if not self._fits(seq, shared=len(blocks)):
                return
            queue.popleft()
            if seq.status is Status.SWAPPED:
                self._swap_in(seq)
            elif blocks:
                self._prefix_claim(seq, blocks, hashes)
            seq.admit()
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
        prefills = [s for s in self.running if s.is_prefilling]
        return prefills if prefills else list(self.running)
