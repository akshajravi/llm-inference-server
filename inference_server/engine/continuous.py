"""Continuous batching — P2 (Days 3-5). Target: M2, >=3x throughput over P1 static.

Thin driver only: owns the background step loop and the futures the HTTP layer awaits.
The actual decisions live in core/scheduler.py and the forward pass in core/executor.py
(NFR2 — an interviewer reads one file).

There is no batch object here, and that is the design. A request is added to the pool and
the loop turns; whether it shares a forward pass with three others or thirty is decided
fresh every step by the scheduler. P1 had to answer "which batch is this request in?" —
here the question does not typecheck.

Threading model (P4, the M4 fix)
--------------------------------
Two locks, with a strict rule about what each may wait on:

  `_lock`        the *step* lock. Held for the duration of a forward pass. Only the
                 step thread and the synchronous `generate()` path ever take it.
  `_inbox_lock`  the *inbox* lock. Held for microseconds: an append, a dict pop, a
                 handful of integer reads. This is the only lock the event-loop thread
                 (uvicorn: `submit()`, `stream()`, `stats()`) is allowed to touch.

The first version of this file had `submit()` take the step lock to call
`scheduler.add()`. Python locks are not fair: the step loop releases `_lock` and
re-acquires it a few hundred nanoseconds later, so a handler waiting on it could lose
the race for many consecutive steps. Under the 20 req/s overload run the event loop
stalled for tens of seconds, uvicorn stopped calling accept(), the 128-entry listen
backlog overflowed, and clients saw connection resets — 152 of them — instead of the
503s the bounded queue exists to send. NOTHING on the event-loop thread may wait for
a forward pass; that is the invariant this file now keeps.

Arrivals therefore go into an inbox. The step thread drains the inbox into
`scheduler.add()` at the top of every iteration, under the step lock. Admission control
(QueueFull -> 503, SequenceTooLong -> 422) is still decided synchronously in `submit()`
so the caller gets its refusal immediately: the queue bound is checked against
`scheduler.queue_depth + len(inbox)` under the inbox lock, and the length check is a
pure function of the request and the pool size. Both checks are copies of the
scheduler's, kept consistent by using the scheduler's own helpers; if the race between
the two still lets `scheduler.add()` raise in the step thread, exactly that request's
waiter is failed with the exception and the HTTP layer maps it as usual.

Exit criteria: M2 met; M1 holds including the alone-vs-crowded-batch test;
p99 reported alongside throughput.
See IMPLEMENTATION_GUIDE.md "Days 3-5".
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import AsyncIterator

from inference_server.config import CONFIG
from inference_server.core.executor import Executor
from inference_server.core.scheduler import QueueFull, Scheduler, SequenceTooLong
from inference_server.core.sequence import Sequence
from inference_server.engine.base import Engine, Request, Result
from inference_server.model import load, sync


class DuplicateRequest(ValueError):
    """A request_id that is already in flight. The server maps this to 400.

    Two `submit()`s with the same id used to overwrite each other's future, and the
    caller whose future was overwritten hung forever. Rejecting the duplicate is the
    honest fix: the id is the caller's handle on the result, and two results cannot
    share one handle. The HTTP layer never trips this (it mints a uuid per request);
    it exists for in-process callers, which is where it was found.
    """

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        super().__init__(f"request_id {request_id!r} is already in flight")


@dataclass
class StreamEvent:
    """One item on a `stream()` iterator. Either a token or the end of the stream.

    Per token: `token_id` and the `text` delta that token added to the decoded output.
    The delta is computed by decoding the *whole* output so far and emitting the new
    suffix — decoding tokens one at a time renders a BPE merge or a multi-byte character
    as garbage, because a single token can be half a code point. A suffix that ends in
    U+FFFD (an incomplete byte sequence) is held back until the next token completes it.
    The deltas concatenate to exactly `result.text`.

    Final: `result` is set, `token_id` is None. It is the same `Result` `submit()`
    would have returned.
    """

    token_id: int | None
    text: str
    result: Result | None = None

    @property
    def done(self) -> bool:
        return self.result is not None


@dataclass
class _Waiter:
    """Who is waiting on a sequence, and how to reach them. Owned by `_inbox_lock`.

    Exactly one of `future` / `queue` is set for callers of `submit()` / `stream()`;
    both are None for `generate()`, which watches its Sequence directly. Everything
    that touches asyncio state goes through `loop.call_soon_threadsafe` because the
    step thread is a plain thread and the future/queue belong to the caller's loop.
    """

    seq: Sequence
    loop: asyncio.AbstractEventLoop | None = None
    future: asyncio.Future | None = None
    queue: asyncio.Queue | None = None
    #: Streaming bookkeeping: how many tokens have been pushed, and the text emitted.
    sent: int = 0
    emitted_text: str = ""


class ContinuousEngine(Engine):
    name = "continuous"

    #: How long the step thread sleeps when the pool is empty. Long enough not to spin a
    #: core, short enough that it is invisible next to a forward pass.
    IDLE_POLL_S = 0.002

    def __init__(self) -> None:
        self.model, self.tokenizer = load()
        self.executor = Executor(self.model, self.tokenizer)
        self.scheduler = Scheduler(self.executor, self.tokenizer.eos_token_id)
        self._init_driver()

    def _init_driver(self) -> None:
        """The driver state, shared verbatim with PagedEngine (which does not call
        `super().__init__()` because it builds a different executor and scheduler)."""
        # One lock over the pool. Steps are serialised anyway — there is one device and
        # one step loop — so a finer-grained scheme would buy nothing and cost the
        # ability to reason about this file. Held only by the step thread and generate().
        self._lock = threading.Lock()

        # The inbox, and everything the event-loop thread is allowed to touch. Held for
        # microseconds, never while anything blocks. See the module docstring.
        self._inbox_lock = threading.Lock()
        self._inbox: list[Sequence] = []
        #: Every accepted sequence's waiter, keyed by seq_id, from the moment it enters
        #: the inbox until its result is handed back. Doubles as the duplicate-id check.
        #: (Named `_futures` for history: it once held bare asyncio futures.)
        self._futures: dict[str, _Waiter] = {}

        self._wake = threading.Event()
        self._stopping = False
        self._thread: threading.Thread | None = None

    # --------------------------------------------------------------------- sync path
    def generate(self, req: Request) -> Result:
        """One request, driven to completion by the caller. This is what M1 calls.

        It drives the same scheduler the async path does rather than taking a shortcut
        around it — a private fast path here would mean M1 was testing code the server
        never runs. It also enters through the same inbox as `submit()`, so admission
        control is one code path with two callers instead of two that can drift.
        """
        seq = self._to_sequence(req)
        self._enqueue(seq)

        while not seq.is_finished:
            with self._lock:
                failed = self._drain_inbox()
                self.scheduler.step()
                self._push_stream_tokens()
            for failed_seq, exc in failed:
                if failed_seq is seq:
                    raise exc

        sync()
        with self._inbox_lock:                  # no-op if the step thread got there first
            self._futures.pop(seq.seq_id, None)
        return self._to_result(seq)

    # -------------------------------------------------------------------- async path
    async def submit(self, req: Request) -> Result:
        """Never waits for a forward pass: admission is decided here, under the inbox
        lock, and the step thread picks the request up on its next iteration."""
        self._ensure_running()
        seq = self._to_sequence(req)
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._enqueue(seq, loop=loop, future=fut)
        return await fut

    def stream(self, req: Request) -> AsyncIterator[StreamEvent]:
        """Tokens as they are sampled (S4). One `StreamEvent` per token, then a final
        event carrying the `Result`.

        Deliberately *not* an `async def`: an async generator's body does not run until
        its first `__anext__`, which would push the QueueFull / SequenceTooLong refusal
        past the point where the HTTP handler can still answer with a status code.
        Admission happens synchronously here; the returned iterator only ever drains.

        The scheduler never learns about streaming. The step loop already sees every
        sequence after every step; for sequences with a queue registered it also pushes
        the newly sampled tokens, via the same `call_soon_threadsafe` bridge `_resolve`
        uses. A consumer that stops iterating (client disconnected) leaves the sequence
        running to completion — its queue is discarded with the waiter when it finishes,
        so nothing leaks. Cancelling in-flight work is future work.
        """
        self._ensure_running()
        seq = self._to_sequence(req)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        self._enqueue(seq, loop=loop, queue=queue)
        return self._drain_queue(queue)

    @staticmethod
    async def _drain_queue(queue: asyncio.Queue) -> AsyncIterator[StreamEvent]:
        while True:
            item = await queue.get()
            if isinstance(item, BaseException):
                raise item
            yield item
            if item.done:
                return

    # ------------------------------------------------------------------- admission
    def _enqueue(
        self,
        seq: Sequence,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        future: asyncio.Future | None = None,
        queue: asyncio.Queue | None = None,
    ) -> _Waiter:
        """Admission control, then into the inbox. The only writer of `_inbox`.

        Order matters: every refusal is raised before the waiter is registered, so a
        refused request leaves nothing behind (test_admission checks `_futures` is empty
        after a burst). The length check needs no lock at all; the queue bound is read
        under the inbox lock so two concurrent submits cannot both see the last slot.
        """
        self._check_length(seq)
        with self._inbox_lock:
            if seq.seq_id in self._futures:
                raise DuplicateRequest(seq.seq_id)
            depth = self.scheduler.queue_depth + len(self._inbox)
            if depth >= CONFIG.max_queue_depth:
                raise QueueFull(depth, CONFIG.max_queue_depth)
            waiter = _Waiter(seq=seq, loop=loop, future=future, queue=queue)
            self._futures[seq.seq_id] = waiter
            self._inbox.append(seq)
        self._wake.set()
        return waiter

    def _check_length(self, seq: Sequence) -> None:
        """`Scheduler.add()`'s SequenceTooLong check, computed here so `submit()` can
        refuse without the step lock. Same formula, same helper (`_blocks_for`), same
        exception — the scheduler still re-checks on `add()`, so a mismatch would
        surface as a failed waiter rather than a wrongly admitted request."""
        allocator = self.scheduler.allocator
        if allocator is None:
            return
        worst = self.scheduler._blocks_for(seq.prompt_len + seq.max_tokens - 1)
        if worst > allocator.num_blocks:
            raise SequenceTooLong(seq, worst, allocator.num_blocks)

    def _drain_inbox(self) -> list[tuple[Sequence, Exception]]:
        """Move arrivals into the scheduler. Caller holds the step lock.

        Returns the arrivals `scheduler.add()` refused, after failing their waiters.
        That list is non-empty only if the admission check in `_enqueue` and the
        scheduler's disagreed — a race on the queue bound — and it exists so a request
        can never hang: a refusal here is delivered to its caller exactly like one at
        the door. `generate()` reads it because it has no future to fail.
        """
        with self._inbox_lock:
            if not self._inbox:
                return []
            arrivals, self._inbox = self._inbox, []
        failed: list[tuple[Sequence, Exception]] = []
        for seq in arrivals:
            try:
                self.scheduler.add(seq)
            except Exception as exc:  # noqa: BLE001 — QueueFull / SequenceTooLong
                self._fail_one(seq, exc)
                failed.append((seq, exc))
        return failed

    # ------------------------------------------------------------------ observability
    def stats(self) -> dict:
        """The scheduler's counters, read WITHOUT the step lock.

        /health is polled from the event-loop thread during the overload run, and a
        poll that waits behind a forward pass is the starvation the module docstring
        describes. So the snapshot may be up to one step stale, and in principle torn
        (a gauge from before the step, a counter from after). For a chart sampled every
        five seconds neither matters. The one real hazard is iterating the preempted
        deque while the step thread mutates it, which raises RuntimeError; that is
        retried and, if it keeps happening, read under the step lock as a last resort.
        """
        for _ in range(3):
            try:
                return self.scheduler.stats()
            except RuntimeError:        # "deque mutated during iteration"
                continue
        with self._lock:
            return self.scheduler.stats()

    # --------------------------------------------------------------- the driving loop
    def _ensure_running(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="step-loop", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        """Turn the crank forever. Never touches asyncio except to hand back results.

        Everything blocking happens on this thread, which is what keeps the guide's
        async-deadlock warning from applying: the step loop cannot be blocked by a
        request handler because it never awaits one — and, since P4, a request handler
        cannot be blocked by the step loop either, because it never waits on `_lock`.
        """
        while not self._stopping:
            try:
                with self._lock:
                    self._drain_inbox()
                    finished = self.scheduler.step() if self.scheduler.has_work else []
                    self._push_stream_tokens()
            except Exception as exc:  # noqa: BLE001
                # Found by mutation testing: breaking eviction made the suite hang for
                # ten minutes rather than fail. The step raised, this thread died, and
                # every caller sat on a future nobody would ever resolve. A serving loop
                # that can die silently is worse than one that crashes, so the failure is
                # propagated to everyone waiting and the loop stops deliberately.
                self._fail_all(exc)
                return

            if not finished:
                if not self.scheduler.has_work and not self._inbox:
                    # Idle. Release the KV cache before sleeping — a finished burst
                    # should not hold its memory hostage until the next arrival.
                    self.executor.reset()
                    self._wake.wait(self.IDLE_POLL_S)
                    self._wake.clear()
                continue

            for seq in finished:
                self._resolve(seq)

    def _push_stream_tokens(self) -> None:
        """Hand every streaming consumer the tokens its sequence gained this step.
        Caller holds the step lock, so `output_token_ids` is stable while it is read.

        Cheap on the common path: a dict scan and an integer compare per waiter. Only
        the sequences that actually grew pay for a decode, and only streaming ones.
        The decode itself is a few microseconds for a short output; it is done here,
        on the step thread, rather than on the consumer's loop, so the event loop stays
        as idle as the module docstring promises.
        """
        with self._inbox_lock:
            streaming = [w for w in self._futures.values() if w.queue is not None]
        for w in streaming:
            seq = w.seq
            if seq.num_generated <= w.sent:
                continue
            new_ids = seq.output_token_ids[w.sent:]
            full = self.tokenizer.decode(seq.output_token_ids, skip_special_tokens=True)
            for i, token_id in enumerate(new_ids):
                last = w.sent + i + 1 == seq.num_generated
                if last:
                    # Hold back a dangling partial code point unless the sequence is
                    # done, in which case whatever is left is flushed with this token.
                    if full.endswith("�") and not seq.is_finished:
                        delta = ""
                    else:
                        delta = full[len(w.emitted_text):]
                        w.emitted_text = full
                else:
                    delta = ""              # several tokens in one step: text rides the last
                w.loop.call_soon_threadsafe(w.queue.put_nowait, StreamEvent(token_id, delta))
            w.sent = seq.num_generated

    def _resolve(self, seq: Sequence) -> None:
        with self._inbox_lock:
            waiter = self._futures.pop(seq.seq_id, None)
        if waiter is None or waiter.loop is None:
            return                      # a generate() caller is watching it directly
        result = self._to_result(seq)
        # The future belongs to the caller's event loop, and this is a plain thread.
        # set_result from here would be a data race; call_soon_threadsafe is the bridge.
        if waiter.future is not None:
            fut = waiter.future
            waiter.loop.call_soon_threadsafe(
                lambda: None if fut.done() else fut.set_result(result)
            )
        if waiter.queue is not None:
            waiter.loop.call_soon_threadsafe(
                waiter.queue.put_nowait, StreamEvent(None, "", result=result)
            )

    def _fail_one(self, seq: Sequence, exc: BaseException) -> None:
        with self._inbox_lock:
            waiter = self._futures.pop(seq.seq_id, None)
        if waiter is not None:
            self._deliver_failure(waiter, exc)

    @staticmethod
    def _deliver_failure(waiter: _Waiter, exc: BaseException) -> None:
        if waiter.loop is None:
            return
        if waiter.future is not None:
            fut = waiter.future
            waiter.loop.call_soon_threadsafe(
                lambda: None if fut.done() else fut.set_exception(exc)
            )
        if waiter.queue is not None:
            waiter.loop.call_soon_threadsafe(waiter.queue.put_nowait, exc)

    def _fail_all(self, exc: BaseException) -> None:
        """Hand the failure to every waiting caller. Nobody is left hanging."""
        with self._lock:
            with self._inbox_lock:
                pending = list(self._futures.values())
                self._futures.clear()
                self._inbox.clear()
            self.scheduler.waiting.clear()
            self.scheduler.preempted.clear()
            self.scheduler.running.clear()
        for waiter in pending:
            self._deliver_failure(waiter, exc)

    def shutdown(self) -> None:
        self._stopping = True
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    # ------------------------------------------------------------------- translations
    def _to_sequence(self, req: Request) -> Sequence:
        return Sequence(
            seq_id=req.request_id,
            prompt_token_ids=self.tokenizer(req.prompt).input_ids,
            max_tokens=req.max_tokens,
            eos_token_id=req.eos_token_id,
        )

    def _to_result(self, seq: Sequence) -> Result:
        return Result(
            request_id=seq.seq_id,
            token_ids=seq.output_token_ids,
            text=self.tokenizer.decode(seq.output_token_ids, skip_special_tokens=True),
            ttft_s=seq.ttft_s,
            latency_s=seq.latency_s,
            finish_reason=seq.finish_reason,
            prompt_len=seq.prompt_len,
            # Unchanged from P0/P1 and for the same reason: HuggingFace still grows one
            # contiguous cache per sequence, so the allocator has not changed and neither
            # has the waste. This is the "before" number P3 is measured against (M3).
            reserved_tokens=CONFIG.max_seq_len,
            used_tokens=seq.total_len,
            # Structurally zero, not measured-as-zero. A finished sequence is evicted at
            # the top of the next step and append_token() raises if one is ever sampled
            # again, so the 77.3% stall P1 measured cannot occur here by construction.
            wasted_steps=0,
        )
