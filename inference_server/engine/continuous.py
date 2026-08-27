"""Continuous batching — P2 (Days 3-5). Target: M2, >=3x throughput over P1 static.

Thin driver only: owns the background step loop and the futures the HTTP layer awaits.
The actual decisions live in core/scheduler.py and the forward pass in core/executor.py
(NFR2 — an interviewer reads one file).

There is no batch object here, and that is the design. A request is added to the pool and
the loop turns; whether it shares a forward pass with three others or thirty is decided
fresh every step by the scheduler. P1 had to answer "which batch is this request in?" —
here the question does not typecheck.

DAY 3 STATE: correct, not yet fast. The executor still runs one sequence per forward
pass (see its docstring); Day 4 makes the pass genuinely batched. M1 is meaningful now,
M2 is not measurable until then.

Exit criteria: M2 met; M1 holds including the alone-vs-crowded-batch test;
p99 reported alongside throughput.
See IMPLEMENTATION_GUIDE.md "Days 3-5".
"""

from __future__ import annotations

import asyncio
import threading

from inference_server.config import CONFIG
from inference_server.core.executor import Executor
from inference_server.core.scheduler import Scheduler
from inference_server.core.sequence import Sequence
from inference_server.engine.base import Engine, Request, Result
from inference_server.model import load, sync


class ContinuousEngine(Engine):
    name = "continuous"

    #: How long the step thread sleeps when the pool is empty. Long enough not to spin a
    #: core, short enough that it is invisible next to a forward pass.
    IDLE_POLL_S = 0.002

    def __init__(self) -> None:
        self.model, self.tokenizer = load()
        self.executor = Executor(self.model, self.tokenizer)
        self.scheduler = Scheduler(self.executor, self.tokenizer.eos_token_id)

        # One lock over the whole pool. Steps are serialised anyway — there is one device
        # and one step loop — so a finer-grained scheme would buy nothing and cost the
        # ability to reason about this file.
        self._lock = threading.Lock()
        self._futures: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Future]] = {}

        self._wake = threading.Event()
        self._stopping = False
        self._thread: threading.Thread | None = None

    # --------------------------------------------------------------------- sync path
    def generate(self, req: Request) -> Result:
        """One request, driven to completion by the caller. This is what M1 calls.

        It drives the same scheduler the async path does rather than taking a shortcut
        around it — a private fast path here would mean M1 was testing code the server
        never runs.
        """
        seq = self._to_sequence(req)
        with self._lock:
            self.scheduler.add(seq)

        while not seq.is_finished:
            with self._lock:
                self.scheduler.step()

        sync()
        return self._to_result(seq)

    # -------------------------------------------------------------------- async path
    async def submit(self, req: Request) -> Result:
        self._ensure_running()
        seq = self._to_sequence(req)
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()

        with self._lock:
            self._futures[seq.seq_id] = (loop, fut)
            self.scheduler.add(seq)
        self._wake.set()

        return await fut

    # --------------------------------------------------------------- the driving loop
    def _ensure_running(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="step-loop", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        """Turn the crank forever. Never touches asyncio except to hand back results.

        Everything blocking happens on this thread, which is what keeps the guide's
        async-deadlock warning from applying: the step loop cannot be blocked by a
        request handler because it never awaits one.
        """
        while not self._stopping:
            try:
                with self._lock:
                    finished = self.scheduler.step() if self.scheduler.has_work else []
            except Exception as exc:  # noqa: BLE001
                # Found by mutation testing: breaking eviction made the suite hang for
                # ten minutes rather than fail. The step raised, this thread died, and
                # every caller sat on a future nobody would ever resolve. A serving loop
                # that can die silently is worse than one that crashes, so the failure is
                # propagated to everyone waiting and the loop stops deliberately.
                self._fail_all(exc)
                return

            if not finished:
                if not self.scheduler.has_work:
                    # Idle. Wait to be woken by an arrival rather than spinning.
                    self._wake.wait(self.IDLE_POLL_S)
                    self._wake.clear()
                continue

            for seq in finished:
                self._resolve(seq)

    def _resolve(self, seq: Sequence) -> None:
        with self._lock:
            entry = self._futures.pop(seq.seq_id, None)
        if entry is None:
            return                      # a generate() caller is watching it directly
        loop, fut = entry
        result = self._to_result(seq)
        # The future belongs to the caller's event loop, and this is a plain thread.
        # set_result from here would be a data race; call_soon_threadsafe is the bridge.
        loop.call_soon_threadsafe(lambda: None if fut.done() else fut.set_result(result))

    def _fail_all(self, exc: BaseException) -> None:
        """Hand the failure to every waiting caller. Nobody is left hanging."""
        with self._lock:
            pending = list(self._futures.values())
            self._futures.clear()
            self.scheduler.waiting.clear()
            self.scheduler.running.clear()
        for loop, fut in pending:
            loop.call_soon_threadsafe(
                lambda f=fut: None if f.done() else f.set_exception(exc)
            )

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
