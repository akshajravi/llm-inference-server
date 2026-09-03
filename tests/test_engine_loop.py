"""P4 — the step loop must never starve the event loop (M4).

The overload run found the server refusing TCP connections instead of sending 503s:
`submit()` and `/health` waited on the step lock, Python locks are not fair, and the
event-loop thread lost the race for seconds at a time. The fix is an inbox: nothing on
the event-loop thread waits for a forward pass. These tests hold the step lock from a
thread — simulating a forward pass that never ends — and check that every event-loop
operation still returns immediately, with the right answer.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time

import pytest

from inference_server.config import CONFIG
from inference_server.core.scheduler import QueueFull, SequenceTooLong
from inference_server.engine import build
from inference_server.engine.base import Request
from inference_server.engine.continuous import DuplicateRequest

#: Long enough that a lock-waiting implementation fails; short enough not to matter.
PROMPT_S = 0.25


def _req(rid: str, golden: dict) -> Request:
    return Request(
        request_id=rid,
        prompt=golden["prompt"],
        max_tokens=golden["max_tokens"],
        eos_token_id=golden.get("eos_token_id"),
    )


@pytest.fixture(scope="module", params=["continuous", "paged"])
def engine(request):
    eng = build(request.param)
    yield eng
    eng.shutdown()


class HeldStepLock:
    """Hold the engine's step lock from another thread until released.

    Stands in for a forward pass in progress. Everything the event loop does while
    this is held must complete without waiting on it."""

    def __init__(self, engine) -> None:
        self.engine = engine
        self._release = threading.Event()
        self._held = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        with self.engine._lock:
            self._held.set()
            self._release.wait()

    def __enter__(self):
        self._thread.start()
        assert self._held.wait(5.0)
        return self

    def __exit__(self, *_):
        self._release.set()
        self._thread.join(5.0)


async def _returns_promptly(fn, budget_s: float = PROMPT_S):
    t0 = time.perf_counter()
    out = fn()
    if inspect.isawaitable(out):
        out = await out
    took = time.perf_counter() - t0
    assert took < budget_s, f"took {took:.3f}s while the step lock was held — it waited on it"
    return out


@pytest.mark.phase("P4")
def test_submit_registers_without_the_step_lock(engine, goldens):
    golden = goldens["cases"]["short_prompt"]

    async def run():
        with HeldStepLock(engine):
            task = asyncio.ensure_future(engine.submit(_req("inbox-1", golden)))
            # The request must be in the inbox before anything steps, i.e. now.
            deadline = time.perf_counter() + PROMPT_S
            while "inbox-1" not in engine._futures and time.perf_counter() < deadline:
                await asyncio.sleep(0.001)
            assert "inbox-1" in engine._futures, "submit did not register while the lock was held"
            assert not task.done()
        return await task

    result = asyncio.run(run())
    assert result.token_ids == golden["token_ids"]
    assert not engine._futures


@pytest.mark.phase("P4")
def test_stats_reads_without_the_step_lock(engine):
    async def run():
        with HeldStepLock(engine):
            stats = await _returns_promptly(engine.stats)
        return stats

    stats = asyncio.run(run())
    for key in ("queue_depth", "num_running", "completed"):
        assert key in stats


@pytest.mark.phase("P4")
def test_queue_full_is_raised_synchronously_counting_the_inbox(engine, goldens, monkeypatch):
    """With the step thread unable to drain, the bound is enforced against
    `queue_depth + len(inbox)`: the third submit is refused at the door, immediately,
    and the two accepted ones complete correctly once the lock is released."""
    monkeypatch.setattr(CONFIG, "max_queue_depth", 2)
    golden = goldens["cases"]["short_prompt"]

    async def run():
        with HeldStepLock(engine):
            accepted = [asyncio.ensure_future(engine.submit(_req(f"qf-{i}", golden))) for i in range(2)]
            await asyncio.sleep(0.01)
            assert len(engine._inbox) == 2
            t0 = time.perf_counter()
            with pytest.raises(QueueFull) as info:
                await engine.submit(_req("qf-2", golden))
            assert time.perf_counter() - t0 < PROMPT_S
            assert info.value.depth == 2 and info.value.bound == 2
            assert "qf-2" not in engine._futures, "a refused request left a waiter behind"
        return await asyncio.gather(*accepted)

    results = asyncio.run(run())
    assert all(r.token_ids == golden["token_ids"] for r in results)
    assert not engine._futures and not engine._inbox


@pytest.mark.phase("P4")
def test_sequence_too_long_is_raised_synchronously(engine):
    if engine.scheduler.allocator is None:
        pytest.skip("SequenceTooLong only exists with a block allocator")
    # One token more than the whole pool holds; tokenised, not guessed.
    prompt = "a " * (engine.allocator.num_blocks * CONFIG.block_size + 1)
    too_long = Request(request_id="huge", prompt=prompt, max_tokens=64)
    prompt_len = len(engine.tokenizer(prompt).input_ids)

    async def run():
        with HeldStepLock(engine):
            with pytest.raises(SequenceTooLong) as info:
                await _returns_promptly(lambda: engine.submit(too_long))
        return info.value

    exc = asyncio.run(run())
    # Same numbers the scheduler itself would have produced.
    assert exc.available == engine.allocator.num_blocks
    assert exc.needed == engine.scheduler._blocks_for(prompt_len + 64 - 1)
    assert "huge" not in engine._futures

    with pytest.raises(SequenceTooLong):
        engine.generate(too_long)                     # the sync path shares the check


@pytest.mark.phase("P4")
def test_add_failure_in_the_step_thread_fails_only_that_request(engine, goldens, monkeypatch):
    """The race the inbox cannot close: `scheduler.add()` refuses a request the engine
    had already accepted. That request's future gets the exception; nothing hangs;
    every other request is untouched."""
    golden = goldens["cases"]["short_prompt"]
    real_add = engine.scheduler.add

    def flaky_add(seq):
        if seq.seq_id == "doomed":
            raise QueueFull(99, 99)
        real_add(seq)

    monkeypatch.setattr(engine.scheduler, "add", flaky_add)

    async def run():
        ok = asyncio.ensure_future(engine.submit(_req("fine", golden)))
        with pytest.raises(QueueFull) as info:
            await asyncio.wait_for(engine.submit(_req("doomed", golden)), timeout=30)
        assert info.value.depth == 99
        return await ok

    result = asyncio.run(run())
    assert result.token_ids == golden["token_ids"]
    assert not engine._futures and not engine._inbox


@pytest.mark.phase("P4")
def test_generate_uses_the_same_admission_path(engine, goldens):
    """`generate()` enters through the inbox too, so it sees the same refusals."""
    golden = goldens["cases"]["short_prompt"]

    async def run():
        pending = asyncio.ensure_future(engine.submit(_req("shared-id", golden)))
        await asyncio.sleep(0)
        with pytest.raises(DuplicateRequest):
            await asyncio.get_running_loop().run_in_executor(
                None, engine.generate, _req("shared-id", golden)
            )
        return await pending

    assert asyncio.run(run()).token_ids == golden["token_ids"]
    # And a clean generate() still produces the golden and leaves nothing behind.
    assert engine.generate(_req("gen-alone", golden)).token_ids == golden["token_ids"]
    assert not engine._futures and not engine._inbox
