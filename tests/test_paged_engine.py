"""P3 (Day 9) — the paged engine end to end: M3, the free list, and late arrivals.

test_correctness / test_batch_invariance / test_http_api already sweep "paged" because
it is in IMPLEMENTED; this file holds what those cannot express:

  - M3 as a measurement rather than arithmetic (test_block_table.py proves the formula;
    this proves the engine actually reports what the formula predicts).
  - Every block coming back after a real workload, through the real executor. The
    FakeExecutor version in test_block_table.py cannot see a leak that only the model
    path introduces.
  - The staggered-arrival case from test_continuous_batching.py, replayed on the paged
    path. There is no padded merge to get wrong here, but there IS a block table per
    newcomer being written next to incumbents deep into decode — the paged analogue.
  - The shared-model contract: the paged executor flips `config._attn_implementation`
    for its forward and must put it back, or the next engine built in this process
    silently runs the paged kernel with no block tables.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from inference_server.bench.metrics import summarize
from inference_server.config import CONFIG
from inference_server.engine import IMPLEMENTED, build
from inference_server.engine.base import Request

pytestmark = pytest.mark.skipif("paged" not in IMPLEMENTED, reason="paged engine not implemented yet")


def _req(rid: str, golden: dict) -> Request:
    return Request(
        request_id=rid,
        prompt=golden["prompt"],
        max_tokens=golden["max_tokens"],
        eos_token_id=golden.get("eos_token_id"),
    )


def _wait_for_free_list(engine, timeout_s: float = 5.0) -> None:
    """Finished sequences give their blocks back at the top of the *next* step, not on
    the step that finished them. The loop takes that step on its own once results are
    resolved, so wait for it rather than asserting the instant the futures return."""
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        with engine._lock:
            if engine.allocator.num_free == engine.pool.num_blocks and not engine.scheduler.has_work:
                return
        time.sleep(0.01)
    with engine._lock:
        held = engine.pool.num_blocks - engine.allocator.num_free
    raise AssertionError(f"{held} blocks still held {timeout_s}s after the pool drained")


# ------------------------------------------------------------------------------ M3
@pytest.mark.phase("P3")
def test_m3_waste_under_ten_percent_on_mixed_lengths(model_and_tokenizer):
    """~20 requests of mixed prompt and output lengths, summarised the way `make bench`
    does. The contiguous engines report 84.4% here; the bar is 10%.

    Lengths follow the bench's `mixed` workload (prompts 16..256, outputs median ~30
    with a tail), because paged waste is the unfilled tail of one block — a fixed
    ~7.5 tokens per sequence — so the percentage is a function of how long sequences
    are. Twenty 10-token requests would report ~40% and prove nothing about the
    allocator; twenty realistic ones must come in under 10%.
    """
    _, tokenizer = model_and_tokenizer
    engine = build("paged")
    try:
        words = "the quick brown fox jumps over the lazy dog and keeps running until ".split()
        reqs = []
        for i in range(20):
            prompt_words = 16 + (i * 37) % 180        # 16..195 words, ~1 token each
            max_tokens = 8 + (i * 23) % 100            # 8..107 tokens
            prompt = " ".join(words[(i + j) % len(words)] for j in range(prompt_words))
            reqs.append(Request(request_id=f"m3-{i}", prompt=prompt, max_tokens=max_tokens))

        async def run():
            return await asyncio.gather(*(engine.submit(r) for r in reqs))

        t0 = time.perf_counter()
        results = asyncio.run(run())
        summary = summarize(results, engine="paged", workload="m3", concurrency=len(reqs),
                            duration_s=time.perf_counter() - t0)

        for r in results:
            assert r.used_tokens == r.prompt_len + r.num_generated
            assert r.reserved_tokens >= r.used_tokens
            assert r.reserved_tokens - r.used_tokens < CONFIG.block_size, (
                f"{r.request_id} over-reserved by a whole block"
            )
            assert r.reserved_tokens % CONFIG.block_size == 0
        assert summary.num_errors == 0
        assert summary.kv_waste_pct < 10.0, f"kv_waste_pct={summary.kv_waste_pct:.1f}% misses M3"

        _wait_for_free_list(engine)
        assert engine.allocator.num_free == engine.pool.num_blocks
    finally:
        engine.shutdown()


# --------------------------------------------------------------------- free list
@pytest.mark.phase("P3")
def test_free_list_returns_to_full_after_the_goldens(goldens):
    """The M4 canary on the real executor. Every golden at once, then everything must
    be back in the allocator — through the real scatter/gather path, on the device."""
    engine = build("paged")
    try:
        cases = sorted(goldens["cases"].items())

        async def all_at_once():
            return await asyncio.gather(*(engine.submit(_req(f"{cid}-fl", g)) for cid, g in cases))

        results = asyncio.run(all_at_once())
        assert all(r.token_ids == g["token_ids"] for r, (_, g) in zip(results, cases))
        _wait_for_free_list(engine)
        assert engine.allocator.num_free == engine.pool.num_blocks
        # And nothing leaked *into* a sequence either: a table freed is a table emptied.
        assert all(s.block_table is None or s.block_table.num_blocks == 0 for s in engine.scheduler.running)
    finally:
        engine.shutdown()


# ---------------------------------------------------------------- late arrivals
@pytest.mark.phase("P3")
def test_joins_a_batch_already_decoding(goldens):
    """Mirror of test_continuous_batching.py::test_joins_a_batch_already_decoding.

    A newcomer prefills into freshly allocated blocks while incumbents are hundreds of
    tokens into their own tables. Both copies of every golden must match: the early
    wave must not be disturbed by a prefill pass landing between its decode steps, and
    the late wave must read only its own blocks.
    """
    engine = build("paged")
    try:
        cases = sorted(goldens["cases"].items())

        async def staggered():
            early = [asyncio.create_task(engine.submit(_req(f"{cid}-early", g))) for cid, g in cases]
            await asyncio.sleep(0.3)
            late = [asyncio.create_task(engine.submit(_req(f"{cid}-late", g))) for cid, g in cases]
            return await asyncio.gather(*early), await asyncio.gather(*late)

        early_results, late_results = asyncio.run(staggered())

        failures = []
        for wave, results in (("early", early_results), ("late", late_results)):
            for (case_id, golden), got in zip(cases, results):
                if got.token_ids != golden["token_ids"]:
                    failures.append(
                        f"\n{case_id} ({wave}): differs from running alone"
                        f"\n  expected {golden['token_ids']}\n  got      {got.token_ids}"
                    )
        assert not failures, "paged engine is not arrival-order invariant:" + "".join(failures)
    finally:
        engine.shutdown()


# ------------------------------------------------------------- shared-model contract
@pytest.mark.phase("P3")
def test_attention_implementation_is_restored_after_every_step(model_and_tokenizer, goldens):
    """`model.load()` is an lru_cache: one model object serves every engine in this
    process. The paged executor swaps its attention function in for the duration of a
    forward and must swap it back, otherwise a continuous engine built afterwards runs
    the paged kernel without a block table and fails M1 in a way that points nowhere
    near the cause."""
    model, _ = model_and_tokenizer
    before = model.config._attn_implementation
    engine = build("paged")
    try:
        case_id, golden = sorted(goldens["cases"].items())[0]
        assert engine.generate(_req(case_id, golden)).token_ids == golden["token_ids"]
    finally:
        engine.shutdown()
    assert model.config._attn_implementation == before
