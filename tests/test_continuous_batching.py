"""Continuous batching's actual claim — P2 (Day 4).

test_batch_invariance.py submits everything at once, so every sequence prefills together
and the batch only ever shrinks. That misses the case continuous batching exists for: a
request arriving while others are *mid-decode*, joining rows that already hold hundreds
of cached tokens.

That path is the ragged one. A newcomer's cache is short, the incumbents' are long, and
the two have to be merged into one rectangular tensor before they can share a forward
pass. Getting the padding, the mask, or the per-row positions wrong there produces no
crash — just a sequence that quietly decodes differently than it does alone.

So: same goldens, but the arrivals are staggered on purpose.
"""

from __future__ import annotations

import asyncio

import pytest

from inference_server.engine import IMPLEMENTED, build
from inference_server.engine.base import Request

pytestmark = pytest.mark.skipif(
    "continuous" not in IMPLEMENTED, reason="continuous batching not implemented yet"
)


def _req(rid: str, golden: dict) -> Request:
    return Request(
        request_id=rid,
        prompt=golden["prompt"],
        max_tokens=golden["max_tokens"],
        eos_token_id=golden.get("eos_token_id"),
    )


@pytest.mark.phase("P2")
def test_joins_a_batch_already_decoding(goldens):
    """Every golden must survive arriving late to a batch already in flight.

    Each case is submitted twice — once in the opening wave, once after a delay long
    enough that the first wave is deep into decode. Both copies must match the golden,
    which means the merge padded and masked correctly in both directions: short row
    joining long incumbents, and long row joining short ones.
    """
    engine = build("continuous")
    try:
        cases = sorted(goldens["cases"].items())

        async def staggered():
            early = [
                asyncio.create_task(engine.submit(_req(f"{cid}-early", g)))
                for cid, g in cases
            ]
            # Long enough for the first wave to prefill and decode a while, short enough
            # that the shortest golden has not already finished — the newcomer has to
            # land in a genuinely mixed-length cache.
            await asyncio.sleep(0.3)
            late = [
                asyncio.create_task(engine.submit(_req(f"{cid}-late", g)))
                for cid, g in cases
            ]
            return await asyncio.gather(*early), await asyncio.gather(*late)

        early_results, late_results = asyncio.run(staggered())

        failures = []
        for wave, results in (("early", early_results), ("late", late_results)):
            for (case_id, golden), got in zip(cases, results):
                if got.token_ids != golden["token_ids"]:
                    failures.append(
                        f"\n{case_id} ({wave}): differs from running alone"
                        f"\n  expected {golden['token_ids']}"
                        f"\n  got      {got.token_ids}"
                    )
        assert not failures, "continuous batching is not arrival-order invariant:" + "".join(failures)
    finally:
        engine.shutdown()


@pytest.mark.phase("P2")
def test_batch_actually_mutates(goldens):
    """Guard against the whole thing passing because nothing ever batched.

    Every correctness test here would also pass on an engine that ran requests strictly
    one at a time — which is exactly what the Day 3 executor did. This asserts the batch
    genuinely held several sequences at once, so a regression to serial execution is a
    test failure rather than a silent performance cliff.
    """
    engine = build("continuous")
    peak = 0

    try:
        cases = sorted(goldens["cases"].items())
        original_step = engine.scheduler.step

        def watched_step():
            nonlocal peak
            peak = max(peak, len(engine.scheduler.running))
            return original_step()

        engine.scheduler.step = watched_step

        async def all_at_once():
            return await asyncio.gather(
                *(engine.submit(_req(f"{cid}-batched", g)) for cid, g in cases)
            )

        asyncio.run(all_at_once())
        assert peak > 1, f"never batched: peak running set was {peak}"
    finally:
        engine.shutdown()
