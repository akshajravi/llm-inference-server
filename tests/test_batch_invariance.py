"""A sequence's output must not depend on who else is in the batch.

Written for P2, but it starts earning its keep on Day 2: static batching is the first
engine that pads, and padding is where this class of bug lives. The failure is silent —
nothing crashes, the text stays fluent, the row just quietly generates something other
than what it generates alone. M1 cannot see it, because M1 runs one request at a time.

The two ways to fail, both exercised here:
  - attention mask wrong -> the row attends to pad tokens as if they were real text
  - position IDs wrong   -> a left-padded row thinks its first token is at position 3

Engines join this test as soon as they can form a batch.
"""

from __future__ import annotations

import asyncio

import pytest

from inference_server.engine import IMPLEMENTED, build
from inference_server.engine.base import Request

ENGINES_UNDER_TEST = [e for e in ("static", "continuous", "paged") if e in IMPLEMENTED]

pytestmark = pytest.mark.skipif(
    not ENGINES_UNDER_TEST, reason="no batching engine implemented yet"
)


def _req(rid: str, golden: dict) -> Request:
    return Request(
        request_id=rid,
        prompt=golden["prompt"],
        max_tokens=golden["max_tokens"],
        eos_token_id=golden.get("eos_token_id"),
    )


@pytest.mark.phase("P1")
@pytest.mark.parametrize("engine_name", ENGINES_UNDER_TEST)
def test_alone_vs_crowded(engine_name, goldens):
    """Every golden case must survive being padded next to all the others.

    Checking one case would let a padding bug hide in whichever row happened to be the
    longest — the row that gets no padding is exactly the row that cannot detect the
    bug. So every case is a target, and the batch is deliberately ragged.
    """
    engine = build(engine_name)
    try:
        cases = sorted(goldens["cases"].items())

        async def crowded():
            return await asyncio.gather(
                *(engine.submit(_req(f"{cid}-crowded", g)) for cid, g in cases)
            )

        results = asyncio.run(crowded())

        failures = []
        for (case_id, golden), got in zip(cases, results):
            if got.token_ids != golden["token_ids"]:
                failures.append(
                    f"\n{case_id}: batched output differs from alone"
                    f"\n  expected {golden['token_ids']}"
                    f"\n  got      {got.token_ids}"
                )
        assert not failures, f"{engine_name} is not batch-invariant:" + "".join(failures)
    finally:
        engine.shutdown()
