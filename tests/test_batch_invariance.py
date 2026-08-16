"""P2 (Days 3-5) — a sequence's output must not depend on who else is in the batch.

Written on Day 3, before it is needed. Ragged batching makes this the most likely place
for a silent correctness regression: the bug does not crash, it just quietly produces
different tokens when the batch is crowded, and no other test would notice.

Enable by adding the engine name to ENGINES_UNDER_TEST once P2 lands.
"""

from __future__ import annotations

import asyncio

import pytest

from inference_server.engine import IMPLEMENTED, build
from inference_server.engine.base import Request

ENGINES_UNDER_TEST = [e for e in ("continuous", "paged") if e in IMPLEMENTED]

pytestmark = pytest.mark.skipif(
    not ENGINES_UNDER_TEST, reason="batching engines land in P2 (Days 3-5)"
)


@pytest.mark.phase("P2")
@pytest.mark.parametrize("engine_name", ENGINES_UNDER_TEST)
def test_alone_vs_crowded(engine_name, goldens):
    engine = build(engine_name)
    try:
        target_id, target = sorted(goldens["cases"].items())[0]
        req = lambda rid: Request(  # noqa: E731
            request_id=rid, prompt=target["prompt"], max_tokens=target["max_tokens"]
        )

        alone = engine.generate(req(f"{target_id}-alone"))

        async def crowded():
            noise = [
                engine.submit(
                    Request(request_id=f"noise-{i}", prompt=g["prompt"], max_tokens=g["max_tokens"])
                )
                for i, (_, g) in enumerate(sorted(goldens["cases"].items()))
            ]
            return await asyncio.gather(engine.submit(req(f"{target_id}-crowded")), *noise)

        crowded_result = asyncio.run(crowded())[0]
        assert crowded_result.token_ids == alone.token_ids
        assert alone.token_ids == goldens["cases"][target_id]["token_ids"]
    finally:
        engine.shutdown()
