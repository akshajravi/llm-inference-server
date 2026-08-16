"""M1 — the tripwire, installed before walking into the minefield.

Every engine must reproduce HuggingFace greedy `generate()` token-for-token. This is
parameterised over IMPLEMENTED engines, so an engine becomes subject to M1 the moment
it is declared shippable — there is no way to add a phase and forget to test it.

On Day 1 this passes trivially (generate() vs generate()). That is the point.
"""

from __future__ import annotations

import pytest

from inference_server.engine import IMPLEMENTED, build
from inference_server.engine.base import Request


@pytest.fixture(scope="module", params=IMPLEMENTED)
def engine(request):
    eng = build(request.param)
    yield eng
    eng.shutdown()


def _cases(goldens: dict):
    return sorted(goldens["cases"].items())


@pytest.mark.phase("P0")
def test_matches_goldens(engine, goldens):
    """Exact token-id equality. Not 'close', not 'same text' — equal."""
    failures = []
    for case_id, golden in _cases(goldens):
        result = engine.generate(
            Request(request_id=case_id, prompt=golden["prompt"], max_tokens=golden["max_tokens"])
        )
        if result.token_ids != golden["token_ids"]:
            failures.append(
                f"\n{case_id}:\n  expected {golden['token_ids']}\n  got      {result.token_ids}"
            )
    assert not failures, f"{engine.name} diverged from goldens:" + "".join(failures)


@pytest.mark.phase("P0")
def test_respects_max_tokens(engine, goldens):
    for case_id, golden in _cases(goldens):
        result = engine.generate(
            Request(request_id=case_id, prompt=golden["prompt"], max_tokens=golden["max_tokens"])
        )
        assert len(result.token_ids) <= golden["max_tokens"], case_id


@pytest.mark.phase("P0")
def test_finish_reason(engine, goldens):
    """A sequence that stops early must say so — P4's preemption logic reads this."""
    for case_id, golden in _cases(goldens):
        result = engine.generate(
            Request(request_id=case_id, prompt=golden["prompt"], max_tokens=golden["max_tokens"])
        )
        assert result.finish_reason == golden["finish_reason"], case_id


@pytest.mark.phase("P0")
def test_determinism(engine, goldens):
    """Same prompt twice, same tokens. Cheap, and catches state leaking between calls."""
    case_id, golden = _cases(goldens)[0]
    req = Request(request_id=case_id, prompt=golden["prompt"], max_tokens=golden["max_tokens"])
    assert engine.generate(req).token_ids == engine.generate(req).token_ids
