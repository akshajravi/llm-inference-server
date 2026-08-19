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


def _req(case_id: str, golden: dict) -> Request:
    """Rebuild the exact request the golden was generated from, stop token included."""
    return Request(
        request_id=case_id,
        prompt=golden["prompt"],
        max_tokens=golden["max_tokens"],
        eos_token_id=golden.get("eos_token_id"),
    )


@pytest.mark.phase("P0")
def test_matches_goldens(engine, goldens):
    """Exact token-id equality. Not 'close', not 'same text' — equal."""
    failures = []
    for case_id, golden in _cases(goldens):
        result = engine.generate(_req(case_id, golden))
        if result.token_ids != golden["token_ids"]:
            failures.append(
                f"\n{case_id}:\n  expected {golden['token_ids']}\n  got      {result.token_ids}"
            )
    assert not failures, f"{engine.name} diverged from goldens:" + "".join(failures)


@pytest.mark.phase("P0")
def test_respects_max_tokens(engine, goldens):
    for case_id, golden in _cases(goldens):
        result = engine.generate(_req(case_id, golden))
        assert len(result.token_ids) <= golden["max_tokens"], case_id


@pytest.mark.phase("P0")
def test_finish_reason(engine, goldens):
    """A sequence that stops early must say so — P4's preemption logic reads this."""
    for case_id, golden in _cases(goldens):
        result = engine.generate(_req(case_id, golden))
        assert result.finish_reason == golden["finish_reason"], case_id


@pytest.mark.phase("P0")
def test_eos_stops_early(engine, goldens):
    """The termination path, tested for real: the sequence must stop *before* the cap.

    Guards against the failure this suite already had once — every golden finishing with
    reason "length", so nothing exercised EOS at all while the suite still looked green.
    """
    eos_cases = {k: g for k, g in goldens["cases"].items() if g["finish_reason"] == "eos"}
    assert eos_cases, "no golden terminates on EOS; regenerate goldens"

    for case_id, golden in eos_cases.items():
        result = engine.generate(_req(case_id, golden))
        assert result.finish_reason == "eos", case_id
        assert len(result.token_ids) < golden["max_tokens"], (
            f"{case_id} hit the cap instead of stopping at EOS"
        )


@pytest.mark.phase("P0")
def test_determinism(engine, goldens):
    """Same prompt twice, same tokens. Cheap, and catches state leaking between calls."""
    case_id, golden = _cases(goldens)[0]
    req = _req(case_id, golden)
    assert engine.generate(req).token_ids == engine.generate(req).token_ids
