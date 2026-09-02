"""M1 over the wire — the same goldens, driven through FastAPI instead of in-process.

This file exists because the in-process suite was green while the server was wrong.
`GenerateRequest` had no `eos_token_id` field, so the per-request stop token was dropped
at the HTTP boundary and both EOS goldens ran to the cap. Every engine passed
`test_eos_stops_early`; the thing users actually talk to did not. A seam that only one
of two callers exercises is a seam that will drift, so the goldens now run on both.

Deliberately narrow: correctness of the HTTP path, not its performance. Throughput is
measured in-process (see bench/loadgen.py) so the numbers isolate scheduling from
uvicorn's own queueing.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from inference_server.engine import IMPLEMENTED, build
from inference_server.server import app as app_module


@pytest.fixture(scope="module", params=IMPLEMENTED)
def client(request):
    """A real ASGI client over the real app.

    The engine is injected rather than built by `lifespan`, because `lifespan` reads the
    ENGINE env var at import time and this fixture needs to sweep all of them in one
    process. Everything below that injection — routing, pydantic validation, JSON
    encoding — is the production path, and the validation layer is exactly what broke.
    """
    engine = build(request.param)
    app_module._engine = engine
    # Not entered as a context manager on purpose: that would run `lifespan`, which
    # builds an engine from the ENGINE env var and would overwrite the injected one.
    yield TestClient(app_module.app)
    engine.shutdown()
    app_module._engine = None


def _post(client, golden: dict) -> dict:
    body = {"prompt": golden["prompt"], "max_tokens": golden["max_tokens"]}
    if golden.get("eos_token_id") is not None:
        body["eos_token_id"] = golden["eos_token_id"]
    resp = client.post("/generate", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.mark.phase("P2")
def test_http_matches_goldens(client, goldens):
    """Token-id equality across the wire. The in-process assertion, one layer out."""
    failures = []
    for case_id, golden in sorted(goldens["cases"].items()):
        body = _post(client, golden)
        if body["token_ids"] != golden["token_ids"]:
            failures.append(
                f"\n{case_id}:\n  expected {golden['token_ids']}\n  got      {body['token_ids']}"
            )
    assert not failures, "HTTP path diverged from goldens:" + "".join(failures)


@pytest.mark.phase("P2")
def test_http_eos_override_crosses_the_wire(client, goldens):
    """The regression that motivated this file, pinned as its own test.

    Without `eos_token_id` on GenerateRequest the request still succeeds — it just
    generates to `max_tokens` using the tokenizer's default stop token. So the symptom
    is a silently longer completion, not an error, which is why it survived until the
    goldens were driven through the server by hand.
    """
    eos_cases = {k: g for k, g in goldens["cases"].items() if g["finish_reason"] == "eos"}
    assert eos_cases, "no golden terminates on EOS; regenerate goldens"

    for case_id, golden in eos_cases.items():
        body = _post(client, golden)
        assert body["finish_reason"] == "eos", case_id
        assert body["num_generated"] < golden["max_tokens"], (
            f"{case_id} hit the cap over HTTP — the stop token did not survive the wire"
        )


@pytest.mark.phase("P2")
def test_health_reports_the_running_engine(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_id"] and body["device"]
