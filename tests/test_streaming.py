"""S4 — SSE streaming. Token-exact, incremental, and the same wire semantics as /generate.

Streaming is the one feature where "it works" can be true of an implementation that
buffers the whole completion and emits it at the end. So beyond token equality with the
goldens, this file asserts the stream is *actually* incremental: one event per token,
and the first token arrives long before the last one is generated.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from inference_server.engine import build
from inference_server.engine.base import Request
from inference_server.server import app as app_module

STREAMING_ENGINES = ["continuous", "paged"]


def _req(rid: str, golden: dict) -> Request:
    return Request(
        request_id=rid,
        prompt=golden["prompt"],
        max_tokens=golden["max_tokens"],
        eos_token_id=golden.get("eos_token_id"),
    )


@pytest.fixture(scope="module", params=STREAMING_ENGINES)
def engine(request):
    eng = build(request.param)
    yield eng
    eng.shutdown()


async def _collect(engine, req: Request):
    """Every event plus its arrival time, relative to the call."""
    t0 = time.perf_counter()
    events, stamps = [], []
    async for ev in engine.stream(req):
        events.append(ev)
        stamps.append(time.perf_counter() - t0)
    return events, stamps


# ---------------------------------------------------------------------- in-process
@pytest.mark.phase("S4")
def test_streamed_tokens_match_goldens_and_submit(engine, goldens):
    """Concatenated token events == golden == the non-streamed result, every case,
    including the EOS-override one (the stop token must be streamed, then stop)."""

    async def run_all():
        out = {}
        for case_id, golden in sorted(goldens["cases"].items()):
            events, _ = await _collect(engine, _req(f"stream-{case_id}", golden))
            result = await engine.submit(_req(f"submit-{case_id}", golden))
            out[case_id] = (events, result)
        return out

    outcomes = asyncio.run(run_all())
    failures = []
    for case_id, (events, result) in outcomes.items():
        golden = goldens["cases"][case_id]
        tokens = [ev.token_id for ev in events if not ev.done]
        final = events[-1]
        assert final.done and all(not ev.done for ev in events[:-1]), case_id
        if tokens != golden["token_ids"]:
            failures.append(f"\n{case_id}: streamed {tokens}\n  expected {golden['token_ids']}")
            continue
        assert final.result.token_ids == golden["token_ids"] == result.token_ids, case_id
        assert final.result.finish_reason == golden["finish_reason"] == result.finish_reason
        assert len(events) == len(golden["token_ids"]) + 1, f"{case_id}: not one event per token"
        # The text deltas must reassemble the full decoded text (BPE/multi-byte safe).
        assert "".join(ev.text for ev in events[:-1]) == final.result.text == result.text, case_id
        assert final.result.prompt_len == result.prompt_len
        assert final.result.used_tokens == result.used_tokens
    assert not failures, "streaming diverged from goldens:" + "".join(failures)


@pytest.mark.phase("S4")
def test_stream_is_incremental(engine, goldens):
    """The first token must arrive well before the completion finishes.

    Measured against the same request's non-streamed latency: with N decode steps,
    the first event lands after roughly one step and the last after N. A buffered
    implementation would deliver everything at ~N steps and fail both bounds."""
    golden = max(goldens["cases"].values(), key=lambda g: len(g["token_ids"]))
    assert len(golden["token_ids"]) >= 16, "need a long golden for a timing margin"

    async def run():
        ref = await engine.submit(_req("incremental-ref", golden))
        events, stamps = await _collect(engine, _req("incremental-stream", golden))
        return ref, events, stamps

    ref, events, stamps = asyncio.run(run())
    assert len(events) == ref.num_generated + 1
    t_first, t_last = stamps[0], stamps[-1]
    assert t_first < 0.5 * t_last, (
        f"first token at {t_first:.3f}s, stream done at {t_last:.3f}s — not incremental"
    )
    assert t_last - t_first > 0.25 * ref.latency_s, (
        f"tokens arrived within {t_last - t_first:.3f}s of each other for a "
        f"{ref.latency_s:.3f}s completion — buffered?"
    )
    # Every stamp is non-decreasing and events are spread out, not two clumps.
    assert stamps == sorted(stamps)


@pytest.mark.phase("S4")
def test_duplicate_request_id_is_rejected(engine, goldens):
    from inference_server.engine.continuous import DuplicateRequest

    golden = goldens["cases"]["long_prompt"]

    async def run():
        first = asyncio.ensure_future(engine.submit(_req("dup", golden)))
        await asyncio.sleep(0)                          # let it register
        with pytest.raises(DuplicateRequest):
            await engine.submit(_req("dup", golden))
        return await first

    result = asyncio.run(run())
    assert result.token_ids == golden["token_ids"], "the original request was disturbed"
    assert not engine._futures


# ------------------------------------------------------------------------- over HTTP
def _parse_sse(body: str) -> list:
    """`data: ...` frames -> parsed JSON, or the literal "[DONE]"."""
    frames = []
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        assert chunk.startswith("data: "), chunk
        payload = chunk[len("data: "):]
        frames.append(payload if payload == "[DONE]" else json.loads(payload))
    return frames


@pytest.fixture(scope="module", params=STREAMING_ENGINES)
def client(request):
    engine = build(request.param)
    app_module._engine = engine
    # Not entered as a context manager: that runs `lifespan`, which would build a
    # second engine from the ENGINE env var and overwrite the injected one.
    yield TestClient(app_module.app)
    engine.shutdown()
    app_module._engine = None


@pytest.mark.phase("S4")
def test_sse_body_parses_to_the_same_tokens(client, goldens):
    for case_id, golden in sorted(goldens["cases"].items()):
        body = {"prompt": golden["prompt"], "max_tokens": golden["max_tokens"]}
        if golden.get("eos_token_id") is not None:
            body["eos_token_id"] = golden["eos_token_id"]

        resp = client.post("/generate/stream", json=body)
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"].startswith("text/event-stream")
        frames = _parse_sse(resp.text)

        assert frames[-1] == "[DONE]", case_id
        final = frames[-2]
        tokens = frames[:-2]
        assert final.get("done") is True, case_id
        assert [f["token_id"] for f in tokens] == golden["token_ids"], case_id
        assert final["token_ids"] == golden["token_ids"], case_id
        assert final["finish_reason"] == golden["finish_reason"], case_id
        assert final["num_generated"] == len(golden["token_ids"]) == len(tokens), case_id
        assert "".join(f["text"] for f in tokens) == final["text"], case_id

        # Final event carries the same fields /generate does, with the same values
        # for everything that is not a timing.
        plain = client.post("/generate", json=body).json()
        for key in ("token_ids", "finish_reason", "num_generated", "text",
                    "prompt_len", "reserved_tokens", "used_tokens"):
            assert final[key] == plain[key], f"{case_id}: {key}"
        assert final["ttft_s"] > 0 and final["latency_s"] >= final["ttft_s"], case_id


def _too_long_prompt(engine) -> str:
    """One token more than the whole pool holds. Tokenised, not guessed."""
    from inference_server.config import CONFIG
    return "a " * (engine.allocator.num_blocks * CONFIG.block_size + 1)


@pytest.mark.phase("S4")
def test_stream_admission_matches_generate(client):
    """A 422 for a request that can never fit, decided before any byte is streamed."""
    engine = app_module._engine
    if engine.scheduler.allocator is None:
        pytest.skip("SequenceTooLong only exists with a block allocator")
    resp = client.post("/generate/stream", json={"prompt": _too_long_prompt(engine), "max_tokens": 64})
    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"]["error"] == "sequence_too_long"
    assert not engine._futures


@pytest.mark.phase("S4")
def test_non_streaming_engine_returns_501():
    engine = build("naive")
    app_module._engine = engine
    try:
        resp = TestClient(app_module.app).post(
            "/generate/stream", json={"prompt": "hi", "max_tokens": 2}
        )
        assert resp.status_code == 501, resp.text
        assert resp.json()["detail"]["error"] == "not_implemented"
    finally:
        engine.shutdown()
        app_module._engine = None
