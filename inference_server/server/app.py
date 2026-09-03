"""FastAPI app (FR1). Engine selected by the ENGINE env var so the same server binary
benchmarks every phase.

    ENGINE=naive python -m uvicorn inference_server.server.app:app

Day 1 ships /generate and /health. P4 adds the 503 path (FR7) and the /health counters
the overload run (M4) charts. /generate/stream (S4) streams tokens over SSE.
"""

from __future__ import annotations

import contextlib
import json
import os
import resource
import sys
import uuid

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from inference_server.config import CONFIG, device_name
from inference_server.core.scheduler import QueueFull, SequenceTooLong
from inference_server.engine import Request, build
from inference_server.engine.continuous import DuplicateRequest
from inference_server.server.schemas import GenerateRequest, GenerateResponse, HealthResponse

ENGINE_NAME = os.environ.get("ENGINE", "naive")
_engine = None


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = build(ENGINE_NAME)      # model loads once, at startup, never per request
    yield
    _engine.shutdown()


app = FastAPI(title="inference-server", lifespan=lifespan)


# ------------------------------------------------------------------ P4 rejections (FR7)
@app.exception_handler(QueueFull)
async def _queue_full(_, exc: QueueFull) -> JSONResponse:
    """503, immediately, with the numbers. The bounded queue's whole argument is that an
    honest error beats unbounded latency; a 503 with no body would be an honest error
    the client cannot act on. Retry-After is a hint, not a promise — one step loop, one
    queue, and the drain rate is whatever the model does."""
    return JSONResponse(
        status_code=503,
        content={
            "detail": {
                "error": "queue_full",
                "message": str(exc),
                "queue_depth": exc.depth,
                "max_queue_depth": exc.bound,
            }
        },
        headers={"Retry-After": "1"},
    )


@app.exception_handler(SequenceTooLong)
async def _too_long(_, exc: SequenceTooLong) -> JSONResponse:
    """422, not 503: this request can never be served by this deployment, so telling
    the client to retry would be a lie. 422 is what FastAPI already uses for a body
    that fails validation, and "prompt + max_tokens exceeds the pool" is a validation
    failure the schema cannot express because it depends on the tokenizer."""
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "error": "sequence_too_long",
                "message": str(exc),
                "blocks_needed": exc.needed,
                "num_blocks": exc.available,
            }
        },
    )


@app.exception_handler(DuplicateRequest)
async def _duplicate(_, exc: DuplicateRequest) -> JSONResponse:
    """400: a request_id already in flight. Unreachable through this app, which mints
    a uuid per request, but the engine raises it for in-process callers and the mapping
    belongs next to the other two."""
    return JSONResponse(
        status_code=400,
        content={"detail": {"error": "duplicate_request", "message": str(exc)}},
    )


# ------------------------------------------------------------------- P4 observability
def _rss_bytes() -> int:
    """Peak resident set size, in bytes on every platform. getrusage reports ru_maxrss
    in bytes on macOS and kibibytes on Linux, which is the kind of discrepancy that
    turns a flat memory chart into a 1000x cliff between the dev box and the GPU box."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def _device_mem_bytes() -> int:
    if CONFIG.device == "cuda":
        return int(torch.cuda.memory_allocated())
    if CONFIG.device == "mps":
        return int(torch.mps.current_allocated_memory())
    return 0


def _to_request(body: GenerateRequest) -> Request:
    return Request(
        request_id=str(uuid.uuid4()),
        prompt=body.prompt,
        max_tokens=body.max_tokens,
        eos_token_id=body.eos_token_id,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest) -> GenerateResponse:
    result = await _engine.submit(_to_request(body))
    return GenerateResponse(
        text=result.text,
        token_ids=result.token_ids,
        num_generated=result.num_generated,
        finish_reason=result.finish_reason,
        ttft_s=result.ttft_s,
        latency_s=result.latency_s,
        prompt_len=result.prompt_len,
        reserved_tokens=result.reserved_tokens,
        used_tokens=result.used_tokens,
    )


# ------------------------------------------------------------------------ S4 streaming
def _sse(payload) -> str:
    return f"data: {payload if isinstance(payload, str) else json.dumps(payload)}\n\n"


@app.post("/generate/stream")
async def generate_stream(body: GenerateRequest):
    """Server-sent events, one per token, as they are sampled.

    Wire format (each line is one `data:` event, blank-line terminated):

        data: {"token_id": 464, "text": "The"}          one per generated token
        data: {"done": true, "finish_reason": "length", "num_generated": 16,
               "ttft_s": 0.01, "latency_s": 0.2, "token_ids": [...], "text": "...",
               "prompt_len": 5, "reserved_tokens": 32, "used_tokens": 21}
        data: [DONE]

    The `text` deltas concatenate to the final event's `text`. Admission is decided
    before any byte is sent, so QueueFull and SequenceTooLong are the same 503 / 422
    JSON responses /generate returns. A failure after the headers are out (the rare
    admission race inside the step thread, or the step loop dying) cannot change the
    status code any more; it is reported as `data: {"error": ...}` followed by [DONE].

    Engines without a step loop (naive, manual, static) cannot stream: 501.
    """
    stream_fn = getattr(_engine, "stream", None)
    if not callable(stream_fn):
        return JSONResponse(
            status_code=501,
            content={"detail": {"error": "not_implemented",
                                "message": f"engine {ENGINE_NAME!r} does not stream"}},
        )
    try:
        events = stream_fn(_to_request(body))     # raises QueueFull / SequenceTooLong here
    except NotImplementedError as exc:
        return JSONResponse(
            status_code=501,
            content={"detail": {"error": "not_implemented", "message": str(exc)}},
        )

    async def body_iter():
        # If the client disconnects, this generator is closed and the sequence keeps
        # running to completion; the engine drops its queue when it finishes.
        # Cancelling in-flight work on disconnect is future work.
        try:
            async for ev in events:
                if ev.done:
                    r = ev.result
                    yield _sse({
                        "done": True,
                        "finish_reason": r.finish_reason,
                        "num_generated": r.num_generated,
                        "ttft_s": r.ttft_s,
                        "latency_s": r.latency_s,
                        "token_ids": r.token_ids,
                        "text": r.text,
                        "prompt_len": r.prompt_len,
                        "reserved_tokens": r.reserved_tokens,
                        "used_tokens": r.used_tokens,
                    })
                else:
                    yield _sse({"token_id": ev.token_id, "text": ev.text})
        except Exception as exc:  # noqa: BLE001 — headers are already out
            yield _sse({"error": type(exc).__name__, "message": str(exc)})
        yield _sse("[DONE]")

    return StreamingResponse(
        body_iter(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness plus the M4 gauges. The scheduler's counters come from `engine.stats()`
    when the engine has one (continuous, paged); the P0/P1 engines have no scheduler
    and report zeros rather than pretending. Process memory is measured here, once,
    for every engine — a naive engine's RSS is still a number worth charting."""
    stats_fn = getattr(_engine, "stats", None)
    stats = stats_fn() if callable(stats_fn) else {}
    return HealthResponse(
        status="ok",
        engine=ENGINE_NAME,
        model_id=CONFIG.model_id,
        device=CONFIG.device,
        hardware=device_name(CONFIG.device),
        rss_bytes=_rss_bytes(),
        device_mem_bytes=_device_mem_bytes(),
        **stats,
    )
