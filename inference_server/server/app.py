"""FastAPI app (FR1). Engine selected by the ENGINE env var so the same server binary
benchmarks every phase.

    ENGINE=naive python -m uvicorn inference_server.server.app:app

Day 1 ships /generate and /health. P4 adds the 503 path (FR7) and /generate/stream (S4).
"""

from __future__ import annotations

import contextlib
import os
import uuid

from fastapi import FastAPI

from inference_server.config import CONFIG, device_name
from inference_server.engine import Request, build
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


@app.post("/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest) -> GenerateResponse:
    result = await _engine.submit(
        Request(request_id=str(uuid.uuid4()), prompt=body.prompt, max_tokens=body.max_tokens)
    )
    return GenerateResponse(
        text=result.text,
        token_ids=result.token_ids,
        num_generated=result.num_generated,
        finish_reason=result.finish_reason,
        ttft_s=result.ttft_s,
        latency_s=result.latency_s,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        engine=ENGINE_NAME,
        model_id=CONFIG.model_id,
        device=CONFIG.device,
        hardware=device_name(CONFIG.device),
    )
