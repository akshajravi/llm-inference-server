"""Wire types (FR1)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from inference_server.config import CONFIG


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=CONFIG.default_max_tokens, ge=1, le=4096)


class GenerateResponse(BaseModel):
    text: str
    token_ids: list[int]
    num_generated: int
    finish_reason: str
    ttft_s: float
    latency_s: float


class HealthResponse(BaseModel):
    status: str
    engine: str
    model_id: str
    device: str
    hardware: str
    queue_depth: int = 0      # FR7: observable from Day 1, real from P4
