"""Wire types (FR1)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from inference_server.config import CONFIG


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=CONFIG.default_max_tokens, ge=1, le=4096)
    #: Override the stop token, mirroring Request.eos_token_id. Without this field the
    #: HTTP path silently dropped the override and fell back to the tokenizer default,
    #: so the EOS goldens passed in-process and failed over the wire. None = default.
    eos_token_id: int | None = Field(default=None, ge=0)


class GenerateResponse(BaseModel):
    text: str
    token_ids: list[int]
    num_generated: int
    finish_reason: str
    ttft_s: float
    latency_s: float
    # --- M3, copied from Result so the HTTP benchmark path (bench/loadgen.py) reports
    # waste% instead of 0.0. reserved is what the engine held for this sequence, used
    # is prompt + generated; waste = 1 - used/reserved.
    prompt_len: int = 0
    reserved_tokens: int = 0
    used_tokens: int = 0


class HealthResponse(BaseModel):
    status: str
    engine: str
    model_id: str
    device: str
    hardware: str
    # --- P4 (FR7 / M4): what the overload run polls and charts. Zeros on engines that
    # have no scheduler (naive, manual, static); cumulative counters never reset.
    queue_depth: int = 0      # FR7: new arrivals waiting; the 503 bound applies to this
    num_running: int = 0
    num_waiting: int = 0      # queue_depth + recompute-preempted sequences awaiting re-admission
    num_swapped: int = 0      # swap-preempted sequences whose KV is on the host
    free_blocks: int = 0
    num_blocks: int = 0
    preemptions: int = 0      # cumulative
    swaps: int = 0            # cumulative
    completed: int = 0        # cumulative
    rss_bytes: int = 0        # process resident set size (peak, from getrusage)
    device_mem_bytes: int = 0 # torch allocator bytes on cuda/mps; 0 on cpu
