"""Central configuration. Everything tunable lives here, nothing is hardcoded downstream.

Seeds and determinism land on Day 1 (NFR3) — before there is anything nondeterministic
to catch. Determinism that gets switched on later never gets switched on.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch

SEED = 1337


def seed_everything(seed: int = SEED) -> None:
    """Fix every RNG we touch. Called once at model load."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> str:
    """cuda for the benchmark box, mps for the dev Mac, cpu as the honest fallback."""
    if forced := os.environ.get("DEVICE"):
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_name(device: str) -> str:
    """Human-readable hardware string. Goes into every results file — a benchmark
    number without a GPU name is not reproducible (NFR3)."""
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return "Apple Silicon (MPS)"
    return "CPU"


@dataclass
class Config:
    # --- model ---
    model_id: str = os.environ.get("MODEL_ID", "gpt2")
    device: str = field(default_factory=pick_device)
    dtype: str = os.environ.get("DTYPE", "float32")  # bf16 on the GPU box

    # --- P2 scheduler ---
    max_running: int = 32          # sequences in flight
    max_queue_depth: int = 256     # bound on WAITING; beyond this -> HTTP 503 (FR7)

    # --- P3 paged KV ---
    block_size: int = 16           # tokens per block
    num_blocks: int = 2048         # sized to fill GPU memory after weights

    # --- generation ---
    default_max_tokens: int = 128

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[self.dtype]


CONFIG = Config()
