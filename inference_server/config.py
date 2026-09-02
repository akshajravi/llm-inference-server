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

    # --- P1 static batching ---
    #: How long a forming batch waits for more arrivals before launching. Static
    #: batching cannot admit anyone once the batch is running, so this window is the
    #: only chance a request has to join — too short and every batch is size 1, too
    #: long and it shows up directly in TTFT.
    batch_window_s: float = 0.01

    # --- P2 scheduler ---
    max_running: int = 32          # sequences in flight
    max_queue_depth: int = 256     # bound on WAITING; beyond this -> HTTP 503 (FR7)

    # --- P3 paged KV ---
    block_size: int = 16           # tokens per block
    #: max_running * max_seq_len / block_size = 32 * 1024 / 16. Deliberately the exact
    #: capacity the contiguous engines reserve, so P3 is measured holding the same worst
    #: case P2 does — the win has to come from packing, not from a smaller pool. For gpt2
    #: at fp32 that is 2.4 GiB (72 KiB/token); PagedKVPool.describe() prints the real
    #: figure, and ModelDims.blocks_for_budget() sizes it from a memory budget instead.
    num_blocks: int = 2048

    # --- generation ---
    default_max_tokens: int = 128
    #: Worst-case sequence length a contiguous allocator must reserve per request.
    #: It cannot know the real output length at admission time, so it reserves this
    #: much every time — which is exactly the waste M3 measures P3 against.
    max_seq_len: int = 1024

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[self.dtype]


CONFIG = Config()
