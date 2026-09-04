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

#: The published pool size (see Config.num_blocks). A module constant so a test can check
#: the derivation even when NUM_BLOCKS has shrunk the live config for a local run.
DEFAULT_NUM_BLOCKS = 2048


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
    #: NUM_BLOCKS overrides it for local test runs only: on the dev Mac the default pool
    #: is 2.25 GiB per engine and the goldens need a few hundred tokens, so
    #: `NUM_BLOCKS=256 make test` runs the whole suite in ~290 MiB. Published benchmarks
    #: never set it — the waste comparison is only fair against the full reservation.
    num_blocks: int = int(os.environ.get("NUM_BLOCKS", DEFAULT_NUM_BLOCKS))
    #: Which paged attention kernel core/attention.py dispatches to. "gather" is the
    #: PyTorch path that ships; "triton" is S3 stretch and raises until Day 14.
    attention: str = os.environ.get("ATTENTION", "gather")
    #: S1. Share full KV blocks between sequences whose token histories start the same
    #: way (core/prefix_cache.py). On by default because it is free when nothing
    #: matches; PREFIX_CACHING=0 is the A/B control for the shared-prefix workload.
    prefix_caching: bool = os.environ.get("PREFIX_CACHING", "1") == "1"

    # --- P4 preemption + admission ---
    #: What happens to a victim's KV when the pool runs dry (FR6, S2). "recompute" frees
    #: it and re-prefills prompt + generated tokens on re-admission; "swap" copies the
    #: blocks to host memory and back. Recompute is the default because it is the only
    #: strategy in the critical path and it wins for short sequences; swap exists so the
    #: crossover (recompute cost vs. the PCIe round trip) is measured, not asserted.
    #: `max_queue_depth` (the FR7 bound) lives in the P2 section where it was declared.
    preemption: str = os.environ.get("PREEMPTION", "recompute")

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
