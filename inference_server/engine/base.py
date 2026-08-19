"""The Engine seam — P0 (Day 1).

Every phase's implementation satisfies this one interface, which is what makes the
P0-vs-P3 comparison apples-to-apples rather than three subtly different harnesses.

Note this is richer than a bare `generate(prompt, max_tokens) -> str`:
  - `token_ids` because M1 compares token IDs, never text (no tokenizer round-trip fuzz)
  - `ttft_s` because the harness needs it, and measuring it inside the engine means
    it denotes the same event for all four implementations
Both are painful to retrofit once four engines exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Request:
    request_id: str
    prompt: str
    max_tokens: int
    #: Override the stop token. Exists because GPT-2 greedy decoding essentially never
    #: emits <|endoftext|>, so without it nothing would exercise the EOS termination
    #: path — and every later phase reimplements that path by hand. None = tokenizer default.
    eos_token_id: int | None = None


@dataclass
class Result:
    request_id: str
    token_ids: list[int]      # generated only, excludes the prompt — this is what M1 asserts on
    text: str
    ttft_s: float             # to first *sampled token*, not first byte on the wire
    latency_s: float
    finish_reason: str        # "eos" | "length"
    prompt_len: int = 0
    # P3 fills these in; P0-P2 leave them at zero. Waste = 1 - used/reserved (M3).
    reserved_tokens: int = 0
    used_tokens: int = 0

    @property
    def num_generated(self) -> int:
        return len(self.token_ids)


@runtime_checkable
class Engine(Protocol):
    """One forward-facing contract for every phase.

    `generate` is synchronous and is what the correctness suite calls.
    `submit` is what the HTTP layer calls; P2/P3 implement it natively over a shared
    step loop, while P0/P1 serialize it behind a lock (honestly — that IS the baseline).
    """

    name: str

    def generate(self, req: Request) -> Result: ...

    async def submit(self, req: Request) -> Result: ...

    def shutdown(self) -> None: ...


class NotBuiltYet(NotImplementedError):
    """Raised by phase stubs so a premature `--engine paged` fails loudly and legibly."""

    def __init__(self, engine: str, phase: str, days: str) -> None:
        super().__init__(
            f"{engine} is not implemented yet — it lands in {phase} ({days}). "
            f"See IMPLEMENTATION_GUIDE.md."
        )
