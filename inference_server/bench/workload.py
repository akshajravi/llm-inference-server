"""Workload definition — the experiment, not a garnish.

If every request generates exactly 100 tokens, static batching has no longest sequence
to stall on and continuous batching shows ~1.2x. The output length distribution is
right-skewed (mass of short outputs, long tail) because that is what real traffic looks
like, and it is the reason P2's number is what it is.

Seeded: the same workload name always produces the same request list, on any machine.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from inference_server.config import SEED

_CORPUS = Path(__file__).parent / "prompts.txt"


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    num_requests: int = 200
    prompt_len_range: tuple[int, int] = (16, 256)
    #: lognormal(mu, sigma) over output length, clipped — the right-skewed tail
    out_len_mu: float = 3.4          # exp(3.4) ~= 30 tokens median
    out_len_sigma: float = 0.9
    out_len_clip: tuple[int, int] = (8, 512)
    #: seconds between arrivals; 0.0 = all-at-once burst
    arrival_rate: float = 0.0


#: Registered workloads. `mixed` is the headline one — P1's degradation and P2's
#: >=3x (M2) are both measured on it. `uniform` exists only to show the contrast.
WORKLOADS: dict[str, WorkloadSpec] = {
    "mixed": WorkloadSpec(name="mixed"),
    "uniform": WorkloadSpec(name="uniform", out_len_mu=3.4, out_len_sigma=0.0),
    "smoke": WorkloadSpec(name="smoke", num_requests=8, out_len_clip=(8, 32)),
    # P4: 10x capacity for 30 minutes (M4)
    "overload": WorkloadSpec(name="overload", num_requests=20_000, arrival_rate=0.005),
}


@dataclass(frozen=True)
class WorkItem:
    request_id: str
    prompt: str
    max_tokens: int
    arrive_at_s: float


def build_workload(spec: WorkloadSpec, tokenizer) -> list[WorkItem]:
    """Deterministic given (spec, seed). Prompts are slices of a fixed corpus so that
    prompt length is controlled rather than incidental."""
    rng = np.random.default_rng(SEED)
    corpus_ids = tokenizer(_CORPUS.read_text(), return_tensors=None).input_ids

    out_lens = np.exp(rng.normal(spec.out_len_mu, spec.out_len_sigma, spec.num_requests))
    out_lens = np.clip(out_lens, *spec.out_len_clip).astype(int)
    prompt_lens = rng.integers(*spec.prompt_len_range, size=spec.num_requests)

    items = []
    for i, (plen, olen) in enumerate(zip(prompt_lens, out_lens)):
        start = int(rng.integers(0, max(1, len(corpus_ids) - plen)))
        items.append(
            WorkItem(
                request_id=f"{spec.name}-{i:05d}",
                prompt=tokenizer.decode(corpus_ids[start : start + plen]),
                max_tokens=int(olen),
                arrive_at_s=i * spec.arrival_rate,
            )
        )
    return items


def describe(spec: WorkloadSpec) -> dict:
    """Goes verbatim into every results file — a number without its workload is noise."""
    return {**asdict(spec), "seed": SEED}


if __name__ == "__main__":
    print(json.dumps({k: describe(v) for k, v in WORKLOADS.items()}, indent=2))
