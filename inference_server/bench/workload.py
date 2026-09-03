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
    #: seconds between arrivals; 0.0 = all-at-once burst. Closed-loop only: the
    #: concurrency semaphore still caps in-flight requests, so this trickles but cannot
    #: overload.
    arrival_rate: float = 0.0
    #: Open-loop arrival process (the overload run). Requests arrive at `arrival_rps`
    #: (Poisson inter-arrivals) for `duration_s` seconds no matter how many are already
    #: in flight — that is what "10x capacity" means, and it is the thing a semaphore-
    #: bounded loop cannot express. `num_requests` becomes a POOL of distinct prompts
    #: the generator cycles through, since 30 min at hundreds of req/s is far more
    #: requests than it is worth tokenizing up front. 0.0 = closed-loop (the default).
    arrival_rps: float = 0.0
    duration_s: float = 0.0
    #: S1. When > 0, every prompt is this many corpus tokens (a fixed slice: the
    #: "system prompt") followed by the usual varied slice. Applied after the RNG draws
    #: so the closed-loop specs' request lists are unchanged.
    shared_prefix_len: int = 0

    @property
    def open_loop(self) -> bool:
        return self.duration_s > 0


#: Registered workloads. `mixed` is the headline one — P1's degradation and P2's
#: >=3x (M2) are both measured on it. `uniform` exists only to show the contrast.
WORKLOADS: dict[str, WorkloadSpec] = {
    "mixed": WorkloadSpec(name="mixed"),
    "uniform": WorkloadSpec(name="uniform", out_len_mu=3.4, out_len_sigma=0.0),
    "smoke": WorkloadSpec(name="smoke", num_requests=8, out_len_clip=(8, 32)),
    # P4: 10x capacity for 30 minutes (M4). `arrival_rps` is left at 0 on purpose:
    # "10x" is 10x whatever *this* hardware sustains, so it is passed as `--rps` from
    # the latest mixed sweep (scripts/headline.py prints the suggestion). The 2000-prompt
    # pool has the same length distribution as `mixed`, so the server is overloaded by
    # the workload it was benchmarked on, not an easier one.
    "overload": WorkloadSpec(name="overload", num_requests=2_000, duration_s=1800.0),
    # Same shape, one minute, short outputs — enough to verify the open-loop path, the
    # /health sampler and the accounting on a laptop before renting the GPU.
    "smoke-overload": WorkloadSpec(
        name="smoke-overload", num_requests=64, out_len_clip=(8, 32), arrival_rps=20.0, duration_s=60.0
    ),
    # S1: a ~128-token system prompt shared by every request, then a 16-128 token tail;
    # outputs as in `mixed`. Run with PREFIX_CACHING=1 vs 0 for the A/B.
    "shared-prefix": WorkloadSpec(name="shared-prefix", prompt_len_range=(16, 128), shared_prefix_len=128),
}


@dataclass(frozen=True)
class WorkItem:
    request_id: str
    prompt: str
    max_tokens: int
    arrive_at_s: float


def build_workload(spec: WorkloadSpec, tokenizer) -> list[WorkItem]:
    """Deterministic given (spec, seed). Prompts are slices of a fixed corpus so that
    prompt length is controlled rather than incidental.

    The RNG draw order (out_lens, prompt_lens, then one start per item) is part of the
    contract: every results file in results/ was produced by it, so it must not change
    for the closed-loop specs. Open-loop specs get the same list — it is the prompt pool
    — and the loadgen owns the arrival times (`arrive_at_s` stays 0 here)."""
    rng = np.random.default_rng(SEED)
    corpus_ids = tokenizer(_CORPUS.read_text(), return_tensors=None).input_ids

    out_lens = np.exp(rng.normal(spec.out_len_mu, spec.out_len_sigma, spec.num_requests))
    out_lens = np.clip(out_lens, *spec.out_len_clip).astype(int)
    prompt_lens = rng.integers(*spec.prompt_len_range, size=spec.num_requests)

    # S1: the shared system prompt is a fixed slice, decoded once. Joined to each tail
    # with a blank line so the tail's first token cannot BPE-merge into the prefix's
    # last one — the prefix must re-tokenize to the same IDs in every request or there
    # is nothing to share. No RNG draw here, so the existing specs are untouched.
    shared = tokenizer.decode(corpus_ids[: spec.shared_prefix_len]) + "\n\n" if spec.shared_prefix_len else ""

    items = []
    for i, (plen, olen) in enumerate(zip(prompt_lens, out_lens)):
        start = int(rng.integers(0, max(1, len(corpus_ids) - plen)))
        items.append(
            WorkItem(
                request_id=f"{spec.name}-{i:05d}",
                prompt=shared + tokenizer.decode(corpus_ids[start : start + plen]),
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
