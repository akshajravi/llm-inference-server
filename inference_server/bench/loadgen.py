"""Load generator (FR8). N requests in flight against an engine.

Drives the engine's async `submit` directly (in-process) rather than over HTTP, so the
measurement isolates scheduling from uvicorn's own queueing. `--http` benchmarks the
full stack when that is the question being asked.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np

from inference_server.bench.workload import WorkItem
from inference_server.config import SEED
from inference_server.engine.base import Engine, Request, Result


def warmup(engine: Engine, rounds: int = 2) -> None:
    """Burn a few short requests before the clock starts.

    The first forward pass on a device pays one-time costs — kernel autotuning/compilation,
    allocator initialisation, weights migrating to the device. Timed, those land entirely
    on the first request and deflate the baseline, which inflates every speedup measured
    against it. Cheap to avoid, invisible if you don't.
    """
    for i in range(rounds):
        engine.generate(Request(request_id=f"warmup-{i}", prompt="warm up the device", max_tokens=8))


async def run_load(
    engine: Engine,
    items: list[WorkItem],
    concurrency: int,
) -> tuple[list[Result], list[BaseException], float]:
    """Returns (results, errors, wall_clock_seconds)."""
    warmup(engine)
    sem = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()

    async def one(item: WorkItem) -> Result:
        if item.arrive_at_s:                       # Poisson-ish trickle vs. all-at-once
            await asyncio.sleep(item.arrive_at_s)
        async with sem:
            return await engine.submit(
                Request(request_id=item.request_id, prompt=item.prompt, max_tokens=item.max_tokens)
            )

    settled = await asyncio.gather(*(one(i) for i in items), return_exceptions=True)
    duration = time.perf_counter() - t0

    results = [r for r in settled if isinstance(r, Result)]
    errors = [r for r in settled if isinstance(r, BaseException)]
    return results, errors, duration


# --------------------------------------------------------------------------------------
# Over the wire. The 503 path (FR7) and the /health counters only exist at the HTTP
# layer, so the overload run (M4) has to go through uvicorn. The in-process path above
# stays the one the throughput comparison uses.
# --------------------------------------------------------------------------------------

#: /health keys copied into every sample, when present. Older engines lack most of
#: them; a missing key is simply absent from that sample rather than a crash.
HEALTH_FIELDS = (
    "queue_depth", "num_running", "num_waiting", "num_swapped",
    "free_blocks", "num_blocks", "preemptions", "swaps", "completed",
    "rss_bytes", "device_mem_bytes",
)


@dataclass
class HttpLoadResult:
    results: list[Result]
    errors: list[BaseException]
    #: HTTP 503s. Counted apart from errors because a 503 is the server doing exactly
    #: what FR7 asks — an honest, immediate refusal — whereas an error is a failure.
    shed: int
    #: Requests handed to the client. M4's "no dropped-but-unacknowledged request" is
    #: the equation submitted == completed + shed + errors; run.py asserts it.
    submitted: int
    #: Wall clock from first arrival to last response (arrivals + drain).
    duration_s: float
    #: Seconds during which arrivals were generated (open-loop only; == duration_s
    #: for closed-loop, where there is no separate drain phase worth naming).
    arrival_window_s: float
    #: [{t_s, ...HEALTH_FIELDS}] polled from /health. "Flat memory" is this, charted.
    samples: list[dict] = field(default_factory=list)


async def _post_one(client, item: WorkItem, out: HttpLoadResult) -> None:
    """One request; every outcome lands in exactly one of results/errors/shed."""
    started = time.perf_counter()
    try:
        resp = await client.post(
            "/generate", json={"prompt": item.prompt, "max_tokens": item.max_tokens}
        )
    except Exception as exc:  # noqa: BLE001 — the harness records, never crashes
        out.errors.append(exc)
        return
    if resp.status_code == 503:
        out.shed += 1
        return
    if resp.status_code != 200:
        out.errors.append(RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}"))
        return
    try:
        body = resp.json()
        out.results.append(
            Result(
                request_id=item.request_id,
                token_ids=body["token_ids"],
                text=body["text"],
                ttft_s=body["ttft_s"],
                latency_s=time.perf_counter() - started,   # end-to-end, client-side
                finish_reason=body["finish_reason"],
                # M3 fields (absent from older servers -> 0, and waste% reads 0.0).
                prompt_len=body.get("prompt_len", 0),
                reserved_tokens=body.get("reserved_tokens", 0),
                used_tokens=body.get("used_tokens", 0),
            )
        )
    except Exception as exc:  # noqa: BLE001 — a 200 with a malformed body is still an error
        out.errors.append(exc)


async def sample_health(client, every_s: float, t0: float, out: list[dict], stop: asyncio.Event) -> None:
    """Poll /health on a timer for the life of the run.

    Runs as its own task so a stalled request cannot stop the memory chart, and takes
    one final sample after `stop` so the drain tail is on the chart too. A failed poll
    is recorded as an error sample rather than raised — during overload, /health not
    answering IS a finding."""
    while True:
        t = time.perf_counter() - t0
        sample: dict = {"t_s": t}
        try:
            resp = await client.get("/health", timeout=10.0)
            body = resp.json()
            sample["status_code"] = resp.status_code
            for k in HEALTH_FIELDS:
                if k in body:
                    sample[k] = body[k]
        except Exception as exc:  # noqa: BLE001
            sample["error"] = repr(exc)
        out.append(sample)
        if stop.is_set():
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=every_s)
        except asyncio.TimeoutError:
            pass


async def run_load_http(
    base_url: str,
    items: list[WorkItem],
    concurrency: int,
    *,
    sample_every_s: float | None = None,
) -> HttpLoadResult:
    """The closed loop from `run_load`, over the wire. Same semaphore, same items, so a
    number measured here differs from the in-process one only by uvicorn."""
    import httpx

    out = HttpLoadResult(results=[], errors=[], shed=0, submitted=len(items),
                         duration_s=0.0, arrival_window_s=0.0)
    sem = asyncio.Semaphore(concurrency)
    stop = asyncio.Event()
    t0 = time.perf_counter()

    async with httpx.AsyncClient(base_url=base_url, timeout=600.0) as client:
        sampler = (
            asyncio.create_task(sample_health(client, sample_every_s, t0, out.samples, stop))
            if sample_every_s else None
        )

        async def one(item: WorkItem) -> None:
            if item.arrive_at_s:
                await asyncio.sleep(item.arrive_at_s)
            async with sem:
                await _post_one(client, item, out)

        await asyncio.gather(*(one(i) for i in items))
        stop.set()
        if sampler:
            await sampler

    out.duration_s = out.arrival_window_s = time.perf_counter() - t0
    return out


async def run_open_loop_http(
    base_url: str,
    pool: list[WorkItem],
    rps: float,
    duration_s: float,
    *,
    sample_every_s: float | None = 5.0,
    seed: int = SEED,
    max_in_flight: int = 10_000,
) -> HttpLoadResult:
    """The overload run (M4). Arrivals are a Poisson process at `rps` for `duration_s`
    seconds, generated whether or not the server has answered anything yet.

    A closed loop with N workers can never present more than N requests, so the server
    never sees more than it can hold and admission control is never exercised. Here the
    offered load is the independent variable: set `rps` to 10x what the engine cleared
    in the mixed sweep and the queue must fill, and the 503s must start.

    Inter-arrival gaps are drawn up front from a seeded RNG so two runs offer the same
    arrival sequence (NFR3). Prompts cycle through `pool`; request ids carry the cycle
    number so no two in-flight requests share an id. `max_in_flight` is a client-side
    guard against exhausting file descriptors if the server stops answering entirely —
    hitting it is recorded as errors, not silently skipped, so the accounting holds.

    After the arrival window closes the generator stops and waits for everything in
    flight to settle (bounded by the client timeout), so `duration_s` in the result
    includes the drain and `arrival_window_s` is the window alone."""
    import httpx

    out = HttpLoadResult(results=[], errors=[], shed=0, submitted=0,
                         duration_s=0.0, arrival_window_s=0.0)
    rng = np.random.default_rng(seed)
    stop = asyncio.Event()
    tasks: set[asyncio.Task] = set()
    t0 = time.perf_counter()

    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=600.0,
        limits=httpx.Limits(max_connections=max_in_flight, max_keepalive_connections=256),
    ) as client:
        sampler = (
            asyncio.create_task(sample_health(client, sample_every_s, t0, out.samples, stop))
            if sample_every_s else None
        )

        n = 0
        next_at = 0.0
        while True:
            next_at += float(rng.exponential(1.0 / rps))
            if next_at >= duration_s:
                break
            delay = next_at - (time.perf_counter() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            base = pool[n % len(pool)]
            item = WorkItem(
                request_id=f"{base.request_id}#{n // len(pool)}",
                prompt=base.prompt, max_tokens=base.max_tokens, arrive_at_s=next_at,
            )
            n += 1
            out.submitted += 1
            if len(tasks) >= max_in_flight:
                out.errors.append(RuntimeError(f"client max_in_flight={max_in_flight} reached"))
                continue
            task = asyncio.create_task(_post_one(client, item, out))
            tasks.add(task)
            task.add_done_callback(tasks.discard)

        out.arrival_window_s = time.perf_counter() - t0
        if tasks:
            await asyncio.gather(*tasks)
        stop.set()
        if sampler:
            await sampler

    out.duration_s = time.perf_counter() - t0
    return out
