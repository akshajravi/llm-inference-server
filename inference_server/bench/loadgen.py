"""Load generator (FR8). N requests in flight against an engine.

Drives the engine's async `submit` directly (in-process) rather than over HTTP, so the
measurement isolates scheduling from uvicorn's own queueing. `--http` benchmarks the
full stack when that is the question being asked.
"""

from __future__ import annotations

import asyncio
import time

from inference_server.bench.workload import WorkItem
from inference_server.engine.base import Engine, Request, Result


async def run_load(
    engine: Engine,
    items: list[WorkItem],
    concurrency: int,
) -> tuple[list[Result], list[BaseException], float]:
    """Returns (results, errors, wall_clock_seconds)."""
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


async def run_load_http(
    base_url: str,
    items: list[WorkItem],
    concurrency: int,
) -> tuple[list[Result], list[BaseException], float]:
    """Same load, over the wire. Used for the P4 overload run, where the 503 path
    (FR7) only exists at the HTTP layer."""
    import httpx

    sem = asyncio.Semaphore(concurrency)
    results: list[Result] = []
    errors: list[BaseException] = []
    t0 = time.perf_counter()

    async with httpx.AsyncClient(base_url=base_url, timeout=600.0) as client:
        async def one(item: WorkItem) -> None:
            if item.arrive_at_s:
                await asyncio.sleep(item.arrive_at_s)
            async with sem:
                started = time.perf_counter()
                try:
                    resp = await client.post(
                        "/generate", json={"prompt": item.prompt, "max_tokens": item.max_tokens}
                    )
                except Exception as exc:  # noqa: BLE001 — the harness records, never crashes
                    errors.append(exc)
                    return
                if resp.status_code != 200:
                    errors.append(RuntimeError(f"HTTP {resp.status_code}"))
                    return
                body = resp.json()
                results.append(
                    Result(
                        request_id=item.request_id,
                        token_ids=body["token_ids"],
                        text=body["text"],
                        ttft_s=body["ttft_s"],
                        latency_s=time.perf_counter() - started,
                        finish_reason=body["finish_reason"],
                    )
                )

        await asyncio.gather(*(one(i) for i in items))

    return results, errors, time.perf_counter() - t0
