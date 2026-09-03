"""Benchmark entry point — the M5 target.

    python -m inference_server.bench.run --engine naive --workload mixed --concurrency 1,4,16

Sweeps every (engine, concurrency) pair, prints a table, writes a dated JSON to
results/. One command on the rented GPU (Day 12) covers every phase on identical
hardware — that is what makes the P0-vs-P3 comparison defensible.

Two more modes, both over HTTP against a server you started yourself (`make serve`):

    --http http://localhost:8000                          closed-loop sweep, over the wire
    --http http://localhost:8000 --workload overload --rps 300 --duration 1800

The second is the overload run (M4): an open-loop arrival process the server cannot
throttle, a /health sampler for the memory chart, and the accounting
`submitted == completed + shed + errors` asserted and written into the file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from inference_server.bench.loadgen import HttpLoadResult, run_load, run_load_http, run_open_loop_http
from inference_server.bench.metrics import Summary, format_table, hardware_info, summarize
from inference_server.bench.workload import WORKLOADS, WorkloadSpec, build_workload, describe
from inference_server.config import CONFIG, SEED
from inference_server.engine import IMPLEMENTED, build
from inference_server.model import load

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="inference-server benchmark harness")
    p.add_argument("--engine", default=",".join(IMPLEMENTED),
                   help="comma-separated engine names, or 'all' for every implemented one "
                        "(ignored with --http: the server picks the engine)")
    p.add_argument("--workload", default="mixed", choices=sorted(WORKLOADS))
    p.add_argument("--concurrency", default="1,4,16", help="comma-separated levels to sweep")
    p.add_argument("--tag", default="", help="suffix for the results filename")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--out-dir", default=str(RESULTS_DIR),
                   help="where the results JSON goes (results/scratch is gitignored)")
    p.add_argument("--http", metavar="BASE_URL", default=None,
                   help="benchmark a running server over HTTP instead of in-process")
    p.add_argument("--rps", type=float, default=None,
                   help="open-loop arrival rate (requests/s). Overrides the workload's "
                        "arrival_rps; the overload spec requires it (--http only)")
    p.add_argument("--duration", type=float, default=None,
                   help="open-loop arrival window in seconds. Overrides the workload's "
                        "duration_s (--http only)")
    p.add_argument("--sample-every", type=float, default=5.0,
                   help="/health sampling period in seconds for the HTTP modes; 0 disables")
    return p.parse_args()


def bench_config() -> dict:
    """The knobs a waste number depends on. Written into every results file so
    scripts/headline.py can turn waste% into 'how many more sequences fit' without
    reading config.py at a later commit where the values may have moved."""
    return {
        "seed": SEED,
        "max_running": CONFIG.max_running,
        "max_queue_depth": CONFIG.max_queue_depth,
        "max_seq_len": CONFIG.max_seq_len,
        "block_size": CONFIG.block_size,
        "num_blocks": CONFIG.num_blocks,
        "num_slots": CONFIG.num_blocks * CONFIG.block_size,
    }


def _http_summary(out: HttpLoadResult, *, engine: str, workload: str, concurrency: int) -> Summary:
    return summarize(
        out.results,
        engine=engine,
        workload=workload,
        concurrency=concurrency,
        duration_s=out.duration_s,
        num_errors=len(out.errors),
        num_shed_503=out.shed,
        num_submitted=out.submitted,
    )


def _print_errors(errors: list[BaseException], limit: int = 5) -> None:
    if not errors:
        return
    kinds: dict[str, int] = {}
    for e in errors:
        kinds[type(e).__name__] = kinds.get(type(e).__name__, 0) + 1
    print(f"  errors: {len(errors)} ({', '.join(f'{k}x{v}' for k, v in kinds.items())})")
    for e in errors[:limit]:
        print(f"    {e!r}")


async def server_engine(base_url: str) -> dict:
    import httpx

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        resp = await client.get("/health")
        resp.raise_for_status()
        return resp.json()


async def run_in_process(engines: list[str], levels: list[int], spec: WorkloadSpec, items) -> list[Summary]:
    summaries: list[Summary] = []
    for engine_name in engines:
        for concurrency in levels:
            # A fresh engine per measurement. Reusing one across concurrency levels lets
            # state accumulate — cache, threads, allocator fragmentation — so a later
            # level is measured on an engine the earlier ones already used. Observed as a
            # 40% throughput spread on identical work; the model itself is cached by
            # load(), so rebuilding costs little.
            engine = build(engine_name)
            try:
                print(f"running {engine_name} @ concurrency={concurrency} ...", flush=True)
                results, errors, duration = await run_load(engine, items, concurrency)
                summaries.append(
                    summarize(
                        results,
                        engine=engine_name,
                        workload=spec.name,
                        concurrency=concurrency,
                        duration_s=duration,
                        num_errors=len(errors),
                    )
                )
            finally:
                engine.shutdown()
    return summaries


async def run_http_closed(base_url: str, engine: str, levels: list[int], spec: WorkloadSpec, items,
                          sample_every: float) -> tuple[list[Summary], list[dict]]:
    """Over the wire, same closed loop as in-process. No fresh engine per level here —
    the server is whoever is listening — so a sweep over HTTP is one engine's story."""
    summaries: list[Summary] = []
    samples: list[dict] = []
    for concurrency in levels:
        print(f"running {engine} @ concurrency={concurrency} over {base_url} ...", flush=True)
        out = await run_load_http(base_url, items, concurrency, sample_every_s=sample_every or None)
        _print_errors(out.errors)
        summaries.append(_http_summary(out, engine=engine, workload=spec.name, concurrency=concurrency))
        samples.extend({"concurrency": concurrency, **s} for s in out.samples)
    return summaries, samples


async def run_http_open(base_url: str, engine: str, spec: WorkloadSpec, pool, rps: float,
                        duration: float, sample_every: float) -> tuple[Summary, HttpLoadResult]:
    print(f"open-loop: {rps:g} req/s for {duration:g}s against {base_url} ({engine}), "
          f"pool of {len(pool)} prompts, sampling /health every {sample_every:g}s ...", flush=True)
    out = await run_open_loop_http(base_url, pool, rps, duration, sample_every_s=sample_every or None)
    _print_errors(out.errors)
    # concurrency is not a knob in the open loop; the offered rate is. Recorded as the
    # rate so the table column still says something true.
    summary = _http_summary(out, engine=engine, workload=spec.name, concurrency=int(round(rps)))
    return summary, out


def accounting(out: HttpLoadResult) -> dict:
    """M4's 'no dropped-but-unacknowledged request', as arithmetic. Every submitted
    request came back as a completion, a 503 or a client-visible error; if the equation
    fails, something was swallowed and the run is not a pass."""
    completed = len(out.results)
    balanced = out.submitted == completed + out.shed + len(out.errors)
    return {
        "submitted": out.submitted,
        "completed": completed,
        "shed_503": out.shed,
        "errors": len(out.errors),
        "balanced": balanced,
        "equation": f"{out.submitted} == {completed} + {out.shed} + {len(out.errors)}",
    }


async def main() -> None:
    args = parse_args()
    levels = [int(c) for c in args.concurrency.split(",")]
    spec = WORKLOADS[args.workload]

    _, tokenizer = load()
    items = build_workload(spec, tokenizer)

    payload: dict = {
        "hardware": hardware_info(),
        "workload": describe(spec),
        "config": bench_config(),
    }
    summaries: list[Summary]

    rps = args.rps if args.rps is not None else spec.arrival_rps
    duration = args.duration if args.duration is not None else spec.duration_s
    open_loop = spec.open_loop or args.rps is not None or args.duration is not None

    if args.http is None:
        if open_loop:
            sys.exit("open-loop workloads need --http: the 503 path only exists at the HTTP layer")
        engines = IMPLEMENTED if args.engine == "all" else args.engine.split(",")
        summaries = await run_in_process(engines, levels, spec, items)
    else:
        health = await server_engine(args.http)
        engine = health.get("engine", "unknown")
        payload["server"] = {"base_url": args.http, **health}
        if open_loop:
            if not rps or not duration:
                sys.exit("open-loop mode needs --rps and --duration (the overload spec sets "
                         "no default rate: pick 10x the req/s from the latest mixed sweep, "
                         "see `make headline`)")
            summary, out = await run_http_open(args.http, engine, spec, items, rps, duration, args.sample_every)
            summaries = [summary]
            acct = accounting(out)
            payload["mode"] = "open-loop"
            payload["open_loop"] = {"rps": rps, "duration_s": duration, "arrival_window_s": out.arrival_window_s,
                                    "drain_s": out.duration_s - out.arrival_window_s,
                                    "sample_every_s": args.sample_every, "pool_size": len(items)}
            payload["accounting"] = acct
            payload["samples"] = out.samples
            print(f"\naccounting: {acct['equation']} -> {'balanced' if acct['balanced'] else 'NOT BALANCED'}")
        else:
            payload["mode"] = "closed-loop-http"
            summaries, samples = await run_http_closed(args.http, engine, levels, spec, items, args.sample_every)
            payload["samples"] = samples

    payload["runs"] = [s.to_dict() for s in summaries]

    print()
    print(format_table(summaries))

    if not args.no_save:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d-%H%M%S")
        suffix = f"-{args.tag}" if args.tag else ""
        path = out_dir / f"{stamp}-{spec.name}{suffix}.json"
        path.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {path}")

    if payload.get("accounting") and not payload["accounting"]["balanced"]:
        sys.exit("M4 accounting failed: a request went unacknowledged")


if __name__ == "__main__":
    asyncio.run(main())
