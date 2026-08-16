"""Benchmark entry point — the M5 target.

    python -m inference_server.bench.run --engine naive --workload mixed --concurrency 1,4,16

Sweeps every (engine, concurrency) pair, prints a table, writes a dated JSON to
results/. One command on the rented GPU (Day 12) covers every phase on identical
hardware — that is what makes the P0-vs-P3 comparison defensible.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from inference_server.bench.loadgen import run_load
from inference_server.bench.metrics import Summary, format_table, hardware_info, summarize
from inference_server.bench.workload import WORKLOADS, build_workload, describe
from inference_server.engine import IMPLEMENTED, build
from inference_server.model import load

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="inference-server benchmark harness")
    p.add_argument("--engine", default=",".join(IMPLEMENTED),
                   help="comma-separated engine names, or 'all' for every implemented one")
    p.add_argument("--workload", default="mixed", choices=sorted(WORKLOADS))
    p.add_argument("--concurrency", default="1,4,16", help="comma-separated levels to sweep")
    p.add_argument("--tag", default="", help="suffix for the results filename")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    engines = IMPLEMENTED if args.engine == "all" else args.engine.split(",")
    levels = [int(c) for c in args.concurrency.split(",")]
    spec = WORKLOADS[args.workload]

    _, tokenizer = load()
    items = build_workload(spec, tokenizer)

    summaries: list[Summary] = []
    for engine_name in engines:
        engine = build(engine_name)
        try:
            for concurrency in levels:
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

    print()
    print(format_table(summaries))

    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        stamp = time.strftime("%Y-%m-%d-%H%M%S")
        suffix = f"-{args.tag}" if args.tag else ""
        path = RESULTS_DIR / f"{stamp}-{spec.name}{suffix}.json"
        path.write_text(
            json.dumps(
                {
                    "hardware": hardware_info(),
                    "workload": describe(spec),
                    "runs": [s.to_dict() for s in summaries],
                },
                indent=2,
            )
        )
        print(f"\nwrote {path}")


if __name__ == "__main__":
    asyncio.run(main())
