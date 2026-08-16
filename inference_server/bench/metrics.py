"""Metrics aggregation.

Percentiles, not averages. 990 requests at 100ms plus 10 at 5s averages to a
healthy-looking 150ms while 1 in 100 users leaves. Sort the latencies.
"""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass, field

import numpy as np

from inference_server.config import CONFIG, device_name
from inference_server.engine.base import Result


@dataclass
class Summary:
    engine: str
    workload: str
    concurrency: int
    num_requests: int
    duration_s: float

    throughput_tok_s: float
    throughput_req_s: float

    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    latency_p50: float
    latency_p95: float
    latency_p99: float

    #: M3 — waste = 1 - used/reserved. Zero until P3 instruments it.
    kv_waste_pct: float = 0.0

    num_errors: int = 0
    num_shed_503: int = 0     # FR7, from P4

    hardware: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _pct(values, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else 0.0


def summarize(
    results: list[Result],
    *,
    engine: str,
    workload: str,
    concurrency: int,
    duration_s: float,
    num_errors: int = 0,
    num_shed_503: int = 0,
) -> Summary:
    ttfts = [r.ttft_s for r in results]
    lats = [r.latency_s for r in results]
    total_tokens = sum(r.num_generated for r in results)

    reserved = sum(r.reserved_tokens for r in results)
    used = sum(r.used_tokens for r in results)

    return Summary(
        engine=engine,
        workload=workload,
        concurrency=concurrency,
        num_requests=len(results),
        duration_s=duration_s,
        throughput_tok_s=total_tokens / duration_s if duration_s else 0.0,
        throughput_req_s=len(results) / duration_s if duration_s else 0.0,
        ttft_p50=_pct(ttfts, 50),
        ttft_p95=_pct(ttfts, 95),
        ttft_p99=_pct(ttfts, 99),
        latency_p50=_pct(lats, 50),
        latency_p95=_pct(lats, 95),
        latency_p99=_pct(lats, 99),
        kv_waste_pct=100.0 * (1 - used / reserved) if reserved else 0.0,
        num_errors=num_errors,
        num_shed_503=num_shed_503,
        hardware=hardware_info(),
    )


def hardware_info() -> dict:
    """NFR3. A number without a GPU name is not reproducible."""
    return {
        "device": CONFIG.device,
        "name": device_name(CONFIG.device),
        "model_id": CONFIG.model_id,
        "dtype": CONFIG.dtype,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def format_table(summaries: list[Summary]) -> str:
    """What `make bench` prints. Kept boring on purpose."""
    head = f"{'engine':<12}{'workload':<10}{'conc':>5}{'tok/s':>10}{'req/s':>8}{'ttft p50':>10}{'p99':>9}{'lat p50':>9}{'p99':>9}{'waste%':>8}"
    rows = [head, "-" * len(head)]
    for s in summaries:
        rows.append(
            f"{s.engine:<12}{s.workload:<10}{s.concurrency:>5}{s.throughput_tok_s:>10.1f}"
            f"{s.throughput_req_s:>8.2f}{s.ttft_p50:>10.3f}{s.ttft_p99:>9.3f}"
            f"{s.latency_p50:>9.3f}{s.latency_p99:>9.3f}{s.kv_waste_pct:>8.1f}"
        )
    return "\n".join(rows)
