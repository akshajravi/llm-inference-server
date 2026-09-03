"""Charts for the writeup, from results/*.json.

    python -m scripts.plot --latest 3            # newest N results files
    python -m scripts.plot results/2026-09-01-204104-mixed-p99sweep-ac.json

Four figures, each a PNG in docs/charts/:
  throughput.png   tok/s vs concurrency, one line per engine       (M2)
  latency.png      p50/p95/p99 end-to-end latency vs concurrency   (FR8)
  kv_waste.png     KV waste % per engine                           (M3)
  overload.png     free_blocks / queue_depth / num_running / memory vs time (M4)

Deterministic on purpose: fixed figure sizes, fixed engine->color map, no timestamp
baked into the image, headless backend. Two runs of `make charts` on the same results
files produce byte-identical PNGs, which is what lets them live in the repo.

Merging: a sweep may be split across files (one engine per run, or reps). Runs are
keyed by (engine, concurrency); when the same key appears in several files the LAST
file given wins — with --latest that is the newest one. Nothing is averaged here;
if reps should be averaged that is a deliberate act for a different script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = ROOT / "docs" / "charts"

#: Fixed order = phase order, so P0 is always the first color and P3 always the fourth,
#: whichever engines happen to be in a file. Never cycled.
ENGINE_ORDER = ["naive", "manual", "static", "continuous", "paged"]
ENGINE_COLOR = {
    "naive": "#2a78d6",
    "manual": "#eb6834",
    "static": "#1baf7a",
    "continuous": "#eda100",
    "paged": "#e87ba4",
}
#: Marker shape carries identity too, so the lines survive greyscale printing.
ENGINE_MARKER = {"naive": "o", "manual": "s", "static": "^", "continuous": "D", "paged": "v"}
FALLBACK_COLOR = "#7a7a72"
TEXT = "#333330"
GRID = "#e4e4df"

FIGSIZE = (7.0, 4.2)
DPI = 150

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.dpi": DPI,
    # Metadata would otherwise embed the matplotlib version, and Software/creation
    # time are what make two renders of the same data differ byte-for-byte.
    "savefig.pad_inches": 0.05,
})
_SAVE_META = {"Software": None}


def _engine_style(engine: str) -> dict:
    return {
        "color": ENGINE_COLOR.get(engine, FALLBACK_COLOR),
        "marker": ENGINE_MARKER.get(engine, "x"),
        "markersize": 5,
        "linewidth": 1.8,
    }


def _sorted_engines(engines) -> list[str]:
    known = [e for e in ENGINE_ORDER if e in engines]
    return known + sorted(e for e in engines if e not in ENGINE_ORDER)


def load_files(paths: list[Path]) -> list[dict]:
    docs = []
    for p in paths:
        try:
            doc = json.loads(p.read_text())
        except (OSError, ValueError) as exc:
            print(f"skip {p}: {exc}")
            continue
        doc["_path"] = str(p)
        docs.append(doc)
    return docs


def latest_files(n: int, directory: Path = RESULTS_DIR) -> list[Path]:
    """Filenames start with a timestamp, so lexical order is chronological."""
    return sorted(directory.glob("*.json"))[-n:]


def sweep_runs(docs: list[dict]) -> dict[str, dict[int, dict]]:
    """{engine: {concurrency: run}} from every closed-loop document. Open-loop files
    are timelines, not sweeps; their single 'run' would put a req/s value on the
    concurrency axis, so they are left out here."""
    out: dict[str, dict[int, dict]] = {}
    for doc in docs:
        if doc.get("mode") == "open-loop":
            continue
        for run in doc.get("runs", []):
            engine = run.get("engine")
            conc = run.get("concurrency")
            if engine is None or conc is None:
                continue
            out.setdefault(engine, {})[int(conc)] = run
    return out


def _workload_label(docs: list[dict]) -> str:
    names = sorted({d.get("workload", {}).get("name", "?") for d in docs if d.get("mode") != "open-loop"})
    return ", ".join(names)


def _hardware_label(docs: list[dict]) -> str:
    names = sorted({d.get("hardware", {}).get("name", "") for d in docs} - {""})
    return " / ".join(names)


def _save(fig, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, metadata=_SAVE_META)
    plt.close(fig)
    return path


def plot_throughput(sweeps: dict[str, dict[int, dict]], docs: list[dict], out_dir: Path) -> Path | None:
    if not sweeps:
        return None
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for engine in _sorted_engines(sweeps):
        pts = sorted(sweeps[engine].items())
        xs = [c for c, _ in pts]
        ys = [r.get("throughput_tok_s", 0.0) for _, r in pts]
        ax.plot(xs, ys, label=engine, **_engine_style(engine))
        ax.annotate(f"{ys[-1]:.0f}", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(5, 0), fontsize=8, color=TEXT, va="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({c for s in sweeps.values() for c in s}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("concurrency (requests in flight)")
    ax.set_ylabel("throughput (generated tok/s)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Throughput vs concurrency — {_workload_label(docs)} workload, {_hardware_label(docs)}",
                 fontsize=9, color=TEXT, loc="left")
    ax.legend(title=None)
    return _save(fig, "throughput.png", out_dir)


def plot_latency(sweeps: dict[str, dict[int, dict]], docs: list[dict], out_dir: Path) -> Path | None:
    """One panel per percentile rather than one per engine: the question the chart
    answers is 'what happens to the tail as load rises', and the tail is a column."""
    if not sweeps:
        return None
    pcts = [("latency_p50", "p50"), ("latency_p95", "p95"), ("latency_p99", "p99")]
    fig, axes = plt.subplots(1, 3, figsize=(FIGSIZE[0] * 1.6, FIGSIZE[1]), sharey=True)
    levels = sorted({c for s in sweeps.values() for c in s})
    for ax, (key, label) in zip(axes, pcts):
        for engine in _sorted_engines(sweeps):
            pts = sorted(sweeps[engine].items())
            ax.plot([c for c, _ in pts], [r.get(key, 0.0) for _, r in pts],
                    label=engine, **_engine_style(engine))
        ax.set_xscale("log", base=2)
        ax.set_xticks(levels)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_yscale("log")
        ax.set_title(label, fontsize=9, color=TEXT)
        ax.set_xlabel("concurrency")
    axes[0].set_ylabel("end-to-end latency (s, log)")
    axes[0].legend()
    fig.suptitle(f"Request latency vs concurrency — {_workload_label(docs)} workload, {_hardware_label(docs)}",
                 fontsize=9, color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, "latency.png", out_dir)


def plot_kv_waste(sweeps: dict[str, dict[int, dict]], docs: list[dict], out_dir: Path) -> Path | None:
    """Waste is a property of the allocator, not the concurrency level (it is constant
    across a sweep for the contiguous engines), so one bar per engine at the highest
    shared concurrency is the honest summary."""
    if not sweeps:
        return None
    engines = _sorted_engines(sweeps)
    vals = []
    for e in engines:
        conc = max(sweeps[e])
        vals.append(float(sweeps[e][conc].get("kv_waste_pct", 0.0)))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(engines, vals, width=0.55, color=[ENGINE_COLOR.get(e, FALLBACK_COLOR) for e in engines])
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8, color=TEXT)
    ax.axhline(10, color=TEXT, linewidth=0.8, linestyle="--")
    ax.annotate("M3 target: <10%", (len(engines) - 0.5, 10), textcoords="offset points",
                xytext=(0, 3), ha="right", fontsize=8, color=TEXT)
    ax.set_ylabel("KV cache waste (% of reserved tokens unused)")
    ax.set_ylim(0, max(100.0, max(vals) * 1.1))
    ax.set_title(f"KV memory waste per engine — {_workload_label(docs)} workload", fontsize=9, color=TEXT, loc="left")
    ax.grid(axis="x", visible=False)
    return _save(fig, "kv_waste.png", out_dir)


OVERLOAD_PANELS = [
    # (title, [(field, label)], transform)
    ("KV pool: free blocks", [("free_blocks", "free_blocks"), ("num_blocks", "num_blocks (total)")], 1.0),
    ("Scheduler", [("queue_depth", "queue_depth"), ("num_running", "num_running"),
                   ("num_waiting", "num_waiting"), ("num_swapped", "num_swapped")], 1.0),
    ("Process memory (MiB)", [("rss_bytes", "rss"), ("device_mem_bytes", "device")], 1.0 / 2**20),
    ("Cumulative", [("completed", "completed"), ("preemptions", "preemptions"), ("swaps", "swaps")], 1.0),
]
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def plot_overload(docs: list[dict], out_dir: Path) -> Path | None:
    """Flat memory over 30 minutes is M4's claim; this is the chart that makes it one.
    Every /health field is optional — an engine that predates a counter simply has no
    line for it, and a panel with no lines says so instead of rendering blank."""
    timelines = [d for d in docs if d.get("mode") == "open-loop" and d.get("samples")]
    if not timelines:
        return None
    doc = timelines[-1]
    samples = doc["samples"]
    ts = [s.get("t_s", 0.0) / 60.0 for s in samples]

    fig, axes = plt.subplots(2, 2, figsize=(FIGSIZE[0] * 1.6, FIGSIZE[1] * 1.7), sharex=True)
    for ax, (title, fields, scale) in zip(axes.flat, OVERLOAD_PANELS):
        drew = False
        for (field, label), color in zip(fields, SERIES_COLORS):
            pts = [(t, s[field] * scale) for t, s in zip(ts, samples) if isinstance(s.get(field), (int, float))]
            if not pts:
                continue
            ax.plot([t for t, _ in pts], [v for _, v in pts], label=label, color=color, linewidth=1.6)
            drew = True
        ax.set_title(title, fontsize=9, color=TEXT, loc="left")
        if drew:
            ax.legend(fontsize=8)
            ax.set_ylim(bottom=0)
        else:
            ax.text(0.5, 0.5, "not reported by this engine's /health", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color=FALLBACK_COLOR)
    for ax in axes[1]:
        ax.set_xlabel("time (min)")
    window = doc.get("open_loop", {})
    if window.get("arrival_window_s"):
        for ax in axes.flat:
            ax.axvline(window["arrival_window_s"] / 60.0, color=TEXT, linewidth=0.7, linestyle=":")
    acct = doc.get("accounting", {})
    run = (doc.get("runs") or [{}])[0]
    fig.suptitle(
        f"Overload run — {run.get('engine', '?')} at {window.get('rps', '?')} req/s offered for "
        f"{window.get('duration_s', '?')}s, {_hardware_label([doc])}\n"
        f"submitted {acct.get('submitted', '?')} = completed {acct.get('completed', '?')} + "
        f"shed(503) {acct.get('shed_503', '?')} + errors {acct.get('errors', '?')}  (dotted line: arrivals stop)",
        fontsize=9, color=TEXT, x=0.01, ha="left",
    )
    fig.tight_layout()
    return _save(fig, "overload.png", out_dir)


def main() -> None:
    p = argparse.ArgumentParser(description="render docs/charts/*.png from results/*.json")
    p.add_argument("paths", nargs="*", type=Path, help="results files (default: --latest)")
    p.add_argument("--latest", type=int, default=None, metavar="N",
                   help="use the newest N files in results/ (default 1 when no paths given)")
    p.add_argument("--out-dir", type=Path, default=CHARTS_DIR)
    args = p.parse_args()

    paths = list(args.paths)
    if args.latest is not None or not paths:
        paths += latest_files(args.latest or 1)
    docs = load_files(paths)
    if not docs:
        raise SystemExit("no results files to plot")
    for d in docs:
        print(f"using {d['_path']}")

    sweeps = sweep_runs(docs)
    written = [
        plot_throughput(sweeps, docs, args.out_dir),
        plot_latency(sweeps, docs, args.out_dir),
        plot_kv_waste(sweeps, docs, args.out_dir),
        plot_overload(docs, args.out_dir),
    ]
    for w in written:
        if w:
            print(f"wrote {w}")
    if not sweeps:
        print("no closed-loop sweep in the given files: throughput/latency/kv_waste skipped")
    if not any(d.get("mode") == "open-loop" for d in docs):
        print("no open-loop run in the given files: overload timeline skipped")


if __name__ == "__main__":
    main()
