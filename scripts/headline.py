"""The headline numbers, computed from a results file rather than typed into the README.

    python -m scripts.headline                       # newest mixed sweep in results/
    python -m scripts.headline results/a.json results/b.json

Prints, in plain text:
  M2  continuous (and paged) tok/s over static tok/s at every concurrency both were
      measured at, and the best of those — with p99 latency at the same points, since
      a throughput win bought with a worse tail is not the claim being made.
  M3  KV waste % per engine, and how many more sequences the freed memory holds.
  M4  the offered rate to use for the overload run: 10x the best req/s in the sweep.

Anything `make bench` cannot reproduce does not go on the resume, and this script is
how that rule is enforced: the README quotes its output, and its input is the file
`make bench` writes. If a number is missing from the file it says which one, rather
than filling it in from the current config.py — the config may have moved since.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

BASELINE = "static"
CANDIDATES = ["continuous", "paged"]


def load(paths: list[Path]) -> list[dict]:
    docs = []
    for p in paths:
        d = json.loads(p.read_text())
        d["_path"] = str(p)
        docs.append(d)
    return docs


def latest_mixed(directory: Path = RESULTS_DIR) -> Path | None:
    """Newest closed-loop `mixed` file. Files are timestamp-prefixed, so lexical order
    is chronological; open-loop files are timelines and cannot be the headline sweep."""
    for p in sorted(directory.glob("*mixed*.json"), reverse=True):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if d.get("mode") != "open-loop" and d.get("runs"):
            return p
    return None


def sweeps(docs: list[dict]) -> dict[str, dict[int, dict]]:
    """{engine: {concurrency: run}}; a later file overrides an earlier one for the
    same key, so pass the newest last."""
    out: dict[str, dict[int, dict]] = {}
    for d in docs:
        if d.get("mode") == "open-loop":
            continue
        for r in d.get("runs", []):
            out.setdefault(r["engine"], {})[int(r["concurrency"])] = r
    return out


def m2(sw: dict[str, dict[int, dict]]) -> None:
    print("M2 — throughput over static batching (mixed workload)")
    base = sw.get(BASELINE)
    if not base:
        print(f"  missing: no '{BASELINE}' runs in the given files")
        return
    for cand in CANDIDATES:
        runs = sw.get(cand)
        if not runs:
            print(f"  {cand}: not in the given files")
            continue
        shared = sorted(set(base) & set(runs))
        if not shared:
            print(f"  {cand}: no concurrency level shared with {BASELINE}")
            continue
        best = None
        for c in shared:
            b, r = base[c], runs[c]
            ratio = r["throughput_tok_s"] / b["throughput_tok_s"] if b["throughput_tok_s"] else float("nan")
            print(f"  {cand:<11} conc {c:>3}: {r['throughput_tok_s']:8.1f} tok/s vs {b['throughput_tok_s']:8.1f} "
                  f"= {ratio:5.2f}x   p99 {r['latency_p99']:7.2f}s vs {b['latency_p99']:7.2f}s"
                  f"   (req/s {r['throughput_req_s']:.2f} vs {b['throughput_req_s']:.2f})")
            if best is None or ratio > best[1]:
                best = (c, ratio)
        c, ratio = best
        verdict = "meets" if ratio >= 3.0 else "does not meet"
        print(f"  {cand}: best {ratio:.2f}x at concurrency {c} — {verdict} the >=3x target")


def m3(sw: dict[str, dict[int, dict]], config: dict | None) -> None:
    print("M3 — KV cache waste")
    waste: dict[str, float] = {}
    for engine, runs in sw.items():
        c = max(runs)
        waste[engine] = float(runs[c].get("kv_waste_pct", 0.0))
        print(f"  {engine:<11} {waste[engine]:5.1f}% waste  (at concurrency {c})")
    if "paged" in waste:
        verdict = "meets" if waste["paged"] < 10.0 else "does not meet"
        print(f"  paged: {waste['paged']:.1f}% — {verdict} the <10% target")

    # How many more sequences fit. The contiguous engines hold exactly max_running
    # sequences in a pool of max_running * max_seq_len tokens, using (1 - waste_base)
    # of what they reserve. An allocator with waste w reserves used / (1 - w) per
    # sequence, so the same pool holds max_running * (1 - w) / (1 - waste_base).
    base_w = waste.get(BASELINE)
    if base_w is None:
        print(f"  additional sequences: missing '{BASELINE}' waste to derive the per-sequence footprint")
        return
    if not config or any(k not in config for k in ("max_running", "max_seq_len", "num_slots")):
        print("  additional sequences: missing 'config' block (max_running, max_seq_len, num_slots) "
              "in the results file — re-run `make bench` at a commit that writes it")
        return
    max_running, max_seq_len, num_slots = config["max_running"], config["max_seq_len"], config["num_slots"]
    reserved_base = max_running * max_seq_len
    if num_slots != reserved_base:
        print(f"  note: pool holds {num_slots} slots but contiguous reservation is {reserved_base}; "
              f"capacities below use the pool size")
    used_per_seq = max_seq_len * (1.0 - base_w / 100.0)
    print(f"  pool: {num_slots} token slots; contiguous engines fit {num_slots // max_seq_len} sequences "
          f"(reserving {max_seq_len} each, using ~{used_per_seq:.0f})")
    for cand in CANDIDATES:
        if cand not in waste:
            continue
        w = waste[cand] / 100.0
        footprint = used_per_seq / (1.0 - w) if w < 1.0 else float("inf")
        fits = int(num_slots // footprint) if footprint else 0
        print(f"  {cand:<11} footprint ~{footprint:.0f} tokens/seq -> {fits} sequences fit, "
              f"{fits - num_slots // max_seq_len:+d} vs contiguous")


def m4(sw: dict[str, dict[int, dict]]) -> None:
    print("M4 — overload run: pick the offered rate")
    best = None
    for engine, runs in sw.items():
        if engine not in CANDIDATES:
            continue
        for c, r in runs.items():
            if best is None or r["throughput_req_s"] > best[2]:
                best = (engine, c, r["throughput_req_s"])
    if best is None:
        print("  missing: no continuous/paged runs to size capacity from")
        return
    engine, c, req_s = best
    print(f"  {engine} peaked at {req_s:.2f} req/s (concurrency {c}); 10x capacity = {10 * req_s:.0f} req/s")
    print(f"  make overload ENGINE={engine} RPS={10 * req_s:.0f}")


def overload_report(docs: list[dict]) -> None:
    runs = [d for d in docs if d.get("mode") == "open-loop"]
    if not runs:
        return
    d = runs[-1]
    acct = d.get("accounting", {})
    ol = d.get("open_loop", {})
    print(f"M4 — overload run in {d['_path']}")
    print(f"  {ol.get('rps')} req/s offered for {ol.get('duration_s')}s: "
          f"{acct.get('equation')} -> {'balanced' if acct.get('balanced') else 'NOT balanced'}")
    samples = [s for s in d.get("samples", []) if "rss_bytes" in s]
    if samples:
        first, last = samples[0]["rss_bytes"], samples[-1]["rss_bytes"]
        peak = max(s["rss_bytes"] for s in samples)
        print(f"  rss: {first / 2**20:.0f} MiB -> {last / 2**20:.0f} MiB (peak {peak / 2**20:.0f} MiB) "
              f"over {len(samples)} samples")
    else:
        print("  rss: not in samples (server /health predates rss_bytes)")


def main() -> None:
    p = argparse.ArgumentParser(description="print the headline M2/M3/M4 numbers from results files")
    p.add_argument("paths", nargs="*", type=Path, help="results files (default: newest mixed sweep)")
    args = p.parse_args()
    paths = list(args.paths)
    if not paths:
        latest = latest_mixed()
        if latest is None:
            raise SystemExit("no mixed sweep in results/; run `make bench` first")
        paths = [latest]
    docs = load(paths)
    for d in docs:
        hw = d.get("hardware", {})
        print(f"file: {d['_path']}  ({hw.get('name', '?')}, {hw.get('model_id', '?')} {hw.get('dtype', '')}, "
              f"workload {d.get('workload', {}).get('name', '?')})")
    print()
    sw = sweeps(docs)
    config = next((d["config"] for d in reversed(docs) if d.get("config")), None)
    if sw:
        m2(sw)
        print()
        m3(sw, config)
        print()
        m4(sw)
        print()
    overload_report(docs)


if __name__ == "__main__":
    main()
