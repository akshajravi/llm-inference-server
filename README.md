# LLM Inference Server

> **Status: Day 1 of a 14-day sprint.** This README is a stub. It becomes a real
> deliverable on Day 13 — architecture diagram, headline numbers with the GPU named,
> three-minute quickstart. See `IMPLEMENTATION_GUIDE.md`.

A from-scratch inference server implementing continuous batching and a paged KV cache,
benchmarked against a naive static-batching baseline.

## Quickstart

```bash
make setup                    # venv + pinned deps
make goldens                  # generate M1 reference token IDs (once)
make test                     # M1 correctness gate
make serve ENGINE=naive       # http://localhost:8000
make bench-one ENGINE=naive   # baseline numbers -> results/
```

## Layout

| Path | What it is |
|---|---|
| `inference_server/engine/` | One implementation per phase, all behind `engine/base.py` |
| `inference_server/core/` | Scheduler, executor, sequence, block allocator — each readable alone |
| `inference_server/bench/` | Load generator, workload spec, percentile metrics |
| `tests/` | M1 correctness suite + goldens; allocator and batch-invariance tests |
| `results/` | Committed benchmark output, dated, hardware named |

## Phase status

| Engine | Phase | Days | Status |
|---|---|---|---|
| `naive` | P0 baseline | 1 | ✅ shipped |
| `manual` | P1 decode loop | 2 | ⬜ |
| `static` | P1 static batching | 2 | ⬜ |
| `continuous` | P2 continuous batching | 3–5 | ⬜ |
| `paged` | P3 paged KV cache | 6–9 | ⬜ |

Stretch (Day 14 only): Triton kernel (S3), prefix caching (S1), CPU-swap preemption
(S2), SSE streaming (S4).

## Headline numbers

_Filled on Day 13 from a single GPU session. Any number `make bench` cannot reproduce
does not go in this table, and does not go on a resume._
