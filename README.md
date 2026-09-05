# LLM Inference Server

A from-scratch inference server implementing the three ideas that make modern serving
systems fast — **continuous batching**, a **paged KV cache**, and **preemption with
admission control** — benchmarked against a naive baseline and a static-batching
baseline, with greedy output verified token-for-token against HuggingFace `generate()`
at every phase.

Python, PyTorch, FastAPI, Triton. No vLLM, no TensorRT: the point is to implement, not
integrate.

## Architecture

```
HTTP layer (FastAPI)          POST /generate, /generate/stream (SSE), GET /health
        │
   inbox ──────────────────── admission control: bounded queue → 503, too-long → 422
        │
   Scheduler                  core/scheduler.py — the whole thesis is in step():
        ├── sequence pool       evict finished → admit waiting (memory-aware) → grow
        │   waiting / running   block tables → preempt on exhaustion → one forward pass
        │   swapped / finished
        ├── Block allocator     core/block_allocator.py — free list + refcounts
        ├── Block tables        core/block_table.py — logical position → physical block
        └── Prefix cache        core/prefix_cache.py — hash-chained full blocks, shared
        │
   Executor                   core/paged_executor.py — one forward pass, no policy
        │
   Paged attention            core/attention.py (PyTorch gather) / attention_triton.py
        │
   KV pool                    core/kv_pool.py — [num_blocks, 16, heads, head_dim] per layer
```

Each `core/` file is meant to be read on its own. Five engines share one interface
(`engine/base.py`) so the benchmark compares schedulers, not harnesses:

| Engine | Phase | What it is |
|---|---|---|
| `naive` | P0 | `model.generate()`, one request at a time. The denominator. |
| `manual` | P1 | Hand-written prefill/decode loop. Same speed as P0; creates the seam every later phase inserts into. |
| `static` | P1 | Left-padded batch, frozen until the slowest row finishes. The stall is measured, not asserted. |
| `continuous` | P2 | Mutable batch: membership re-decided between every forward pass. |
| `paged` | P3+P4 | Continuous batching over a block-allocated KV pool, with preemption, prefix sharing, and admission control. |

## Quickstart

```bash
make setup                    # venv + pinned deps
make goldens                  # M1 reference token IDs from HF generate() (once)
make test-light               # full suite, small KV pool (dev laptop)
make serve ENGINE=paged       # http://localhost:8000
curl -s localhost:8000/generate -d '{"prompt":"The capital of France is","max_tokens":16}' -H 'content-type: application/json'
curl -N localhost:8000/generate/stream -d '{"prompt":"Once upon a time","max_tokens":32}' -H 'content-type: application/json'
```

Reproducing the numbers (M5), on any box:

```bash
make bench                    # sweep every engine on the mixed workload → results/*.json
make headline                 # M2 / M3 ratios and the suggested overload rate
make serve ENGINE=paged &     # then, in another shell:
make overload RPS=<10x capacity>   # 30 min open-loop, /health sampled every 5 s
make charts                   # docs/charts/*.png from the newest results
```

Knobs, all environment variables: `ENGINE`, `DEVICE` (cuda/mps/cpu), `MODEL_ID`,
`DTYPE`, `PREEMPTION` (recompute/swap), `PREFIX_CACHING` (1/0), `ATTENTION`
(gather/triton), `NUM_BLOCKS` (local test runs only).

## Headline numbers

**GPU: not yet run.** Every number below is from the development machine (Apple M-series,
MPS, GPT-2 124M, float32) and is there to show direction. The GPU sweep on a rented
4090/A10 replaces this table, and only numbers `make bench` reproduces go on a resume.

Mixed workload (200 requests, prompts 16–256 tokens, lognormal outputs, median ~30,
tail to 512), Apple Silicon, `results/2026-09-04-094441-mixed.json`:

| | naive | static | continuous | paged |
|---|---|---|---|---|
| Throughput at concurrency 4 (tok/s) | 132 | 115 | **350** | 106 (gather path) |
| Throughput at concurrency 32 (tok/s) | 135 | 188 | 369 | 214 |
| p99 latency at concurrency 4 | 4.5 s | 21.3 s | 2.3 s | 8.1 s |
| Continuous vs static, best | | | **3.06x** (conc 4) | |
| KV cache waste | 84.4% | 84.4% | 84.4% | **4.4%** |
| Sequences that fit in the 32k-slot pool | 32 | 32 | 32 | **196** |
| Static-batching stall (row-steps wasted) at concurrency 32 | 0 | 77.3% | 0 | 0 |

The paged engine's throughput is honest: the PyTorch gather path re-gathers every
layer's K/V each step. That is the Triton kernel's job, and the kernel is written but
can only be verified on CUDA (see `docs/design.md`, "Scoped out").

Overload (M4), paged engine, open-loop Poisson arrivals over HTTP at 49 req/s — 10x
measured capacity — for 30 minutes *(Mac, `results/2026-09-04-104334-overload-overload.json`)*:

| Submitted | Completed | Shed (503) | Errors | Device memory t=0 → t=30 min | Free blocks min |
|---|---|---|---|---|---|
| 88,165 | 5,867 | 82,298 | 0 | 2780 MiB → 2780 MiB | 1585 / 2048 |

`submitted == completed + shed + errors` is asserted by the harness and written into
every results file. No OOM, no crash, no connection errors, memory flat
(`docs/charts/overload.png`).

Prefix caching, shared 128-token system prompt, concurrency 8 *(Mac)*: TTFT p50 95 ms →
84 ms, 8 of 9 prompt blocks served from cache, 25.5k tokens not recomputed per 200
requests.

## Correctness

`tests/goldens/gpt2_greedy.json` holds token IDs from HuggingFace greedy `generate()`
for seven prompts (short, long, EOS-terminated, cap-terminated, single token, repeated
structure, whitespace edges). Every engine must reproduce them exactly — in-process,
over HTTP, alone, crowded into a ragged batch, arriving mid-decode, streamed, and under
forced preemption with both strategies. `make test` runs all of it; the goldens are
never regenerated to make a phase pass.

## Layout

| Path | What it is |
|---|---|
| `inference_server/engine/` | One implementation per phase behind `engine/base.py` |
| `inference_server/core/` | Scheduler, executor, sequence, allocator, block table, prefix cache, attention — each readable alone |
| `inference_server/server/` | FastAPI app: `/generate`, `/generate/stream`, `/health` |
| `inference_server/bench/` | Load generator (closed- and open-loop), workload specs, percentile metrics |
| `scripts/` | `make_goldens`, `plot` (charts), `headline` (M2/M3 ratios) |
| `tests/` | M1 goldens + allocator, block table, attention, preemption, swap, admission, streaming, prefix cache |
| `results/` | Committed benchmark output, dated, hardware named |
| `docs/design.md` | Every decision with its rejected alternative and the number that decided it |

## Status

| Requirement | Status |
|---|---|
| M1 exact greedy equivalence at every phase | ✅ enforced by test |
| M2 ≥3x over static batching | ✅ 3.06x *(Mac)*; GPU pending |
| M3 <10% KV waste | ✅ 4.4% vs 84.4% |
| M4 30-min overload without OOM/crash/silent drop | ✅ 30 min at 10x, 0 errors, memory flat *(Mac)* |
| M5 `make bench` reproduces published numbers | ✅ one command; GPU run pending |
| S1 prefix caching with refcounts | ✅ |
| S2 both preemption strategies + comparison | ✅ implemented; comparison in `docs/design.md` |
| S3 Triton paged attention | written, **unverified** (no Triton on macOS) |
| S4 SSE streaming | ✅ |
