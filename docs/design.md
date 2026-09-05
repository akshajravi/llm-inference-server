# Design Document

Every non-obvious decision in this repo, with the alternative that lost and the number
that decided it. Numbers marked *(Mac)* were measured on Apple Silicon (MPS, GPT-2 124M,
float32) and exist to check direction, not to be quoted; the GPU run replaces them (see
README "Headline numbers").

## The three ideas, in one paragraph each

**Continuous batching.** A forward pass over 32 sequences costs about the same as one
over 1, because decode is bound by reading the weights, not by arithmetic. Static
batching captures that but freezes membership until the *slowest* row finishes; on
right-skewed output lengths the short rows sit idle in their slots. Measured: 77.3% of
row-steps wasted at concurrency 32 on the mixed workload *(Mac)*. Continuous batching
re-decides membership between every forward pass, so the stall is structurally zero
(`core/scheduler.py`).

**Paged KV cache.** A contiguous allocator cannot know a request's output length at
admission, so it reserves the worst case (`max_seq_len` = 1024) every time. Measured
waste: 84.4% of reserved cache *(Mac)*. Paging cuts the pool into 16-token blocks and
gives each sequence a block table, so its only over-reservation is the tail of its last
block. Measured waste: 4.4% on the mixed workload *(Mac)* — a property of the
arithmetic (bounded by 15 tokens per sequence), not a tuning result
(`core/block_allocator.py`, `core/block_table.py`, `core/paged_executor.py`).

**Preemption and admission control.** When the pool runs dry a running sequence is
evicted and re-admitted later, never dropped (FR6); when the wait queue hits its bound,
new requests get an immediate 503 instead of an unbounded wait (FR7). Together they are
what let the server run at 10x its capacity without OOM or silent loss (M4).

## Decisions

| Decision | Choice | Rejected | Why it lost | Measured |
|---|---|---|---|---|
| Batch mutability | Continuous (per-step) | Static (per-batch) | Static's stall is 50–77% of row-steps on mixed traffic; at concurrency 4 it did 115 tok/s with a 21.3 s p99 against continuous batching's 350 tok/s and 2.3 s *(Mac)* | Day 2, Day 5 |
| M2 denominator | P1 static batching | Naive P0 | P0 leaves >99% of the GPU idle; beating it by 3x measures "batching works", not "continuous batching works". Both numbers are reported; static leads. | Day 1 |
| KV allocation | Paged, 16-token blocks | Contiguous to `max_seq_len` | 84.4% → 4.4% waste *(Mac)*. 16 tokens: small enough that the tail waste is <10% at realistic lengths, large enough that the block table stays short and each gather reads a contiguous 16×heads×dim tile. | Day 6, Day 9 |
| Baseline KV reservation | `max_seq_len` per request | `prompt_len + max_tokens` | A contiguous allocator does not know output length at admission. Charging only what was asked understates baseline waste to ~0% and erases M3's "before" number. | Day 1 |
| Pool sizing | Fixed 2048 blocks = 32 × 1024 slots | Fill free GPU memory | The pool is deliberately *equal* to the contiguous engines' worst case, so P3 is measured holding exactly the same reservation P2 does. The win has to come from packing, not from a bigger pool. `NUM_BLOCKS` shrinks it for local test runs only. | Day 6 |
| Paged attention hook | Register a custom attention function + a `Cache` subclass on the HF model | Hand-written GPT-2 forward | The hook is architecture-agnostic (Llama/Qwen use the same interface), keeps M1's reference and the paged path on the same weights, and cost 280 lines instead of a second model implementation. The K/V that HF hands the attention function is ignored; everything is read back through the block table, so a scatter/gather mismatch fails M1 immediately rather than silently. | Day 8 |
| Batch layout | All sequences' new tokens packed into one row with per-token positions | Padded `[batch, width]` | Padding is what the P2 executor did; under paging the kernel already handles ragged lengths through `context_lens`/`query_lens`, so packing costs nothing and a prefill of 200 tokens next to a decode of 1 wastes no compute on pad columns. | Day 8 |
| Kernel language | Triton (S3) with a PyTorch gather fallback that ships | Raw CUDA | Comparable performance for this access pattern at a fraction of the development cost. The gather path is the shipping path on the Mac; it runs at 0.3–0.6x of P2's throughput because it re-gathers every layer's K/V each step and runs prefill per sequence *(Mac)*. The Triton kernel is written but **unverified** on the author's Mac (no macOS wheels); its tiling and online-softmax loop is mirrored in pure PyTorch and that mirror matches the gather path to 1e-5. First run on CUDA: `ATTENTION=triton make test`. | Day 8, Day 14 |
| Prefill scheduling | Prefill-priority, alternating | Chunked prefill | Simpler. The cost is real: decoding sequences stall for one pass whenever a newcomer prefills, which shows up in p99 at high concurrency. Chunked prefill is named future work. | Day 5 |
| Preemption default | Recompute | CPU swap | Both are implemented (`PREEMPTION=recompute|swap`). Recompute is the default: it needs no host memory, no pinned buffers, and wins whenever the victim's history is short. Swap wins once re-prefilling costs more than the PCIe round trip; the crossover is a measured comparison (below), not an assertion. | Day 10 |
| Victim policy | Most recently admitted | Oldest / largest | Preserves progress on the oldest work and gives a termination argument: the oldest running sequence is only ever a victim when it is alone, and any admitted sequence fits alone (`SequenceTooLong` at `add()` guarantees it), so it always finishes. | Day 10 |
| Preemption-loop guard | Reserve the running set's next-step growth before admitting | Admit whenever a block is free | Without the reservation a newcomer takes the last free block, the next decode finds the pool empty, and the newcomer (youngest, hence victim) is evicted before producing a token — a re-prefill for nothing, every lap. Reserving makes it wait one step instead. | Day 10 |
| Overload behavior | Bounded queue (256) + 503 with `Retry-After` | Unbounded queue | Unbounded queues convert overload into unbounded latency, which is worse than an explicit error. A request that can never fit (prompt + max_tokens > pool) gets 422, not 503, because a retry cannot help. | Day 11 |
| Engine threading | Two locks: a step lock only the step thread holds, an inbox lock the event loop may touch | One lock over the pool | The one-lock version starved the HTTP thread: Python locks are unfair and the step loop re-acquires within nanoseconds, so at 20 req/s the event loop stalled for tens of seconds, uvicorn stopped accepting, and clients saw 152 connection resets instead of 503s. With the inbox: 0 errors, 208 clean 503s at 20 req/s, 742 at 40 req/s *(Mac)*. Nothing on the event-loop thread may wait for a forward pass. | Day 11 |
| Prefix sharing granularity | Full 16-token blocks, hash-chained | Token-granular with copy-on-write | Sharing only full blocks means a sequence's writes always start at a block boundary in a block it owns, so copy-on-write is never triggered by the policy. `BlockTable.ensure_private` is implemented and unit-tested as the guard for a future partial-block policy, and the executor calls it before every write; under this policy it is one refcount read per written block. | Day 14 |
| Prefix cache lifetime | Entry dies with the block (refcount → 0) | Keep evicted blocks as an LRU of still-hashed blocks (vLLM) | The allocator stays a free list of integers with no knowledge of hashes. Cost: a prefix is reusable only while some sequence still holds it, so sharing is between *concurrent* requests. The LRU is the named next step. | Day 14 |
| Engine interface | Token IDs + TTFT in `Result` | `generate(prompt) -> str` | Text comparison hides off-by-one errors; TTFT is unmeasurable after the fact. | Day 1 |

## Measurements behind the table

### Static batching's stall (P1 → P2)

`Result.wasted_steps` counts forward-pass steps a finished sequence spent occupying a
slot. On the mixed workload *(Mac, `results/2026-09-04-094441-mixed.json`)*:

| Concurrency | Static stall % | Static tok/s | Continuous tok/s | Ratio | p99 static | p99 continuous |
|---|---|---|---|---|---|---|
| 4 | 50.2 | 114.5 | 350.4 | 3.06x | 21.3 s | 2.3 s |
| 8 | 62.8 | 209.5 | 431.7 | 2.06x | 4.8 s | 3.7 s |
| 16 | 70.3 | 212.7 | 299.6 | 1.41x | 6.8 s | 11.1 s |
| 32 | 77.3 | 187.6 | 368.6 | 1.97x | 12.5 s | 14.8 s |

The ratio is noisy on MPS (the device is shared with the OS); the GPU sweep is the one
the README quotes. The direction is not noisy: the stall grows with concurrency and
continuous batching's stall is zero by construction.

### KV waste (P2 → P3)

Waste = 1 − used/reserved, summed over the workload. Contiguous engines reserve
`max_seq_len` per request; the paged engine reserves its block table's capacity.

| Engine | Waste | Note |
|---|---|---|
| naive / manual / static / continuous | 84.4% | mixed workload, any concurrency |
| paged | 4.4% | mixed workload, every concurrency *(Mac)* |

What the freed memory buys (the "N" in the resume bullet): the 32,768-slot pool holds
32 sequences under contiguous reservation and 196 at the paged engine's measured
~167-token footprint — 164 more concurrent sequences on the same memory.

The paged engine's throughput is where the honesty is: 106 tok/s vs 350 for the P2
executor at concurrency 4, 0.36x of even the naive baseline at concurrency 1. The
PyTorch gather path re-reads every layer's K/V through the block table on every step
and runs prefill per-sequence in Python; that is the cost the Triton kernel exists to
remove, and it is reported as measured rather than hidden behind the kernel.

The paged figure is a function of sequence length: waste per sequence is a fixed
0–15-token tail, so very short requests (~48 tokens) read ~14% and the bench-shaped
mix reads 4.4%.

### Prefix caching (S1)

`shared-prefix` workload: every prompt starts with the same 128-token system prompt,
then a 16–128-token tail; outputs as in `mixed`. Paged engine, concurrency 8, 200
requests, 512-block pool, two repeats *(Mac)*:

| Prefix caching | tok/s | TTFT p50 | TTFT p95 | Latency p50 | Blocks shared | Tokens not recomputed |
|---|---|---|---|---|---|---|
| on | 149.5 / 147.8 | 84 ms | 141 / 126 ms | 1.51 s | 1596 | 25,536 |
| off | 144.4 / 140.9 | 95 / 98 ms | 155 / 160 ms | 1.60 s | 0 | 0 |

Eight of each prompt's nine full blocks come from the cache (the ninth is the last full
block before the tail, which is always fed so the model produces a logit). TTFT drops
~11%; throughput rises ~4% because prefill is a small share of total work on this
workload. The memory saved is 1596 blocks × 16 tokens over the run; at any instant it
is the shared prefix once instead of once per running sequence — 7 × 8 blocks with 8
in flight. The free list returns to full after the run under both settings.

### Recompute vs swap (S2)

Mixed workload at concurrency 16 with a 96-block pool (1,536 slots — deliberately too
small for 16 sequences, so the scheduler preempts throughout), two repeats *(Mac)*:

| Strategy | tok/s | TTFT p50 | Latency p50 | p99 | Preemptions | Free list after |
|---|---|---|---|---|---|---|
| recompute | 150.8 / 155.1 | 1.70 / 1.64 s | 3.41 / 3.39 s | 11.6 / 11.7 s | 49 / 49 | 96 / 96 |
| swap | 152.6 / 160.6 | 1.74 / 1.65 s | 3.43 / 3.22 s | 11.4 / 10.8 s | 46 / 46 | 96 / 96 |

Every request completed under both strategies with token-exact output (the goldens
run under forced preemption in `tests/test_preemption.py`), and the free list returned
to full. Swap is 1–4% ahead here, within run-to-run noise: on Apple Silicon host and
device memory are the same pool, so "swap" is a memcpy and the PCIe cost that makes
the comparison interesting does not exist. The expected crossover on a discrete GPU —
swap wins once a victim's history is long enough that re-prefilling it costs more than
copying `blocks × 72 KiB` each way over PCIe — is the measurement the GPU run adds.
Recompute stays the default because it needs no host memory and loses nothing on
short histories, which is what the mixed workload mostly has.

### Overload (M4)

Open-loop Poisson arrivals over HTTP, paged engine, 30-second smoke runs *(Mac)*:

| Offered load | Submitted | Completed | 503 | Errors | Queue max | Free blocks min |
|---|---|---|---|---|---|---|
| 20 req/s, one-lock engine | 672 | 520 | 0 | 152 (connection resets) | — | — |
| 20 req/s, inbox engine | 672 | 464 | 208 | 0 | 256 | 1744 |
| 40 req/s, inbox engine | 1269 | 527 | 742 | 0 | 256 | 0 |

The 30-minute run (`make overload RPS=49`, 10x the paged engine's 4.9 req/s peak from
the sweep; `results/2026-09-04-104334-overload-overload.json`, chart in
`docs/charts/overload.png`) *(Mac)*:

| | |
|---|---|
| Submitted / completed / shed (503) / errors | 88,165 / 5,867 / 82,298 / 0 — balanced |
| Server throughput | 137.8 tok/s, 3.08 req/s |
| Queue depth | pinned at the 256 bound for the whole window, drained to 0 in 103 s after |
| Free blocks | 1585–2048 of 2048; never exhausted |
| Device memory (MPS allocated) | 2780 MiB at t=0, 2780 MiB at t=1800 s, peak 2956 MiB |
| Process RSS | 659 MiB → 1007 MiB in the first five minutes, then flat (peak RSS, so flat by construction after warm-up) |
| Preemptions | 0 |
| `/health` samples | 382, none failed |

`submitted == completed + shed + errors` is asserted by the harness and written into
every results file; that equation is the "no dropped-but-unacknowledged request"
requirement made checkable. Two honest notes. First, preemption never fired: with the
full 2048-block pool and `max_running` = 32, admission control shed load long before
memory ran dry, so this run proves the bounded queue and the flat-memory claim, while
preemption's correctness under pressure is proven by the small-pool comparison above and
the goldens-under-preemption tests. Second, completed requests waited a median 82 s in
the queue (TTFT p50) — the price of a 256-deep queue at 3 req/s, and the argument for
sizing the bound to a latency target rather than a count, which is future work.

## Scoped out, and why

| Item | Status | Note |
|---|---|---|
| Triton kernel verification | Written, unverified | No macOS wheels. The PyTorch mirror of the kernel's loop passes; the kernel itself is verified the first time `ATTENTION=triton make test` runs on CUDA. Reported honestly as unverified until then. |
| Prefix cache that outlives its referents | Future work | Needs the allocator to know about hashes (free list becomes an LRU). Would turn sharing-between-concurrent-requests into sharing-across-time. |
| Chunked prefill | Future work | Removes the one-pass stall decoders pay when a newcomer prefills. |
| Cancelling in-flight work on client disconnect | Future work | A disconnected streaming client's sequence runs to completion; it does not leak the queue. |
| Multi-GPU, quantization, speculative decoding, sampling strategies | Non-goals | Per the PRD. Greedy only, which is what lets M1 assert exact token IDs. |

## Open questions

- **GPU numbers.** Every number above is from the dev Mac. The README's headline table
  is empty until `make bench` runs on the rented GPU; that run also produces
  `requirements-gpu.txt` and verifies the Triton kernel.
- **Swap on unified memory.** On Apple Silicon "host" and "device" memory are the same
  pool, so the swap comparison there measures copy cost, not PCIe. The crossover only
  means something on a discrete GPU.
