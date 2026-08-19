# Design Document

> **Status: skeleton (Day 1).** Written properly on Day 13. Capture decisions here *as
> they are made* — reconstructing rationale at the end of a sprint produces vague prose,
> and the rejected alternative is the part that gets forgotten first.

## How to use this file

Every non-obvious decision gets a row. The rejected alternative is named, and the reason
it lost is what was *measured*, not what was assumed. The table below starts from the
decisions fixed during planning; each row expands as its phase produces a number.

## Decisions

| Decision | Choice | Rejected | Why it lost | Measured? |
|---|---|---|---|---|
| Batch mutability | Continuous (per-step) | Static (per-batch) | _P1 mixed-length degradation: TBD_ | Day 2 |
| KV allocation | Paged, 16-token blocks | Contiguous to max length | _Baseline waste: TBD%_ | Day 6 |
| Prefill scheduling | Prefill-priority, alternating | Chunked prefill | Simpler; TTFT cost documented not optimised | Day 5 |
| Preemption | Recompute | CPU swap | Simpler; swap scoped out (see below) | Day 10 |
| Overload behavior | Bounded queue + 503 | Unbounded queue | Unbounded queues turn overload into unbounded latency | Day 11 |
| Engine interface | Token IDs + TTFT in `Result` | `generate(prompt) -> str` | Text comparison hides off-by-one; TTFT unmeasurable after the fact | Day 1 |
| M2 denominator | P1 static batching | Naive P0 | P0 leaves >99% of the GPU idle, so beating it by 3x measures "batching works", not "continuous batching works". Both numbers reported; the static one leads. | Day 1 |
| Baseline KV reservation | `max_seq_len` per request | `prompt_len + max_tokens` | A contiguous allocator cannot know output length at admission time. Charging only what was asked understates baseline waste to ~0% and erases M3's "before" number. | Day 1 |

## Scoped out, and why

A 14-day sprint cannot hold everything at defensible quality. These were cut from the
critical path deliberately, before the sprint started, rather than abandoned mid-flight:

| Item | Why cut | What I'd expect if built |
|---|---|---|
| S3 — Triton paged attention | Named top schedule risk in the PRD; in 14 days it eats the sprint. The PyTorch gather path is correct and unblocks P4. | _fill in the expected speedup and where it comes from_ |
| S1 — Prefix caching | Second allocator-correctness surface (refcounts + COW). Refcounts are in the allocator anyway. | _memory saved on shared-system-prompt traffic_ |
| S2 — CPU swap preemption | Recompute alone satisfies M4; swap exists to produce a comparison chart. | _crossover: swap wins once recompute cost exceeds the PCIe round trip_ |
| S4 — Streaming (SSE) | Cheapest stretch item; first to pick up on Day 14. | _perceived-latency improvement, no throughput change_ |

## Open questions

- _log them here as they come up, with the day they were resolved_
