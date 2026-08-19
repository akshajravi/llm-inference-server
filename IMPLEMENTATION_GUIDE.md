# Implementation Guide — 2-Week Sprint

Step-by-step build plan for the inference server. Companion to the project PRD (kept locally, not in this repo) — the PRD says *what* and *why*, this says *in what order* and *how to know you're done*.

**Rule that governs everything:** the correctness suite (M1) passes at the end of every phase. If it fails, you are not allowed to start the next phase. Optimizations that change output are not optimizations, they're bugs.

**Second rule, specific to a 2-week sprint:** when a day runs long, cut scope, not correctness. Every phase below has a named "minimum shippable" and a named "cut first" — decide against those, not in the moment at 1am.

---

## Sprint shape

14 days. Days are the unit; assume ~4–6 focused hours each.

| Days | Phase | Deliverable |
|---|---|---|
| 1 | P0 — Baseline + harness | Naive server, correctness goldens, load generator, baseline numbers |
| 2 | P1 — Manual decode + static batching | Own the loop; mixed-length degradation quantified |
| 3–5 | P2 — Continuous batching | Scheduler / sequence / executor split; M2 (≥3x over static) |
| 6–9 | P3 — Paged KV cache | Block allocator, block tables, PyTorch paged attention; M3 (<10% waste) |
| 10–11 | P4 — Preemption + admission control | Recompute preemption, bounded queue, overload run; M4 |
| 12–13 | P5 — Writeup + GPU benchmark day | README, design doc, charts, `make bench`; M5 |
| 14 | Buffer | Slip absorption, or stretch goals if you're somehow ahead |

### Scope decisions made up front

Must-haves (M1–M5) are all in the critical path. Everything in the PRD's "should have" table is **stretch**, and stretch work is only allowed on Day 14 or when a phase finishes early:

| Item | Status | Why it's cut from the critical path |
|---|---|---|
| S3 — Triton paged attention kernel | **Stretch** | The PRD already names it the top schedule risk. In 14 days it's the risk that eats the sprint. The PyTorch gather fallback is correct and unblocks everything downstream. |
| S1 — Prefix caching | **Stretch** | Refcounting + copy-on-write is a second allocator-correctness surface. Nice bullet, not a must-have. |
| S2 — CPU swap preemption | **Stretch** | Recompute alone satisfies M4. Swap only exists to make a comparison chart. |
| S4 — Streaming (SSE) | **Stretch, cheap** | ~1 hour once the scheduler exists. First stretch item to pick up — it demos well per token of effort. |

Write this table into the design doc as-is. "I scoped these out deliberately and here's the tradeoff" is a stronger interview answer than a half-finished Triton kernel.

---

## Table of contents

- [Day 1 — P0: Baseline + harness](#day-1--p0-baseline--harness)
- [Day 2 — P1: Manual decode loop + static batching](#day-2--p1-manual-decode-loop--static-batching)
- [Days 3–5 — P2: Continuous batching](#days-35--p2-continuous-batching)
- [Days 6–9 — P3: Paged KV cache](#days-69--p3-paged-kv-cache)
- [Days 10–11 — P4: Preemption + admission control](#days-1011--p4-preemption--admission-control)
- [Days 12–13 — P5: Writeup + GPU benchmark day](#days-1213--p5-writeup--gpu-benchmark-day)
- [Day 14 — Buffer / stretch](#day-14--buffer--stretch)
- [Appendix: reference numbers](#appendix-reference-numbers)

---

## Day 1 — P0: Baseline + harness

**Goal:** produce the denominator. Every speedup you claim is measured against what you build today, so build it honestly and don't optimize it.

### Steps

1. **Repo skeleton.** Pin dependencies (`requirements.txt` with exact versions), fix all seeds, add `Makefile` with `make test` and `make bench` targets from day one.

2. **Naive server.** FastAPI app, model loaded once at startup, one endpoint:
   ```
   POST /generate  { prompt, max_tokens, ... }  ->  { text }
   ```
   Body is `model.generate(..., do_sample=False)`. One request at a time. No batching. **Deliberately naive** — any cleverness here shrinks your headline number for free.

3. **Define the engine interface.** One thin abstraction that today's naive server satisfies and that P2's scheduler and P3's paged engine will also satisfy:
   ```python
   class Engine(Protocol):
       def generate(self, prompt: str, max_tokens: int) -> str: ...
   ```
   Do this now. Retrofitting it on Day 9 means three subtly different harnesses and no defensible apples-to-apples comparison.

4. **Correctness suite.** Fixed prompt set → run through HuggingFace `generate()` greedy → save resulting **token IDs** (not text — avoids tokenizer round-trip fuzz) as golden files. Test asserts exact list equality.

   Cover these cases explicitly:
   - short prompt / long prompt
   - stops naturally at EOS
   - stops at `max_tokens` instead
   - single-token output
   - prompt with repeated structure (matters if you pick up prefix caching as stretch)

   It passes trivially today (`generate()` vs `generate()`). That's fine. You're installing the tripwire before walking into the minefield.

5. **Load generator.** Configurable:
   - **concurrency** — number of in-flight requests (the knob you sweep)
   - **arrival pattern** — all-at-once vs. Poisson trickle; pick one, document it
   - **length distribution** — prompt lengths and output lengths

   The length distribution is the experiment, not a garnish. If every request generates exactly 100 tokens, static batching has no longest sequence to stall on and continuous batching will show ~1.2x. Use a wide, right-skewed output distribution (mass of short outputs, long tail) — that's what real traffic looks like.

6. **Metrics.** Record per request, aggregate at the end:

   | Metric | Question it answers |
   |---|---|
   | Throughput (tok/s, req/s) | How much can this box serve? |
   | TTFT | Does it feel alive to a user? |
   | End-to-end latency | Total wait |
   | p50 / p95 / p99 | How often does it feel broken? |

   Percentiles, not averages. 990 requests at 100ms + 10 requests at 5s averages to a healthy-looking 150ms while 1 in 100 users leaves. Sort the latencies; p99 is the 99th-percentile value.

7. **Record the baseline.** Commit the numbers as a dated file. Include: exact GPU, model, workload distribution, concurrency levels.

**Minimum shippable today:** naive server + goldens + a load generator that prints throughput and percentiles. **Cut first:** Poisson arrivals (all-at-once is defensible if documented), fancy result formatting.

### Watch out for

- Building the harness *after* the optimization. It gets unconsciously shaped to flatter what you built. Neutral instrument = built first.
- Forgetting to record hardware. A number without a GPU name is not reproducible.
- Spending the day making the harness beautiful. It needs to be *neutral*, not nice.

### Exit criteria

- [ ] `make bench` prints throughput, TTFT, p50/p95/p99 against the naive server
- [ ] `make test` passes
- [ ] Baseline numbers committed with hardware + workload documented

---

## Day 2 — P1: Manual decode loop + static batching

**Goal:** stop calling `generate()`. Own the loop. You cannot schedule a batch you don't control, and `generate()` gives you no seams to cut along.

### Steps

1. **Manual prefill.** Run the model over the full prompt in one forward pass with `use_cache=True`. Capture the returned KV cache. Sample the first token from the last position's logits.

2. **Manual decode loop.** Per step: feed **only the single new token** plus the existing cache, get logits, argmax (greedy), append token, append the new K/V to the cache, check termination (EOS or `max_tokens`).

   ```python
   logits, kv = model(input_ids=prompt_ids, use_cache=True)
   next_id = logits[:, -1].argmax(-1)
   while not done:
       logits, kv = model(input_ids=next_id, past_key_values=kv, use_cache=True)
       next_id = logits[:, -1].argmax(-1)
   ```

3. **Run the correctness suite.** This is the first real test. Token-for-token match with the Day 1 goldens, or you have a bug — most likely in position IDs, the attention mask, or off-by-one on which logit position you sample from.

4. **Static batching.** Batch N requests together. Left-pad prompts to equal length, build the attention mask correctly so padding is ignored, run them as a unit until **all** finish. Sequences that hit EOS early keep occupying their slot doing dead work.

5. **Measure the degradation deliberately.** Benchmark static batching twice:
   - **uniform** output lengths → looks great
   - **mixed** output lengths → the stall shows up

   The gap between those two runs is your project's motivation, measured rather than asserted. Record it.

**Minimum shippable today:** manual decode loop passing M1, plus the mixed-length static-batching number. **Cut first:** the uniform-length run (it's the flattering one, and P2 doesn't need it) — but it's cheap, so run it if the day holds.

### Watch out for

- **Padding + position IDs.** The single most common source of P1 correctness failures. Left-padded sequences need position IDs that account for the padding, and the attention mask must exclude pad tokens. Budget real debugging time here; this is the most likely place Day 2 becomes Day 3.
- Sampling from the wrong logit index when the batch has mixed prompt lengths.

### Exit criteria

- [ ] M1 holds against Day 1 goldens
- [ ] Static-batching numbers recorded for the mixed-length workload
- [ ] The mixed-length degradation is quantified (this is the number P2 is going to beat)

---

## Days 3–5 — P2: Continuous batching

**Goal:** ≥3x throughput over the **P1 static-batching** baseline (M2). The batch becomes mutable — evict finished sequences and admit waiting ones *between every step*, so the batch never drains.

**On the denominator.** Static batching, not naive P0. Against P0 you will clear 3x on Day 3 without trying — batch-1 decode leaves >99% of the GPU idle, so that comparison measures "batching works," not "continuous batching works." Static batching is the honest bar and the one an interviewer assumes you meant. Report both numbers; lead with the static one.

**Day split:** Day 3 = the three components and the step loop, correctness on a single sequence. Day 4 = ragged batching (the hard part) and M1 under mixed batches. Day 5 = server integration and benchmark. If Day 4's ragged masking is still broken at end of day, that's your signal to spend Day 14 here instead of on stretch goals.

### Steps

1. **Split into three components** (NFR2 — each must be readable in isolation, an interviewer reads one file):

   - **`Sequence`** — one request's state: prompt IDs, generated IDs, its KV cache, status (`WAITING` / `RUNNING` / `FINISHED`), `max_tokens`, arrival time.
   - **`Scheduler`** — owns the sequence pool and decides what runs each step.
   - **`Executor`** — given a set of sequences, runs exactly one forward pass. Knows nothing about scheduling.

   Do this split now even though it feels like overhead on a 2-week timeline. P3 and P4 both edit exactly one of these files; a monolith makes those phases slower, not faster.

2. **Implement the step loop:**
   ```
   1. Evict sequences that hit EOS or max_tokens; free their resources
   2. Admit waiting sequences if there's capacity
   3. Run one forward pass over the current running set
   4. Sample one token per sequence; append; check termination
   5. Repeat
   ```

3. **Handle prefill vs. decode.** New admissions need a prefill pass (many tokens, compute-bound); running sequences need a decode pass (one token, memory-bound). Simplest correct approach per the PRD: **prefill-priority with alternation** — run prefills for newly admitted sequences, then decode steps. Document the TTFT cost rather than optimizing it away; chunked prefill is named future work.

4. **Ragged batching.** Sequences now have different lengths *and* different KV cache lengths within one batch. This is the core mechanical difficulty of P2 and the single largest correctness risk in the sprint. Pad to the max in the current batch and mask correctly.

5. **Server integration.** The HTTP layer submits to the scheduler and awaits completion; a single background loop drives steps continuously.

6. **Benchmark.** Same harness, same workload distribution, against P0. Report throughput **at equal p99 latency** — winning throughput by making everyone wait longer is not a win.

**Minimum shippable by end of Day 5:** a mutable batch that passes M1 and beats static batching. If M2's 3x isn't there, report the real number and diagnose it in the writeup — a measured 2.4x with an explanation beats a fabricated 3x. **Cut first:** any prefill/decode scheduling sophistication beyond simple alternation.

### Watch out for

- **Correctness drift as batch composition changes.** A sequence's output must not depend on who else is in the batch with it. Add a test that runs the same prompt alone and in a crowded batch and asserts identical output. Write this test on Day 3, before you need it.
- Starvation: if you always admit the newest/shortest request, old ones never run. FIFO is fine and defensible.
- Async deadlocks in the server integration — the step loop must never block on request handlers.

### Exit criteria

- [ ] M1 still holds, including the alone-vs-crowded-batch test
- [ ] M2 met: ≥3x throughput over the P1 static-batching baseline on the mixed-length workload (or the real number, recorded with a diagnosis)
- [ ] The P0 comparison also recorded, labelled as such — it is a different, much larger number
- [ ] p99 latency reported alongside throughput, not instead of it
- [ ] Scheduler, sequence pool, and executor are separately readable

---

## Days 6–9 — P3: Paged KV cache

**Goal:** <10% memory waste (M3). Hardest phase, and it gets the largest block of the sprint. Steal virtual memory from operating systems.

| OS concept | Your equivalent |
|---|---|
| Page (4 KB) | Block (16 tokens) |
| Page table | Block table |
| Virtual address space | A sequence's logical token positions |
| Physical RAM | The preallocated KV tensor |
| Free page list | Block allocator free list |
| Shared pages + copy-on-write | Prefix sharing, refcounted (S1 — stretch) |
| Page fault → swap | Out of blocks → preempt (P4) |

**Day split:** Day 6 = waste measurement + preallocated tensor + allocator (with its unit tests). Day 7 = block tables wired into the engine. Days 8–9 = paged attention via PyTorch gather, and getting M1 green again. Days 8–9 are deliberately generous; this is where correctness bugs hide.

### Steps

1. **Measure the baseline waste first.** Instrument P2 to record, per request, `reserved_tokens` vs `used_tokens`. You need the "before" number or M3 is unmeasurable. Expect 60–80%. Half an hour of work; do not skip it to save time, because it deletes an entire resume bullet.

2. **Preallocate the KV tensor.** One big tensor, shaped `[num_blocks, block_size, num_heads, head_dim]` per layer (K and V). Size it to fill available GPU memory after weights and activations.

3. **Block allocator.** Free list of block indices, `allocate()` / `free()`, refcounts per block. Small, self-contained, heavily unit-tested — this is a data structure, test it like one. Refcounts go in now even though prefix sharing is stretch; retrofitting them later is worse.

4. **Block tables.** Each sequence holds a list mapping logical block number → physical block index. Allocate a new block only when the current one fills (every 16 tokens). Max waste per sequence is now 15 tokens instead of `max_tokens`.

5. **PyTorch gather fallback for attention — this is the shipping target, not a stepping stone.** Attention can no longer assume contiguity. Write the correct-but-slow version that walks the block table and gathers K/V. **Verify M1 here.** This unblocks P4 and satisfies M3, which is what the sprint is graded on.

6. **Measure.** Waste before vs. after, and the additional concurrent sequences the freed memory buys you (that's the "N" in the resume bullet).

**Deferred to stretch:** the Triton kernel (S3) and prefix caching (S1). Both are in the design doc as named future work with the reasoning above. Do not start either before P4 is done and M4 is met — a fast kernel with no overload story is the wrong trade.

**Minimum shippable by end of Day 9:** allocator + block tables + gather attention, M1 green, waste before/after recorded. **Cut first:** any attempt to make the gather fast.

### Watch out for

- **The gather path will be slow, and that's fine.** Report throughput honestly; note the kernel as the named optimization you scoped out. A working end-to-end system with a slow attention path beats a fast kernel with no system around it — that's the whole reason S3 is stretch.
- Refcount bugs → use-after-free or leaked blocks. Add a test that asserts the free list returns to full size after all sequences complete.
- Off-by-one on block boundaries, especially the partially-filled last block.
- If M3 slips past Day 9, take Day 14 for it and drop stretch entirely. M3 is a must-have; the kernel is not.

### Exit criteria

- [ ] M1 still holds with the gather-based paged attention
- [ ] M3 met: <10% waste vs. the measured P2 baseline
- [ ] Free-list-returns-to-full test passes
- [ ] Allocator has standalone unit tests

---

## Days 10–11 — P4: Preemption + admission control

**Goal:** survive 30 minutes at 10x capacity without OOM, crash, or a silently dropped request (M4). This is the phase that separates "I made it fast" from "I know what it does when it breaks."

**Day split:** Day 10 = preemption + bounded queue. Day 11 = the overload run, which is 30 minutes of wall clock you can spend writing README while it goes.

### Steps

1. **Preemption trigger.** A running sequence needs a new block; the free list is empty. Pick a victim (simplest defensible policy: most recently admitted — preserves progress on older work and avoids starvation).

2. **Recompute strategy.** Free the victim's blocks entirely, move it back to `WAITING`, keep its token IDs. On re-admission, re-prefill over prompt + already-generated tokens. Simple; wins for short sequences. This is the only strategy in the critical path.

   CPU swap (S2) is stretch. Say so in the design doc, and name the crossover you *expect* — swap wins once recomputation costs more than the PCIe round trip — so the reader sees you understand the tradeoff you didn't have time to measure.

3. **Bounded queue + shedding (FR7).** Fixed-size waiting queue. Beyond the bound, return **HTTP 503** immediately. An unbounded queue converts overload into unbounded latency, which is worse than an honest error. Expose queue depth.

4. **Guarantee: preempted ≠ dropped (FR6).** A preempted sequence is always re-admitted. The only requests that ever fail are ones explicitly rejected with a 503 at admission time.

5. **The overload run.** 10x capacity, 30 minutes sustained. Assert: no OOM, no crash, every request either completed or got a 503, memory usage flat over time. Log free-list size on a timer so "flat memory" is a chart, not a claim.

**Minimum shippable by end of Day 11:** recompute preemption + bounded queue + a clean 30-minute overload run with a memory chart. **Cut first:** nothing — this phase is already at minimum. If you're behind, take from Day 14.

### Watch out for

- **Preemption loops.** If the victim is immediately re-admitted and preempted again, no progress happens. Prevent it (e.g. don't re-admit a sequence preempted in the same step; consider a small backoff).
- **Memory leak under churn.** Flat memory over a 30-minute run is the actual test. If it isn't flat, the bug is almost certainly a block leak on the preemption path — the free-list test from Day 9 is where to start.

### Exit criteria

- [ ] M4 met: 30 min at 10x, no OOM/crash/silent drop, flat memory (charted)
- [ ] Recompute preemption implemented; swap named as scoped-out with expected crossover
- [ ] Bounded queue returns 503s; queue depth observable
- [ ] M1 still holds

---

## Days 12–13 — P5: Writeup + GPU benchmark day

**Goal:** M5 — a stranger with a rented GPU reproduces every published number from a clean checkout.

The README and design doc are first-class deliverables. The secondary user of this repo is a technical interviewer, and they will read one file for three minutes. On a 2-week sprint the temptation is to spend these days on more code; don't. Unwritten work is unhireable work.

**Day split:** Day 12 = the GPU run. Rent the box, run every benchmark across P0/P1/P2/P3 in one session, dump raw results to the repo. Day 13 = README, design doc, charts, `make bench` verification. Batching all GPU work into one day is also the cheapest way to hit the PRD's $40–60 budget.

### Steps

1. **Day 12: one clean GPU session.** Fresh checkout on the rented box. Run every phase's benchmark back to back on identical hardware and workload. Commit raw output, not just the summary. Anything you forget here means renting the box again.

2. **README.** Architecture comprehensible in under three minutes. Diagram, headline numbers with the GPU named, quickstart.

3. **Design doc.** Every non-obvious decision with **the rejected alternative named and why it lost**. The decision table in `docs/design.md` is the skeleton — expand each row with what you actually measured. Add a **"Scoped out and why"** section covering the four stretch items; that section is doing real work in an interview.

4. **Benchmark charts.** Throughput vs. concurrency across P0/P1/P2/P3. Latency percentile distributions. Memory waste before/after. Free-list size over the overload run.

5. **`make bench`.** One command, clean checkout, reproduces every published number. Pinned deps, fixed seeds, documented hardware. Verify this by actually re-cloning into a temp directory.

6. **Resume bullets.** Fill in the real X/Y/Z/N. **Any number `make bench` can't reproduce does not go on the resume** — you are pre-committing to only claiming what survives a follow-up question.

### Exit criteria

- [ ] M5 met: `make bench` reproduces all published numbers from a clean checkout
- [ ] Design doc names a rejected alternative for every key decision, plus a scoped-out section
- [ ] Charts committed
- [ ] Every resume number traceable to a benchmark run

---

## Day 14 — Buffer / stretch

Default assumption: **this day gets eaten**, most likely by P2's ragged batching or P3's block-boundary bugs. That's the plan working, not failing.

If you genuinely arrive here with everything green, pick up stretch work in this order — highest value per hour first:

1. **S4 — Streaming (SSE).** ~1 hour on top of an existing scheduler. Emits tokens as generated, improves perceived latency, demos well.
2. **S1 — Prefix caching.** Hash block contents; when a new sequence's prompt shares a prefix, point at the same physical blocks and bump the refcount. Copy-on-write on divergence. Free at refcount 0. The allocator already has refcounts from Day 6.
3. **S2 — CPU swap preemption.** Copy the victim's blocks to pinned host memory, free the GPU blocks, mark `SWAPPED`; copy back on re-admission. Then benchmark against recompute and find the crossover — the comparison is the deliverable, not the code.
4. **S3 — Triton paged attention kernel.** Each program instance handles one sequence's attention, reading block indices from the block table. Verify against the PyTorch fallback numerically on random inputs, *then* run M1 end-to-end. Only start this if you have a real day; a half-finished kernel is worth less than the fallback plus a written explanation.

Whatever you pick up, M1 runs before you commit, and anything unfinished at end of day gets reverted — not left half-merged with a broken test.

---

## Appendix: reference numbers

Useful for sanity-checking that your measurements are physically plausible.

**GPT-2 124M, bf16:**
- Weights: ~250 MB
- KV cache: ~36 KB/token (`2 × 12 layers × 768 dims × 2 bytes`)
- 1,024-token sequence: ~37 MB of cache
- 32 such sequences: ~1.2 GB — roughly **5x the model weights**

**RTX 4090 (approximate):**
- Memory bandwidth: ~1 TB/s
- bf16 compute: ~165 TFLOP/s
- Break-even arithmetic intensity: **~165 ops/byte**

**Key relationship:** for decode, arithmetic intensity ≈ batch size (model size cancels out). So batch < ~165 is memory-bound — adding sequences is nearly free. Above that, compute-bound and extra sequences cost real time. The KV cache's own memory traffic lowers this crossover, more so as sequences get longer.

**Single-token decode, batch 1, GPT-2 on a 4090:**
- Weight loading: ~250 µs
- Arithmetic: ~1.5 µs
- → **>99% of the GPU sits idle.** This is the waste the entire project exists to recover.
