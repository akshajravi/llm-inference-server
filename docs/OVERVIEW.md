# What This Project Is, In Plain Language

A companion to the README. The README tells you how to run things; this explains *what
problem the code solves and why it's built the way it is* — including the parts that
sound intimidating.

**Last updated:** Day 1 complete (2026-08-19)

---

## The one-sentence version

When you send a prompt to an AI model, some server decides how to run your request
alongside everyone else's on the same GPU. This project builds that server from scratch,
to learn how the fast ones actually work.

---

## Part 1: The problem

### GPUs are shaped wrong for this job

Think of a GPU as a factory with 10,000 workers standing at a very long assembly line.
It's phenomenally good at doing the *same operation* to *thousands of pieces of data* at
once. That's what it's for.

Now here's what generating text actually looks like:

```
"The capital of France is" → "Paris"
"The capital of France is Paris" → "."
"The capital of France is Paris." → <stop>
```

Each word depends on the one before it. **You cannot compute the third word until you
have the second.** It's inherently a one-at-a-time process.

So to produce a *single* token, the GPU has to load every parameter in the model out of
memory, run it, and produce one word. For GPT-2 (small, 124M parameters) on a decent GPU:

| Step | Time |
|---|---|
| Reading the model weights out of memory | ~250 microseconds |
| Actually doing the math on them | ~1.5 microseconds |

Read that again. **The GPU spends 99.4% of its time waiting for memory and 0.6% of it
computing.** Our factory of 10,000 workers has one person doing something and everyone
else standing around.

That idle 99% is what this entire project is about recovering.

### The obvious fix: do many at once

If you have to haul all the weights out of memory anyway, why apply them to just one
request? Load them once, apply them to 32 requests simultaneously. The expensive part
(the memory read) is paid once instead of 32 times.

This is **batching**, and it's close to free. The useful mental model: you're driving a
truck to the warehouse. The trip costs the same whether you bring back one box or a
hundred. So bring back a hundred.

---

## Part 2: Why naive batching isn't good enough

Here's where it gets interesting, and where this project starts being about *scheduling*
rather than math.

Naive ("static") batching groups 32 requests and runs them together until **all** of
them finish. Sounds fine. But real traffic looks like this:

```
Request A: "What's 2+2?"           → 4 tokens,   done immediately
Request B: "Hi"                     → 8 tokens,   done immediately
Request C: "Write me an essay on…"  → 500 tokens, still going
...
```

Most responses are short. A few are very long. This is called a **right-skewed
distribution**, and it's what real users actually produce.

With static batching, A and B finish in a fraction of a second — and then **sit in their
slots doing nothing for the next 490 steps** while C finishes. Their space is reserved.
Nobody else can use it.

> **The bus analogy:** a bus that won't leave the stop until every passenger has reached
> their destination. Someone going one block waits for the person going across the city.

### The fix: continuous batching

Make the batch **mutable**. Between *every single token*, the scheduler:

1. Kicks out any sequence that finished
2. Pulls a waiting request into the freed slot
3. Runs one step
4. Repeats

The bus now drops people off and picks new people up at every corner. The seats stay
full. This is the single biggest win in the project, and it's pure scheduling — no new
math, no new kernels, just deciding what runs when.

---

## Part 3: Then memory becomes the problem

Solving the scheduling problem exposes the next one.

### What the KV cache is

When a model generates token 100, it needs to "look back" at tokens 1–99. Recomputing
that from scratch every step would be brutally wasteful, so instead the model **caches**
its intermediate work for every token it's seen. That's the **KV cache** (keys and
values, terms from how attention works internally).

The cache grows by one entry per token, per layer. And it gets big — often **larger than
the model itself**:

| | Size |
|---|---|
| GPT-2 weights | ~250 MB |
| KV cache, 32 sequences × 1024 tokens | ~1.2 GB |

So the cache is roughly **5x the model weights**. Memory, not compute, is what limits
how many users you can serve at once.

### Why the naive approach wastes most of it

Here's the trap. When a request arrives, you don't know how long its answer will be.
Could be 5 tokens, could be 500. But if you're allocating one contiguous block of
memory, you have to reserve for the **worst case** right now — say 1024 tokens.

Then the request generates 40 tokens and finishes. The other 984 slots were reserved and
never touched. Nobody else could use them.

**We measured this on our own baseline: 84% of reserved cache memory is wasted.**

> **The parking analogy:** every car that pulls in gets a space big enough for a
> semi-truck, just in case. Most cars are Civics. You fill the lot with air.

### The fix: steal virtual memory from operating systems

This is the part worth understanding well, because it's a genuinely elegant idea and it's
the strongest thing in the project.

Operating systems solved this exact problem decades ago. Programs need memory but don't
know how much in advance. So the OS:

- Cuts physical memory into fixed-size **pages** (4 KB each)
- Gives each program a **page table** — a lookup saying "your chunk #3 lives at physical
  page #847"
- Hands out pages **only as needed**

The program *thinks* it has one long continuous stretch of memory. Physically it's
scattered all over. The page table maintains the illusion.

We do exactly this, with a direct one-to-one mapping:

| Operating system | This project |
|---|---|
| Page (4 KB) | Block (16 tokens) |
| Page table | Block table |
| Physical RAM | One big preallocated tensor |
| Free page list | Block allocator's free list |
| Shared pages + copy-on-write | Prefix sharing between requests |
| Page fault → swap to disk | Out of blocks → preempt a request |

Now a request that generates 40 tokens gets 3 blocks (48 tokens of space) instead of
1024. **Worst-case waste drops from "everything you didn't use" to at most 15 tokens.**

That's the difference between 84% waste and roughly 4%. Same GPU, ~5x more users.

---

## Part 4: What happens when you run out anyway

Eventually you fill the GPU. A running request needs one more block, and there are none.

You have three options, and only one is acceptable:

1. **Crash.** No.
2. **Reject the request.** But it's already half-finished — the user has been waiting.
3. **Preempt.** Pick a victim, take its memory away, put it back in the queue, resume it
   later.

Option 3 is what real systems do, and it's the same thing an OS does when RAM fills up:
evict something to disk and bring it back later.

Two ways to preempt, with a genuine tradeoff:

- **Recompute** — throw the victim's cache away entirely, keep just its text. Regenerate
  the cache when it resumes. Cheap to implement; fine if the victim hasn't generated much.
- **Swap** — copy the victim's cache to regular system RAM, copy it back later. Costs
  time moving data, but avoids redoing work. Better for long sequences.

Which wins depends on how much the victim has already generated. Measuring that crossover
is more interesting than either implementation.

There's also an important guarantee: **preempted is never dropped.** A request that gets
preempted always comes back. The only requests that fail are ones we explicitly reject at
the door with an HTTP 503 when the queue is full — because an unbounded queue doesn't
prevent failure, it just converts it into an unbounded wait, which is worse than an honest
error.

---

## Part 5: The thing that makes it real engineering

Everything above is about going faster. Here's the rule that constrains all of it:

> **An optimization that changes the output isn't an optimization. It's a bug that runs
> faster.**

It is *very* easy to write a paged attention kernel that's 10x faster and subtly wrong —
off by one token position, mishandling the last partial block. It'll produce
plausible-looking text. You won't notice for days.

So before writing any optimization, we built the tripwire:

1. Pick 7 fixed prompts, each targeting a specific failure mode
2. Run them through HuggingFace's reference implementation
3. Freeze the exact output — **as token ID numbers, not text**
4. Every phase must reproduce those numbers exactly, forever

Why token IDs rather than text? Because two different token sequences can decode to the
same string. Comparing text would hide exactly the off-by-one bugs we're hunting.

The 7 cases: short prompt, long prompt (catches position bugs), stops at EOS, stops at
the token limit, single-token output, repeated structure, whitespace edges.

---

## Part 6: What actually exists right now

### Built and working

**The baseline server (P0).** A deliberately naive server: one request at a time, no
batching, just calling HuggingFace's `generate()`. This sounds like a strange thing to
build on purpose. It has three jobs:

1. **It's the number we divide by.** "3x faster" is meaningless without a *than what*. And
   critically — building the baseline *first*, before knowing what we'll optimize, keeps us
   from unconsciously building a weak one to flatter our results later.
2. **It's the correctness reference.** It defines what the right answer is.
3. **It proves the measuring tools work** against something simple.

**The benchmark harness.** Load generator, workload definitions, metrics. It reports
percentiles (p50/p95/p99), not averages — because 990 fast requests plus 10 terrible ones
averages out to "looks fine" while 1 in 100 users has an awful experience. The workload
uses the right-skewed length distribution described in Part 2, since a uniform workload
would make continuous batching look pointless.

**The correctness suite.** 5 tests, currently passing.

**The skeleton for everything else.** Every file that later phases will need already
exists as a stub, with a docstring naming its phase, its exit criteria, and its most
likely bug. Day 6 starts by deleting a `raise NotImplementedError`, not by deciding where
things go.

### The measured baseline

Apple Silicon (MPS), GPT-2, 200 requests, mixed-length workload:

| Concurrency | Throughput | p99 latency | KV waste |
|---|---|---|---|
| 1 | 130.5 tok/s | 2.12 s | 84.4% |
| 4 | 128.8 tok/s | 1.74 s | 84.4% |
| 16 | 136.4 tok/s | 1.69 s | 84.4% |

**The flat line is the point.** Adding 16x the concurrency changes throughput by ~4%,
because the naive server handles requests strictly one at a time. That flatness is the
opportunity everything else exploits.

*(These are development numbers on a laptop. Published numbers come from a rented NVIDIA
GPU on Day 12.)*

### Three bugs that only appeared when we ran it

Worth recording, because each was invisible to code review:

1. **The EOS test was fake.** All 7 golden cases were finishing by hitting the token
   limit — GPT-2 almost never emits a natural stop token when decoding greedily. So the
   suite *looked* green while testing nothing about early termination, which every later
   phase reimplements by hand. Fixed by letting a test case override which token means
   "stop."

2. **The benchmark had no warmup.** The first request paid one-time setup costs (kernel
   compilation, memory allocator init) *inside the timed region*. This made a
   strictly-serialized engine appear to get 38% faster with more concurrency — impossible.
   It would have understated our baseline and inflated every future speedup.

3. **Memory waste measured as ~0%.** We were counting "reserved" as what each request
   *asked for*, but a contiguous allocator can't know that in advance — it reserves the
   worst case every time. Wrong accounting made the baseline look efficient and would have
   erased the entire memory-savings result.

All three were cheap to fix on Day 1 and would have been expensive or embarrassing later.

---

## Part 7: What's still ahead

| Days | Phase | What it adds |
|---|---|---|
| 2 | Manual decode loop | Stop calling `generate()`; own the loop, so there's something to schedule |
| 3–5 | Continuous batching | The mutable batch — the main throughput win |
| 6–9 | Paged KV cache | The block allocator and the memory win |
| 10–11 | Preemption | Survive overload without crashing or dropping requests |
| 12–13 | Benchmarks + writeup | Real GPU numbers, charts, design doc |
| 14 | Buffer | Absorb slippage, or stretch goals |

Deliberately scoped out: a custom Triton GPU kernel, prefix sharing between requests,
CPU-swap preemption, and streaming output. Each is genuinely interesting; none is
necessary to demonstrate the core ideas in two weeks. `docs/design.md` records why —
the reasoning behind what got cut is as much a part of the project as the code.

---

## Glossary

| Term | Meaning |
|---|---|
| **Token** | A chunk of text, roughly ¾ of a word. Models read and write these, not characters |
| **Prefill** | Processing your whole prompt at once, before generating anything |
| **Decode** | Generating one token at a time after prefill |
| **KV cache** | Saved intermediate work, so the model doesn't reprocess earlier tokens |
| **TTFT** | Time To First Token — how long until you see *something* |
| **p99 latency** | 99% of requests were at least this fast. Catches the bad tail that averages hide |
| **Throughput** | Total tokens per second across all users |
| **Greedy decoding** | Always pick the highest-probability next token. Deterministic, which is what makes correctness testing possible |
| **Preemption** | Taking resources away from a running request to give to another |
| **Block** | 16 tokens' worth of cache space — our unit of memory allocation |
