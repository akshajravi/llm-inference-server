"""P4 (Day 10) — preemption, FR6. Model-free.

A tiny pool (8 blocks of 4 tokens) and a handful of sequences that cannot all fit at
once, driven through the real Scheduler with a fake executor. The fake keeps a private
"KV cache" per sequence — the running sum of every token it has been fed — and samples
the next token from it. That is what makes these tests sharp: a recompute preemption
that re-fed only the prompt, or fed the history twice, or lost a generated token, would
change the sum and produce different tokens from the unpreempted reference run. The
output comparison against a huge pool is therefore the M1-style assertion for this
phase: preemption is invisible to the result, or it is a bug.

The property that matters most is the last one — the free list returns to full. A leak
on the preemption path is silent here and shows up as an upward-sloping memory chart 30
minutes into the M4 overload run.
"""

from __future__ import annotations

import pytest

from inference_server.config import CONFIG
from inference_server.core.block_allocator import BlockAllocator
from inference_server.core.scheduler import QueueFull, Scheduler, SequenceTooLong
from inference_server.core.sequence import Sequence, Status

BS = 4            # block size, monkeypatched into CONFIG so hand arithmetic stays legible
POOL = 8          # blocks — 32 token slots
HUGE = 4096       # a pool nothing here can exhaust: the reference run


class CachingFakeExecutor:
    """A forward pass whose output depends on the whole fed history, via a cache.

    `state[seq]` plays the role of the KV cache: it is rebuilt from scratch on prefill
    (from everything fed, which after a recompute preemption is prompt + generated) and
    extended by one token on decode. The sampled token is a function of it, so a
    scheduler that mishandled the cache's contents on the preemption path would sample
    differently from the unpreempted reference.
    """

    def __init__(self) -> None:
        self.state: dict[str, int] = {}
        self.steps = 0

    def execute(self, seqs: list[Sequence]) -> list[int]:
        self.steps += 1
        out = []
        for s in seqs:
            fed = s.next_input_ids
            if s.needs_prefill:
                self.state[s.seq_id] = sum(fed)
            else:
                self.state[s.seq_id] += fed[0]
            s.num_cached += len(fed)
            out.append(1 + (self.state[s.seq_id] * 7) % 49)     # never 0, the stop token
        return out

    def reset(self) -> None:
        pass


def drive(scheduler: Scheduler, limit: int = 10_000) -> None:
    for _ in range(limit):
        if not scheduler.has_work:
            return
        scheduler.step()
    raise AssertionError("scheduler did not drain — preemption livelock?")


def workload(n: int = 6) -> list[Sequence]:
    """Prompts of 5-9 tokens, 6-10 outputs: 3-5 blocks each, ~24 blocks in total for a
    pool of 8, so several preemptions are inevitable and no single sequence needs more
    than the pool holds."""
    return [
        Sequence(
            seq_id=f"s{i}",
            prompt_token_ids=[10 + (i * 3 + j) % 17 for j in range(5 + i % 5)],
            max_tokens=6 + (i * 5) % 5,
        )
        for i in range(n)
    ]


def run(seqs: list[Sequence], num_blocks: int) -> tuple[Scheduler, BlockAllocator]:
    alloc = BlockAllocator(num_blocks)
    sched = Scheduler(CachingFakeExecutor(), eos_token_id=0, allocator=alloc)
    for s in seqs:
        sched.add(s)
    drive(sched)
    return sched, alloc


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch):
    monkeypatch.setattr(CONFIG, "block_size", BS)
    monkeypatch.setattr(CONFIG, "preemption", "recompute")
    monkeypatch.setattr(CONFIG, "max_running", 32)


# ------------------------------------------------------------------- the FR6 contract
@pytest.mark.phase("P4")
def test_preemption_replaces_the_memory_error_and_changes_nothing():
    """The whole phase in one assertion: the small pool preempts instead of raising,
    every sequence finishes, and every output is identical to the run that never ran
    out of memory."""
    ref_seqs = workload()
    reference, _ = run(ref_seqs, HUGE)
    assert reference.stats()["preemptions"] == 0, "the reference run must not preempt"

    seqs = workload()
    sched, alloc = run(seqs, POOL)

    assert sched.stats()["preemptions"] > 0, "the pool was big enough — the test proves nothing"
    for got, want in zip(seqs, ref_seqs):
        assert got.is_finished and got.finish_reason == want.finish_reason
        assert got.output_token_ids == want.output_token_ids, got.seq_id
    assert alloc.num_free == POOL, f"leaked {POOL - alloc.num_free} blocks on the preemption path"
    assert not sched.preempted and not sched.waiting and not sched.running


@pytest.mark.phase("P4")
def test_victim_is_the_most_recently_admitted():
    """Stepped by hand so the admission order is observable. Whenever the preemption
    counter ticks, the sequence that left `running` must be the one that was at the
    back of it — never the oldest, whose progress is what the policy protects."""
    alloc = BlockAllocator(POOL)
    sched = Scheduler(CachingFakeExecutor(), eos_token_id=0, allocator=alloc)
    for s in workload():
        sched.add(s)

    victims_seen = 0
    for _ in range(10_000):
        if not sched.has_work:
            break
        before = list(sched.running)
        count = sched.num_preemptions
        sched.step()
        if sched.num_preemptions == count:
            continue
        evicted = [s for s in before if s not in sched.running and not s.is_finished]
        assert evicted, "preemption counted but nobody left the running set"
        # Victims leave youngest-first, so the evicted set is a suffix of `before`.
        assert evicted == before[-len(evicted):], (evicted, before)
        assert all(s.status is Status.WAITING for s in evicted)
        victims_seen += len(evicted)
    assert victims_seen > 0


@pytest.mark.phase("P4")
def test_preempted_is_not_dropped_and_not_readmitted_on_the_same_step():
    """FR6: the victim is parked in `preempted`, ahead of every new arrival, and holds
    no blocks while it waits. The guide's loop guard: it is not back in `running` at
    the end of the step it was evicted on."""
    alloc = BlockAllocator(POOL)
    sched = Scheduler(CachingFakeExecutor(), eos_token_id=0, allocator=alloc)
    seqs = workload()
    for s in seqs:
        sched.add(s)

    while sched.num_preemptions == 0:
        assert sched.has_work
        sched.step()

    victim = sched.preempted[0]
    assert victim.status is Status.WAITING
    assert victim.num_cached == 0                       # recompute: cache is gone
    assert victim.output_token_ids or victim.needs_prefill
    assert victim.block_table.num_blocks == 0           # ...and so are its blocks
    assert victim not in sched.running and victim not in sched.waiting
    assert victim.preempted_step == sched.step_count

    drive(sched)
    assert all(s.is_finished for s in seqs), "a preempted sequence was dropped"


@pytest.mark.phase("P4")
def test_free_list_returns_to_full_under_sustained_churn():
    """Waves of work against a pool that forces preemption on every wave. Any block lost
    on the preempt/re-admit path would accumulate across waves."""
    alloc = BlockAllocator(POOL)
    sched = Scheduler(CachingFakeExecutor(), eos_token_id=0, allocator=alloc)
    for wave in range(8):
        for s in workload(5):
            s.seq_id = f"{wave}-{s.seq_id}"
            sched.add(s)
        drive(sched)
        assert alloc.num_free == POOL, f"leaked {POOL - alloc.num_free} blocks by wave {wave}"
    assert sched.stats()["preemptions"] > 0
    assert sched.stats()["completed"] == 40


# ------------------------------------------------------------------- loop prevention
@pytest.mark.phase("P4")
def test_no_livelock_when_every_sequence_needs_most_of_the_pool():
    """Three sequences that each need 6 of 8 blocks can only run one at a time. The
    oldest is never preempted while a younger one exists, so it always finishes, so
    the run always drains — that is the termination argument, tested."""
    seqs = [Sequence(seq_id=f"big{i}", prompt_token_ids=[i] * 12, max_tokens=12) for i in range(3)]
    sched, alloc = run(seqs, POOL)
    assert all(s.is_finished for s in seqs)
    assert alloc.num_free == POOL


@pytest.mark.phase("P4")
def test_no_livelock_when_the_victim_is_the_sequence_that_needed_the_block():
    """Prompts that exactly fill their blocks need a fresh one on the very first decode.
    With two of them in a 4-block pool the youngest evicts itself; it must still get
    back in and finish once the older one is done."""
    seqs = [Sequence(seq_id=f"edge{i}", prompt_token_ids=[i] * 8, max_tokens=5) for i in range(2)]
    sched, alloc = run(seqs, num_blocks=4)
    assert all(s.is_finished for s in seqs)
    assert sched.stats()["preemptions"] > 0
    assert alloc.num_free == 4


@pytest.mark.phase("P4")
def test_a_sequence_that_can_never_fit_is_rejected_at_add():
    """32 slots. A 40-token prompt can never prefill; 20 + 20 tokens can never finish.
    Either would be admitted, evicted as its own victim, re-admitted, forever — so they
    are refused at the door. 20 + 13 = 33 tokens is fine: the last sampled token is
    never cached, so the sequence peaks at exactly 32."""
    sched = Scheduler(CachingFakeExecutor(), eos_token_id=0, allocator=BlockAllocator(POOL))
    with pytest.raises(SequenceTooLong):
        sched.add(Sequence(seq_id="prompt", prompt_token_ids=[1] * 40, max_tokens=1))
    with pytest.raises(SequenceTooLong):
        sched.add(Sequence(seq_id="total", prompt_token_ids=[1] * 20, max_tokens=20))
    sched.add(Sequence(seq_id="fits", prompt_token_ids=[1] * 20, max_tokens=13))
    assert sched.queue_depth == 1
    drive(sched)


@pytest.mark.phase("P4")
def test_rejections_are_not_queue_full():
    """The two refusals mean different things to the client (503 retry vs 422 never),
    so they must be distinct types."""
    assert not issubclass(SequenceTooLong, QueueFull)
    assert not issubclass(QueueFull, SequenceTooLong)


# ------------------------------------------------------------------------- counters
@pytest.mark.phase("P4")
def test_stats_reports_exactly_the_health_fields():
    sched, alloc = run(workload(), POOL)
    stats = sched.stats()
    assert set(stats) == {
        "queue_depth", "num_running", "num_waiting", "num_swapped",
        "free_blocks", "num_blocks", "preemptions", "swaps", "completed",
    }
    assert stats["free_blocks"] == stats["num_blocks"] == POOL
    assert stats["queue_depth"] == stats["num_running"] == stats["num_waiting"] == 0
    assert stats["swaps"] == 0                          # recompute never swaps
    assert stats["preemptions"] > 0
    assert stats["completed"] == 6


@pytest.mark.phase("P4")
def test_without_an_allocator_nothing_here_exists():
    """P2 path: no allocator, no memory check, no preemption, stats read as zeros."""
    sched = Scheduler(CachingFakeExecutor(), eos_token_id=0)
    seqs = workload()
    for s in seqs:
        sched.add(s)
    drive(sched)
    assert all(s.is_finished and s.block_table is None for s in seqs)
    stats = sched.stats()
    assert stats["preemptions"] == stats["free_blocks"] == stats["num_blocks"] == 0


# ------------------------------------------------------- M1 under preemption, real model
@pytest.mark.phase("P4")
@pytest.mark.parametrize("strategy", ["recompute", "swap"])
def test_goldens_survive_preemption_on_the_paged_engine(monkeypatch, goldens, strategy):
    """The model-backed version of the first test: a pool of 12 blocks (192 slots, when
    the goldens together need ~350) forces several preemptions while every golden runs
    at once through the real paged engine, and every output must still match. This is
    the assertion that preemption is invisible to the user under both strategies."""
    monkeypatch.setattr(CONFIG, "block_size", 16)       # the real block size, not BS
    monkeypatch.setattr(CONFIG, "num_blocks", 12)
    monkeypatch.setattr(CONFIG, "preemption", strategy)
    from inference_server.engine.paged import PagedEngine  # noqa: PLC0415

    engine = PagedEngine()
    try:
        seqs = []
        for case_id, case in goldens["cases"].items():
            seq = Sequence(
                seq_id=case_id,
                prompt_token_ids=engine.tokenizer(case["prompt"]).input_ids,
                max_tokens=case["max_tokens"],
                eos_token_id=case.get("eos_token_id"),
            )
            seqs.append(seq)
            engine.scheduler.add(seq)
        drive(engine.scheduler, limit=2_000)
    finally:
        engine.shutdown()

    stats = engine.stats()
    assert stats["preemptions"] > 0, "12 blocks did not force a preemption; shrink the pool"
    assert stats["swaps"] == (stats["preemptions"] if strategy == "swap" else 0)
    assert stats["free_blocks"] == stats["num_blocks"] == 12
    for seq in seqs:
        assert seq.output_token_ids == goldens["cases"][seq.seq_id]["token_ids"], seq.seq_id
