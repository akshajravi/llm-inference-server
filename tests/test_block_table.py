"""P3 (Day 7) — block tables and the scheduler integration that uses them.

Two halves. The first is arithmetic: does a table hold exactly the blocks its token
count needs, and does it give them all back. The second drives a real Scheduler with a
fake executor, so the allocate-grow-free lifecycle is tested end to end without loading
a model — the same trick that makes the allocator's tests instant.

The test that matters most is `test_free_list_returns_to_full_after_a_full_workload`.
Every other failure here is loud; a block leak is silent until M4's 30-minute overload
run produces a memory graph that slopes upward with no obvious cause.
"""

from __future__ import annotations

import math

import pytest

from inference_server.config import CONFIG
from inference_server.core.block_allocator import BlockAllocator
from inference_server.core.block_table import BlockTable
from inference_server.core.scheduler import Scheduler
from inference_server.core.sequence import Sequence, Status

BS = 4          # a small block size makes boundary cases visible by hand


def table(num_blocks: int = 16, block_size: int = BS) -> BlockTable:
    return BlockTable(BlockAllocator(num_blocks), block_size)


# --------------------------------------------------------------------- the arithmetic
@pytest.mark.phase("P3")
@pytest.mark.parametrize(
    "tokens,blocks",
    [(0, 0), (1, 1), (3, 1), (4, 1), (5, 2), (8, 2), (9, 3)],
)
def test_block_count_is_ceiling_not_floor(tokens, blocks):
    """The boundary the guide warns about. 4 tokens with block_size 4 needs exactly one
    block — allocating a second for a sequence sitting on the boundary wastes a whole
    block; allocating one fewer means the next write has nowhere to go."""
    t = table()
    t.ensure_capacity(tokens)
    assert t.num_blocks == blocks


@pytest.mark.phase("P3")
def test_growth_is_incremental_and_idempotent():
    """Re-asking for capacity already held must allocate nothing. This runs on every
    decode step, so a table that grew unconditionally would burn a block per token and
    exhaust the pool in `block_size` steps."""
    t = table()
    assert len(t.ensure_capacity(4)) == 1
    assert t.ensure_capacity(4) == []
    assert t.ensure_capacity(1) == []          # shrinking is not a thing
    assert len(t.ensure_capacity(5)) == 1
    assert t.num_blocks == 2


@pytest.mark.phase("P3")
def test_slots_stop_at_the_live_tokens():
    """`slots` returns num_tokens entries, not `capacity`.

    The tail of the final block is allocated but never written. Handing those slots to
    attention would make it read uninitialized pool memory — garbage that is not even
    deterministic, so M1 would fail intermittently rather than consistently.
    """
    t = table()
    t.ensure_capacity(6)
    assert t.capacity == 8
    assert len(t.slots(6)) == 6
    with pytest.raises(IndexError):
        t.slots(9)


@pytest.mark.phase("P3")
def test_slot_addressing_matches_the_physical_layout():
    """flat slot == physical_block * block_size + offset, for every token."""
    t = table()
    t.ensure_capacity(10)
    for pos, flat in enumerate(t.slots(10)):
        block, offset = t.slot(pos)
        assert block == t.blocks[pos // BS]
        assert offset == pos % BS
        assert flat == block * BS + offset


@pytest.mark.phase("P3")
def test_distinct_sequences_never_share_a_slot():
    """Two tables drawing on one allocator must partition the pool, not overlap it."""
    alloc = BlockAllocator(16)
    a, b = BlockTable(alloc, BS), BlockTable(alloc, BS)
    a.ensure_capacity(10)
    b.ensure_capacity(10)
    assert not set(a.slots(10)) & set(b.slots(10))


# ------------------------------------------------------------------------- exhaustion
@pytest.mark.phase("P3")
def test_growth_raises_when_the_pool_is_empty():
    """Out of blocks is P4's preemption trigger, so it must be observable at exactly one
    call site rather than swallowed here."""
    t = table(num_blocks=2)
    t.ensure_capacity(8)
    with pytest.raises(MemoryError):
        t.ensure_capacity(9)
    assert t.num_blocks == 2, "a failed growth kept a partially allocated table"


# ------------------------------------------------------------------------------ free
@pytest.mark.phase("P3")
def test_free_returns_everything_and_runs_twice_safely():
    alloc = BlockAllocator(16)
    t = BlockTable(alloc, BS)
    t.ensure_capacity(10)
    assert alloc.num_free == 13
    t.free()
    assert alloc.num_free == 16
    t.free()                                    # eviction paths are easy to run twice
    assert alloc.num_free == 16


# ----------------------------------------------------------------------------- waste
@pytest.mark.phase("P3")
def test_waste_is_bounded_by_one_block_regardless_of_length():
    """M3 restated as a property. Contiguous reservation wastes
    max_seq_len - total_len, which grows with how wrong the guess was; paged waste is
    the unfilled tail of one block and cannot exceed block_size - 1 at any length."""
    t = table(num_blocks=64, block_size=BS)
    for tokens in range(1, 60):
        t.free()
        t.ensure_capacity(tokens)
        assert 0 <= t.waste(tokens) <= BS - 1


@pytest.mark.phase("P3")
def test_paged_waste_clears_the_m3_bar_on_realistic_lengths():
    """The M3 claim as arithmetic, before attention exists to measure it end to end.

    Lengths are the mixed workload's shape: short prompts, lognormal outputs. Against
    the same worst case P2 reserves (max_seq_len per request, measured at 84.4% waste),
    paging has to come in under 10%.
    """
    lengths = [16 + (i * 37) % 300 for i in range(200)]     # deterministic, 16..315
    block_size = CONFIG.block_size

    reserved_contiguous = len(lengths) * CONFIG.max_seq_len
    reserved_paged = sum(math.ceil(n / block_size) * block_size for n in lengths)
    used = sum(lengths)

    contiguous_waste = 1 - used / reserved_contiguous
    paged_waste = 1 - used / reserved_paged

    assert contiguous_waste > 0.6, "the baseline should be badly wasteful; check max_seq_len"
    assert paged_waste < 0.10, f"paged waste {paged_waste:.1%} misses M3"


# ------------------------------------------------------------- scheduler integration
class FakeExecutor:
    """Advances `num_cached` exactly as the real executor does, and nothing else.

    Standing in for the model here is the point: the block lifecycle is scheduler
    bookkeeping, and testing it against a real forward pass would make these tests slow
    and would hide a leak behind attention bugs on Days 8-9.
    """

    def __init__(self) -> None:
        self.steps = 0

    def execute(self, seqs: list[Sequence]) -> list[int]:
        self.steps += 1
        for s in seqs:
            # Prefill caches whatever was fed, not `prompt_len`: after a recompute
            # preemption (P4) the fed history is prompt + generated tokens.
            s.num_cached = len(s.next_input_ids) if s.needs_prefill else s.num_cached + 1
        return [7] * len(seqs)          # 7 is never the stop token below

    def reset(self) -> None:
        pass


def drive(scheduler: Scheduler, limit: int = 10_000) -> None:
    """Turn the crank until the pool empties, the way the engine's loop does."""
    for _ in range(limit):
        if not scheduler.has_work:
            return
        scheduler.step()
    raise AssertionError("scheduler did not drain")


@pytest.mark.phase("P3")
def test_scheduler_without_an_allocator_is_untouched():
    """P2 must be bit-for-bit unchanged. The allocator is opt-in, so a sequence run
    through the contiguous path never grows a block table at all."""
    sched = Scheduler(FakeExecutor(), eos_token_id=0)
    seq = Sequence(seq_id="a", prompt_token_ids=[1, 2, 3], max_tokens=5)
    sched.add(seq)
    drive(sched)
    assert seq.is_finished
    assert seq.block_table is None


@pytest.mark.phase("P3")
def test_blocks_track_the_token_count_across_a_whole_lifetime():
    alloc = BlockAllocator(256)
    sched = Scheduler(FakeExecutor(), eos_token_id=0, allocator=alloc)
    seq = Sequence(seq_id="a", prompt_token_ids=list(range(20)), max_tokens=30)
    sched.add(seq)

    seen: list[tuple[int, int]] = []
    while not seq.is_finished:
        sched.step()
        if seq.block_table is not None and seq.block_table.blocks:
            seen.append((seq.num_cached, seq.block_table.num_blocks))

    bs = CONFIG.block_size
    for cached, blocks in seen:
        assert blocks == math.ceil(cached / bs), f"{blocks} blocks holding {cached} tokens"


@pytest.mark.phase("P3")
def test_free_list_returns_to_full_after_a_full_workload():
    """The M4 canary at the integration level.

    Many sequences of mixed length, admitted and evicted through the real scheduler.
    Every block handed out must come back — the leak this catches is invisible in every
    other test here, and would otherwise surface as an upward-sloping memory graph half
    an hour into the overload run.
    """
    alloc = BlockAllocator(512)
    sched = Scheduler(FakeExecutor(), eos_token_id=0, allocator=alloc)

    for wave in range(5):
        for i in range(40):
            sched.add(
                Sequence(
                    seq_id=f"{wave}-{i}",
                    prompt_token_ids=list(range(3 + (i * 11) % 60)),
                    max_tokens=1 + (i * 7) % 25,
                )
            )
        drive(sched)
        assert alloc.num_free == 512, f"leaked {512 - alloc.num_free} blocks after wave {wave}"


@pytest.mark.phase("P3")
def test_a_sequence_finishing_mid_batch_releases_its_blocks_immediately():
    """The paged form of the P2 thesis: a slot comes back the step after it is free,
    and under paging 'slot' means physical memory, not just a batch row."""
    alloc = BlockAllocator(256)
    sched = Scheduler(FakeExecutor(), eos_token_id=0, allocator=alloc)

    short = Sequence(seq_id="short", prompt_token_ids=[1, 2], max_tokens=1)
    long = Sequence(seq_id="long", prompt_token_ids=[1, 2], max_tokens=40)
    sched.add(short)
    sched.add(long)

    while not short.is_finished:
        sched.step()
    held_with_short = 256 - alloc.num_free
    sched.step()                                  # the eviction at the top of this step
    assert 256 - alloc.num_free < held_with_short
    assert short.block_table is not None and short.block_table.num_blocks == 0
    assert long.status is Status.RUNNING
