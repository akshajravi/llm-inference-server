"""P3 (Days 6-9) — the allocator is a data structure, so test it like one.

These run without a model or a GPU, which makes them the fastest feedback loop in the
repo. The leak test is the one that matters: if the free list does not return to full
size, P4's 30-minute overload run (M4) will fail with a memory graph that slopes upward
and no obvious cause.
"""

from __future__ import annotations

import pytest

from inference_server.core.block_allocator import BlockAllocator


@pytest.mark.phase("P3")
def test_allocate_then_free_returns_blocks():
    alloc = BlockAllocator(num_blocks=8)
    blocks = [alloc.allocate() for _ in range(8)]
    assert len(set(blocks)) == 8, "allocator handed out a duplicate block"
    assert alloc.num_free == 0
    for b in blocks:
        alloc.free(b)
    assert alloc.num_free == 8


@pytest.mark.phase("P3")
def test_exhaustion_signals_rather_than_corrupts():
    """Out of blocks is a normal condition (it triggers P4 preemption), not a crash."""
    alloc = BlockAllocator(num_blocks=1)
    alloc.allocate()
    with pytest.raises(MemoryError):
        alloc.allocate()


@pytest.mark.phase("P3")
def test_refcount_defers_free():
    """S1 groundwork: a shared block survives until the last referent releases it."""
    alloc = BlockAllocator(num_blocks=4)
    block = alloc.allocate()
    alloc.incref(block)
    alloc.free(block)
    assert alloc.num_free == 3, "freed a block that another sequence still references"
    alloc.free(block)
    assert alloc.num_free == 4


@pytest.mark.phase("P3")
def test_no_leak_after_full_churn():
    """The M4 canary. Run many alloc/free cycles; the free list must come back whole."""
    alloc = BlockAllocator(num_blocks=16)
    for _ in range(100):
        held = [alloc.allocate() for _ in range(16)]
        for b in held:
            alloc.free(b)
    assert alloc.num_free == 16


@pytest.mark.phase("P3")
def test_allocate_many_is_all_or_nothing():
    """The partial-allocation leak, pinned.

    A prefill asks for every block it needs in one call. If that call could take four
    blocks, then discover it needs a fifth and raise, the caller would abandon the
    request while four blocks stayed held with nothing pointing at them — a leak that
    only shows up 30 minutes into the M4 overload run.
    """
    alloc = BlockAllocator(num_blocks=8)
    alloc.allocate_many(5)
    with pytest.raises(MemoryError):
        alloc.allocate_many(4)
    assert alloc.num_free == 3, "a failed allocate_many kept blocks it never returned"


@pytest.mark.phase("P3")
def test_double_free_raises_rather_than_duplicating():
    """The bug this guard exists to prevent is not the crash — it is the silence.

    Pushing an index onto the free list twice lets the allocator hand one physical
    block to two live sequences. They then overwrite each other's KV, and the symptom
    is wrong tokens in an unrelated request many steps later.
    """
    alloc = BlockAllocator(num_blocks=4)
    block = alloc.allocate()
    alloc.free(block)
    with pytest.raises(ValueError):
        alloc.free(block)
    assert alloc.num_free == 4


@pytest.mark.phase("P3")
def test_incref_on_a_free_block_is_use_after_free():
    alloc = BlockAllocator(num_blocks=4)
    block = alloc.allocate()
    alloc.free(block)
    with pytest.raises(ValueError):
        alloc.incref(block)


@pytest.mark.phase("P3")
def test_out_of_range_block_is_rejected():
    alloc = BlockAllocator(num_blocks=4)
    for bad in (-1, 4, 99):
        with pytest.raises(IndexError):
            alloc.free(bad)


@pytest.mark.phase("P3")
def test_free_list_and_refcounts_never_disagree():
    """Invariant 1, checked directly: a block is free iff its refcount is zero.

    Every other test here asserts a consequence of this; this one asserts the thing
    itself, across a churn of allocate / incref / free in mixed order.
    """
    alloc = BlockAllocator(num_blocks=12)
    held: list[int] = []
    for round_ in range(30):
        while alloc.num_free and len(held) < 12:
            held.append(alloc.allocate())
        for b in held[::3]:                       # some blocks get a second referent
            alloc.incref(b)
        for b in list(held):
            while alloc.refcount(b):
                alloc.free(b)
            held.remove(b)
        free_by_refcount = sum(1 for b in range(12) if alloc.refcount(b) == 0)
        assert alloc.num_free == free_by_refcount, f"diverged on round {round_}"
    assert alloc.num_free == 12
