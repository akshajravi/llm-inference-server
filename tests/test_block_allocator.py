"""P3 (Days 6-9) — the allocator is a data structure, so test it like one.

These run without a model or a GPU, which makes them the fastest feedback loop in the
repo. The leak test is the one that matters: if the free list does not return to full
size, P4's 30-minute overload run (M4) will fail with a memory graph that slopes upward
and no obvious cause.
"""

from __future__ import annotations

import pytest

pytest.skip("BlockAllocator lands in P3 (Days 6-9)", allow_module_level=True)

from inference_server.core.block_allocator import BlockAllocator  # noqa: E402


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
