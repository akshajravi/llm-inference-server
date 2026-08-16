"""Block allocator — P3 (Days 6-9). Virtual memory, stolen from operating systems.

    Page (4KB)          -> Block (16 tokens)
    Page table          -> Block table (per sequence)
    Physical RAM        -> The one preallocated KV tensor
    Free page list      -> self.free_list
    Shared + COW pages  -> Prefix sharing, refcounted (S1, stretch)
    Page fault -> swap  -> Out of blocks -> preempt (P4)

Refcounts land now even though prefix sharing is stretch — retrofitting them later is
worse than carrying them unused.

This is a data structure. Test it like one: allocate/free/refcount unit tests, plus the
leak test that asserts the free list returns to full size after every sequence completes.

See IMPLEMENTATION_GUIDE.md "Days 6-9".
"""

from __future__ import annotations


class BlockAllocator:
    """Lands in P3. Free list of physical block indices with per-block refcounts."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("BlockAllocator lands in P3 (Days 6-9)")
