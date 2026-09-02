"""Block allocator — P3 (Days 6-9). Virtual memory, stolen from operating systems.

    Page (4KB)          -> Block (16 tokens)
    Page table          -> Block table (per sequence)
    Physical RAM        -> The one preallocated KV tensor
    Free page list      -> self._free
    Shared + COW pages  -> Prefix sharing, refcounted (S1, stretch)
    Page fault -> swap  -> Out of blocks -> preempt (P4)

Refcounts land now even though prefix sharing is stretch — retrofitting them later is
worse than carrying them unused.

**No torch in this file.** The allocator hands out integers; kv_pool.py owns the tensors
those integers index into. That split is what lets these tests run in milliseconds with
no model and no device, which matters because this is the data structure whose bugs
show up four days later as a memory graph that slopes upward.

See IMPLEMENTATION_GUIDE.md "Days 6-9".
"""

from __future__ import annotations


class BlockAllocator:
    """A free list of physical block indices with per-block refcounts.

    Two invariants hold at all times, and every method is written to preserve them:

    1. A block index is in `_free` **iff** its refcount is 0. No block is ever both
       handed out and available; none is ever lost from both.
    2. `num_free + sum(held)` == `num_blocks`. This is what the leak test asserts, and
       it is why `allocate_many` checks capacity up front rather than failing midway.
    """

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        self.num_blocks = num_blocks
        # LIFO. A just-freed block is the most likely to still be resident in cache, and
        # a deterministic order keeps allocation traces comparable between runs.
        self._free: list[int] = list(reversed(range(num_blocks)))
        self._refcount: list[int] = [0] * num_blocks

    # ------------------------------------------------------------------ inspection
    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_allocated(self) -> int:
        return self.num_blocks - self.num_free

    def refcount(self, block: int) -> int:
        return self._refcount[self._check(block)]

    # ------------------------------------------------------------------ allocation
    def allocate(self) -> int:
        """Take one block. Refcount starts at 1 — the caller is the first referent.

        Raises MemoryError when the pool is empty. That is a *normal* condition, not a
        crash: P4 catches it and preempts a victim to make room. It is an exception
        rather than a `None` return so that a caller who forgets to check cannot
        silently write into block index `None`.
        """
        if not self._free:
            raise MemoryError(f"block pool exhausted ({self.num_blocks} blocks, all held)")
        block = self._free.pop()
        self._refcount[block] = 1
        return block

    def allocate_many(self, count: int) -> list[int]:
        """All-or-nothing. A sequence's prefill needs ceil(prompt_len / block_size)
        blocks at once, and a partial success is the worst outcome available: the
        caller sees MemoryError and abandons the request, while the blocks it already
        took stay marked held forever. Capacity is therefore checked *before* the first
        block leaves the free list, so the failure path never has anything to undo.
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if count > self.num_free:
            raise MemoryError(f"need {count} blocks, {self.num_free} free")
        return [self.allocate() for _ in range(count)]

    # ---------------------------------------------------------------------- release
    def incref(self, block: int) -> int:
        """Register another referent. S1 groundwork: two sequences sharing a prompt
        prefix point at the same physical block, and neither may free it alone."""
        self._check(block)
        if self._refcount[block] == 0:
            raise ValueError(f"incref on free block {block} — use-after-free")
        self._refcount[block] += 1
        return self._refcount[block]

    def free(self, block: int) -> int:
        """Drop one reference. The block returns to the free list only at zero.

        A double free raises rather than pushing the index twice, which would let the
        allocator hand the same physical block to two live sequences — silent KV
        corruption that surfaces as wrong tokens, nowhere near the actual bug.
        """
        self._check(block)
        if self._refcount[block] == 0:
            raise ValueError(f"double free of block {block}")
        self._refcount[block] -= 1
        if self._refcount[block] == 0:
            self._free.append(block)
        return self._refcount[block]

    def free_many(self, blocks: list[int]) -> None:
        for block in blocks:
            self.free(block)

    # ------------------------------------------------------------------- internals
    def _check(self, block: int) -> int:
        if not isinstance(block, int) or not 0 <= block < self.num_blocks:
            raise IndexError(f"block {block!r} outside [0, {self.num_blocks})")
        return block

    def __repr__(self) -> str:
        return f"BlockAllocator({self.num_allocated}/{self.num_blocks} held)"
