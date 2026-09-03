"""Block table — P3 (Day 7). One sequence's logical->physical block mapping.

    Virtual address space  -> a sequence's logical token positions
    Page table             -> this file
    Free page list         -> block_allocator.py

The allocator is global and knows nothing about sequences; a block table is per-sequence
and knows nothing about who else is allocating. Splitting them is what keeps the
allocator's tests model-free and this file's arithmetic in one place.

**The one idea.** Logical token position `p` lives at::

    physical_block = blocks[p // block_size]
    offset         = p %  block_size

Nothing else in P3 needs to know how KV is laid out. That indirection is the entire
mechanism: a sequence's tokens are contiguous in *logical* space and scattered in
physical space, so growing by one token costs one slot rather than reserving
`max_seq_len` up front.

**What it buys.** A contiguous allocator cannot know a request's output length at
admission time, so it reserves the worst case every time — 84.4% waste, measured. Here a
sequence over-reserves only the unused tail of its final block, so waste per sequence is
bounded by `block_size - 1` tokens no matter how long `max_seq_len` is. That bound is
M3, and it is a property of this arithmetic rather than a tuning result.

No torch here either. See IMPLEMENTATION_GUIDE.md "Days 6-9".
"""

from __future__ import annotations

from inference_server.config import CONFIG
from inference_server.core.block_allocator import BlockAllocator


class BlockTable:
    """The blocks one sequence holds, in logical order.

    Grows on demand and never shrinks while the sequence lives; `free()` returns
    everything at once. Growing is the only operation that can fail, and it fails with
    MemoryError so P4 has a single well-defined place to hang preemption.
    """

    def __init__(self, allocator: BlockAllocator, block_size: int = CONFIG.block_size) -> None:
        self.allocator = allocator
        self.block_size = block_size
        self.blocks: list[int] = []

    # ------------------------------------------------------------------- capacity
    @property
    def capacity(self) -> int:
        """Token slots this sequence currently holds, used or not."""
        return len(self.blocks) * self.block_size

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    def blocks_for(self, num_tokens: int) -> int:
        """Ceiling division, written out because this is the off-by-one that hurts.

        17 tokens needs 2 blocks, not 1. 16 needs exactly 1, not 2 — a sequence sitting
        precisely on a boundary must not allocate a block it will never write to.
        """
        return -(-num_tokens // self.block_size)      # ceil without float rounding

    def ensure_capacity(self, num_tokens: int) -> list[int]:
        """Hold at least `num_tokens` slots. Returns the blocks newly allocated.

        Called before every forward pass with the count the pass will *end* at, so the
        growth happens ahead of the write rather than after it. Allocating one block per
        16 tokens means this is a no-op on 15 of every 16 decode steps.

        Raises MemoryError when the pool is empty. That is the P4 preemption trigger and
        the reason growth is a single call: there is exactly one line in the system where
        "out of KV memory" can be observed.
        """
        needed = self.blocks_for(num_tokens) - len(self.blocks)
        if needed <= 0:
            return []
        fresh = self.allocator.allocate_many(needed)
        self.blocks.extend(fresh)
        return fresh

    # -------------------------------------------------------------------- addressing
    def slot(self, position: int) -> tuple[int, int]:
        """Logical token position -> (physical block, offset within it)."""
        if not 0 <= position < self.capacity:
            raise IndexError(f"position {position} outside [0, {self.capacity})")
        return self.blocks[position // self.block_size], position % self.block_size

    def slots(self, num_tokens: int) -> list[int]:
        """Flat slot indices for this sequence's first `num_tokens` tokens.

        Flat means `physical_block * block_size + offset`, which is what indexing a
        pool tensor viewed as [num_blocks * block_size, heads, dim] wants. The Day 8
        gather builds its index tensor from exactly this list.

        Note it returns *num_tokens* entries, not `capacity` — the unused tail of the
        final block is allocated but holds nothing, and handing it out would let
        attention read uninitialized memory.
        """
        if num_tokens > self.capacity:
            raise IndexError(f"{num_tokens} tokens requested, capacity is {self.capacity}")
        return [
            self.blocks[p // self.block_size] * self.block_size + (p % self.block_size)
            for p in range(num_tokens)
        ]

    # ------------------------------------------------------------------------ release
    def free(self) -> None:
        """Return every block. Idempotent, because eviction paths are easy to run twice.

        Explicit rather than garbage-collected: nothing is reclaimed automatically out of
        a preallocated tensor, so a missed call here is a permanent leak that shows up as
        an upward-sloping memory graph 30 minutes into the M4 run.
        """
        if not self.blocks:
            return
        self.allocator.free_many(self.blocks)
        self.blocks = []

    # ------------------------------------------------------------------------ metrics
    def waste(self, num_tokens: int) -> int:
        """Slots held but unused — the unfilled tail of the last block. Bounded by
        block_size - 1, which is the whole M3 claim."""
        return self.capacity - num_tokens

    def __len__(self) -> int:
        return len(self.blocks)

    def __repr__(self) -> str:
        return f"BlockTable({len(self.blocks)} blocks, {self.capacity} slots)"
