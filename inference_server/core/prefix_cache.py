"""Prefix cache — S1 (Day 14). Shared KV blocks across requests with equal prompt prefixes.

    Shared pages + copy-on-write  ->  this file (FR5)

Two requests that start with the same 128-token system prompt compute the same K/V for
those 128 tokens. With paging that work is already in the pool, in blocks someone else
holds; the only thing missing is a way to *find* it. This is that index:

    content hash of a FULL block  ->  physical block index

**The hash is a chain.** Block `i`'s hash covers every token from position 0 through
the end of block `i` (it is `H(hash(block i-1), tokens of block i)`), so two blocks
with equal hashes hold KV computed over equal full prefixes — which is the only case
where the KV is interchangeable, because attention at position p depends on every
token before p. A flat per-block hash would let "the quick brown fox" at positions
16-31 of one prompt alias the same words at 16-31 of a different prompt, and the KV
would be silently wrong.

**Only full blocks are shared.** A partially filled block is still being written by
its owner, so nobody else may point at it; the partial last block of a prompt is
always private. Consequence: a sequence that matched `n` blocks starts writing at
position `n * block_size` — a block boundary, in a block nobody else holds — so
copy-on-write is never triggered by this policy. `BlockTable.ensure_private` exists
anyway as the guard for a future policy that shares partial blocks.

**Entries live as long as the block does.** A cache entry pointing at a block that has
been returned to the free list would be a use-after-free — the allocator hands that
index to someone else, who overwrites it, and a later match reads their KV as ours.
So the allocator tells the cache the moment a block's refcount hits zero and the
entry goes with it (`BlockAllocator.on_release`). The cost is that the cache does not
outlive its last referent: a prefix is only reusable while some sequence still holds
it. Keeping evicted-but-intact blocks around (vLLM's approach: the free list becomes an
LRU of still-hashed blocks) is the named next step; it needs the allocator to know
about hashes, and this sprint keeps the allocator a free list of integers.

**Torch-free, model-free.** Hashes are over token IDs; the copy that COW needs is done
by whoever owns the pool (the executor). This file only ever handles integers, which
is what lets the tests below run in milliseconds.

See PRD FR5/S1 and IMPLEMENTATION_GUIDE.md "Day 14".
"""

from __future__ import annotations

import hashlib
from array import array

from inference_server.core.block_allocator import BlockAllocator
from inference_server.core.block_table import BlockTable

#: 128-bit digests. Python's built-in `hash()` of an int tuple is 64-bit and, more to
#: the point, a collision here is not a slowdown but silently wrong tokens, so the
#: digest is a real one. Cost: one blake2b over ~64 bytes per full block, once.
_DIGEST_BYTES = 16


def block_hash(prev: bytes | None, tokens: list[int]) -> bytes:
    """One link of the chain: H(previous link, this block's tokens)."""
    h = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    h.update(prev if prev is not None else b"\x00" * _DIGEST_BYTES)
    h.update(array("q", tokens).tobytes())
    return h.digest()


def hash_chain(tokens: list[int], block_size: int, prefix: list[bytes] = ()) -> list[bytes]:
    """Hashes of every FULL block of `tokens`, in order, continuing from `prefix`
    (already-known hashes for the first `len(prefix)` blocks, so a decode step costs
    one link rather than re-hashing the whole history)."""
    chain = list(prefix)
    prev = chain[-1] if chain else None
    for i in range(len(chain), len(tokens) // block_size):
        prev = block_hash(prev, tokens[i * block_size : (i + 1) * block_size])
        chain.append(prev)
    return chain


class PrefixCache:
    """hash -> physical block, plus the counters the design notes report."""

    def __init__(self, allocator: BlockAllocator, block_size: int) -> None:
        self.allocator = allocator
        self.block_size = block_size
        self._by_hash: dict[bytes, int] = {}
        self._by_block: dict[int, bytes] = {}
        # Cumulative. `hits` and `misses` count full prompt blocks at admission time:
        # served from the cache vs. had to be prefilled. `blocks_shared` == hits, kept
        # under its own name because it is the number the memory-saved figure divides.
        self.hits = 0
        self.misses = 0
        self.blocks_shared = 0
        allocator.on_release = self._forget

    # ---------------------------------------------------------------------- lookup
    @property
    def entries(self) -> int:
        return len(self._by_hash)

    def match(self, tokens: list[int]) -> tuple[list[int], list[bytes]]:
        """Pure lookup: the longest cached prefix of `tokens`, as (blocks, hashes).

        Capped at `len(tokens) - 1` rounded down to a block boundary. The model must
        run over at least the last token to produce a logit for it, so a prompt that
        is cached in its entirety still feeds its final block — the whole final block,
        because a partial one cannot be shared and the write must start on a boundary.
        No side effects: admission may look and then decline for lack of memory, and a
        refcount taken by a sequence that is not admitted would be a leak.
        """
        limit = (len(tokens) - 1) // self.block_size
        blocks: list[int] = []
        hashes: list[bytes] = []
        prev: bytes | None = None
        for i in range(max(0, limit)):
            prev = block_hash(prev, tokens[i * self.block_size : (i + 1) * self.block_size])
            block = self._by_hash.get(prev)
            if block is None:
                break
            blocks.append(block)
            hashes.append(prev)
        return blocks, hashes

    def claim(self, table: BlockTable, blocks: list[int], hashes: list[bytes], full_blocks: int) -> int:
        """Point `table` at the matched blocks (one refcount each) and count the outcome.
        `full_blocks` is how many full blocks the prompt had in total, for the miss
        count. Returns the number of tokens now covered by the table."""
        assert not table.blocks, "prefix match applies to an empty table only"
        table.adopt(blocks, hashes)
        self.hits += len(blocks)
        self.blocks_shared += len(blocks)
        self.misses += max(0, full_blocks - len(blocks))
        return len(blocks) * self.block_size

    # ---------------------------------------------------------------- registration
    def register(self, table: BlockTable, tokens: list[int], num_cached: int) -> int:
        """After a pass: hash the table's newly completed full blocks and publish any
        whose hash is not yet known. Returns how many were published.

        `tokens` is the sequence's whole history (prompt + generated); only the first
        `num_cached` of them have KV in the pool, so only blocks entirely below that
        line are full. Generated-token blocks are registered too: it is the same
        arithmetic, and it is what lets a multi-turn follow-up (prompt = previous
        prompt + previous answer + new turn) reuse the earlier turn's KV.

        Two sequences prefilled in the same pass with the same prefix both compute it
        privately (neither could match the other before the pass ran); the first to
        register wins and the second keeps its private copy, unshared. If the winner
        finishes first its entries vanish and the second publishes its copy on its next
        step, so the cache heals rather than staying empty. That is what
        `table.published` is for: it is the lowest block whose hash is *not* in the
        cache, so the common case (everything below `full` is in) is one comparison,
        and the duplicate case re-checks only the blocks that were passed over.
        """
        full = min(num_cached, len(tokens)) // self.block_size
        if full <= table.published:
            return 0                                # nothing new crossed a boundary
        if full > len(table.hashes):
            table.hashes = hash_chain(tokens[: full * self.block_size], self.block_size, table.hashes)
        published = 0
        cursor_done = False
        for i in range(table.published, full):
            h = table.hashes[i]
            block = table.blocks[i]
            if h in self._by_hash:
                pass                                # ours, or an identical one someone else holds
            elif block not in self._by_block:
                self._by_hash[h] = block
                self._by_block[block] = h
                published += 1
            else:                                   # cannot happen: one block, one content
                raise RuntimeError(f"block {block} registered under two hashes")
            if not cursor_done and h in self._by_hash and self._by_hash[h] == block:
                table.published = i + 1
            elif not cursor_done:
                # Another block holds this content; stop advancing so it is rechecked
                # once that block is gone. Later blocks are still published now.
                cursor_done = True
        return published

    def _forget(self, block: int) -> None:
        """Allocator callback: refcount reached zero, the contents are about to be
        someone else's."""
        h = self._by_block.pop(block, None)
        if h is not None:
            del self._by_hash[h]

    # --------------------------------------------------------------------- metrics
    def stats(self) -> dict:
        return {
            "prefix_hits": self.hits,
            "prefix_misses": self.misses,
            "prefix_blocks_shared": self.blocks_shared,
            "prefix_entries": self.entries,
            "prefix_tokens_saved": self.blocks_shared * self.block_size,
        }

    def __repr__(self) -> str:
        return f"PrefixCache({self.entries} entries, {self.hits} hits, {self.misses} misses)"
