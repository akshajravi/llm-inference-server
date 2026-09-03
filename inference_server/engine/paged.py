"""Paged KV cache engine — P3 (Days 6-9). Target: M3, <10% memory waste.

Continuous batching plus a block allocator: KV lives in one preallocated tensor cut
into 16-token blocks, and each sequence holds a block table. Max waste per sequence
becomes 15 tokens instead of max_tokens.

This file is deliberately small. Everything P2 built — the step loop, the futures, the
lock, the sync and async paths — is inherited unchanged from ContinuousEngine, because
paging changes *where KV lives*, not *how the batch is driven*. The three things that
differ are exactly the three things overridden here:

  1. Construction: a PagedKVPool is allocated once, a BlockAllocator hands out its
     indices, and the scheduler is built WITH that allocator so it grows and frees
     block tables around every step (core/scheduler.py `_grow`/`_evict`).
  2. The executor: PagedExecutor writes and reads KV through block tables
     (core/paged_executor.py) instead of a padded per-row cache.
  3. `_to_result`: `reserved_tokens` is now what the sequence *actually held* — its
     block table's capacity — rather than the max_seq_len worst case. That is the M3
     measurement, and it is a measurement rather than a formula because the table is
     still allocated when the result is built (eviction frees it at the top of the
     next step), so we read the real number off the real structure.

Ships with the PyTorch gather attention path (core/attention.py). The Triton kernel
(S3) is stretch — a working system with a slow attention path beats a fast kernel with
no system around it. Throughput of the gather path is reported as measured.

Exit criteria: M3 met; M1 holds; free-list-returns-to-full test passes.
See IMPLEMENTATION_GUIDE.md "Days 6-9".
"""

from __future__ import annotations

from inference_server.config import CONFIG
from inference_server.core.block_allocator import BlockAllocator
from inference_server.core.kv_pool import PagedKVPool
from inference_server.core.paged_executor import PagedExecutor
from inference_server.core.prefix_cache import PrefixCache
from inference_server.core.scheduler import Scheduler
from inference_server.core.sequence import Sequence
from inference_server.engine.base import Result
from inference_server.engine.continuous import ContinuousEngine
from inference_server.model import load


class PagedEngine(ContinuousEngine):
    name = "paged"

    def __init__(self) -> None:
        # Not super().__init__(): that would build a contiguous Executor and a
        # scheduler without an allocator, only to throw both away. The driver state
        # (locks, inbox, waiters, step thread) is ContinuousEngine's, set up by the
        # same helper so the two files cannot drift.
        self.model, self.tokenizer = load()

        self.pool = PagedKVPool.from_model(
            self.model,
            num_blocks=CONFIG.num_blocks,
            block_size=CONFIG.block_size,
            device=CONFIG.device,
        )
        # Printed once so every benchmark run records the pool a waste figure was
        # measured against; the number is meaningless without it.
        print(f"[paged] pool: {self.pool.describe()}", flush=True)

        self.allocator = BlockAllocator(self.pool.num_blocks)
        # S1: hash -> block index over the same allocator. Off (PREFIX_CACHING=0) means
        # every block is private and the engine is bit-for-bit the P3/P4 one.
        self.prefix_cache = PrefixCache(self.allocator, CONFIG.block_size) if CONFIG.prefix_caching else None
        self.executor = PagedExecutor(self.model, self.tokenizer, self.pool)
        self.scheduler = Scheduler(
            self.executor,
            self.tokenizer.eos_token_id,
            allocator=self.allocator,
            pool=self.pool,
            prefix_cache=self.prefix_cache,
        )

        self._init_driver()

    def _to_result(self, seq: Sequence) -> Result:
        """Same as ContinuousEngine's, except the two M3 fields are measured.

        `reserved_tokens` is the block table's capacity at the moment the sequence
        finished. It is read here rather than computed because the table is still held
        — the scheduler frees it at the top of the *next* step — and a measured number
        cannot drift from the allocator the way a formula can. The formula is the
        fallback only for a table that is already gone, which happens if a result is
        built after another step has run (a `generate()` caller racing the loop).

        One correction to the measurement, in the conservative direction. The last
        sampled token is chosen but never fed back through the model, so it owns no KV
        slot: the table holds `num_cached = total_len - 1` tokens. On the 1-in-16
        sequences where that count sits exactly on a block boundary, the measured table
        is one block short of holding `total_len`, and waste would come out negative.
        `reserved` is therefore rounded up to the capacity that holds every token the
        sequence owns — the same `used_tokens = total_len` the contiguous engines
        report against, so the two waste figures divide the same numerator.
        """
        bs = CONFIG.block_size
        holds_all = -(-seq.total_len // bs) * bs
        table = seq.block_table
        if table is not None and table.blocks:
            reserved = max(table.capacity, holds_all)
        else:
            reserved = holds_all
        return Result(
            request_id=seq.seq_id,
            token_ids=seq.output_token_ids,
            text=self.tokenizer.decode(seq.output_token_ids, skip_special_tokens=True),
            ttft_s=seq.ttft_s,
            latency_s=seq.latency_s,
            finish_reason=seq.finish_reason,
            prompt_len=seq.prompt_len,
            reserved_tokens=reserved,
            used_tokens=seq.total_len,
            wasted_steps=0,     # structurally zero, as in ContinuousEngine
        )
