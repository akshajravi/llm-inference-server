"""P4 (S2) — swap preemption. Model-free, but not pool-free.

Two layers. `PagedKVPool.swap_out` / `swap_in` on a cpu pool with toy dims: the bytes
that leave must be the bytes that come back, into *different* blocks. Then the
scheduler under PREEMPTION=swap, with an executor that reads and writes its "KV"
straight into the pool at the sequence's block-table slots — exactly the property the
real paged executor relies on (no per-batch state; the pool is the truth). If the
swap path restored the wrong blocks, the wrong layer, or stale contents, the sampled
tokens would diverge from the never-preempted reference.
"""

from __future__ import annotations

import pytest
import torch

from inference_server.config import CONFIG
from inference_server.core.block_allocator import BlockAllocator
from inference_server.core.kv_pool import HostBlocks, ModelDims, PagedKVPool
from inference_server.core.scheduler import Scheduler
from inference_server.core.sequence import Sequence, Status

BS = 4
POOL = 8
HUGE = 512
DIMS = ModelDims(num_layers=2, num_kv_heads=2, head_dim=4, dtype=torch.float32)


def make_pool(num_blocks: int) -> PagedKVPool:
    return PagedKVPool(DIMS, num_blocks=num_blocks, block_size=BS, device="cpu")


# ----------------------------------------------------------------------- the copies
@pytest.mark.phase("P4")
def test_swap_round_trip_lands_the_same_bytes_in_different_blocks():
    pool = make_pool(POOL)
    for t in pool.k + pool.v:
        t.copy_(torch.randn_like(t))
    k0 = [t.clone() for t in pool.k]
    v0 = [t.clone() for t in pool.v]

    host = pool.swap_out([1, 5, 2])
    assert isinstance(host, HostBlocks) and len(host) == 3
    assert all(t.device.type == "cpu" for t in host.k + host.v)
    assert host.nbytes == 3 * DIMS.bytes_per_block(BS)

    for t in pool.k + pool.v:                      # the source blocks get reused by others
        t[[1, 5, 2]] = 0
    pool.swap_in(host, [6, 0, 3])

    for layer in range(DIMS.num_layers):
        assert torch.equal(pool.k[layer][[6, 0, 3]], k0[layer][[1, 5, 2]])
        assert torch.equal(pool.v[layer][[6, 0, 3]], v0[layer][[1, 5, 2]])
        # Untouched blocks are untouched.
        assert torch.equal(pool.k[layer][4], k0[layer][4])
        assert torch.equal(pool.v[layer][7], v0[layer][7])


@pytest.mark.phase("P4")
def test_swap_in_refuses_a_mismatched_block_count():
    """Fewer blocks would drop the sequence's tail; more would leave garbage that
    attention reads. Neither is clamped."""
    pool = make_pool(POOL)
    host = pool.swap_out([0, 1])
    with pytest.raises(ValueError):
        pool.swap_in(host, [2])
    with pytest.raises(ValueError):
        pool.swap_in(host, [2, 3, 4])


@pytest.mark.phase("P4")
def test_swap_out_does_not_touch_the_free_list():
    """The pool moves bytes; the allocator moves indices. A pool that freed as a side
    effect would double-free when the scheduler frees, or leak when it forgot."""
    alloc = BlockAllocator(POOL)
    pool = make_pool(POOL)
    blocks = alloc.allocate_many(3)
    pool.swap_out(blocks)
    assert alloc.num_free == POOL - 3


# ------------------------------------------------------------------- the scheduler
class PoolExecutor:
    """A fake forward pass whose only state is the pool, like the real paged executor.

    Each fed token is written into layer 0's K at that token's slot; the next token is
    sampled from the sum over every slot the sequence has cached. After a swap-in the
    sequence's block indices are different, and this reads through the *new* block
    table — so stale, missing, or misplaced contents change the output.
    """

    def __init__(self, pool: PagedKVPool) -> None:
        self.pool = pool
        self.flat_k = pool.k[0].view(-1, DIMS.num_kv_heads, DIMS.head_dim)

    def execute(self, seqs: list[Sequence]) -> list[int]:
        out = []
        for s in seqs:
            fed = s.next_input_ids
            start = s.num_cached
            slots = s.block_table.slots(start + len(fed))[start:]
            for slot, tok in zip(slots, fed):
                self.flat_k[slot] = float(tok)
            s.num_cached = start + len(fed)
            total = self.flat_k[s.block_table.slots(s.num_cached)][:, 0, 0].sum().item()
            out.append(1 + (int(total) * 7) % 49)
        return out

    def reset(self) -> None:
        pass


def workload(n: int = 6) -> list[Sequence]:
    return [
        Sequence(
            seq_id=f"s{i}",
            prompt_token_ids=[10 + (i * 3 + j) % 17 for j in range(5 + i % 5)],
            max_tokens=6 + (i * 5) % 5,
        )
        for i in range(n)
    ]


def run(seqs: list[Sequence], num_blocks: int, check=None) -> tuple[Scheduler, BlockAllocator]:
    alloc = BlockAllocator(num_blocks)
    pool = make_pool(num_blocks)
    sched = Scheduler(PoolExecutor(pool), eos_token_id=0, allocator=alloc, pool=pool)
    for s in seqs:
        sched.add(s)
    for _ in range(10_000):
        if not sched.has_work:
            return sched, alloc
        sched.step()
        if check is not None:
            check(sched)
    raise AssertionError("scheduler did not drain")


@pytest.fixture(autouse=True)
def swap_mode(monkeypatch):
    monkeypatch.setattr(CONFIG, "block_size", BS)
    monkeypatch.setattr(CONFIG, "preemption", "swap")
    monkeypatch.setattr(CONFIG, "max_running", 32)


@pytest.mark.phase("P4")
def test_swap_preemption_finishes_everything_with_identical_outputs():
    ref = workload()
    run(ref, HUGE)

    def swapped_hold_nothing(sched: Scheduler) -> None:
        """No GPU block is held while SWAPPED — the invariant the free list depends on."""
        for s in sched.preempted:
            if s.status is Status.SWAPPED:
                assert s.block_table.num_blocks == 0, s
                assert s.host_kv is not None and len(s.host_kv) > 0
                assert s.num_cached > 0                        # swap keeps the cache
        held = sum(s.block_table.num_blocks for s in sched.running if s.block_table)
        assert held == sched.allocator.num_allocated

    seqs = workload()
    sched, alloc = run(seqs, POOL, check=swapped_hold_nothing)

    stats = sched.stats()
    assert stats["swaps"] > 0
    assert stats["preemptions"] >= stats["swaps"]
    for got, want in zip(seqs, ref):
        assert got.is_finished
        assert got.output_token_ids == want.output_token_ids, got.seq_id
    assert alloc.num_free == POOL
    assert all(s.host_kv is None for s in seqs), "a host copy outlived its swap-in"


@pytest.mark.phase("P4")
def test_swap_victim_resumes_as_decode_not_prefill():
    """The one-line difference between the strategies: `num_cached` survives, so the
    sequence re-enters one token wide rather than re-prefilling its history."""
    alloc = BlockAllocator(POOL)
    pool = make_pool(POOL)
    sched = Scheduler(PoolExecutor(pool), eos_token_id=0, allocator=alloc, pool=pool)
    seqs = workload()
    for s in seqs:
        sched.add(s)
    while sched.num_swaps == 0:
        assert sched.has_work
        sched.step()
    victim = sched.preempted[0]
    assert victim.status is Status.SWAPPED
    assert not victim.needs_prefill and len(victim.next_input_ids) == 1
    assert len(victim.host_kv) == -(-victim.num_cached // BS)


@pytest.mark.phase("P4")
def test_swap_mode_without_a_pool_degrades_to_recompute():
    """An engine that passed an allocator but no pool cannot swap. It still preempts —
    correctness first — and the counters say which strategy actually ran."""
    class HistoryExecutor:                      # output depends on the whole history
        def execute(self, seqs):
            out = []
            for s in seqs:
                s.num_cached += len(s.next_input_ids)
                out.append(1 + (sum(s.prompt_token_ids + s.output_token_ids) * 7) % 49)
            return out

    alloc = BlockAllocator(POOL)
    sched = Scheduler(HistoryExecutor(), eos_token_id=0, allocator=alloc)
    seqs = workload()
    for s in seqs:
        sched.add(s)
    for _ in range(10_000):
        if not sched.has_work:
            break
        sched.step()
    assert all(s.is_finished for s in seqs)
    assert sched.stats()["preemptions"] > 0 and sched.stats()["swaps"] == 0
    assert alloc.num_free == POOL
