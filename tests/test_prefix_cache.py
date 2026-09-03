"""S1 (Day 14) — prefix caching, FR5.

Model-free first: the hash chain, two sequences sharing exactly the blocks their
prompts have in common (refcount 2, freed only by the last referent, cache entry gone
at zero), the one-token floor on what is fed, copy-on-write as a unit, and the
interaction with both P4 preemption strategies on a pool too small to hold everyone.
The fake executor writes each fed token into the pool at its slot and samples from
the sum over every slot the sequence can see, so a shared block holding the *wrong*
KV — someone else's prefix, a freed block reused, a swap that lost a shared block —
changes the output against the unshared reference.

Then the real thing: every golden run twice through the paged engine, the second wave
admitted while the first still holds its blocks, token-exact with `prefix_hits > 0`;
and the same with caching off.
"""

from __future__ import annotations

import pytest
import torch

from inference_server.config import CONFIG
from inference_server.core.block_allocator import BlockAllocator
from inference_server.core.block_table import BlockTable
from inference_server.core.kv_pool import ModelDims, PagedKVPool
from inference_server.core.prefix_cache import PrefixCache, block_hash, hash_chain
from inference_server.core.scheduler import Scheduler
from inference_server.core.sequence import Sequence

BS = 4
POOL = 8
HUGE = 512
DIMS = ModelDims(num_layers=2, num_kv_heads=2, head_dim=4, dtype=torch.float32)
HEAD = [10, 11, 12, 13, 14, 15, 16, 17]          # two full blocks, shared by every prompt


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch):
    monkeypatch.setattr(CONFIG, "block_size", BS)
    monkeypatch.setattr(CONFIG, "preemption", "recompute")
    monkeypatch.setattr(CONFIG, "max_running", 32)


class PoolExecutor:
    """Like test_swap's: the pool is the only state, read through the block table."""

    def __init__(self, pool: PagedKVPool) -> None:
        self.pool = pool
        self.flat_k = pool.k[0].view(-1, DIMS.num_kv_heads, DIMS.head_dim)
        self.fed: dict[str, int] = {}

    def execute(self, seqs: list[Sequence]) -> list[int]:
        out = []
        for s in seqs:
            fed = s.next_input_ids
            start = s.num_cached
            for slot, tok in zip(s.block_table.slots(start + len(fed))[start:], fed):
                self.flat_k[slot] = float(tok)
            s.num_cached = start + len(fed)
            self.fed[s.seq_id] = self.fed.get(s.seq_id, 0) + len(fed)
            total = self.flat_k[s.block_table.slots(s.num_cached)][:, 0, 0].sum().item()
            out.append(1 + (int(total) * 7) % 49)
        return out

    def reset(self) -> None:
        pass


def make(num_blocks: int, caching: bool = True) -> tuple[Scheduler, BlockAllocator, PrefixCache | None]:
    alloc = BlockAllocator(num_blocks)
    pool = PagedKVPool(DIMS, num_blocks=num_blocks, block_size=BS, device="cpu")
    cache = PrefixCache(alloc, BS) if caching else None
    sched = Scheduler(PoolExecutor(pool), eos_token_id=0, allocator=alloc, pool=pool, prefix_cache=cache)
    return sched, alloc, cache


def drive(sched: Scheduler, limit: int = 10_000) -> None:
    for _ in range(limit):
        if not sched.has_work:
            return
        sched.step()
    raise AssertionError("scheduler did not drain")


def workload(n: int = 6) -> list[Sequence]:
    """Same 8-token head, tails of 1-5 tokens, outputs 6-10: 3-5 blocks each."""
    return [
        Sequence(seq_id=f"s{i}", prompt_token_ids=HEAD + [20 + i + j for j in range(1 + i % 5)], max_tokens=6 + (i * 5) % 5)
        for i in range(n)
    ]


# ------------------------------------------------------------------ the hash chain
@pytest.mark.phase("S1")
def test_hash_chain_is_deterministic_and_prefix_sensitive():
    a = list(range(12))
    assert hash_chain(a, BS) == hash_chain(list(a), BS)
    assert len(hash_chain(a, BS)) == 3 and len(hash_chain(a[:11], BS)) == 2   # full blocks only
    b = [99] + a[1:]                                   # one token differs, in block 0
    assert all(x != y for x, y in zip(hash_chain(a, BS), hash_chain(b, BS))), "later blocks must see the change"
    c = a[:8] + [99] + a[9:]                           # differs in block 2 only
    ca, cc = hash_chain(a, BS), hash_chain(c, BS)
    assert ca[:2] == cc[:2] and ca[2] != cc[2]
    # Continuing from a known prefix is the same chain as hashing from scratch.
    assert hash_chain(a, BS, prefix=ca[:1]) == ca
    assert block_hash(None, [1, 2]) != block_hash(None, [2, 1])


# --------------------------------------------------------------------- sharing
@pytest.mark.phase("S1")
def test_two_sequences_share_exactly_the_common_full_blocks():
    sched, alloc, cache = make(HUGE)
    s1 = Sequence(seq_id="a", prompt_token_ids=HEAD + [1, 2, 3], max_tokens=4)
    s2 = Sequence(seq_id="b", prompt_token_ids=HEAD + [7, 8], max_tokens=8)
    sched.add(s1)
    sched.step()                                        # s1 prefills alone and publishes
    assert cache.entries == 2 and cache.hits == 0
    sched.add(s2)
    sched.step()                                        # s2 admitted: matches 2 blocks
    assert s2.block_table.blocks[:2] == s1.block_table.blocks[:2]
    assert s2.block_table.blocks[2:] and not set(s2.block_table.blocks[2:]) & set(s1.block_table.blocks)
    assert all(alloc.refcount(b) == 2 for b in s1.block_table.blocks[:2])
    assert all(alloc.refcount(b) == 1 for b in s1.block_table.blocks[2:] + s2.block_table.blocks[2:])
    assert cache.hits == 2 and cache.blocks_shared == 2 and cache.misses == 0
    assert sched.executor.fed["b"] == 2, "only the tail past the shared prefix is fed"
    assert sched.stats()["prefix_hits"] == 2

    while not s1.is_finished:
        sched.step()
    sched.step()                                        # eviction frees s1's blocks
    assert all(alloc.refcount(b) == 1 for b in s2.block_table.blocks[:2]), "shared blocks survive the first referent"
    assert cache.entries >= 2, "entries stay while a referent lives"
    drive(sched)
    assert alloc.num_free == HUGE
    assert cache.entries == 0, "refcount zero must drop the entry"


@pytest.mark.phase("S1")
def test_at_least_one_prompt_token_is_always_fed():
    """A prompt that is entirely cached still feeds its whole last block: the model
    must run over the final prompt token to produce the first logit."""
    sched, alloc, cache = make(HUGE)
    s1 = Sequence(seq_id="a", prompt_token_ids=list(HEAD), max_tokens=3)
    sched.add(s1)
    sched.step()
    s2 = Sequence(seq_id="b", prompt_token_ids=list(HEAD), max_tokens=3)
    sched.add(s2)
    sched.step()
    assert s2.block_table.blocks[0] == s1.block_table.blocks[0]
    assert s2.block_table.blocks[1] != s1.block_table.blocks[1]
    assert sched.executor.fed["b"] == BS                # one block, not zero tokens
    assert cache.hits == 1
    drive(sched)
    assert s1.output_token_ids == s2.output_token_ids
    assert alloc.num_free == HUGE


@pytest.mark.phase("S1")
def test_outputs_match_an_unshared_run_and_generated_blocks_are_reusable():
    """Sharing is invisible: same outputs as a run with no cache. A follow-up whose
    prompt is a finished sequence's whole history (prompt + answer, still held by a
    live sibling) reuses generated-token blocks too."""
    ref = workload()
    sched_ref, _, _ = make(HUGE, caching=False)
    for s in ref:
        sched_ref.add(s)
    drive(sched_ref)

    seqs = workload()
    sched, alloc, cache = make(HUGE)
    sched.add(seqs[0])
    sched.step()
    for s in seqs[1:]:
        sched.add(s)
    drive(sched)
    for got, want in zip(seqs, ref):
        assert got.output_token_ids == want.output_token_ids, got.seq_id
    assert cache.hits > 0 and alloc.num_free == HUGE and cache.entries == 0

    # Multi-turn: keep the first turn alive, ask for its history back.
    turn1 = Sequence(seq_id="t1", prompt_token_ids=HEAD + [1], max_tokens=8)
    keeper = Sequence(seq_id="keep", prompt_token_ids=HEAD + [1], max_tokens=40)
    sched.add(turn1)
    sched.step()
    sched.add(keeper)
    while not turn1.is_finished:
        sched.step()
    history = turn1.prompt_token_ids + turn1.output_token_ids       # 17 tokens: 4 full blocks
    sched.step()                                                    # turn1 evicted; keeper still holds
    turn2 = Sequence(seq_id="t2", prompt_token_ids=history + [5, 6], max_tokens=2)
    before = cache.hits
    sched.add(turn2)
    sched.step()
    assert cache.hits - before == len(history) // BS
    assert turn2.block_table.blocks[: len(history) // BS] == keeper.block_table.blocks[: len(history) // BS]
    drive(sched)
    assert alloc.num_free == HUGE


# ------------------------------------------------------------------- preemption
@pytest.mark.phase("S1")
@pytest.mark.parametrize("strategy", ["recompute", "swap"])
def test_sharing_survives_preemption_under_both_strategies(monkeypatch, strategy):
    monkeypatch.setattr(CONFIG, "preemption", strategy)
    ref = workload()
    sched_ref, _, _ = make(HUGE, caching=False)
    for s in ref:
        sched_ref.add(s)
    drive(sched_ref)
    assert sched_ref.stats()["preemptions"] == 0

    seqs = workload()
    sched, alloc, cache = make(POOL)
    for s in seqs:
        sched.add(s)
    drive(sched)
    stats = sched.stats()
    assert stats["preemptions"] > 0, "the pool was big enough — the test proves nothing"
    assert stats["prefix_hits"] > 0, "nothing was shared — the test proves nothing"
    assert stats["swaps"] == (stats["preemptions"] if strategy == "swap" else 0)
    for got, want in zip(seqs, ref):
        assert got.is_finished and got.output_token_ids == want.output_token_ids, got.seq_id
    assert alloc.num_free == POOL, f"leaked {POOL - alloc.num_free} blocks"
    assert cache.entries == 0


@pytest.mark.phase("S1")
def test_free_list_returns_to_full_under_shared_churn():
    sched, alloc, cache = make(POOL)
    for wave in range(6):
        for s in workload(5):
            s.seq_id = f"{wave}-{s.seq_id}"
            sched.add(s)
        drive(sched)
        assert alloc.num_free == POOL, f"leaked by wave {wave}"
        assert cache.entries == 0
    assert sched.stats()["prefix_hits"] > 0 and sched.stats()["completed"] == 30


# ------------------------------------------------------------------ copy-on-write
@pytest.mark.phase("S1")
def test_ensure_private_copies_a_shared_block_and_fixes_refcounts():
    alloc = BlockAllocator(4)
    contents = [None] * 4                                # a "pool" of one value per block
    a, b = BlockTable(alloc, BS), BlockTable(alloc, BS)
    a.ensure_capacity(BS)
    contents[a.blocks[0]] = "shared"
    b.adopt(list(a.blocks), [b"h"])
    assert alloc.refcount(a.blocks[0]) == 2

    def copy(src, dst):
        contents[dst] = contents[src]

    moved = b.ensure_private(0, copy)
    assert moved is not None and moved[0] == a.blocks[0] and moved[1] == b.blocks[0]
    assert a.blocks[0] != b.blocks[0]
    assert alloc.refcount(a.blocks[0]) == 1 and alloc.refcount(b.blocks[0]) == 1
    contents[b.blocks[0]] = "mine"
    assert contents[a.blocks[0]] == "shared", "the writer's copy diverged, the original did not"
    assert b.ensure_private(0, copy) is None            # already private: no-op
    assert alloc.num_free == 2
    a.free()
    b.free()
    assert alloc.num_free == 4


@pytest.mark.phase("S1")
def test_cow_is_never_triggered_by_full_block_sharing():
    """The admission policy only ever hands out full blocks, so the block a sequence
    writes into is always its own: no copy is made during a whole shared run."""
    sched, alloc, cache = make(HUGE)
    copies = []
    orig = BlockTable.ensure_private

    def spy(self, logical, copy=None):
        out = orig(self, logical, copy)
        if out is not None:
            copies.append(out)
        return out

    BlockTable.ensure_private = spy
    try:
        seqs = workload()
        sched.add(seqs[0])
        sched.step()
        for s in seqs[1:]:
            sched.add(s)
        # Ask for privacy on every block each sequence is about to write, as the executor does.
        for _ in range(10_000):
            if not sched.has_work:
                break
            for s in sched.running:
                if s.block_table is not None and s.block_table.blocks:
                    for logical in range(s.num_cached // BS, min(len(s.block_table.blocks), (s.cached_after_next_pass - 1) // BS + 1)):
                        s.block_table.ensure_private(logical)
            sched.step()
    finally:
        BlockTable.ensure_private = orig
    assert cache.hits > 0 and copies == []


# ---------------------------------------------------------- real model, real engine
def _golden_seqs(engine, goldens, suffix: str) -> list[Sequence]:
    return [
        Sequence(
            seq_id=f"{case_id}{suffix}",
            prompt_token_ids=engine.tokenizer(case["prompt"]).input_ids,
            max_tokens=case["max_tokens"],
            eos_token_id=case.get("eos_token_id"),
        )
        for case_id, case in sorted(goldens["cases"].items())
    ]


@pytest.mark.phase("S1")
@pytest.mark.parametrize("caching", [True, False])
def test_goldens_twice_on_the_paged_engine(monkeypatch, goldens, caching):
    """Every golden, then every golden again one step later — the second wave is
    admitted while the first still holds its blocks, so each long enough prompt hits
    the cache. Token-exact either way; the counters say whether sharing happened."""
    monkeypatch.setattr(CONFIG, "block_size", 16)
    monkeypatch.setattr(CONFIG, "prefix_caching", caching)
    from inference_server.engine.paged import PagedEngine  # noqa: PLC0415

    engine = PagedEngine()
    try:
        first = _golden_seqs(engine, goldens, "-1")
        for s in first:
            engine.scheduler.add(s)
        engine.scheduler.step()                          # wave 1 prefills, publishes
        second = _golden_seqs(engine, goldens, "-2")
        for s in second:
            engine.scheduler.add(s)
        drive(engine.scheduler, limit=2_000)
        stats = engine.stats()
    finally:
        engine.shutdown()

    for seq in first + second:
        want = goldens["cases"][seq.seq_id.rsplit("-", 1)[0]]["token_ids"]
        assert seq.output_token_ids == want, seq.seq_id
    assert stats["free_blocks"] == stats["num_blocks"]
    if caching:
        assert stats["prefix_hits"] > 0 and stats["prefix_entries"] == 0
        assert stats["prefix_tokens_saved"] == stats["prefix_blocks_shared"] * 16
    else:
        assert "prefix_hits" not in stats


@pytest.mark.phase("S1")
def test_repeated_structure_golden_is_token_exact_with_and_without_caching(monkeypatch, goldens):
    from inference_server.engine import build
    from inference_server.engine.base import Request

    case = goldens["cases"]["repeated_structure"]
    for caching in (True, False):
        monkeypatch.setattr(CONFIG, "prefix_caching", caching)
        engine = build("paged")
        try:
            for i in range(2):
                req = Request(request_id=f"rs-{i}", prompt=case["prompt"], max_tokens=case["max_tokens"],
                              eos_token_id=case.get("eos_token_id"))
                assert engine.generate(req).token_ids == case["token_ids"], (caching, i)
        finally:
            engine.shutdown()
