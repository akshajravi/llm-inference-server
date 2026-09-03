"""P4 (Day 10) — admission control, FR7.

Two levels. `Scheduler.add()` raises QueueFull at the bound, model-free. Then the same
thing over the wire: a burst against the continuous engine with a queue bound of two
and one running slot, where some requests must get a 503 with a JSON body and every
request that was accepted must complete with the golden output. That second half is
M4 in miniature — the only way a request fails is an explicit 503 at the door, and
nothing accepted is ever dropped.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from inference_server.config import CONFIG
from inference_server.core.scheduler import QueueFull, Scheduler
from inference_server.core.sequence import Sequence
from inference_server.engine import build
from inference_server.server import app as app_module

HEALTH_FIELDS = [
    "queue_depth", "num_running", "num_waiting", "num_swapped", "free_blocks",
    "num_blocks", "preemptions", "swaps", "completed", "rss_bytes", "device_mem_bytes",
]


class NoopExecutor:
    def execute(self, seqs):
        for s in seqs:
            s.num_cached += len(s.next_input_ids)
        return [7] * len(seqs)

    def reset(self):
        pass


# ------------------------------------------------------------------------ in-process
@pytest.mark.phase("P4")
def test_add_raises_queue_full_at_the_bound(monkeypatch):
    monkeypatch.setattr(CONFIG, "max_queue_depth", 3)
    sched = Scheduler(NoopExecutor(), eos_token_id=0)
    for i in range(3):
        sched.add(Sequence(seq_id=f"q{i}", prompt_token_ids=[1], max_tokens=1))
    with pytest.raises(QueueFull) as info:
        sched.add(Sequence(seq_id="q3", prompt_token_ids=[1], max_tokens=1))
    assert info.value.depth == 3 and info.value.bound == 3
    assert sched.queue_depth == 3, "a rejected request must not occupy the queue"


@pytest.mark.phase("P4")
def test_preempted_sequences_never_count_against_the_bound(monkeypatch):
    """FR6 meets FR7: a victim is re-admitted, never rejected, so it lives in a queue the
    bound does not see. Otherwise an overloaded server would 503 its own victims."""
    monkeypatch.setattr(CONFIG, "max_queue_depth", 2)
    sched = Scheduler(NoopExecutor(), eos_token_id=0)
    sched.add(Sequence(seq_id="w0", prompt_token_ids=[1], max_tokens=1))
    for i in range(5):
        sched.preempted.append(Sequence(seq_id=f"p{i}", prompt_token_ids=[1], max_tokens=1))
    sched.add(Sequence(seq_id="w1", prompt_token_ids=[1], max_tokens=1))      # still room
    with pytest.raises(QueueFull):
        sched.add(Sequence(seq_id="w2", prompt_token_ids=[1], max_tokens=1))
    assert sched.stats()["queue_depth"] == 2
    assert sched.stats()["num_waiting"] == 7


# ------------------------------------------------------------------------- over HTTP
@pytest.fixture(scope="module")
def engine():
    eng = build("continuous")
    app_module._engine = eng
    yield eng
    eng.shutdown()
    app_module._engine = None


@pytest.fixture
def client(engine):
    # Not entered as a context manager: that runs `lifespan`, which would build a
    # second engine from the ENGINE env var and overwrite the injected one.
    return TestClient(app_module.app)


@pytest.mark.phase("P4")
def test_burst_yields_503s_and_every_accepted_request_completes(client, engine, goldens, monkeypatch):
    monkeypatch.setattr(CONFIG, "max_queue_depth", 2)
    monkeypatch.setattr(engine.scheduler, "max_running", 1)
    golden = goldens["cases"]["stops_at_max_tokens"]
    body = {"prompt": golden["prompt"], "max_tokens": golden["max_tokens"]}

    n = 12
    with ThreadPoolExecutor(max_workers=n) as tp:
        responses = list(tp.map(lambda _: client.post("/generate", json=body), range(n)))

    codes = [r.status_code for r in responses]
    assert set(codes) <= {200, 503}, codes
    assert codes.count(503) >= 1, "one slot and a queue of two absorbed twelve requests?"
    assert codes.count(200) >= 3, "the slot plus the queue must have served at least three"

    for r in responses:
        if r.status_code == 503:
            detail = r.json()["detail"]
            assert detail["error"] == "queue_full"
            assert detail["max_queue_depth"] == 2
            assert detail["queue_depth"] >= 2
            assert r.headers.get("retry-after")
        else:
            assert r.json()["token_ids"] == golden["token_ids"], "an accepted request was corrupted"

    # Nothing left behind: no future waits on a request that was 503'd at the door.
    assert not engine._futures
    assert not engine.scheduler.has_work


@pytest.mark.phase("P4")
def test_health_reports_the_p4_fields(client, engine):
    body = client.get("/health").json()
    for name in HEALTH_FIELDS:
        assert name in body, name
        assert isinstance(body[name], int) and body[name] >= 0, name
    assert body["rss_bytes"] > 0
    assert body["completed"] == engine.stats()["completed"]
    assert body["queue_depth"] == 0
