"""Sequence state machine — P2 (Day 3).

Unit tests, no model. Sequence is where the scheduler and the executor agree about what
is true, so a bug here surfaces as a scheduling mystery three files away. It is cheap to
pin down in isolation and expensive to debug inside a step loop.

The two things worth asserting: termination is decided exactly once and by exactly one
place, and `num_cached` drives the prefill/decode split correctly across the boundary.
"""

from __future__ import annotations

import pytest

from inference_server.core.sequence import Sequence, Status

EOS = 50256


def _seq(prompt=(1, 2, 3), max_tokens=4, **kw) -> Sequence:
    return Sequence(seq_id="s0", prompt_token_ids=list(prompt), max_tokens=max_tokens, **kw)


@pytest.mark.phase("P2")
def test_starts_waiting_and_needing_prefill():
    s = _seq()
    assert s.status is Status.WAITING
    assert s.needs_prefill
    assert s.next_input_ids == [1, 2, 3]      # prefill hands over the whole prompt
    assert s.next_position == 0


@pytest.mark.phase("P2")
def test_prefill_boundary_flips_to_single_token_decode():
    """The one off-by-one that matters: after prefill, the pass is one token wide.

    Getting `next_position` wrong here is the failure that produces fluent, plausible,
    wrong text — no crash, no exception, just a sequence that decodes as if it were
    somewhere else in its own history.
    """
    s = _seq()
    s.num_cached = s.prompt_len              # what the executor does after prefill
    s.append_token(99, EOS)

    assert not s.needs_prefill
    assert s.next_input_ids == [99]          # only the freshly sampled token
    assert s.next_position == 3              # not 4 — token 99 is not cached yet

    s.num_cached += 1                        # executor ran the decode pass
    assert s.next_position == 4


@pytest.mark.phase("P2")
def test_stops_at_max_tokens():
    s = _seq(max_tokens=2)
    s.append_token(10, EOS)
    assert not s.is_finished
    s.append_token(11, EOS)
    assert s.is_finished
    assert s.finish_reason == "length"
    assert s.output_token_ids == [10, 11]


@pytest.mark.phase("P2")
def test_stops_at_eos_before_max_tokens():
    s = _seq(max_tokens=10)
    s.append_token(10, EOS)
    s.append_token(EOS, EOS)
    assert s.is_finished
    assert s.finish_reason == "eos"


@pytest.mark.phase("P2")
def test_per_request_eos_override_wins():
    """FR: the request may name its own stop token, because gpt2 greedy never emits the
    tokenizer default and the EOS path would otherwise be untested everywhere."""
    s = _seq(max_tokens=10, eos_token_id=7)
    s.append_token(EOS, EOS)                 # the *default* stop token must not stop it
    assert not s.is_finished
    s.append_token(7, EOS)
    assert s.is_finished
    assert s.finish_reason == "eos"


@pytest.mark.phase("P2")
def test_sampling_a_finished_sequence_is_an_error():
    """Under static batching, feeding a finished row was normal and wasted 77% of the
    compute. Under continuous batching it means eviction did not happen, so it must be
    loud — a silent version of this bug looks like a throughput regression, not a defect.
    """
    s = _seq(max_tokens=1)
    s.append_token(10, EOS)
    assert s.is_finished
    with pytest.raises(RuntimeError, match="finished"):
        s.append_token(11, EOS)


@pytest.mark.phase("P2")
def test_ttft_measured_from_arrival_not_admission():
    """Queue time is the user's time. A TTFT that starts counting at admission hides
    exactly the delay continuous batching exists to reduce."""
    s = _seq()
    s.arrival_s -= 1.0                       # pretend it queued for a second
    s.append_token(10, EOS)
    assert s.ttft_s >= 1.0
