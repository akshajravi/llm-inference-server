"""Sequence — P2 (Days 3-5). One request's state, and nothing else.

Holds: prompt IDs, generated IDs, KV cache (P2) or block table (P3), status, max_tokens,
arrival time. Knows how to answer "am I done?"; knows nothing about who else is running.

    WAITING --admit--> RUNNING --eos/max_tokens--> FINISHED
       ^                  |
       +---preempt (P4)---+          (SWAPPED is the S2 stretch path)

See IMPLEMENTATION_GUIDE.md "Days 3-5".
"""

from __future__ import annotations

import enum


class Status(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"      # P4 stretch (S2): blocks parked in pinned host memory
    FINISHED = "finished"


class Sequence:
    """Lands in P2. See module docstring for the state machine it implements."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Sequence lands in P2 (Days 3-5)")
