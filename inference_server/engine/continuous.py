"""Continuous batching — P2 (Days 3-5). Target: M2, >=3x throughput over P0.

Thin driver only: owns the background step loop and the futures the HTTP layer awaits.
The actual decisions live in core/scheduler.py and the forward pass in core/executor.py
(NFR2 — an interviewer reads one file).

Exit criteria: M2 met; M1 holds including the alone-vs-crowded-batch test;
p99 reported alongside throughput.
See IMPLEMENTATION_GUIDE.md "Days 3-5".
"""

from __future__ import annotations

from inference_server.engine.base import Engine, NotBuiltYet, Request, Result


class ContinuousEngine(Engine):
    name = "continuous"

    def __init__(self) -> None:
        raise NotBuiltYet("continuous", "P2", "Days 3-5")

    def generate(self, req: Request) -> Result: ...

    async def submit(self, req: Request) -> Result: ...

    def shutdown(self) -> None: ...
