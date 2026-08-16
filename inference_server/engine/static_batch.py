"""Static batching — P1 (Day 2). The straw man, built honestly.

Left-pad N prompts to equal length, mask padding, run as a unit until *all* finish.
Sequences that hit EOS early keep their slot doing dead work — that stall is the
project's motivation, and P1 measures it rather than asserting it.

Exit criteria: M1 holds; mixed-length degradation quantified (the number P2 beats).
See IMPLEMENTATION_GUIDE.md "Day 2".
"""

from __future__ import annotations

from inference_server.engine.base import Engine, NotBuiltYet, Request, Result


class StaticBatchEngine(Engine):
    name = "static"

    def __init__(self) -> None:
        raise NotBuiltYet("static", "P1", "Day 2")

    def generate(self, req: Request) -> Result: ...

    async def submit(self, req: Request) -> Result: ...

    def shutdown(self) -> None: ...
