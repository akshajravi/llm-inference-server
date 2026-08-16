"""Manual decode loop — P1 (Day 2).

Stop calling generate(); own prefill and decode explicitly so there are seams to
schedule along. Prefill the whole prompt with use_cache=True, then feed one token at
a time plus the cache.

Exit criteria: M1 holds against the Day 1 goldens.
Most likely bug: position IDs, attention mask, or off-by-one on the sampled logit index.
See IMPLEMENTATION_GUIDE.md "Day 2".
"""

from __future__ import annotations

from inference_server.engine.base import Engine, NotBuiltYet, Request, Result


class ManualEngine(Engine):
    name = "manual"

    def __init__(self) -> None:
        raise NotBuiltYet("manual", "P1", "Day 2")

    def generate(self, req: Request) -> Result: ...

    async def submit(self, req: Request) -> Result: ...

    def shutdown(self) -> None: ...
