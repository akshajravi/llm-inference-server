"""Paged KV cache engine — P3 (Days 6-9). Target: M3, <10% memory waste.

Continuous batching plus a block allocator: KV lives in one preallocated tensor cut
into 16-token blocks, and each sequence holds a block table. Max waste per sequence
becomes 15 tokens instead of max_tokens.

Ships with the PyTorch gather attention path (core/attention.py). The Triton kernel
(S3) is stretch — a working system with a slow attention path beats a fast kernel with
no system around it.

Exit criteria: M3 met; M1 holds; free-list-returns-to-full test passes.
See IMPLEMENTATION_GUIDE.md "Days 6-9".
"""

from __future__ import annotations

from inference_server.engine.base import Engine, NotBuiltYet, Request, Result


class PagedEngine(Engine):
    name = "paged"

    def __init__(self) -> None:
        raise NotBuiltYet("paged", "P3", "Days 6-9")

    def generate(self, req: Request) -> Result: ...

    async def submit(self, req: Request) -> Result: ...

    def shutdown(self) -> None: ...
