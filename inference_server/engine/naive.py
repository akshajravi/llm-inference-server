"""Naive engine — P0 (Day 1). The denominator.

Deliberately naive: `model.generate()`, one request at a time, no batching. Any
cleverness here shrinks the headline speedup for free, so there is none. The lock is
not an oversight — serialization is exactly the baseline behaviour being measured.
"""

from __future__ import annotations

import asyncio
import threading
import time

from inference_server.config import CONFIG
from inference_server.engine.base import Engine, Request, Result
from inference_server.model import load, sync as _sync


class NaiveEngine(Engine):
    name = "naive"

    def __init__(self) -> None:
        self.model, self.tokenizer = load()
        self._lock = threading.Lock()

    def generate(self, req: Request) -> Result:
        with self._lock:
            return self._generate_locked(req)

    def _generate_locked(self, req: Request) -> Result:
        start = time.perf_counter()
        input_ids = self.tokenizer(req.prompt, return_tensors="pt").input_ids.to(CONFIG.device)
        prompt_len = input_ids.shape[1]

        eos_id = req.eos_token_id if req.eos_token_id is not None else self.tokenizer.eos_token_id
        out = self.model.generate(
            input_ids,
            max_new_tokens=req.max_tokens,
            do_sample=False,                      # greedy — M1 compares against exactly this
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=eos_id,
        )
        _sync()
        end = time.perf_counter()

        generated = out[0, prompt_len:].tolist()
        finish = "eos" if generated and generated[-1] == eos_id else "length"
        return Result(
            request_id=req.request_id,
            token_ids=generated,
            text=self.tokenizer.decode(generated, skip_special_tokens=True),
            # generate() is a black box: no per-token hook, so TTFT == full latency here.
            # That is honest for a baseline that cannot stream, and it is why P1 exists.
            ttft_s=end - start,
            latency_s=end - start,
            finish_reason=finish,
            prompt_len=prompt_len,
            # M3's "before" number. A contiguous allocator does not know the output
            # length at admission time, so it reserves max_seq_len for every request
            # regardless of what the request actually asks for. Charging only
            # (prompt_len + max_tokens) would understate the baseline to near zero and
            # quietly delete the number P3 exists to beat.
            reserved_tokens=CONFIG.max_seq_len,
            used_tokens=prompt_len + len(generated),
        )

    async def submit(self, req: Request) -> Result:
        return await asyncio.to_thread(self.generate, req)

    def shutdown(self) -> None:
        pass
