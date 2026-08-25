"""Manual decode loop — P1 (Day 2).

Stop calling generate(); own prefill and decode explicitly so there are seams to
schedule along. Prefill the whole prompt with use_cache=True, then feed one token at
a time plus the cache.

Nothing here is faster than P0 — it is the same arithmetic in the same order. What it
buys is a line of our own code between step N and step N+1. Every later phase is
something inserted at that line: P2 mutates the batch there, P3 allocates blocks there,
P4 preempts there. None of that is reachable from inside generate().

Exit criteria: M1 holds against the Day 1 goldens.
Most likely bug: position IDs, attention mask, or off-by-one on the sampled logit index.
See IMPLEMENTATION_GUIDE.md "Day 2".
"""

from __future__ import annotations

import asyncio
import threading
import time

import torch

from inference_server.config import CONFIG
from inference_server.engine.base import Engine, Request, Result
from inference_server.model import load, sync


class ManualEngine(Engine):
    name = "manual"

    def __init__(self) -> None:
        self.model, self.tokenizer = load()
        # Still serialized, like P0. P1 is about owning the loop, not about concurrency —
        # changing both at once would make an M1 failure ambiguous.
        self._lock = threading.Lock()

    def generate(self, req: Request) -> Result:
        with self._lock:
            return self._generate_locked(req)

    @torch.inference_mode()
    def _generate_locked(self, req: Request) -> Result:
        # Not optional, and not merely a micro-optimisation. `torch.set_grad_enabled` in
        # model.py is THREAD-LOCAL, and `submit()` hands work to an asyncio worker
        # thread — so the global switch flipped at load time does not apply here.
        # Without this decorator the loop builds an autograd graph across every decode
        # step and retains each step's activations: measured 33 tok/s instead of 143,
        # with memory growing linearly in output length. P0 never showed the bug because
        # HuggingFace's generate() carries its own @torch.no_grad() internally.
        # Every engine that owns its own loop must declare this for itself.
        start = time.perf_counter()
        device = CONFIG.device
        input_ids = self.tokenizer(req.prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = input_ids.shape[1]
        eos_id = req.eos_token_id if req.eos_token_id is not None else self.tokenizer.eos_token_id

        # --- PREFILL: the whole prompt in one pass ---------------------------------
        # Position IDs are passed explicitly rather than left to the default. For a
        # single unpadded sequence the default is identical, but P1 static batching
        # left-pads, and there the default is wrong. Making the contract visible now
        # means the padded case is a change of one line rather than a rediscovery.
        position_ids = torch.arange(prompt_len, device=device).unsqueeze(0)
        out = self.model(input_ids=input_ids, position_ids=position_ids, use_cache=True)
        kv = out.past_key_values

        # The logits tensor is [batch, seq_len, vocab] — a next-token prediction at
        # *every* prompt position. All but the last are byproducts of how attention
        # works and get discarded; only position -1 has seen the whole prompt.
        # Sampling from the wrong index here is the classic P1 bug: it produces fluent,
        # plausible, wrong text that no eyeball test catches.
        next_id = out.logits[:, -1, :].argmax(-1)

        sync()
        # TTFT is a real measurement now. P0 reported it as full latency because
        # generate() offered no per-token hook — this is the first phase where the
        # number means what its name says.
        ttft = time.perf_counter() - start

        generated: list[int] = [int(next_id.item())]
        finish = "eos" if generated[0] == eos_id else ""

        # --- DECODE: one token per pass, cache carries the rest ---------------------
        while not finish and len(generated) < req.max_tokens:
            # The token just produced sits at position prompt_len + (len(generated) - 1):
            # the cache already holds positions 0 .. prompt_len + len(generated) - 2.
            pos = prompt_len + len(generated) - 1
            out = self.model(
                input_ids=next_id.view(1, 1),
                position_ids=torch.tensor([[pos]], device=device),
                past_key_values=kv,
                use_cache=True,
            )
            kv = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(-1)
            generated.append(int(next_id.item()))
            if generated[-1] == eos_id:
                finish = "eos"

        if not finish:
            finish = "length"

        sync()
        latency = time.perf_counter() - start

        return Result(
            request_id=req.request_id,
            token_ids=generated,
            text=self.tokenizer.decode(generated, skip_special_tokens=True),
            ttft_s=ttft,
            latency_s=latency,
            finish_reason=finish,
            prompt_len=prompt_len,
            # Unchanged from P0 and for the same reason: this engine still hands the
            # cache to HuggingFace, which grows one contiguous buffer per sequence. The
            # allocator has not changed, so neither has the waste. P3 is what moves it.
            reserved_tokens=CONFIG.max_seq_len,
            used_tokens=prompt_len + len(generated),
        )

    async def submit(self, req: Request) -> Result:
        return await asyncio.to_thread(self.generate, req)

    def shutdown(self) -> None:
        pass
