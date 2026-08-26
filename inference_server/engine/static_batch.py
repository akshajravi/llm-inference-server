"""Static batching — P1 (Day 2). The straw man, built honestly.

Left-pad N prompts to equal length, mask padding, run as a unit until *all* finish.
Sequences that hit EOS early keep their slot doing dead work — that stall is the
project's motivation, and P1 measures it rather than asserting it.

This is the M2 denominator. It is built to be as good as static batching legitimately
gets — a weak straw man here would inflate every speedup claimed in P2, so the batch
window, the batch size, and the padding are all real rather than crippled.

Exit criteria: M1 holds; mixed-length degradation quantified (the number P2 beats).
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


class StaticBatchEngine(Engine):
    name = "static"

    def __init__(self) -> None:
        self.model, self.tokenizer = load()
        self.max_batch = CONFIG.max_running
        self._lock = threading.Lock()          # one batch on the device at a time
        self._queue: asyncio.Queue | None = None
        self._runner: asyncio.Task | None = None

    # ---------------------------------------------------------------- sync path
    def generate(self, req: Request) -> Result:
        """Single request as a batch of one. This is what the M1 suite calls.

        Passing M1 here proves the *unbatched* path is right, which is necessary but
        not sufficient — padding bugs only appear with a ragged batch, and that is
        what tests/test_batch_invariance.py exists to catch.
        """
        with self._lock:
            return self._run_batch([req])[0]

    # --------------------------------------------------------------- async path
    async def submit(self, req: Request) -> Result:
        if self._queue is None:
            self._queue = asyncio.Queue()
            self._runner = asyncio.create_task(self._batch_loop())

        arrived = time.perf_counter()
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._queue.put((req, fut))
        result = await fut
        # Latency is measured from arrival, not from batch start. Under static batching
        # a request may wait out a whole batch it was too late to join; charging only
        # the batch it eventually ran in would hide exactly the cost being measured.
        result.latency_s = time.perf_counter() - arrived
        return result

    async def _batch_loop(self) -> None:
        """Form a batch, run it to completion, form the next one.

        The defining property of static batching lives here: once `_run_batch` is
        entered the membership is frozen. Anything that arrives one microsecond later
        waits for every sequence in the current batch to finish, including the longest.
        """
        assert self._queue is not None
        loop = asyncio.get_running_loop()
        while True:
            first = await self._queue.get()
            batch = [first]

            # Collect arrivals for a short window so batches are actually batches.
            deadline = loop.time() + CONFIG.batch_window_s
            while len(batch) < self.max_batch:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), remaining))
                except (asyncio.TimeoutError, TimeoutError):
                    break

            reqs = [r for r, _ in batch]
            try:
                results = await asyncio.to_thread(self._locked_run, reqs)
            except Exception as exc:  # noqa: BLE001 — one bad batch must not kill the loop
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)
                continue
            for (_, fut), res in zip(batch, results):
                if not fut.done():
                    fut.set_result(res)

    def _locked_run(self, reqs: list[Request]) -> list[Result]:
        with self._lock:
            return self._run_batch(reqs)

    # ------------------------------------------------------------------ the batch
    @torch.inference_mode()  # thread-local; see the note in manual.py
    def _run_batch(self, reqs: list[Request]) -> list[Result]:
        device = CONFIG.device
        start = time.perf_counter()
        n = len(reqs)

        # Tokenizer pads LEFT (set in model.py) so every row's newest token sits in the
        # final column. That is what makes `logits[:, -1]` the right sample point for
        # every row at once, regardless of how ragged the prompts are.
        enc = self.tokenizer([r.prompt for r in reqs], return_tensors="pt", padding=True)
        input_ids = enc.input_ids.to(device)
        attn = enc.attention_mask.to(device)

        # Position IDs must count *real* tokens, not columns. A left-padded row has its
        # first real token sitting several columns in; numbering by column would tell
        # the model that token is at position 3 when the sequence thinks it is at 0,
        # and the row would generate different text than it does alone. cumsum over the
        # mask gives each real token its rank among real tokens; pad slots clamp to 0
        # and are masked out anyway.
        position_ids = (attn.cumsum(-1) - 1).clamp(min=0)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attn,
            position_ids=position_ids,
            use_cache=True,
        )
        kv = out.past_key_values
        next_ids = out.logits[:, -1, :].argmax(-1)          # [n]
        sync()
        ttft = time.perf_counter() - start

        eos_ids = [
            r.eos_token_id if r.eos_token_id is not None else self.tokenizer.eos_token_id
            for r in reqs
        ]
        prompt_lens = attn.sum(-1).tolist()
        generated: list[list[int]] = [[] for _ in range(n)]
        finish: list[str] = [""] * n

        def record(row: int, token: int) -> None:
            if finish[row]:
                return
            generated[row].append(token)
            if token == eos_ids[row]:
                finish[row] = "eos"
            elif len(generated[row]) >= reqs[row].max_tokens:
                finish[row] = "length"

        first_tokens = next_ids.tolist()
        for i in range(n):
            record(i, first_tokens[i])

        cur_pos = position_ids[:, -1] + 1                    # [n], next position per row

        # The stall, made explicit: the loop runs until *all* rows finish, not until each
        # one does. A row that set finish[] on step 2 keeps being fed through the model for
        # every remaining step. Its slot is not reusable, because the batch tensor was
        # shaped when the batch was formed and nothing can be swapped into it now. Counting
        # those wasted row-steps is the whole point of building this engine, so they are
        # attributed per row and reported on the Result rather than tallied and dropped.
        wasted = [0] * n
        while not all(finish):
            attn = torch.cat([attn, torch.ones(n, 1, dtype=attn.dtype, device=device)], dim=1)
            out = self.model(
                input_ids=next_ids.view(n, 1),
                attention_mask=attn,
                position_ids=cur_pos.view(n, 1),
                past_key_values=kv,
                use_cache=True,
            )
            kv = out.past_key_values
            next_ids = out.logits[:, -1, :].argmax(-1)
            cur_pos = cur_pos + 1

            tokens = next_ids.tolist()
            for i in range(n):
                if finish[i]:
                    wasted[i] += 1
                else:
                    record(i, tokens[i])

        sync()
        latency = time.perf_counter() - start

        return [
            Result(
                request_id=reqs[i].request_id,
                token_ids=generated[i],
                text=self.tokenizer.decode(generated[i], skip_special_tokens=True),
                ttft_s=ttft,
                # Everyone in the batch shares the batch's wall clock, because nobody is
                # released until the slowest row finishes. The short requests paying the
                # long request's latency is not a measurement artifact — it is the
                # behaviour, and it is what shows up in p99.
                latency_s=latency,
                finish_reason=finish[i],
                prompt_len=int(prompt_lens[i]),
                reserved_tokens=CONFIG.max_seq_len,
                used_tokens=int(prompt_lens[i]) + len(generated[i]),
                wasted_steps=wasted[i],
            )
            for i in range(n)
        ]

    def shutdown(self) -> None:
        if self._runner is not None:
            self._runner.cancel()
            self._runner = None
