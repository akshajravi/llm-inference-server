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
        self._loop: asyncio.AbstractEventLoop | None = None

    # ---------------------------------------------------------------- sync path
    def generate(self, req: Request) -> Result:
        """Single request as a batch of one. This is what the M1 suite calls.

        Passing M1 here proves the *unbatched* path is right, which is necessary but
        not sufficient — padding bugs only appear with a ragged batch, and that is
        what tests/test_batch_invariance.py exists to catch.
        """
        with self._lock:
            return self._run_batch([req], [time.perf_counter()])[0]

    # --------------------------------------------------------------- async path
    async def submit(self, req: Request) -> Result:
        # The queue and the batch loop belong to whichever event loop called first, and
        # a Queue is only usable from that loop. Under uvicorn there is exactly one loop
        # for the life of the process, so binding once looks fine — until a second loop
        # appears, at which point `put` lands in a queue whose consumer died with its
        # loop and the caller awaits a future nobody will ever resolve. Found by driving
        # the goldens through FastAPI's TestClient, which opens a fresh loop per request:
        # request one passed, request two hung forever. Rebinding costs one identity
        # check and turns a deadlock into a new batch loop.
        loop = asyncio.get_running_loop()
        if self._queue is None or self._loop is not loop:
            self._queue = asyncio.Queue()
            self._loop = loop
            self._runner = loop.create_task(self._batch_loop())

        # Arrival travels with the request. Under static batching a request may wait out
        # an entire batch it was too late to join, and BOTH its numbers have to include
        # that wait: charging only the batch it eventually ran in would hide exactly the
        # cost this engine exists to measure. TTFT used to start at batch launch, which
        # quietly flattered the baseline against P2 — the comparison is only honest if
        # both engines measure from the same event.
        arrived = time.perf_counter()
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._queue.put((req, fut, arrived))
        return await fut

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

            reqs = [r for r, _, _ in batch]
            arrivals = [a for _, _, a in batch]
            try:
                results = await asyncio.to_thread(self._locked_run, reqs, arrivals)
            except Exception as exc:  # noqa: BLE001 — one bad batch must not kill the loop
                for _, fut, _ in batch:
                    if not fut.done():
                        fut.set_exception(exc)
                continue
            for (_, fut, _), res in zip(batch, results):
                if not fut.done():
                    fut.set_result(res)

    def _locked_run(self, reqs: list[Request], arrivals: list[float]) -> list[Result]:
        with self._lock:
            return self._run_batch(reqs, arrivals)

    # ------------------------------------------------------------------ the batch
    @torch.inference_mode()  # thread-local; see the note in manual.py
    def _run_batch(self, reqs: list[Request], arrivals: list[float]) -> list[Result]:
        device = CONFIG.device
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
        first_token_at = time.perf_counter()

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
        done_at = time.perf_counter()

        return [
            Result(
                request_id=reqs[i].request_id,
                token_ids=generated[i],
                text=self.tokenizer.decode(generated[i], skip_special_tokens=True),
                ttft_s=first_token_at - arrivals[i],
                # Everyone in the batch shares the batch's wall clock, because nobody is
                # released until the slowest row finishes. The short requests paying the
                # long request's latency is not a measurement artifact — it is the
                # behaviour, and it is what shows up in p99.
                latency_s=done_at - arrivals[i],
                finish_reason=finish[i],
                prompt_len=int(prompt_lens[i]),
                reserved_tokens=CONFIG.max_seq_len,
                used_tokens=int(prompt_lens[i]) + len(generated[i]),
                wasted_steps=wasted[i],
            )
            for i in range(n)
        ]

    def shutdown(self) -> None:
        # Only cancellable from its own loop; if that loop is already gone the task went
        # with it, so dropping the reference is the whole cleanup.
        if self._runner is not None and self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._runner.cancel)
        self._runner = None
        self._queue = None
        self._loop = None
