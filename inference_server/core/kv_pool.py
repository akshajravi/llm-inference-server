"""The physical KV memory — P3 (Days 6-9). One tensor pair per layer, allocated once.

This is the "physical RAM" row of the OS analogy. `block_allocator.py` hands out indices
into it; nothing here knows which sequence owns what, and nothing here knows about
scheduling. Its whole job is: reserve the memory up front, and answer how much of it
there is.

    per layer:  k[num_blocks, block_size, num_kv_heads, head_dim]
                v[num_blocks, block_size, num_kv_heads, head_dim]

**Why block-major rather than token-major.** Indexing `k[block]` yields the whole block
contiguously, which is what the Day 8 gather path wants — one index_select over a
sequence's block table instead of a scatter over individual token positions.

**Why allocate up front.** The contiguous engines (P0-P2) reserve `max_seq_len` per
request because they cannot know the real output length at admission time, which is the
84.4% waste M3 exists to beat. Paging does not remove the reservation, it moves it: one
process-lifetime allocation whose size is known, instead of a per-request one whose size
is a guess. Steady-state memory becomes flat by construction, which is also what M4's
30-minute overload run has to demonstrate.

Day 6 scope is the reservation and the arithmetic. Reads and writes against a block
table land on Days 7-8; see attention.py.

See IMPLEMENTATION_GUIDE.md "Days 6-9".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from inference_server.config import CONFIG


@dataclass(frozen=True)
class ModelDims:
    """The four numbers that decide how big a block is. Read off the model config so a
    different `MODEL_ID` resizes the pool without anyone editing a constant."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype

    @classmethod
    def from_model(cls, model, dtype: torch.dtype | None = None) -> "ModelDims":
        cfg = model.config
        heads = getattr(cfg, "num_attention_heads", None) or cfg.n_head
        hidden = getattr(cfg, "hidden_size", None) or cfg.n_embd
        return cls(
            num_layers=getattr(cfg, "num_hidden_layers", None) or cfg.n_layer,
            # Grouped-query models cache far fewer heads than they attend with. gpt2 has
            # no such attribute and falls back to full multi-head, which is correct for
            # it — but reading the attribute means a Llama-family model sizes correctly
            # instead of over-reserving by the group factor.
            num_kv_heads=getattr(cfg, "num_key_value_heads", None) or heads,
            head_dim=getattr(cfg, "head_dim", None) or hidden // heads,
            dtype=dtype or CONFIG.torch_dtype,
        )

    def bytes_per_token(self) -> int:
        """K and V, every layer, one token's worth."""
        itemsize = torch.empty(0, dtype=self.dtype).element_size()
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * itemsize

    def bytes_per_block(self, block_size: int = CONFIG.block_size) -> int:
        return self.bytes_per_token() * block_size

    def blocks_for_budget(self, budget_bytes: int, block_size: int = CONFIG.block_size) -> int:
        """How many blocks fit in a memory budget.

        This is the sizing decision the guide describes as "fill available GPU memory
        after weights and activations" — deliberately a function of an explicit budget
        rather than an auto-probe, because the amount of headroom activations need is a
        judgement call that belongs in the caller, not buried in the allocator.
        """
        return max(0, budget_bytes // self.bytes_per_block(block_size))


@dataclass
class HostBlocks:
    """A sequence's blocks, parked in host memory while it is SWAPPED (P4, S2).

    Per-layer CPU tensors shaped [len(blocks), block_size, num_kv_heads, head_dim], one
    list for K and one for V — the pool's own layout minus the block dimension's
    meaning, so the copy back is one `index_copy_` per layer and nothing is reshaped.
    Which physical blocks these came from is deliberately *not* recorded: by the time
    the sequence is re-admitted those indices belong to someone else, and the new set
    only has to be the same length.
    """

    k: list[torch.Tensor]
    v: list[torch.Tensor]

    def __len__(self) -> int:
        """Number of blocks — what the scheduler must allocate before swapping back in."""
        return int(self.k[0].shape[0]) if self.k else 0

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)


class PagedKVPool:
    """The preallocated blocks. Indices come from BlockAllocator; meaning comes later."""

    def __init__(
        self,
        dims: ModelDims,
        num_blocks: int = CONFIG.num_blocks,
        block_size: int = CONFIG.block_size,
        device: str = CONFIG.device,
    ) -> None:
        self.dims = dims
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = device
        shape = (num_blocks, block_size, dims.num_kv_heads, dims.head_dim)
        # A list of per-layer tensors, not one stacked [layer, ...] tensor: the model is
        # called layer by layer, and a stacked tensor would make every per-layer view a
        # slice of a much larger allocation for no gain.
        self.k = [torch.zeros(shape, dtype=dims.dtype, device=device) for _ in range(dims.num_layers)]
        self.v = [torch.zeros(shape, dtype=dims.dtype, device=device) for _ in range(dims.num_layers)]

    @classmethod
    def from_model(cls, model, **kwargs) -> "PagedKVPool":
        return cls(ModelDims.from_model(model), **kwargs)

    @property
    def num_slots(self) -> int:
        """Total token slots. This is the ceiling on concurrent context the server can
        hold, and it is the number M3's "how many more sequences fit" claim divides."""
        return self.num_blocks * self.block_size

    @property
    def nbytes(self) -> int:
        return self.num_blocks * self.dims.bytes_per_block(self.block_size)

    # ------------------------------------------------------------- swap (P4, S2)
    def swap_out(self, blocks: list[int]) -> HostBlocks:
        """Copy the named blocks to host memory. The caller frees them afterwards.

        Gathers by index (`index_select`) rather than iterating blocks, so a victim with
        forty blocks costs one device->host transfer per layer, not forty. On cuda the
        destination is pinned: a pageable copy would stage through a pinned bounce
        buffer anyway, so pinning up front is the same bytes with one fewer copy and it
        is what lets swap_in be `non_blocking`. Nothing is pinned on cpu/mps because
        there is no transfer to accelerate and mps does not support it.

        Does not touch the free list. The pool knows tensors, the allocator knows
        indices, and a swap that freed as a side effect would be the one place in the
        system where memory changed hands invisibly — precisely the kind of path that
        leaks under churn.
        """
        idx = torch.as_tensor(blocks, dtype=torch.long, device=self.device)
        pin = self.device == "cuda"

        def to_host(layer: torch.Tensor) -> torch.Tensor:
            gathered = layer.index_select(0, idx)
            host = torch.empty(gathered.shape, dtype=gathered.dtype, device="cpu", pin_memory=pin)
            host.copy_(gathered)
            return host

        return HostBlocks(k=[to_host(t) for t in self.k], v=[to_host(t) for t in self.v])

    def swap_in(self, host: HostBlocks, blocks: list[int]) -> None:
        """Copy a host copy back into `blocks`, which need not be the ones it left.

        Length must match exactly. A shorter target would silently drop the tail of the
        sequence's cache and a longer one would leave uninitialised blocks that
        attention reads as garbage, so both are refused loudly rather than clamped.
        """
        if len(blocks) != len(host):
            raise ValueError(f"swap_in: {len(host)} host blocks into {len(blocks)} device blocks")
        idx = torch.as_tensor(blocks, dtype=torch.long, device=self.device)
        non_blocking = self.device == "cuda"
        for dst, src in zip(self.k, host.k):
            dst.index_copy_(0, idx, src.to(self.device, non_blocking=non_blocking))
        for dst, src in zip(self.v, host.v):
            dst.index_copy_(0, idx, src.to(self.device, non_blocking=non_blocking))

    def release(self) -> None:
        """Drop the tensors. The pool object survives so `describe()` and the sizes still
        answer, but the memory is returned to the device allocator — which only takes
        effect once the caller also empties the device cache (PagedEngine.shutdown)."""
        self.k = []
        self.v = []

    def describe(self) -> str:
        """Goes in the results file and the writeup — a waste number is not meaningful
        without the pool size it was measured against."""
        gib = self.nbytes / 1024**3
        return (
            f"{self.num_blocks} blocks x {self.block_size} tokens = {self.num_slots} slots, "
            f"{gib:.2f} GiB on {self.device} ({self.dims.num_layers}L "
            f"x {self.dims.num_kv_heads}H x {self.dims.head_dim}D, {self.dims.dtype})"
        )

    def __repr__(self) -> str:
        return f"PagedKVPool({self.describe()})"
