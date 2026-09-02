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
