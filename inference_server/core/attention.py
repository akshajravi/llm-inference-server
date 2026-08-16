"""Paged attention — P3 (Days 6-9).

Attention can no longer assume KV is contiguous; it walks the block table and gathers.

Two implementations, deliberately ordered:
  1. paged_attention_gather()  — PyTorch, correct and slow. THIS is the shipping target.
     It satisfies M3 and unblocks P4, which is what the sprint is graded on.
  2. paged_attention_triton()  — S3, stretch (Day 14 only). Verify numerically against
     the gather path on random inputs *first*, then run M1 end-to-end.

Report the gather path's throughput honestly and name the kernel as scoped-out work.

See IMPLEMENTATION_GUIDE.md "Days 6-9" and "Day 14".
"""

from __future__ import annotations


def paged_attention_gather(*args, **kwargs):
    """Lands in P3 (Days 6-9). Correct-but-slow; the shipping path."""
    raise NotImplementedError("paged_attention_gather lands in P3 (Days 6-9)")


def paged_attention_triton(*args, **kwargs):
    """S3, stretch (Day 14). Must match paged_attention_gather within float tolerance."""
    raise NotImplementedError("Triton kernel is stretch (S3) — see IMPLEMENTATION_GUIDE.md 'Day 14'")
