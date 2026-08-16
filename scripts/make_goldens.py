"""Generate the M1 golden token IDs from HuggingFace `generate()`.

    python -m scripts.make_goldens          # or: make goldens

Run this ONCE on Day 1, then treat the output as immutable. Regenerating goldens to make
a failing test pass is the single most effective way to destroy this project's value —
if a phase disagrees with the goldens, the phase is wrong.

Token IDs, not text: comparing decoded strings introduces tokenizer round-trip fuzz that
hides real off-by-one bugs.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from inference_server.config import CONFIG
from inference_server.model import load

GOLDENS_DIR = Path(__file__).resolve().parents[1] / "tests" / "goldens"
PROMPTS = GOLDENS_DIR / "prompts.json"


def main() -> None:
    spec = json.loads(PROMPTS.read_text())
    if spec["model_id"] != CONFIG.model_id:
        raise SystemExit(
            f"goldens are for {spec['model_id']!r} but MODEL_ID is {CONFIG.model_id!r}. "
            "Changing models means regenerating goldens deliberately, not by accident."
        )

    model, tokenizer = load()
    out = {"model_id": CONFIG.model_id, "cases": {}}

    for case in spec["cases"]:
        ids = tokenizer(case["prompt"], return_tensors="pt").input_ids.to(CONFIG.device)
        with torch.no_grad():
            generated = model.generate(
                ids,
                max_new_tokens=case["max_tokens"],
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_ids = generated[0, ids.shape[1]:].tolist()
        out["cases"][case["id"]] = {
            "prompt": case["prompt"],
            "max_tokens": case["max_tokens"],
            "token_ids": new_ids,
            "text": tokenizer.decode(new_ids, skip_special_tokens=True),
            "finish_reason": "eos" if new_ids and new_ids[-1] == tokenizer.eos_token_id else "length",
        }
        print(f"{case['id']:<24} {len(new_ids):>3} tokens  {out['cases'][case['id']]['finish_reason']}")

    dest = GOLDENS_DIR / f"{CONFIG.model_id.replace('/', '_')}_greedy.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
