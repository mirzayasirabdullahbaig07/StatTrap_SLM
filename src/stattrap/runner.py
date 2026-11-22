"""Resumable evaluation runner: appends one CSV row per (item, model, style, temperature, decode_seed)."""
from __future__ import annotations

import csv
from pathlib import Path

from .backends import parse_choice

FIELDS = ["item_id", "base_id", "family", "perm", "framing", "model", "style", "temperature", "decode_seed",
          "chosen", "correct", "trapped", "answer", "trap", "max_new_tokens", "latency_s", "prompt_tokens",
          "completion_tokens", "truncated", "response"]


def run_eval(items, models, styles, backend, out_csv, max_new_tokens=256, temperatures=(0.0,), decode_seeds=(0,),
             progress_every=50) -> int:
    out = Path(out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    key = lambda r: (r["item_id"], r["model"], r["style"], str(float(r["temperature"])), str(int(r["decode_seed"])))
    done = set()
    if out.exists():
        with out.open(encoding="utf-8", newline="") as f:
            done = {key(r) for r in csv.DictReader(f)}
    new, header = 0, not out.exists()
    with out.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if header:
            w.writeheader()
        for model in models:                                   # model-major: each model is loaded once
            for style in styles:
                for temp in temperatures:
                    for ds in (decode_seeds if temp > 0 else (decode_seeds[0],)):   # greedy needs one seed only
                        for it in items:
                            if (it.id, model, style, str(float(temp)), str(int(ds))) in done:
                                continue
                            text, lat, pt, ct = backend.generate(model, it, style, max_new_tokens, temp, ds)
                            ch = parse_choice(text, style, it)
                            w.writerow(dict(item_id=it.id, base_id=it.base_id, family=it.family, perm=it.perm,
                                            framing=it.framing, model=model, style=style, temperature=temp,
                                            decode_seed=ds, chosen=ch or "", correct=int(ch == it.answer),
                                            trapped=int(ch == it.trap), answer=it.answer, trap=it.trap,
                                            max_new_tokens=max_new_tokens, latency_s=round(lat, 4), prompt_tokens=pt,
                                            completion_tokens=ct, truncated=int(ct >= max_new_tokens - 1),
                                            response=text.replace("\n", " ")[:600]))
                            f.flush(); new += 1
                            if new % progress_every == 0:
                                print(f"[{model}/{style}/T={temp}] +{new} rows", flush=True)
    return new
