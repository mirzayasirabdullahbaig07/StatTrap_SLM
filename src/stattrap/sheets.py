"""Spreadsheets for the HUMAN steps of the research (no model can do these for you)."""
from __future__ import annotations

import numpy as np
import pandas as pd

ERROR_CATEGORIES = ["fell_for_trap", "computation_error", "misread_question", "wrong_concept_other",
                    "correct_reasoning_wrong_letter", "truncated_output", "format_failure", "other"]


def verification_sheet(items, n_per_family=4, seed=0) -> pd.DataFrame:
    """Items for an independent statistician to answer BLIND, then compare with the key.
    The verifier's columns are left empty; key columns are at the end so they can be hidden."""
    rng = np.random.default_rng(seed); rows = []
    fams = sorted({i.family for i in items})
    for fam in fams:
        pool = [i for i in items if i.family == fam and i.perm == 0]
        for it in [pool[j] for j in rng.permutation(len(pool))[:n_per_family]]:
            rows.append(dict(item_id=it.id, family=fam, question=it.question, **{f"option_{l}": t for l, t in it.options.items()},
                             verifier_choice="", verifier_confident_yes_no="", verifier_comment="",
                             KEY_correct=it.answer, KEY_trap=it.trap))
    return pd.DataFrame(rows)


def error_sheet(df: pd.DataFrame, items, n=60, seed=0, style="cot") -> pd.DataFrame:
    """Sample wrong answers for manual error-type coding (qualitative analysis section)."""
    by_id = {i.id: i for i in items}
    w = df[(df["style"] == style) & (df.correct == 0)]
    w = w.sample(min(n, len(w)), random_state=seed) if len(w) else w
    rows = []
    for _, r in w.iterrows():
        it = by_id[r.item_id]
        rows.append(dict(item_id=r.item_id, family=r.family, model=r.model, question=it.question,
                         correct_text=it.options[it.answer], trap_text=it.options[it.trap],
                         chosen_text=it.options.get(r.chosen, "(unparsed)"), picked_trap=int(r.trapped),
                         auto_flag="truncated" if r.truncated else ("unparsed" if r.chosen == "" else ""),
                         response=r.response, error_category="", notes=""))
    out = pd.DataFrame(rows)
    out.attrs["categories"] = ERROR_CATEGORIES
    return out
