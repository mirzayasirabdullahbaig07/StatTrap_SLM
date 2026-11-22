"""Surface-cue audit: can the benchmark be solved WITHOUT statistical knowledge?

Each heuristic below ignores the meaning of the question. On a valid item set, every heuristic
should score ~25% (chance for 4 options). A family where one scores clearly higher has an
ARTIFACT that a model could exploit; fix the generator before running any model.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .tasks import _num, _words


def heuristic_picks(item) -> dict:
    o = item.options; L = list(o)
    out = {"longest": max(L, key=lambda l: len(o[l])), "shortest": min(L, key=lambda l: len(o[l])),
           "always_A": "A", "always_D": "D"}
    qw = _words(item.question)
    out["max_word_overlap_with_question"] = max(L, key=lambda l: len(_words(o[l]) & qw))
    vals = {l: _num(o[l]) for l in L}
    if all(v is not None for v in vals.values()):
        order = sorted(L, key=lambda l: vals[l])
        out.update({"num_min": order[0], "num_lower_median": order[1], "num_upper_median": order[2], "num_max": order[3]})
    return out


def audit_items(items, flag_acc=0.40, flag_p=1e-3) -> pd.DataFrame:
    """Return rows (family, heuristic, n, acc, p_vs_chance, flagged)."""
    hits: dict = {}
    for it in items:
        for h, pick in heuristic_picks(it).items():
            hits.setdefault((it.family, h), []).append(pick == it.answer)
    rows = []
    for (fam, h), v in sorted(hits.items()):
        k, n = int(np.sum(v)), len(v)
        p = binomtest(k, n, 0.25, alternative="greater").pvalue
        p_lo = binomtest(k, n, 0.25, alternative="less").pvalue
        rows.append(dict(family=fam, heuristic=h, n=n, acc=k / n, p_vs_chance=p, flagged=(k / n >= flag_acc and p < flag_p),
                         # 'anti' cue: heuristic is almost NEVER right, so a model could eliminate that option (<= 33% ceiling)
                         anti_cue=(k / n <= 0.10 and p_lo < flag_p)))
    return pd.DataFrame(rows)


def trap_position_check(items) -> pd.DataFrame:
    """Chi-square-free sanity: share of answers / traps in each letter position (should be ~25%)."""
    ans = pd.Series([i.answer for i in items]).value_counts(normalize=True).reindex(list("ABCD"), fill_value=0)
    trp = pd.Series([i.trap for i in items]).value_counts(normalize=True).reindex(list("ABCD"), fill_value=0)
    return pd.DataFrame({"answer_share": ans, "trap_share": trp})
