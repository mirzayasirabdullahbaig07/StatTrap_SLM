"""Statistics for the paper, computed from the results CSV.

Independence: permuted copies of one base item are NOT independent observations. All confidence
intervals and tests therefore resample / sign-flip at the BASE-ITEM (cluster) level, never at row level.
The 'primary' analysis uses the lowest temperature and lowest decode seed (greedy by default).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .registry import REGISTRY

CHANCE = 0.25
B = 4000


def primary(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df.temperature == df.temperature.min()) & (df.decode_seed == df.decode_seed.min())]


def _cluster_means(g: pd.DataFrame, col: str) -> np.ndarray:
    return g.groupby("base_id")[col].mean().to_numpy(float)


def cluster_boot(x: np.ndarray, seed=0, level=0.95, null=None):
    """Mean, percentile CI and (optionally) one-sided p for H1: mean > null, resampling clusters."""
    rng = np.random.default_rng(seed); n = len(x)
    means = x[rng.integers(0, n, (B, n))].mean(1)
    a = (1 - level) / 2
    p = None if null is None else float((np.sum(means <= null) + 1) / (B + 1))
    return float(x.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a)), p


def sign_flip_p(d: np.ndarray, seed=0, n_flip=20000) -> float:
    """Two-sided paired sign-flip permutation test on cluster-level differences."""
    d = np.asarray(d, float); d = d[d != 0]
    if len(d) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.sum()); signs = rng.choice([-1.0, 1.0], (n_flip, len(d)))
    return float((np.sum(np.abs((signs * d).sum(1)) >= obs - 1e-12) + 1) / (n_flip + 1))


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i])); adj[i] = run
    return adj


def summary(df: pd.DataFrame) -> pd.DataFrame:
    d = primary(df); rows = []
    for (model, style), g in d.groupby(["model", "style"]):
        acc = cluster_boot(_cluster_means(g, "correct"), null=CHANCE)
        trap = cluster_boot(_cluster_means(g, "trapped"))
        rows.append(dict(model=model, style=style, n_rows=len(g), n_base_items=g.base_id.nunique(),
                         acc=acc[0], acc_lo=acc[1], acc_hi=acc[2], p_acc_gt_chance=acc[3],
                         trap_rate=trap[0], trap_lo=trap[1], trap_hi=trap[2],
                         unparsed=float((g.chosen == "").mean()), truncated=float(g.truncated.mean())))
    return pd.DataFrame(rows)


def by_family(df: pd.DataFrame) -> pd.DataFrame:
    d = primary(df); rows = []
    for (model, style, fam), g in d.groupby(["model", "style", "family"]):
        acc = cluster_boot(_cluster_means(g, "correct"))
        rows.append(dict(model=model, style=style, family=fam, n_base_items=g.base_id.nunique(), acc=acc[0],
                         acc_lo=acc[1], acc_hi=acc[2], trap_rate=float(g.trapped.mean())))
    return pd.DataFrame(rows)


def position_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Chosen-letter frequency (should be ~25% each) and accuracy by position of the correct answer."""
    d = primary(df)
    freq = (d[d.chosen != ""].groupby(["model", "style"]).chosen.value_counts(normalize=True)
            .unstack().reindex(columns=list("ABCD")).fillna(0).add_prefix("freq_chosen_"))
    acc = d.groupby(["model", "style", "answer"]).correct.mean().unstack().reindex(columns=list("ABCD")).add_prefix("acc_if_ans_")
    return freq.join(acc).reset_index()


def consistency(df: pd.DataFrame, items) -> pd.DataFrame:
    """Share of base items where the model picks the SAME option text under every option order."""
    text = {(it.id, l): t for it in items for l, t in it.options.items()}
    d = primary(df); d = d[d.chosen != ""].copy()
    d["chosen_text"] = [text[(i, c)] for i, c in zip(d.item_id, d.chosen)]
    g = d.groupby(["model", "style", "base_id"]).agg(n_perm=("perm", "nunique"), n_text=("chosen_text", "nunique"))
    g = g[g.n_perm > 1]
    if g.empty:
        return pd.DataFrame(columns=["model", "style", "choice_consistency"])
    return (g.assign(consistent=(g.n_text == 1)).groupby(["model", "style"]).consistent.mean()
            .rename("choice_consistency").reset_index())


def cot_vs_direct(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Paired CoT-minus-direct accuracy per model (optionally per family): cluster bootstrap CI + sign-flip p."""
    d = primary(df); rows = []
    keys = ["model"] + ([by] if by else [])
    for k, g in d.groupby(keys):
        w = g.groupby(["base_id", "style"]).correct.mean().unstack()
        if not {"direct", "cot"} <= set(w.columns):
            continue
        diff = (w["cot"] - w["direct"]).dropna().to_numpy()
        m, lo, hi, _ = cluster_boot(diff)
        rows.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)), n_base_items=len(diff), acc_direct=float(w["direct"].mean()),
                         acc_cot=float(w["cot"].mean()), cot_gain=m, gain_lo=lo, gain_hi=hi, p_sign_flip=sign_flip_p(diff)))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out.p_sign_flip.to_numpy())
    return out


def pairwise_models(df: pd.DataFrame, style="direct") -> pd.DataFrame:
    """All model pairs, paired over base items, Holm-adjusted across the pairs."""
    d = primary(df); d = d[d["style"] == style]
    w = d.groupby(["base_id", "model"]).correct.mean().unstack(); models = list(w.columns); rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            diff = (w[models[i]] - w[models[j]]).dropna().to_numpy()
            if len(diff):
                m, lo, hi, _ = cluster_boot(diff)
                rows.append(dict(style=style, model_a=models[i], model_b=models[j], n=len(diff), diff_a_minus_b=m,
                                 lo=lo, hi=hi, p_sign_flip=sign_flip_p(diff)))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out.p_sign_flip.to_numpy())
    return out


def scaling(df: pd.DataFrame, registry=REGISTRY) -> pd.DataFrame:
    """Per model: params, accuracy, trap rate. Spearman rho of log-size vs metrics is attached (needs >= 3 models)."""
    s = summary(df); s["params_b"] = s.model.map(lambda m: registry.get(m, {}).get("params_b", np.nan))
    s["family"] = s.model.map(lambda m: registry.get(m, {}).get("family", "unknown"))
    out = []
    for style, g in s.groupby("style"):
        g = g.dropna(subset=["params_b"]).copy()
        if len(g) >= 3:
            for col in ("acc", "trap_rate"):
                r, p = spearmanr(np.log10(g.params_b), g[col])
                g[f"spearman_{col}"], g[f"spearman_{col}_p"] = r, p
        out.append(g)
    return pd.concat(out) if out else s


def robustness(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy by temperature (mean over decode seeds; SD across seeds) - does sampling change the picture?"""
    g = df.groupby(["model", "style", "temperature", "decode_seed"]).correct.mean().reset_index()
    return (g.groupby(["model", "style", "temperature"]).correct.agg(acc="mean", sd_across_seeds="std", n_seeds="count")
            .reset_index())


def compare_runs(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a="A", label_b="B") -> pd.DataFrame:
    """Paired difference in accuracy between two runs over the SAME base items (e.g. framing 0 vs framing 2)."""
    rows = []
    a, b = primary(df_a), primary(df_b)
    for (model, style), ga in a.groupby(["model", "style"]):
        gb = b[(b.model == model) & (b["style"] == style)]
        wa, wb = ga.groupby("base_id").correct.mean(), gb.groupby("base_id").correct.mean()
        diff = (wb - wa).dropna().to_numpy()
        if len(diff):
            m, lo, hi, _ = cluster_boot(diff)
            rows.append(dict(model=model, style=style, n_base_items=len(diff), diff=f"{label_b}-{label_a}", mean_diff=m,
                             lo=lo, hi=hi, p_sign_flip=sign_flip_p(diff)))
    return pd.DataFrame(rows)


def truncation_report(df: pd.DataFrame) -> pd.DataFrame:
    return (primary(df).groupby(["model", "style"]).agg(truncated=("truncated", "mean"), unparsed=("chosen", lambda s: float((s == "").mean())),
                                                         mean_completion_tokens=("completion_tokens", "mean")).reset_index())
