"""Programmatic statistical-reasoning items with a KNOWN correct answer and a KNOWN trap.

Each item is a 4-option multiple-choice question about a classic statistical mistake.

Design guarantees (each is enforced by tests, see tests/):
  * ground truth is COMPUTED where possible (Bayes, Bonferroni recount, verified Simpson reversal);
  * option TEXT LENGTH is uninformative: every option has several phrasings of different length,
    chosen independently, so "pick the longest/shortest option" scores at chance;
  * numeric options are not guessable by rank ("pick the smallest/median/largest number");
  * every base item can be re-asked with options permuted (position bias / consistency);
  * three light question framings for a robustness check (framing 0/1/2).
`stattrap audit-items` re-measures all surface-cue heuristics and flags any family where one beats chance.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass

import numpy as np

LETTERS = "ABCD"


@dataclass
class Item:
    id: str
    base_id: str
    family: str
    question: str
    options: dict          # letter -> text
    answer: str            # correct letter
    trap: str              # letter of the tempting WRONG option
    perm: int = 0
    framing: int = 0

    def prompt(self, style: str = "direct") -> str:
        opts = "\n".join(f"{l}. {t}" for l, t in self.options.items())
        tail = ("Answer with a single letter." if style == "direct" else
                "Think step by step briefly, then finish with a final line 'Answer: <letter>'.")
        return f"{self.question}\n\n{opts}\n\n{tail}"


def _rng(*parts) -> np.random.Generator:
    """Seed from a SHA-256 of all parts (never from a truncated string)."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _pick(rng, variants):
    return variants[int(rng.integers(len(variants)))]


def _pick_n(rng, pool, n, exclude=()):
    pool = [p for p in dict.fromkeys(pool) if p not in set(exclude)]
    idx = rng.permutation(len(pool))[:n]
    return [pool[i] for i in idx]


def _arrange(base_id, family, question, correct, trap, others, perm, seed, framing):
    """Shuffle option order deterministically per (base item, perm)."""
    texts = [correct, trap] + list(others)
    assert len(set(texts)) == 4, f"duplicate options in {family}: {texts}"
    rng = _rng(seed, base_id, perm)
    order = rng.permutation(4)
    options = {LETTERS[i]: texts[j] for i, j in enumerate(order)}
    ans = next(l for l, t in options.items() if t == correct)
    trp = next(l for l, t in options.items() if t == trap)
    return Item(f"{base_id}#p{perm}", base_id, family, question, options, ans, trp, perm, framing)


def _words(t):
    return {w for w in re.findall(r"[a-z]{4,}", t.lower())}


def _num(t):
    try:
        return float(re.sub(r"[%,]", "", t))
    except ValueError:
        return None


def _balanced(rng, question, c_vars, t_vars, o_pool, numeric=False):
    """Choose option texts by rejection sampling so surface cues carry no information:
    the correct option's LENGTH rank (or numeric VALUE rank) is uniform over 0..3, and the option that
    shares the most words with the question is the correct one only ~25% of the time.
    Falls back to relaxed constraints (then to an unconstrained draw) if a family cannot satisfy them."""
    qw = _words(question)
    for attempt in range(8):
        target, want_hit = int(rng.integers(4)), bool(rng.random() < 0.25)
        for _ in range(300):
            c, t = _pick(rng, c_vars), _pick(rng, t_vars)
            o = _pick_n(rng, o_pool, 2, exclude=(c, t))
            texts = [c, t] + o
            if len(o) < 2 or len(set(texts)) < 4:
                continue
            key = [_num(x) for x in texts] if numeric else [len(x) for x in texts]
            if len(set(key)) < 4:
                continue
            if sorted(key).index(key[0]) != target:
                continue
            if attempt < 4 and not numeric:
                ov = [len(_words(x) & qw) for x in texts]
                top = [i for i, v in enumerate(ov) if v == max(ov)]
                if len(top) == 1 and ((top[0] == 0) != want_hit):
                    continue
            return c, t, o
    c, t = _pick(rng, c_vars), _pick(rng, t_vars)
    return c, t, _pick_n(rng, o_pool, 2, exclude=(c, t))


FRAMINGS = {
    0: lambda q: q,
    1: lambda q: f"Scenario: {q}\nWhich option is best?",
    2: lambda q: f"{q} Choose the single best answer.",
}


# ------------------------------------------------------------------ families
# each returns (question, correct_variants, trap_variants, distractor_pool, numeric?)
# `_balanced` then picks concrete texts so length / numeric rank / word-overlap cues are uninformative.
def f_paired(rng, i):
    meas = _pick(rng, ["systolic blood pressure", "reaction time", "exam score", "resting heart rate"])
    subj = _pick(rng, ["patients", "students", "workers", "volunteers"])
    n = int(rng.integers(12, 60))
    q = (f"{n} {subj} each had their {meas} measured before and again after an intervention. "
         f"The researcher wants to test whether the intervention changed the mean {meas}. Which test is most appropriate?")
    return (q, ["Paired t-test on the within-subject differences", "Paired t-test",
                "A paired-samples t-test, since each subject is measured twice", "Paired t-test of the changes"],
            ["Independent two-sample t-test", "Two-sample t-test comparing the before group with the after group",
             "Welch's t-test between the two sets of measurements", "Unpaired t-test"],
            ["Chi-square test of independence", "One-way ANOVA across three groups", "Chi-square test", "One-way ANOVA",
             "Kruskal-Wallis test across three independent groups", "Fisher's exact test on the two sets of measurements",
             "Mann-Whitney U test on the pooled scores", "Pearson correlation between subjects' ages and scores"], False)


def f_bonferroni(rng, i):
    k_true = int(rng.integers(3, 9)); m = int(rng.integers(max(16, k_true + 10), 41))
    p = rng.uniform(0.02, 1.0, m)
    p[:k_true] = rng.uniform(1e-6, 0.045 / m, k_true)            # clearly significant even after correction
    near = int(rng.integers(1, 5))
    p[k_true:k_true + near] = rng.uniform(0.01, 0.049, near)     # significant only WITHOUT correction
    p = np.round(p, 6); p = rng.permutation(p)
    sig_bonf = int((p < 0.05 / m).sum()); sig_naive = int((p < 0.05).sum())
    pl = ", ".join(f"{x:g}" for x in p)
    ask_not = bool(rng.random() < 0.5)                           # flip direction so ranks are not one-sided
    word = "NOT significant" if ask_not else "significant"
    correct, naive = (m - sig_bonf, m - sig_naive) if ask_not else (sig_bonf, sig_naive)
    q = (f"A study ran {m} independent hypothesis tests and obtained these p-values: {pl}. "
         f"Using the Bonferroni correction at family-wise alpha = 0.05, how many results are {word}?")
    cand = [str(v) for v in range(0, m + 1) if abs(v - correct) <= 4 or abs(v - naive) <= 4]
    return q, [str(correct)], [str(naive)], cand, True


def f_simpson(rng, i):
    for _ in range(1000):
        nm_a, nm_b = int(rng.integers(8, 20)), int(rng.integers(60, 120))
        ns_a, ns_b = int(rng.integers(60, 120)), int(rng.integers(8, 20))
        rm_a, rm_b = rng.uniform(0.80, 0.95), rng.uniform(0.65, 0.85)
        rs_a, rs_b = rng.uniform(0.40, 0.60), rng.uniform(0.25, 0.45)
        sm_a, sm_b, ss_a, ss_b = (round(nm_a * rm_a), round(nm_b * rm_b), round(ns_a * rs_a), round(ns_b * rs_b))
        if (sm_a / nm_a > sm_b / nm_b and ss_a / ns_a > ss_b / ns_b and
                (sm_a + ss_a) / (nm_a + ns_a) < (sm_b + ss_b) / (nm_b + ns_b)):
            break
    q = ("Two treatments were compared in an observational study; physicians assigned patients by judged severity. "
         f"Mild cases: Treatment A {sm_a}/{nm_a} recovered, Treatment B {sm_b}/{nm_b} recovered. "
         f"Severe cases: Treatment A {ss_a}/{ns_a} recovered, Treatment B {ss_b}/{ns_b} recovered. "
         "Which conclusion do the data best support?")
    return (q, ["Treatment A is better, because it does better within both severity groups",
                "A is better in each severity group, so A should be preferred", "Treatment A",
                "Treatment A, since it wins in the mild and the severe group alike"],
            ["Treatment B is better, because its overall recovery rate is higher",
             "B, since it has the higher pooled recovery rate", "Treatment B",
             "Treatment B, since more of its patients recovered overall"],
            ["The treatments are equally effective", "Recovery depends only on severity, so the treatments cannot be compared",
             "No difference between treatments", "Treatment B helps mild cases while Treatment A helps severe cases",
             "Neither treatment can be said to work", "Treatment A helps mild cases while Treatment B helps severe cases"], False)


def f_baserate(rng, i):
    sens = float(rng.choice([0.90, 0.95, 0.99])); spec = float(rng.choice([0.90, 0.95, 0.98]))
    prev = float(rng.choice([0.001, 0.005, 0.01, 0.02]))
    ppv = sens * prev / (sens * prev + (1 - spec) * (1 - prev))
    fmt = lambda v: f"{min(max(v, 0.0005), 0.999):.1%}"
    base = (f"A screening test has sensitivity {sens:.0%} and specificity {spec:.0%}. The condition affects {prev:.1%} of the "
            "population. A randomly chosen person tests positive. ")
    if rng.random() < 0.5:
        q = base + "About how likely is it that they truly have the condition?"
        c, t = fmt(ppv), fmt(sens)
        pool = [fmt(v) for v in (spec, 1 - ppv, prev * 10, ppv * 2, ppv / 2, ppv * 3, 1 - spec, sens * spec, ppv / 4, 1 - sens)]
    else:
        q = base + "About what percentage of all positive results are false alarms?"
        c, t = fmt(1 - ppv), fmt(1 - spec)
        pool = [fmt(v) for v in (sens, ppv, 1 - sens, (1 - ppv) / 2, 1 - ppv / 2, spec, prev * 10, 1 - sens * spec, 1 - ppv * 2)]
    return q, [c], [t], [x for x in pool if x not in (c, t)], True


def f_spurious(rng, i):
    r2 = rng.uniform(0.70, 0.95); n = int(rng.integers(120, 600))
    q = (f"An analyst regresses one trending daily series (a cumulative total) on another unrelated-looking trending series, "
         f"n = {n}. The slope has p < 0.001 and R-squared = {r2:.2f}, but the residuals are strongly autocorrelated. "
         "Which conclusion is most justified?")
    return (q, ["The fit may be spurious because of non-stationarity; difference the series or test for cointegration first",
                "Likely spurious: the series are non-stationary, so difference them or test for cointegration",
                "Possibly a spurious regression; check stationarity and cointegration before trusting it",
                "Probably spurious; test for cointegration"],
            ["The tiny p-value shows the two series are genuinely related",
             "The two series are significantly related, as the p-value shows",
             "A strong, significant relationship exists (high R-squared and p < 0.001)", "They are clearly related"],
            ["The result is impossible, since p-values below 0.001 cannot occur", "Adding more trending predictors will confirm the relationship",
             "The sample is too large for the regression to be valid", "Raise the significance level to 0.10 to be safe",
             "The high R-squared proves the model overfits and must be discarded", "The residuals do not matter once p is small",
             "More data points will remove the problem"], False)


def f_ci(rng, i):
    mu = round(float(rng.uniform(5, 50)), 1); half = round(float(rng.uniform(0.5, 3)), 1)
    q = (f"A study reports a sample mean of {mu} with a 95% confidence interval of [{round(mu - half, 1)}, {round(mu + half, 1)}]. "
         "Which interpretation is correct under standard frequentist inference?")
    return (q, ["If the study were repeated many times, about 95% of such intervals would contain the true mean",
                "About 95% of intervals built this way would capture the true mean",
                "The procedure captures the true mean in 95% of repeated studies",
                "Across repeated samples, 95% of such intervals cover the true mean"],
            ["There is a 95% probability that the true mean lies inside this particular interval",
             "The true mean lies in this interval with 95% probability",
             "This interval has a 95% chance of containing the true mean", "The true mean is 95% likely to be in this range"],
            ["95% of individual observations fall inside this interval", "The sample mean equals the population mean with 95% probability",
             "95% of future sample means will fall inside this interval",
             "The population mean is within 1.96 standard deviations of every observation",
             "The interval shows where 95% of the population lies", "A wider interval means a more precise estimate"], False)


def f_pvalue(rng, i):
    p = float(rng.choice([0.01, 0.02, 0.03, 0.04]))
    q = f"A two-sided test returns p = {p}. Which of these statements about the result is correct?"
    return (q, [f"If the null hypothesis were true, data at least this extreme would occur with probability {p}",
                f"Assuming H0, results this extreme arise {p:.0%} of the time", f"The probability of data this extreme under H0 is {p}",
                f"Under the null, a result at least as extreme arises about {p:.0%} of the time",
                f"Were H0 true, such extreme data would be seen with probability {p}", f"P(data this extreme | H0) = {p}"],
            [f"There is a {p:.0%} probability that the null hypothesis is true, given the data we observed",
             f"The null hypothesis has a {p:.0%} chance of being true", f"The chance that the result is a fluke is {p:.0%}",
             f"The null hypothesis is {p:.0%} likely to be correct", f"Given these data, H0 is true with probability {p}",
             f"There is only a {p:.0%} probability that the finding is due to chance alone, so H0 is unlikely"],
            [f"The effect is real with {1 - p:.0%} probability", f"The result would replicate in {1 - p:.0%} of future studies",
             f"The effect size is {1 - p:.0%} of the maximum", f"{1 - p:.0%} of the variance is explained",
             f"The alternative hypothesis has a {p:.0%} probability of being false", "The study is guaranteed to replicate",
             f"If the study were repeated, it would reach p < 0.05 in {1 - p:.0%} of repetitions",
             f"The probability of a Type I error in this particular study is exactly {p}", f"The test has {1 - p:.0%} power",
             f"The observed effect is {1 - p:.0%} likely to be larger than zero in the population"], False)


def f_rtm(rng, i):
    n = int(rng.integers(30, 120)); gain = round(float(rng.uniform(3, 12)), 1)
    q = (f"The {n} students with the lowest scores on a first test were given extra tutoring. On a second test their mean score rose by "
         f"{gain} points. There was no control group. Which statement is most appropriate?")
    return (q, ["Part of the rise may be regression to the mean, so a control group is needed to estimate the tutoring effect",
                "Regression to the mean could explain some of the gain; a control group is needed",
                "The gain may partly reflect regression to the mean", "Some of this gain is probably regression to the mean"],
            ["The tutoring caused the improvement, because scores rose after it", f"Tutoring raised scores by {gain} points",
             "The rise shows the tutoring worked", "The tutoring clearly produced the whole gain"],
            ["Scores always rise on a second test because of practice effects alone", "The improvement is too small to be analysed",
             "The lowest scorers must have been cheating on the first test", "A second test is invalid because the same students took it",
             "The gain is exactly what random guessing would give", "Selecting the lowest scorers makes the comparison fair"], False)


def f_survivor(rng, i):
    n = int(rng.integers(30, 80)); pct = int(rng.integers(65, 92))
    q = (f"A consultant studies the {n} best-performing companies still listed today and finds that {pct}% had a founder who took "
         "big risks. She concludes that bold risk-taking drives high returns. Which criticism is most important?")
    return (q, ["Survivorship bias: failed companies with equally risk-taking founders are missing from the sample",
                "Failed firms are excluded, so the sample is biased toward winners",
                "The sample omits companies that no longer exist, so the link may be an artefact",
                "Survivorship bias in the sample"],
            [f"Nothing is wrong: {pct}% is a large majority, so risk-taking raises returns",
             "Risk-taking clearly drives returns because most top performers were bold",
             "The conclusion is sound because the percentage is high", "Most winners were bold, so boldness works"],
            ["A larger sample of the same listed companies would fix the problem", "Percentages cannot be compared across companies",
             "The study should have used the average rather than the best performers",
             "Founders' risk-taking cannot be measured, so no study is possible", "The sample is too small to compute a percentage",
             "Returns should have been measured in a different currency"], False)


def f_confound(rng, i):
    r = round(float(rng.uniform(0.6, 0.9)), 2); n = int(rng.integers(24, 120))
    x, y, z = _pick(rng, [("ice-cream sales", "drowning incidents", "hot weather"),
                          ("shoe size", "reading score among school children", "age"),
                          ("coffee shops per district", "laptops sold per district", "population size")])
    q = f"Across {n} observations, {x} and {y} are strongly correlated (r = {r}). Which interpretation is best supported?"
    return (q, [f"A common cause such as {z} may drive both; the correlation alone does not show causation",
                f"{z.capitalize()} could be a confounder behind both variables", "Correlation does not establish causation",
                f"Both may be driven by {z}"],
            [f"{x.capitalize()} cause changes in {y}", f"Increasing {x} will increase {y}",
             f"The strong correlation shows that {x} drive {y}", f"{x.capitalize()} are the reason for {y}"],
            [f"{y.capitalize()} cause changes in {x}", f"An r of {r} is too high to be real", "The variables are independent because r is below 1",
             "The correlation is spurious because the sample is under 200", "A correlation this large proves a perfect linear law",
             f"Reducing {y} would be impossible"], False)


def f_power(rng, i):
    n = int(rng.integers(8, 20)); p = float(rng.choice([0.12, 0.18, 0.21, 0.27, 0.34]))
    q = (f"A small trial with {n} patients per arm compares a new drug with placebo and reports p = {p} for the primary outcome. "
         "Which conclusion is justified?")
    return (q, ["The study may be underpowered; a non-significant result does not show the drug has no effect",
                "Absence of significance is not evidence of no effect with so few patients", "The result is inconclusive",
                "With so few patients the trial cannot rule out a real effect"],
            ["The drug has no effect", "The trial proves the drug does not work", "Because p is above 0.05, the drug is ineffective",
             "The drug is shown to be useless"],
            [f"The null hypothesis is true with {1 - p:.0%} probability", "The drug is harmful",
             "The p-value should be halved because the trial is small", "Any effect must be larger than the one observed",
             "The placebo arm must have been mislabelled", "A larger p-value means a larger effect"], False)


FAMILIES = {"paired_design": f_paired, "multiple_testing": f_bonferroni, "simpson": f_simpson,
            "base_rate": f_baserate, "spurious_regression": f_spurious, "ci_meaning": f_ci,
            "pvalue_meaning": f_pvalue, "regression_to_mean": f_rtm, "survivorship": f_survivor,
            "confounding": f_confound, "underpowered_null": f_power}


def make_items(n_per_family: int = 30, seed: int = 0, n_perms: int = 1, framing: int = 0,
               families: list[str] | None = None) -> list[Item]:
    """n_per_family base items per family, each asked in `n_perms` option orders, with question `framing` 0/1/2."""
    items = []
    for fam in (families or FAMILIES):
        fn = FAMILIES[fam]
        rng = _rng(seed, fam)
        for i in range(n_per_family):
            q, c_vars, t_vars, pool, numeric = fn(rng, i)
            c, t, others = _balanced(rng, q, c_vars, t_vars, pool, numeric)
            base = f"{fam}-{i:03d}"
            for p in range(n_perms):
                items.append(_arrange(base, fam, FRAMINGS[framing](q), c, t, others, p, seed, framing))
    return items


def items_hash(items: list[Item]) -> str:
    return hashlib.sha256(json.dumps([asdict(i) for i in items], sort_keys=True).encode()).hexdigest()[:16]


def export_items(items: list[Item], path) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
    return items_hash(items)
