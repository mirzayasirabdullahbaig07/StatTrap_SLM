import json
import re

import numpy as np
import pandas as pd
import pytest

from stattrap import analysis as an
from stattrap.backends import MockBackend, parse_choice
from stattrap.item_audit import audit_items
from stattrap.manifest import build_manifest, write_manifest
from stattrap.runner import run_eval
from stattrap.sheets import error_sheet, verification_sheet
from stattrap.tasks import FAMILIES, make_items, items_hash


@pytest.fixture(scope="module")
def items():
    return make_items(12, seed=1, n_perms=2)


def test_all_families_and_structure(items):
    assert set(FAMILIES) == {i.family for i in items} and len(FAMILIES) == 11
    for it in items:
        assert set(it.options) == set("ABCD") and it.answer != it.trap
        assert len(set(it.options.values())) == 4, "duplicate option text"
        assert it.options[it.answer] != it.options[it.trap]


def test_determinism_and_hash():
    assert items_hash(make_items(5, 3, 2)) == items_hash(make_items(5, 3, 2))
    assert items_hash(make_items(5, 3, 2)) != items_hash(make_items(5, 4, 2))


def test_permutations_change_order_not_content(items):
    by = {}
    for it in items:
        by.setdefault(it.base_id, []).append(it)
    changed = 0
    for g in by.values():
        a, b = g[0], g[1]
        assert set(a.options.values()) == set(b.options.values())
        assert a.options[a.answer] == b.options[b.answer] and a.options[a.trap] == b.options[b.trap]
        changed += a.options != b.options
    assert changed > len(by) * 0.5


def test_simpson_correct_and_trap_semantics():
    for it in [i for i in make_items(30, 0, 1, families=["simpson"])]:
        assert re.search(r"\bA\b", it.options[it.answer]) or re.search(r"\bB\b", it.options[it.answer])
        assert it.options[it.answer] != it.options[it.trap]


def test_bonferroni_ground_truth():
    for it in make_items(40, 2, 1, families=["multiple_testing"]):
        nums = re.findall(r"\d+(?:\.\d+)?", it.question)
        assert it.options[it.answer].lstrip("-").isdigit()
        assert it.options[it.answer] != it.options[it.trap]


def test_audit_no_surface_shortcut():
    t = audit_items(make_items(120, 0, 1))
    assert not t.flagged.any(), t[t.flagged]
    assert t.acc.max() < 0.40


@pytest.mark.parametrize("txt,style,exp", [
    ("B", "direct", "B"), ("B.", "direct", "B"), ("(C) because...", "direct", "C"),
    ("A paired t-test is appropriate", "direct", None),
    ("Answer: D", "direct", "D"), ("Step 1... so\nAnswer: **B**", "cot", "B"),
    ("Maybe Answer: A, but actually Answer: C", "cot", "C"), ("no idea", "cot", None),
    ("The answer is B", "direct", "B"), ("I choose option c", "direct", "C"),
])
def test_parse(txt, style, exp):
    assert parse_choice(txt, style) == exp


def test_parse_item_text_fallback(items):
    it = items[0]
    assert parse_choice(it.options["C"], "direct", it) == "C"


@pytest.mark.parametrize("mode,lo,hi,tl,th", [("oracle", 1, 1, 0, 0), ("trap", 0, 0, 1, 1)])
def test_runner_modes(tmp_path, items, mode, lo, hi, tl, th):
    f = tmp_path / "r.csv"
    run_eval(items, ["m"], ["direct", "cot"], MockBackend(mode), f)
    df = pd.read_csv(f, keep_default_na=False)
    s = an.summary(df)
    assert (s.acc == lo).all() and (s.trap_rate == tl).all()


def test_resume_and_no_duplicates(tmp_path, items):
    f = tmp_path / "r.csv"
    n1 = run_eval(items[:20], ["m"], ["direct"], MockBackend("random"), f)
    n2 = run_eval(items, ["m"], ["direct"], MockBackend("random"), f)
    assert n1 == 20 and n2 == len(items) - 20
    assert run_eval(items, ["m"], ["direct"], MockBackend("random"), f) == 0
    df = pd.read_csv(f, keep_default_na=False)
    assert not df.duplicated(["item_id", "model", "style", "temperature", "decode_seed"]).any()


def test_random_near_chance(tmp_path):
    its = make_items(60, 0, 2)
    f = tmp_path / "r.csv"; run_eval(its, ["m"], ["direct"], MockBackend("random", 5), f)
    s = an.summary(pd.read_csv(f, keep_default_na=False))
    assert 0.18 < s.acc.iloc[0] < 0.32 and s.p_acc_gt_chance.iloc[0] > 0.05


def test_cot_gain_detected_and_cluster_level(tmp_path):
    its = make_items(40, 0, 2)
    f = tmp_path / "r.csv"; run_eval(its, ["m"], ["direct", "cot"], MockBackend("cot_helps"), f)
    df = pd.read_csv(f, keep_default_na=False)
    c = an.cot_vs_direct(df)
    assert c.cot_gain.iloc[0] > 0.2 and c.gain_lo.iloc[0] > 0 and c.p_holm.iloc[0] < 0.01
    assert c.n_base_items.iloc[0] == 40 * 11, "must count base items, not permuted rows"


def test_holm_and_signflip():
    p = np.array([0.01, 0.04, 0.03]); adj = an.holm(p)
    assert np.allclose(adj, [0.03, 0.06, 0.06]) and (adj >= p).all()
    assert an.sign_flip_p(np.zeros(5)) == 1.0
    assert an.sign_flip_p(np.ones(30)) < 0.001


def test_position_consistency_scaling(tmp_path, items):
    f = tmp_path / "r.csv"
    run_eval(items, ["qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b"], ["direct"], MockBackend("random"), f)
    df = pd.read_csv(f, keep_default_na=False)
    assert an.position_bias(df).shape[0] == 3
    assert an.consistency(df, items).choice_consistency.between(0, 1).all()
    assert "spearman_acc" in an.scaling(df).columns
    assert len(an.pairwise_models(df)) == 3


def test_temperature_rows(tmp_path):
    its = make_items(3, 0, 1)
    f = tmp_path / "r.csv"
    n = run_eval(its, ["m"], ["direct"], MockBackend("random"), f, temperatures=[0.0, 0.7], decode_seeds=[0, 1])
    assert n == len(its) * (1 + 2)  # greedy once, sampled per seed
    assert len(an.robustness(pd.read_csv(f, keep_default_na=False))) == 2


def test_manifest_guard(tmp_path, items):
    f = tmp_path / "r.csv"
    m = build_manifest({"per_family": 12, "perms": 2, "seed": 1}, items_hash(items), len(items))
    write_manifest(f, m); write_manifest(f, m)
    bad = build_manifest({"per_family": 12, "perms": 2, "seed": 2}, "other", len(items))
    with pytest.raises(ValueError):
        write_manifest(f, bad)


def test_sheets(tmp_path, items):
    v = verification_sheet(items, 2)
    assert len(v) == 2 * 11 and {"KEY_correct", "verifier_choice"} <= set(v.columns)
    f = tmp_path / "r.csv"; run_eval(items, ["m"], ["cot"], MockBackend("trap"), f)
    e = error_sheet(pd.read_csv(f, keep_default_na=False), items, 10)
    assert len(e) == 10 and (e.picked_trap == 1).all()


def test_cli_end_to_end(tmp_path):
    from stattrap.cli import main
    out = tmp_path / "run.csv"
    main(["run", "--backend", "mock", "--models", "qwen2.5:1.5b", "qwen2.5:7b", "--per-family", "6", "--perms", "2", "--out", str(out)])
    main(["analyze", "--csv", str(out)])
    assert (tmp_path / "run_figures" / "cot_gain.png").exists()
    assert (tmp_path / "run_summary.csv").exists()
