# StatTrap-SLM

**Question:** Do small (<=7B) open language models fall for classic statistical traps, and does step-by-step
(chain-of-thought, CoT) prompting fix it?

**Status in one line:** the benchmark generator, audit, runner, statistics, figures and 28 tests are built and verified on a *mock*
model. **No real model has been evaluated yet, so there are zero research results.** See [Honest status](#honest-status).

## Why this is a reasonable paper (and what is NOT claimed)
- Statistical-reasoning benchmarks exist (e.g. StatQA, StatABench arXiv:2606.22977, SDABench 2607.11079), mostly testing larger models
  on test-selection, data-analysis or computation. This project's intended angle is narrow and cheap: *named misconception traps*
  (every wrong option that a lured reader picks is labelled), run on locally runnable <=7B models, with **trap rate** (picking the
  specific lure) separated from plain accuracy, plus the CoT effect, option-order consistency and scaling.
- **Novelty is NOT verified.** I only saw search-result listings of related work, not full papers. Before writing, read those
  and check whether trap-rate on small models is already covered. The honest framing is a *careful small-model evaluation study / workshop-level
  paper*, not a new method.
- Results are unknown. A null result ("CoT does not help small models here") is still publishable if the benchmark is valid.

## What is built
| Piece | File | Notes |
|---|---|---|
| 11 trap families with computed ground truth | `tasks.py` | paired design, multiple testing, Simpson, base rate, spurious regression, CI meaning, p-value meaning, regression to the mean, survivorship, confounding, underpowered null |
| Deterministic generation | `tasks.py` | SHA-256 seeding; item-set hash; option permutations; 3 light paraphrase framings |
| Shortcut audit | `item_audit.py`, `stattrap audit-items` | 9 cue heuristics must be ~25% (chance) |
| Backends | `backends.py` | Mock (tested); Ollama and Hugging Face (**untested live**) |
| Resumable runner + manifest | `runner.py`, `manifest.py` | CSV per (item, model, style, temperature, seed); refuses mixing different item sets |
| Statistics | `analysis.py` | cluster (base-item) bootstrap CIs, paired sign-flip tests, Holm, McNemar-style CoT gain, scaling (Spearman), position bias, consistency, temperature robustness |
| Figures | `report.py` | trap heatmap, CoT-gain forest, position bias, scaling |
| Human-step sheets | `sheets.py` | blind verification sheet; error-coding sheet |
| Tests | `tests/` | 28 passing |

## Quick start (mock, proves the pipeline only)
```bash
pip install -e . --break-system-packages
stattrap audit-items --per-family 100 --perms 1
stattrap run --backend mock --models qwen2.5:1.5b qwen2.5:7b --per-family 10 --perms 2 --out results/demo.csv
stattrap analyze --csv results/demo.csv
pytest -q
```
Real runs: `docs/RUNNING_REAL_MODELS.md`.

## Method choices (and why)
- **Programmatic items with computed answers**: no label noise from annotators; each item's key and trap are derived from the numbers.
- **Cluster-level statistics**: option-permuted copies of one base item are not independent; all CIs/tests resample base items.
- **Audit before any model run**: a benchmark answerable by "pick the longest option" measures nothing. The first generator version
  failed this (longest option correct on 100% of 5/7 families); it was rebuilt with rejection sampling and re-audited.
- **Unparsed answers are counted, never guessed**; truncation is flagged.

## Honest status
| Area | State |
|---|---|
| Generator, audit, runner, analysis, figures, CLI, manifest | **Done, tested on mock** (28 tests) |
| Surface-cue audit | **Passes**: max heuristic 35.7% (chance 25%) at n=300/family, none flagged (`docs/item_audit_n300.csv`) |
| Ollama / HF backends, answer parsing on real outputs | **Written, never run on a live model** |
| Real model results | **Not started (0%)** |
| Human check of item correctness by a statistician | **Not done** (sheet generator exists) |
| Error taxonomy of real CoT failures | **Not done** (needs real outputs) |
| Related-work verification | **Not done** (only search listings seen) |
| Paper draft | **Not started** |

### Known limitations / where stuck
1. **No real-model evidence**; no Ollama/GPU in the build environment. This is the one blocker that matters, and it needs your laptop or Kaggle.
2. **Residual cues**: in 5 family/heuristic pairs the correct option is almost *never* the word-overlap-top or shortest option
   (rates 0-8%), so a model could eliminate one option (ceiling ~33%). Documented by the audit's WARN; not fixed.
   `multiple_testing` correct answers cannot be uniformly ranked numerically (rank-1/2 over-represented; best numeric heuristic ~33%).
3. **Templated items**: 11 families x randomised numbers/wording gives limited linguistic diversity; framings 1-2 are light paraphrases only.
   A reviewer may say a model memorised the template - mitigated, not eliminated.
4. **Ground truth is my own derivation**; a second human has not confirmed all items (some families, e.g. `confounding`, `ci_meaning`, have
   conceptual keys, not computed ones).
5. Nominal parameter counts; default Ollama tags are often 4-bit quantised - record it.

See `ROADMAP.md` for the checklist.
