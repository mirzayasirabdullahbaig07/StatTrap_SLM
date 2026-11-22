# Experiment log

| date | command | seed | data version | result | next step |
|---|---|---|---|---|---|

## 2026-11-22 - build log (mock only, no research results)
- Heuristic audit of v1 generator: longest option correct on 100% of 5/7 families -> rebuilt with rejection sampling (`_balanced`).
- Seeding bug (first 4 chars of string) found by test, fixed with SHA-256.
- Audit n=300/family: max heuristic 35.7%, nothing flagged; 5 elimination-cue WARNs documented.
- pvalue_meaning had longest-option 43% at n=120 -> options rebalanced; now 25%.
- pandas `DataFrame.style` attribute collision (`d.style == x`) caused KeyError; fixed with `d["style"]`.
- 28 tests pass. Backends Ollama/HF untested live.
