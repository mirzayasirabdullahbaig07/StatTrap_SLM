# Roadmap - StatTrap-SLM (honest progress)

Legend: [x] done and verified | [~] written, not verified on real data | [ ] not started

## Phase 0 - Build the instrument (DONE, ~100% of what can be done without real models)
- [x] 11 trap families, computed keys, named traps
- [x] Deterministic seeding, item hash, option permutations, 3 framings
- [x] Surface-cue audit + CLI + tests; generator rebuilt after the audit found shortcuts
- [x] Resumable runner, manifest (versions, hardware, hash), mixing guard
- [x] Cluster bootstrap, sign-flip, Holm, scaling, consistency, position bias, temperature robustness
- [x] Figures (CVD-validated colours) and tables
- [x] 28 tests passing; end-to-end mock CLI test
- [~] Ollama and HF backends; parser robust on synthetic strings only

## Phase 1 - Real runs (0%) - YOU, ~1 week
- [ ] Day 1: install Ollama, pull 2 small models, smoke test (`--limit 20`), read 20 raw responses, fix parser if needed
- [ ] Day 2-3: main run, 4-6 models, direct + CoT, `--per-family 60 --perms 2`
- [ ] Day 4: check unparsed/truncated rates (`*_truncation.csv`); rerun with larger `--max-new-tokens` if CoT is cut off
- [ ] Day 5: 7B model on Kaggle (4-bit) if laptop cannot
- GATE 1: unparsed < 5% for every model, else fix parsing before interpreting anything

## Phase 2 - Validity (0%) - ~1 week
- [ ] Statistician (or you, blind) answers `verify-sheet`; report agreement; fix/remove disputed items
- [ ] Framing-2 rerun + `stattrap compare`; sampling rerun with 3 decode seeds
- [ ] Read StatQA / StatABench / SDABench full papers; write the real novelty statement
- GATE 2: if human agreement < 90% on any family, drop or rewrite that family. If related work already measures trap rate on small models, reposition as replication/extension.

## Phase 3 - Analysis and writing (0%) - ~2 weeks
- [ ] `analyze`; interpret CoT gain, scaling, family differences (report nulls honestly)
- [ ] `error-sheet`: code 60 wrong CoT answers (fell for trap / arithmetic / misread / format / truncated)
- [ ] Paper: intro, benchmark design + audit, results, error analysis, limitations (list from README), release items + code
- GATE 3: advisor/peer read before posting to arXiv

## Where the work really stands
Instrument: essentially complete. Research: not begun, because no model has been run. Realistic time to a draft: about 4 weeks of part-time work.
