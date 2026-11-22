# Running StatTrap on real models (NOT yet done - this is the next step)

Everything here is **untested against live models** (the sandbox that built this had no Ollama/GPU/Hugging Face access).
Expect to fix small issues; always smoke-test first.

## A. Laptop with Ollama
```bash
pip install -e .
ollama pull qwen2.5:1.5b && ollama pull llama3.2:3b          # then others from registry.py
ollama show qwen2.5:1.5b                                      # note the quantisation, put it in the paper
stattrap audit-items --per-family 100 --perms 1               # must print "Flagged: NONE"
stattrap run --backend ollama --models qwen2.5:1.5b --per-family 5 --perms 2 --limit 20 --out results/smoke.csv
# open results/smoke.csv: are `chosen` filled? is `response` sensible? is `truncated` mostly 0?
stattrap run --backend ollama --models qwen2.5:1.5b qwen2.5:3b qwen2.5:7b llama3.2:1b llama3.2:3b gemma2:2b \
      --per-family 60 --perms 2 --out results/main.csv       # resumable: re-run the same command after any interruption
stattrap analyze --csv results/main.csv
```
Rows: 11 families x 60 x 2 perms = 1,320 items x 2 prompt styles x models. CoT is the slow part (up to 384 tokens/row).

## B. Kaggle / Colab (free GPU)
```python
!pip install -q -e . transformers accelerate bitsandbytes
!stattrap run --backend hf --models Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-7B-Instruct --load-in-4bit \
   --per-family 60 --perms 2 --out /kaggle/working/main.csv
```
Kaggle sessions end after ~12 h: download `main.csv` + `main.manifest.json` before it ends; resume by uploading them back.
Llama and Gemma need a (free) Hugging Face licence acceptance and token.

## C. Then, in this order
1. `stattrap analyze --csv results/main.csv` -> tables + figures next to the CSV.
2. Robustness: re-run with `--framing 2` into a new CSV and `stattrap compare --a results/main.csv --b results/framing2.csv`.
3. Sampling: add `--temperatures 0.7 --decode-seeds 0 1 2` (new CSV; reported in `*_robustness_temperature.csv`).
4. `stattrap verify-sheet` -> give to a statistician (without KEY_ columns). Report agreement with the key.
5. `stattrap error-sheet --csv results/main.csv` -> hand-code error categories for the qualitative section.
