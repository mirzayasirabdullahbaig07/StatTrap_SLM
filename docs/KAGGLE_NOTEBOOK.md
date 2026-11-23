# Kaggle run (copy each block into one notebook cell) - HF backend is UNTESTED live, so do step 2 first

Notebook settings: Accelerator = GPU T4 x2 (or P100), Internet = ON, Persistence = Files only.
Repo must be public (or use a Kaggle secret token in the clone URL).

```python
# 1. setup
!git clone https://github.com/mirzayasirabdullahbaig07/StatTrap_SLM.git
%cd StatTrap_SLM
!pip install -q -e . accelerate bitsandbytes
!stattrap audit-items --per-family 100 --perms 1 --out /tmp/a.csv | tail -3
```
```python
# 2. SMOKE TEST (2-3 min). Look at the printed table before anything else.
!stattrap run --backend hf --models Qwen/Qwen2.5-0.5B-Instruct --per-family 5 --perms 2 --limit 33 --max-new-tokens 1024 --out /kaggle/working/smoke.csv
import pandas as pd; d = pd.read_csv('/kaggle/working/smoke.csv', keep_default_na=False)
print(d.groupby('style').agg(n=('correct','size'), unparsed=('chosen', lambda s: (s=='').mean()), trunc=('truncated','mean')))
print(d.response.head(4).tolist())
```
Go on only if unparsed is small for direct. (Tiny models may legitimately ramble in CoT - that is a result, not a bug.)

```python
# 3. MAIN RUN, pass 1 (resumable: if the session dies, re-run the same cell; the CSV in /kaggle/working is kept)
!stattrap run --backend hf --per-family 30 --perms 2 --max-new-tokens 1024 --out /kaggle/working/main.csv \
  --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct
```
```python
# 4. the 7B in 4-bit, separate cell/session (same --out file, same item flags!)
!stattrap run --backend hf --load-in-4bit --per-family 30 --perms 2 --max-new-tokens 1024 --out /kaggle/working/main.csv \
  --models Qwen/Qwen2.5-7B-Instruct
```
```python
# 5. analysis, then download main.csv + main.manifest.json + main_figures/ from the Output panel
!stattrap analyze --csv /kaggle/working/main.csv
```
Rows: 11 x 30 x 2 = 660 items x 2 styles per model. Rough time: sub-3B models well under 1 h each; the 7B in 4-bit
possibly 3-4 h (CoT dominates). The manifest refuses a re-run with different --per-family/--perms/--seed into the same file.
Note: the 4-bit 7B is quantised, the others are not; say so in the paper (or run 3B in 4-bit too for a fair size trend).
Download the files BEFORE the session ends; copy them into this repo's results/ folder (git-ignored) and back them up elsewhere.
