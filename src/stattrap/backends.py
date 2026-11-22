"""Model backends: mock (tests), Ollama (laptop), Hugging Face (Kaggle/Colab).

STATUS: MockBackend and parse_choice are unit-tested. OllamaBackend and HFBackend need real runtimes
and have NOT been run in the environment where this code was written - smoke-test with --limit 20.
"""
from __future__ import annotations

import hashlib
import re
import time


class MockBackend:
    """Deterministic fake. mode: 'oracle' | 'trap' | 'random' | 'cot_helps'."""

    def __init__(self, mode="random", seed=0):
        self.mode, self.seed = mode, seed

    def generate(self, model, item, style="direct", max_new_tokens=256, temperature=0.0, seed=0):
        h = int(hashlib.md5(f"{self.seed}|{model}|{item.id}|{style}|{temperature}|{seed}".encode()).hexdigest(), 16)
        if self.mode == "oracle":
            pick = item.answer
        elif self.mode == "trap":
            pick = item.trap
        elif self.mode == "cot_helps":
            pick = item.answer if (h % 1000) / 1000 < (0.8 if style == "cot" else 0.4) else item.trap
        else:
            pick = "ABCD"[h % 4]
        text = f"Reasoning...\nAnswer: {pick}" if style == "cot" else pick
        return text, 0.01, 10, 3


class OllamaBackend:
    def __init__(self, base_url="http://localhost:11434", timeout=600):
        import requests
        self.r, self.url, self.timeout = requests, base_url.rstrip("/"), timeout

    def version(self):
        try:
            return self.r.get(f"{self.url}/api/version", timeout=5).json().get("version")
        except Exception:
            return None

    def generate(self, model, item, style="direct", max_new_tokens=256, temperature=0.0, seed=0):
        t0 = time.perf_counter()
        r = self.r.post(f"{self.url}/api/generate", timeout=self.timeout, json={
            "model": model, "prompt": item.prompt(style), "stream": False,
            "options": {"temperature": temperature, "seed": seed, "num_predict": max_new_tokens}})
        r.raise_for_status(); j = r.json()
        return j.get("response", ""), time.perf_counter() - t0, j.get("prompt_eval_count", 0), j.get("eval_count", 0)


class HFBackend:
    """Greedy (temperature 0) or sampled decoding with transformers; one model resident at a time."""

    def __init__(self, load_in_4bit: bool = False):
        self.cur = self.tok = self.model = None
        self.load_in_4bit = load_in_4bit

    def _load(self, name):
        if self.cur == name:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        kw = dict(device_map="auto", torch_dtype="auto")
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, **kw)
        self.cur = name

    def generate(self, model, item, style="direct", max_new_tokens=256, temperature=0.0, seed=0):
        import torch
        self._load(model)
        ids = self.tok.apply_chat_template([{"role": "user", "content": item.prompt(style)}],
                                           add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        with torch.no_grad():
            kw = dict(do_sample=True, temperature=temperature) if temperature > 0 else dict(do_sample=False)
            out = self.model.generate(ids, max_new_tokens=max_new_tokens, **kw)
        new = out[0, ids.shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True), time.perf_counter() - t0, int(ids.shape[1]), int(len(new))


_ANS = re.compile(r"(?:final\s+)?answer(?:\s+is)?\s*[:\-]?\s*\**\s*\(?\s*([A-D])\b", re.I)
_OPT = re.compile(r"\b(?:option|choice)\s*\(?([A-D])\b", re.I)


def parse_choice(text: str, style: str = "direct", item=None) -> str | None:
    """Extract the chosen letter, conservatively (None = unparsed, counted and reported, never guessed).

    Order: 'Answer: X' phrase (last one wins for CoT) -> bare letter at the very start followed by a delimiter or end
    (so 'A paired t-test...' is NOT read as option A) -> 'option X' -> exactly one option's full text quoted."""
    t = text.strip()
    m = _ANS.findall(t)
    if m:
        return (m[-1] if style == "cot" else m[0]).upper()
    if style == "direct":
        m0 = re.match(r"^[\s\*\(]*([A-D])(?:\s*[\.\):\*]|\s*$)", t)
        if m0:
            return m0.group(1).upper()
    m = _OPT.findall(t)
    if m:
        return m[-1].upper()
    if item is not None:
        low = t.lower()
        hits = [l for l, o in item.options.items() if o.lower() in low]
        if len(hits) == 1:
            return hits[0]
    return None
