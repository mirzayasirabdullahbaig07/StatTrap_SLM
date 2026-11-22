"""Model metadata for the scaling analysis. Edit to match what you actually ran.

params_b = nominal parameter count in billions (from the model name). `quant` is deliberately 'check':
default Ollama tags are typically 4-bit quantised GGUF, but verify with `ollama show <tag>`; HF ids run in the
dtype you loaded (record it in the manifest/paper)."""
REGISTRY = {
    "qwen2.5:0.5b": dict(params_b=0.5, family="qwen2.5", quant="check ollama show"),
    "qwen2.5:1.5b": dict(params_b=1.5, family="qwen2.5", quant="check ollama show"),
    "qwen2.5:3b": dict(params_b=3.0, family="qwen2.5", quant="check ollama show"),
    "qwen2.5:7b": dict(params_b=7.0, family="qwen2.5", quant="check ollama show"),
    "llama3.2:1b": dict(params_b=1.0, family="llama3.2", quant="check ollama show"),
    "llama3.2:3b": dict(params_b=3.0, family="llama3.2", quant="check ollama show"),
    "gemma2:2b": dict(params_b=2.0, family="gemma2", quant="check ollama show"),
    "Qwen/Qwen2.5-0.5B-Instruct": dict(params_b=0.5, family="qwen2.5", quant="HF dtype as loaded"),
    "Qwen/Qwen2.5-1.5B-Instruct": dict(params_b=1.5, family="qwen2.5", quant="HF dtype as loaded"),
    "Qwen/Qwen2.5-3B-Instruct": dict(params_b=3.0, family="qwen2.5", quant="HF dtype as loaded"),
    "Qwen/Qwen2.5-7B-Instruct": dict(params_b=7.0, family="qwen2.5", quant="HF dtype as loaded"),
    "meta-llama/Llama-3.2-1B-Instruct": dict(params_b=1.0, family="llama3.2", quant="HF dtype as loaded"),
    "meta-llama/Llama-3.2-3B-Instruct": dict(params_b=3.0, family="llama3.2", quant="HF dtype as loaded"),
    "google/gemma-2-2b-it": dict(params_b=2.0, family="gemma2", quant="HF dtype as loaded"),
}
