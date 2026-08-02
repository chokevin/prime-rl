"""GPU-container-only scripts: build/finalize the frozen eval manifest against the real
model/datasets, and replay it against a running vLLM server. Not importable outside the
prime-rl image (needs torch/vllm/datasets/verifiers + network access to the HF Hub).
"""
