"""Replays a frozen eval manifest against a running vLLM OpenAI-compatible server and
writes a `<label>-rewards.json` file for `tau/eval_tools/compare.py`.

Must run inside the prime-rl GPU container image — it imports `verifiers`,
`math500_v1`, and `openai`, and expects a live inference server. Not importable or unit
tested in this repo's macOS dev sandbox (see ../tests/ for the pure logic that is).

Reloads the `math500-v1` taskset the same way `freeze_manifest_live.py` did (same
pinned `dataset_revision`, same deterministic ordering) rather than trusting stored
text, and asserts every freshly-hashed prompt/answer still matches the frozen
manifest's `ExampleRecord` before evaluating — this is what makes "identical example
set" true independent of anything this script's caller passes in.

Usage (inside the container, cwd=/app, baked venv active; the calling wrapper script
owns starting/stopping the inference server and passes its base URL):

    uv run --no-sync python -m tau.eval_tools.live.run_frozen_eval_live \\
        --manifest /data/pretraining-data/prime-rl-math-7b-h200/manifest/frozen-eval-manifest.json \\
        --base-url http://localhost:8000/v1 \\
        --served-model-name Qwen/Qwen2.5-7B-Instruct \\
        --label baseline \\
        --output /data/pretraining-data/prime-rl-math-7b-h200-eval-baseline/rewards.json

For the post-training eval, pass `--lora-name` (the adapter loaded via
`prime_rl.utils.client.load_lora_adapter` by the calling wrapper) so completions are
requested against the fine-tuned adapter instead of the base model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from tau.eval_tools.hashing import hash_text
from tau.eval_tools.manifest import FrozenEvalManifest


def _reload_and_verify_examples(manifest: FrozenEvalManifest) -> list[tuple[int, str, str]]:
    """Reload math500-v1 at the manifest's pinned revision and return
    `(id, prompt, answer)` triples in manifest order, after asserting every freshly
    computed hash still matches the frozen `ExampleRecord`."""
    import verifiers.v1 as vf
    from math500_v1.taskset import Math500Taskset

    if manifest.eval_taskset.id != "math500-v1":
        raise NotImplementedError(f"only math500-v1 eval replay is implemented, got {manifest.eval_taskset.id!r}")

    tasks = Math500Taskset(vf.TasksetConfig()).select()
    by_id = {t.data.idx: t for t in tasks}

    triples: list[tuple[int, str, str]] = []
    for record in manifest.examples:
        task = by_id.get(record.id)
        if task is None:
            raise RuntimeError(f"reloaded math500-v1 is missing example id {record.id} present in the frozen manifest")
        prompt, answer = task.data.prompt, str(task.data.answer)
        if hash_text(prompt) != record.prompt_hash or hash_text(answer) != record.answer_hash:
            raise RuntimeError(
                f"example id {record.id}: reloaded math500-v1 content no longer matches the frozen "
                "manifest's hashes — the pinned dataset_revision may have drifted upstream. Aborting "
                "rather than silently evaluating against different content."
            )
        triples.append((record.id, prompt, answer))
    return triples


async def _evaluate(
    triples: list[tuple[int, str, str]],
    manifest: FrozenEvalManifest,
    *,
    base_url: str,
    served_model_name: str,
    max_concurrency: int,
) -> dict[str, float]:
    from openai import AsyncOpenAI
    from verifiers.v1 import verify_boxed_math_answer

    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    semaphore = asyncio.Semaphore(max_concurrency)
    decoding = manifest.decoding
    rewards: dict[str, float] = {}

    async def _one(example_id: int, prompt: str, answer: str) -> None:
        async with semaphore:
            extra_body = {"seed": decoding.seed}
            response = await client.chat.completions.create(
                model=served_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=decoding.temperature,
                top_p=decoding.top_p if decoding.top_p is not None else 1.0,
                max_tokens=decoding.max_completion_tokens,
                extra_body=extra_body,
            )
            completion = response.choices[0].message.content or ""
            rewards[str(example_id)] = float(verify_boxed_math_answer(completion, answer))

    await asyncio.gather(*(_one(i, p, a) for i, p, a in triples))
    return rewards


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--served-model-name", required=True, help="Model name as registered on the running server.")
    parser.add_argument(
        "--lora-name",
        default=None,
        help="If set, request completions against this LoRA adapter name instead of --served-model-name.",
    )
    parser.add_argument("--label", required=True, choices=["baseline", "post"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-concurrency", type=int, default=16)
    args = parser.parse_args(argv)

    manifest = FrozenEvalManifest.load(Path(args.manifest))
    triples = _reload_and_verify_examples(manifest)
    model_name = args.lora_name or args.served_model_name

    rewards = asyncio.run(
        _evaluate(
            triples,
            manifest,
            base_url=args.base_url,
            served_model_name=model_name,
            max_concurrency=args.max_concurrency,
        )
    )

    record = {
        "manifest_identity_hash": manifest.identity_hash(),
        "model_label": args.label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rewards": rewards,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2) + "\n")

    mean_reward = sum(rewards.values()) / len(rewards) if rewards else 0.0
    print(f"ok: wrote {len(rewards)} {args.label} rewards (mean={mean_reward:.4f}) to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
