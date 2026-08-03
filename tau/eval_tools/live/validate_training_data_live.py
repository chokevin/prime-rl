from __future__ import annotations

import argparse
import json
import os
import re
import tomllib
from pathlib import Path

from tau.eval_tools.hashing import hash_text
from tau.eval_tools.live.run_frozen_eval_live import _reload_and_verify_examples
from tau.eval_tools.manifest import (
    EXPECTED_TRAIN_TASKSET_ID,
    FrozenEvalManifest,
    validate_training_prompt_hashes,
    validate_training_snapshot,
)


def _reload_training_prompt_hashes(manifest: FrozenEvalManifest) -> list[str]:
    from math_env_v1.taskset import MathConfig, MathTaskset

    taskset = manifest.train_taskset
    local_path = validate_training_snapshot(taskset)
    config = MathConfig(
        dataset_name=str(local_path),
        dataset_subset=taskset.dataset_subset,
        dataset_split=taskset.dataset_split,
    )
    prompt_hashes = [hash_text(task.data.prompt) for task in MathTaskset(config).select()]
    return prompt_hashes


def _write_pinned_training_config(
    source_path: Path,
    output_path: Path,
    manifest: FrozenEvalManifest,
) -> None:
    source_text = Path(source_path).read_text()
    config = tomllib.loads(source_text)
    sources = config["orchestrator"]["train"]["source"]
    matching_sources = [
        source for source in sources if source.get("env", {}).get("taskset", {}).get("id") == EXPECTED_TRAIN_TASKSET_ID
    ]
    if len(sources) != 1 or len(matching_sources) != 1:
        raise ValueError("training config must contain exactly one math-env-v1 training source")
    taskset = matching_sources[0]["env"]["taskset"]
    expected_identity = {
        "dataset_name": manifest.train_taskset.dataset_name,
        "dataset_subset": manifest.train_taskset.dataset_subset,
        "dataset_split": manifest.train_taskset.dataset_split,
    }
    actual_identity = {key: taskset.get(key) for key in expected_identity}
    if actual_identity != expected_identity:
        raise ValueError(f"training config dataset identity is {actual_identity}, expected {expected_identity}")
    dataset_pattern = re.compile(
        rf"(?m)^dataset_name\s*=\s*{re.escape(json.dumps(manifest.train_taskset.dataset_name))}\s*$"
    )
    pinned_text, replacements = dataset_pattern.subn(
        f"dataset_name = {json.dumps(manifest.train_taskset.dataset_local_path)}",
        source_text,
    )
    if replacements != 1:
        raise ValueError(f"expected one canonical training dataset_name assignment, replaced {replacements}")
    pinned_config = tomllib.loads(pinned_text)
    pinned_taskset = pinned_config["orchestrator"]["train"]["source"][0]["env"]["taskset"]
    if pinned_taskset["dataset_name"] != manifest.train_taskset.dataset_local_path:
        raise RuntimeError("rendered training config did not preserve the pinned dataset snapshot path")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as output:
        output.write(pinned_text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-config", required=True)
    args = parser.parse_args(argv)

    if os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise RuntimeError("HF_DATASETS_OFFLINE=1 is required while validating the pinned training snapshot")
    manifest = FrozenEvalManifest.load(Path(args.manifest))
    if manifest.state != "finalized":
        raise ValueError("training-data validation requires a finalized manifest")
    eval_examples = _reload_and_verify_examples(manifest)
    prompt_hashes = _reload_training_prompt_hashes(manifest)
    validate_training_prompt_hashes(manifest, prompt_hashes)
    _write_pinned_training_config(Path(args.config), Path(args.output_config), manifest)
    print(
        f"ok: revalidated {len(eval_examples)} eval examples and {len(prompt_hashes)} "
        f"training prompts (sha256={manifest.training_data.prompt_hash_digest}); "
        f"wrote pinned RL config to {args.output_config}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
