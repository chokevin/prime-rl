from pathlib import Path

import pytest

from tau.eval_tools.output_paths import (
    evidence_generation,
    expected_output_path,
    prepare_output_directory,
    validate_generation_references,
)

SOURCE_A = "a" * 40
SOURCE_B = "b" * 40


def _references(mode, eval_label, source_revision, data_root):
    generation = evidence_generation(source_revision, data_root=data_root)
    if mode in {"freeze-draft", "freeze-finalize", "train"}:
        references = {"manifest_dir": generation.manifest}
    elif mode == "eval" and eval_label == "baseline":
        references = {
            "manifest_path": generation.draft_manifest,
            "baseline_rewards_path": generation.baseline_rewards,
        }
    elif mode == "eval" and eval_label == "post":
        references = {
            "manifest_path": generation.frozen_manifest,
            "baseline_rewards_path": generation.baseline_rewards,
            "training_result_path": generation.training_result,
            "training_output_dir": generation.train,
            "lora_adapter_path": generation.final_adapter,
        }
    else:
        references = {}
    if mode == "freeze-finalize":
        references["baseline_rewards_path"] = generation.baseline_rewards
    return references


@pytest.mark.parametrize(
    ("mode", "eval_label", "leaf"),
    [
        ("smoke", None, "smoke"),
        ("freeze-draft", None, "manifest"),
        ("freeze-finalize", None, "manifest"),
        ("eval", "baseline", "eval-baseline"),
        ("eval", "post", "eval-post"),
        ("train", None, "train"),
    ],
)
def test_prepare_output_directory_creates_only_exact_source_mode_path(tmp_path, mode, eval_label, leaf):
    data_root = tmp_path / "data"
    data_root.mkdir()
    expected = data_root / "pretraining-data" / "prime-rl-math-7b-h200" / "generations" / SOURCE_A / leaf

    prepared = prepare_output_directory(
        mode,
        expected,
        SOURCE_A,
        eval_label=eval_label,
        data_root=data_root,
        **_references(mode, eval_label, SOURCE_A, data_root),
    )

    assert prepared == expected
    assert prepared.is_dir()
    assert prepared.resolve(strict=True) == expected


def test_source_generations_are_disjoint_and_preserve_prior_artifacts(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    first = evidence_generation(SOURCE_A, data_root=data_root)
    second = evidence_generation(SOURCE_B, data_root=data_root)
    prepare_output_directory("smoke", first.smoke, SOURCE_A, data_root=data_root)
    prior = first.smoke / "smoke-result.json"
    prior.write_bytes(b"immutable generation A")

    prepare_output_directory("smoke", second.smoke, SOURCE_B, data_root=data_root)

    assert first.root != second.root
    assert first.smoke != second.smoke
    assert prior.read_bytes() == b"immutable generation A"
    assert not (second.smoke / prior.name).exists()


def test_prepared_eval_and_train_outputs_support_first_runtime_use(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    generation = evidence_generation(SOURCE_A, data_root=data_root)

    prepare_output_directory(
        "eval",
        generation.eval_baseline,
        SOURCE_A,
        eval_label="baseline",
        manifest_path=generation.draft_manifest,
        baseline_rewards_path=generation.baseline_rewards,
        data_root=data_root,
    )
    (generation.eval_baseline / "inference.log").write_text("started")
    prepare_output_directory(
        "train",
        generation.train,
        SOURCE_A,
        manifest_dir=generation.manifest,
        data_root=data_root,
    )

    assert (generation.eval_baseline / "inference.log").read_text() == "started"
    assert generation.train.resolve(strict=True) == generation.train


@pytest.mark.parametrize("source_revision", ["", "abc", "A" * 40, "g" * 40, "a" * 39])
def test_prepare_output_directory_rejects_malformed_source_before_creation(tmp_path, source_revision):
    data_root = tmp_path / "data"
    data_root.mkdir()

    with pytest.raises(ValueError, match="source revision"):
        prepare_output_directory("smoke", data_root / "unused", source_revision, data_root=data_root)

    assert not list(data_root.iterdir())


@pytest.mark.parametrize(
    ("mode", "eval_label"),
    [
        ("unknown", None),
        ("eval", None),
        ("eval", "other"),
        ("train", "baseline"),
    ],
)
def test_prepare_output_directory_rejects_mode_or_label_before_creation(
    tmp_path,
    mode,
    eval_label,
):
    data_root = tmp_path / "data"
    data_root.mkdir()
    supplied = evidence_generation(SOURCE_A, data_root=data_root).train

    with pytest.raises(ValueError):
        prepare_output_directory(
            mode,
            supplied,
            SOURCE_A,
            eval_label=eval_label,
            data_root=data_root,
        )

    assert not list(data_root.iterdir())


def test_prepare_output_directory_rejects_wrong_source_output_before_creation(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    wrong_generation = evidence_generation(SOURCE_B, data_root=data_root)

    with pytest.raises(ValueError, match="expected exactly"):
        prepare_output_directory("smoke", wrong_generation.smoke, SOURCE_A, data_root=data_root)

    assert not list(data_root.iterdir())


def test_prepare_output_directory_rejects_noncanonical_path_before_creation(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    noncanonical = f"{data_root}//pretraining-data/prime-rl-math-7b-h200/generations/{SOURCE_A}/smoke"

    with pytest.raises(ValueError, match="canonical"):
        prepare_output_directory("smoke", noncanonical, SOURCE_A, data_root=data_root)

    assert not list(data_root.iterdir())


@pytest.mark.parametrize(
    ("mode", "eval_label", "field"),
    [
        ("freeze-finalize", None, "baseline_rewards_path"),
        ("train", None, "manifest_dir"),
        ("eval", "baseline", "manifest_path"),
        ("eval", "post", "training_result_path"),
        ("eval", "post", "training_output_dir"),
        ("eval", "post", "lora_adapter_path"),
    ],
)
def test_cross_mode_references_cannot_mix_source_generations(tmp_path, mode, eval_label, field):
    data_root = tmp_path / "data"
    data_root.mkdir()
    references = _references(mode, eval_label, SOURCE_A, data_root)
    wrong = _references(mode, eval_label, SOURCE_B, data_root)
    references[field] = wrong[field]

    with pytest.raises(ValueError, match="expected exactly"):
        validate_generation_references(
            mode,
            SOURCE_A,
            eval_label=eval_label,
            data_root=data_root,
            **references,
        )

    assert not list(data_root.iterdir())


def test_comparison_output_must_be_directly_under_matching_post_generation(tmp_path):
    data_root = tmp_path / "data"
    generation = evidence_generation(SOURCE_A, data_root=data_root)
    references = _references("eval", "post", SOURCE_A, data_root)

    validate_generation_references(
        "eval",
        SOURCE_A,
        eval_label="post",
        comparison_output_path=generation.eval_post / "comparison-retry.json",
        data_root=data_root,
        **references,
    )
    with pytest.raises(ValueError, match="directly under"):
        validate_generation_references(
            "eval",
            SOURCE_A,
            eval_label="post",
            comparison_output_path=evidence_generation(SOURCE_B, data_root=data_root).eval_post / "comparison.json",
            data_root=data_root,
            **references,
        )


def test_prepare_output_directory_rejects_symlinked_component_without_touching_target(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (data_root / "pretraining-data").symlink_to(external, target_is_directory=True)
    expected = expected_output_path("train", SOURCE_A, data_root=data_root)

    with pytest.raises(OSError):
        prepare_output_directory(
            "train",
            expected,
            SOURCE_A,
            manifest_dir=evidence_generation(SOURCE_A, data_root=data_root).manifest,
            data_root=data_root,
        )

    assert not list(external.iterdir())


def test_prepare_output_directory_rejects_non_directory_component(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    component = data_root / "pretraining-data"
    component.write_text("not a directory")
    expected = expected_output_path("train", SOURCE_A, data_root=data_root)

    with pytest.raises(OSError):
        prepare_output_directory(
            "train",
            expected,
            SOURCE_A,
            manifest_dir=evidence_generation(SOURCE_A, data_root=data_root).manifest,
            data_root=data_root,
        )

    assert component.read_text() == "not a directory"


def test_prepare_output_directory_preserves_existing_artifacts(tmp_path):
    data_root = tmp_path / "data"
    generation = evidence_generation(SOURCE_A, data_root=data_root)
    generation.eval_baseline.mkdir(parents=True)
    rewards = generation.baseline_rewards
    rewards.write_bytes(b"immutable")

    prepare_output_directory(
        "eval",
        generation.eval_baseline,
        SOURCE_A,
        eval_label="baseline",
        manifest_path=generation.draft_manifest,
        baseline_rewards_path=generation.baseline_rewards,
        data_root=data_root,
    )

    assert rewards.read_bytes() == b"immutable"


def test_wrapper_passes_verified_source_and_prepares_generation_before_mode_first_use():
    script = (Path(__file__).parents[2] / "scripts/run-prime-rl.sh").read_text()
    verified = script.index('[ "$resolved_sha" = "$PRIME_RL_REPO_SHA" ]')
    prepare = script.index("tau.eval_tools.output_paths", verified)
    source_argument = script.index('--source-revision "$resolved_sha"', prepare)
    dispatch = script.index('case "$PRIME_RL_RUN_MODE" in', source_argument)
    finalize_train_path = script.index('--output-dir "${GENERATION_ROOT}/train"', dispatch)
    inference_redirect = script.index('>"${TAU_OUTPUT_DIR}/inference.log"', dispatch)
    training_supervisor = script.index("tau.eval_tools.live.training_supervisor_live run", dispatch)

    assert verified < prepare < source_argument < dispatch
    assert dispatch < finalize_train_path
    assert dispatch < inference_redirect
    assert dispatch < training_supervisor
