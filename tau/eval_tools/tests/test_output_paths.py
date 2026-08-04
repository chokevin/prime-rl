import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

import tau.eval_tools.live.inference_launcher_live as inference_launcher
from tau.eval_tools.live.inference_launcher_live import (
    attempt_log_name,
    open_inference_log,
    promote_inference_log,
)
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
    elif mode == "tier-curve":
        references = {
            "tier_curve_model_source_revision": source_revision,
            "tier_curve_model_manifest_path": generation.frozen_manifest,
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
        ("tier-curve", None, "tier-curve"),
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
        ("tier-curve", None, "tier_curve_model_manifest_path"),
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


def test_tier_curve_requires_explicit_model_generation_before_creation(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    output = evidence_generation(SOURCE_A, data_root=data_root).tier_curve

    with pytest.raises(ValueError, match="MODEL_SOURCE_REVISION"):
        prepare_output_directory("tier-curve", output, SOURCE_A, data_root=data_root)

    assert not list(data_root.iterdir())


def test_tier_curve_model_references_are_rejected_for_other_modes(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    model_generation = evidence_generation(SOURCE_B, data_root=data_root)

    with pytest.raises(ValueError, match="only for tier-curve"):
        validate_generation_references(
            "smoke",
            SOURCE_A,
            tier_curve_model_source_revision=SOURCE_B,
            tier_curve_model_manifest_path=model_generation.frozen_manifest,
            data_root=data_root,
        )


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


def test_interrupted_inference_attempt_preserves_partial_log_and_retry_can_start(tmp_path):
    output_dir = tmp_path / "eval-baseline"
    output_dir.mkdir()
    first_attempt = "pod-first"
    first_path = output_dir / attempt_log_name(first_attempt)
    environment = os.environ | {"PYTHONPATH": str(Path(__file__).parents[3])}
    child = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tau.eval_tools.live.inference_launcher_live",
            "launch",
            "--output-dir",
            str(output_dir),
            "--attempt-id",
            first_attempt,
            "--",
            sys.executable,
            "-c",
            "import time; print('partial', flush=True); time.sleep(60)",
        ],
        env=environment,
    )
    try:
        for _ in range(500):
            if first_path.exists() and first_path.stat().st_size:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("inference launcher did not write its partial log")
        child.send_signal(signal.SIGTERM)
        child.wait(timeout=5)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)

    second_attempt = "pod-second"
    descriptor = open_inference_log(output_dir, second_attempt)
    os.write(descriptor, b"retry\n")
    os.close(descriptor)

    assert first_path.read_bytes() == b"partial\n"
    assert (output_dir / attempt_log_name(second_attempt)).read_bytes() == b"retry\n"
    assert not (output_dir / "inference.log").exists()


def test_inference_attempt_id_collision_preserves_existing_log(tmp_path):
    output_dir = tmp_path.resolve()
    descriptor = open_inference_log(output_dir, "pod-same")
    os.write(descriptor, b"partial\n")
    os.close(descriptor)

    with pytest.raises(FileExistsError):
        open_inference_log(output_dir, "pod-same")

    assert (output_dir / attempt_log_name("pod-same")).read_bytes() == b"partial\n"


def test_inference_log_successfully_promotes_closed_attempt(tmp_path):
    output_dir = tmp_path.resolve()
    descriptor = open_inference_log(output_dir, "pod-success")
    os.write(descriptor, b"complete\n")
    metadata = os.fstat(descriptor)
    os.close(descriptor)

    snapshot = promote_inference_log(output_dir, "pod-success")

    final = output_dir / "inference.log"
    assert final.read_bytes() == b"complete\n"
    assert (snapshot.device, snapshot.inode, snapshot.size) == (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
    )
    assert not (output_dir / attempt_log_name("pod-success")).exists()


def test_inference_log_final_collision_is_immutable_and_preserves_losing_attempt(tmp_path):
    output_dir = tmp_path.resolve()
    first = open_inference_log(output_dir, "pod-winner")
    os.write(first, b"winner\n")
    os.close(first)
    promote_inference_log(output_dir, "pod-winner")
    second_path = output_dir / attempt_log_name("pod-loser")
    second = open_inference_log(output_dir, "pod-loser")
    os.write(second, b"loser\n")
    os.close(second)

    with pytest.raises(FileExistsError):
        promote_inference_log(output_dir, "pod-loser")

    assert (output_dir / "inference.log").read_bytes() == b"winner\n"
    assert second_path.read_bytes() == b"loser\n"


def test_concurrent_inference_log_promotions_publish_one_closed_attempt(tmp_path, monkeypatch):
    output_dir = tmp_path.resolve()
    attempts = ("pod-old", "pod-new")
    for attempt in attempts:
        descriptor = open_inference_log(output_dir, attempt)
        os.write(descriptor, f"{attempt}\n".encode())
        os.close(descriptor)
    barrier = Barrier(2)
    monkeypatch.setattr(
        inference_launcher,
        "_after_inference_log_validation",
        lambda _path: barrier.wait(timeout=10),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(promote_inference_log, output_dir, attempt) for attempt in attempts]
    outcomes = []
    for future in futures:
        try:
            future.result()
        except FileExistsError:
            outcomes.append("lost")
        else:
            outcomes.append("won")

    assert sorted(outcomes) == ["lost", "won"]
    final = (output_dir / "inference.log").read_text().strip()
    assert final in attempts
    losing_attempt = next(attempt for attempt in attempts if attempt != final)
    assert (output_dir / attempt_log_name(losing_attempt)).read_text() == f"{losing_attempt}\n"


@pytest.mark.parametrize("dangling", [False, True])
def test_inference_attempt_log_rejects_symlink_without_touching_target(tmp_path, dangling):
    output_dir = tmp_path / "eval-baseline"
    output_dir.mkdir()
    target = tmp_path / "external.log"
    if not dangling:
        target.write_bytes(b"external")
    attempt_path = output_dir / attempt_log_name("pod-symlink")
    attempt_path.symlink_to(target)

    with pytest.raises(FileExistsError):
        open_inference_log(output_dir, "pod-symlink")

    if dangling:
        assert not target.exists()
    else:
        assert target.read_bytes() == b"external"
    assert attempt_path.is_symlink()


def test_inference_final_symlink_blocks_promotion_without_touching_target(tmp_path):
    output_dir = tmp_path.resolve()
    target = tmp_path / "external.log"
    target.write_bytes(b"external")
    (output_dir / "inference.log").symlink_to(target)
    attempt_path = output_dir / attempt_log_name("pod-final-symlink")
    descriptor = open_inference_log(output_dir, "pod-final-symlink")
    os.write(descriptor, b"owned\n")
    os.close(descriptor)

    with pytest.raises(FileExistsError):
        promote_inference_log(output_dir, "pod-final-symlink")

    assert target.read_bytes() == b"external"
    assert (output_dir / "inference.log").is_symlink()
    assert attempt_path.read_bytes() == b"owned\n"


def test_inference_attempt_swap_after_validation_fails_closed(tmp_path, monkeypatch):
    output_dir = tmp_path.resolve()
    attempt_path = output_dir / attempt_log_name("pod-swapped")
    displaced = output_dir / "displaced-owned.log"
    descriptor = open_inference_log(output_dir, "pod-swapped")
    os.write(descriptor, b"owned\n")
    os.close(descriptor)

    def swap_attempt(path):
        path.rename(displaced)
        path.write_bytes(b"replacement\n")

    monkeypatch.setattr(inference_launcher, "_after_inference_log_validation", swap_attempt)
    with pytest.raises(RuntimeError, match="changed after validation"):
        promote_inference_log(output_dir, "pod-swapped")

    assert not (output_dir / "inference.log").exists()
    assert displaced.read_bytes() == b"owned\n"
    assert attempt_path.read_bytes() == b"replacement\n"


def test_inference_final_swap_before_verification_is_quarantined(tmp_path, monkeypatch):
    output_dir = tmp_path.resolve()
    displaced = output_dir / "displaced-installed.log"
    descriptor = open_inference_log(output_dir, "pod-final-swapped")
    os.write(descriptor, b"owned\n")
    os.close(descriptor)

    def swap_final(path):
        path.rename(displaced)
        path.write_bytes(b"replacement\n")

    monkeypatch.setattr(inference_launcher, "_after_inference_log_install", swap_final)
    with pytest.raises(RuntimeError, match="does not match validated staging"):
        promote_inference_log(output_dir, "pod-final-swapped")

    assert not (output_dir / "inference.log").exists()
    assert displaced.read_bytes() == b"owned\n"
    quarantines = list(output_dir.glob("inference.log.quarantine-*"))
    assert len(quarantines) == 1
    assert quarantines[0].read_bytes() == b"replacement\n"


@pytest.mark.parametrize("attempt_id", ["", "UPPER", "../escape", "two.parts", "-start", "end-", "a" * 64])
def test_inference_attempt_id_must_be_strict_dns_label(tmp_path, attempt_id):
    with pytest.raises(ValueError, match="DNS label"):
        open_inference_log(tmp_path.resolve(), attempt_id)


def test_wrapper_passes_verified_source_and_prepares_generation_before_mode_first_use():
    script = (Path(__file__).parents[2] / "scripts/run-prime-rl.sh").read_text()
    verified = script.index('[ "$resolved_sha" = "$PRIME_RL_REPO_SHA" ]')
    prepare = script.index("tau.eval_tools.output_paths", verified)
    source_argument = script.index('--source-revision "$resolved_sha"', prepare)
    dispatch = script.index('case "$PRIME_RL_RUN_MODE" in', source_argument)
    finalize_train_path = script.index('--output-dir "${GENERATION_ROOT}/train"', dispatch)
    inference_launcher = script.index("tau.eval_tools.live.inference_launcher_live", dispatch)
    training_supervisor = script.index("tau.eval_tools.live.training_supervisor_live run", dispatch)

    assert verified < prepare < source_argument < dispatch
    assert dispatch < finalize_train_path
    assert dispatch < inference_launcher
    assert dispatch < training_supervisor


def test_eval_wrapper_closes_writer_and_publishes_log_after_result_evidence():
    script = (Path(__file__).parents[2] / "scripts/run-prime-rl.sh").read_text()
    eval_dispatch = script.index("eval)")
    launch = script.index("inference_launcher_live launch", eval_dispatch)
    rewards = script.index("run_frozen_eval_live", launch)
    comparison = script.index("tau.eval_tools.cli compare", rewards)
    stop = script.index("stop_inference", comparison)
    promote = script.index("inference_launcher_live promote", stop)

    assert launch < rewards < comparison < stop < promote
    assert 'inference_attempt_id="${HOSTNAME:?HOSTNAME must identify this pod}"' in script


def test_tier_curve_wrapper_closes_writer_before_log_promotion():
    script = (Path(__file__).parents[2] / "scripts/run-prime-rl.sh").read_text()
    tier_dispatch = script.index("tier-curve)")
    launch = script.index("inference_launcher_live launch", tier_dispatch)
    evaluate = script.index("run_harder_tier_curve_live", launch)
    stop = script.index("stop_inference", evaluate)
    promote = script.index("inference_launcher_live promote", stop)

    assert launch < evaluate < stop < promote


@pytest.mark.parametrize(
    ("inference_body", "expected_status"),
    [
        ("import time; print('ready', flush=True); time.sleep(60)", 143),
        (
            "import signal, threading; "
            "done = threading.Event(); "
            "signal.signal(signal.SIGTERM, lambda *_: done.set()); "
            "print('ready', flush=True); "
            "done.wait()",
            0,
        ),
    ],
)
def test_uv_run_inference_sigterm_status_matches_wrapper_contract(tmp_path, inference_body, expected_status):
    executable = tmp_path / "inference"
    executable.write_text(f"#!/usr/bin/env python3\n{inference_body}\n")
    executable.chmod(0o755)
    environment = os.environ | {"PATH": f"{tmp_path}:{os.environ['PATH']}"}
    child = subprocess.Popen(
        ["uv", "run", "--no-project", "inference"],
        env=environment,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=5) == expected_status
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
