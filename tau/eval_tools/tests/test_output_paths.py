from pathlib import Path

import pytest

from tau.eval_tools.output_paths import expected_output_path, prepare_output_directory


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
def test_prepare_output_directory_creates_only_exact_mode_path(tmp_path, mode, eval_label, leaf):
    data_root = tmp_path / "data"
    data_root.mkdir()
    expected = data_root / "pretraining-data" / "prime-rl-math-7b-h200" / leaf

    prepared = prepare_output_directory(
        mode,
        expected,
        eval_label=eval_label,
        data_root=data_root,
    )

    assert prepared == expected
    assert prepared.is_dir()
    assert prepared.resolve(strict=True) == expected


def test_prepared_eval_and_train_outputs_support_first_runtime_use(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    eval_output = expected_output_path("eval", eval_label="baseline", data_root=data_root)
    train_output = expected_output_path("train", data_root=data_root)

    prepare_output_directory("eval", eval_output, eval_label="baseline", data_root=data_root)
    (eval_output / "inference.log").write_text("started")
    prepare_output_directory("train", train_output, data_root=data_root)

    assert (eval_output / "inference.log").read_text() == "started"
    assert train_output.resolve(strict=True) == train_output


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
    supplied = data_root / "pretraining-data" / "prime-rl-math-7b-h200" / "train"

    with pytest.raises(ValueError):
        prepare_output_directory(
            mode,
            supplied,
            eval_label=eval_label,
            data_root=data_root,
        )

    assert not list(data_root.iterdir())


def test_prepare_output_directory_rejects_wrong_or_noncanonical_path_before_creation(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    wrong = data_root / "pretraining-data" / "prime-rl-math-7b-h200" / "eval-post"
    noncanonical = f"{data_root}//pretraining-data/prime-rl-math-7b-h200/eval-baseline"

    with pytest.raises(ValueError, match="expected exactly"):
        prepare_output_directory("eval", wrong, eval_label="baseline", data_root=data_root)
    with pytest.raises(ValueError, match="canonical"):
        prepare_output_directory("eval", noncanonical, eval_label="baseline", data_root=data_root)

    assert not list(data_root.iterdir())


def test_prepare_output_directory_rejects_symlinked_component_without_touching_target(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (data_root / "pretraining-data").symlink_to(external, target_is_directory=True)
    expected = expected_output_path("train", data_root=data_root)

    with pytest.raises(OSError):
        prepare_output_directory("train", expected, data_root=data_root)

    assert not list(external.iterdir())


def test_prepare_output_directory_rejects_non_directory_component(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    component = data_root / "pretraining-data"
    component.write_text("not a directory")
    expected = expected_output_path("train", data_root=data_root)

    with pytest.raises(OSError):
        prepare_output_directory("train", expected, data_root=data_root)

    assert component.read_text() == "not a directory"


def test_prepare_output_directory_preserves_existing_artifacts(tmp_path):
    data_root = tmp_path / "data"
    output = expected_output_path("eval", eval_label="baseline", data_root=data_root)
    output.mkdir(parents=True)
    rewards = output / "rewards.json"
    rewards.write_bytes(b"immutable")

    prepare_output_directory("eval", output, eval_label="baseline", data_root=data_root)

    assert rewards.read_bytes() == b"immutable"


def test_wrapper_prepares_output_before_mode_specific_first_use():
    script = (Path(__file__).parents[2] / "scripts/run-prime-rl.sh").read_text()
    prepare = script.index("tau.eval_tools.output_paths")
    dispatch = script.index('case "$PRIME_RL_RUN_MODE" in', prepare)
    inference_redirect = script.index('>"${TAU_OUTPUT_DIR}/inference.log"', dispatch)
    training_supervisor = script.index("tau.eval_tools.live.training_supervisor_live run", dispatch)

    assert prepare < dispatch < inference_redirect
    assert prepare < dispatch < training_supervisor
