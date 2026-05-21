import asyncio

import pytest

from prime_rl.utils.pathing import durable_touch, sync_wait_for_path, validate_output_dir, wait_for_path


def test_nonexistent_dir_passes(tmp_path):
    output_dir = tmp_path / "does_not_exist"
    validate_output_dir(output_dir, resuming=False, clean=False)


def test_empty_dir_passes(tmp_path):
    output_dir = tmp_path / "empty"
    output_dir.mkdir()
    validate_output_dir(output_dir, resuming=False, clean=False)


def test_dir_with_only_logs_passes(tmp_path):
    output_dir = tmp_path / "has_logs"
    output_dir.mkdir()
    (output_dir / "logs").mkdir()
    (output_dir / "logs" / "trainer.log").touch()
    validate_output_dir(output_dir, resuming=False, clean=False)


def test_dir_with_checkpoints_raises(tmp_path):
    output_dir = tmp_path / "has_ckpt"
    output_dir.mkdir()
    (output_dir / "checkpoints").mkdir()
    (output_dir / "checkpoints" / "step_0").mkdir()
    with pytest.raises(FileExistsError, match="already contains checkpoints"):
        validate_output_dir(output_dir, resuming=False, clean=False)


def test_dir_with_checkpoints_passes_when_resuming(tmp_path):
    output_dir = tmp_path / "has_ckpt"
    output_dir.mkdir()
    (output_dir / "checkpoints").mkdir()
    (output_dir / "checkpoints" / "step_0").mkdir()
    validate_output_dir(output_dir, resuming=True, clean=False)


def test_dir_with_checkpoints_cleaned_when_flag_set(tmp_path):
    output_dir = tmp_path / "has_ckpt"
    output_dir.mkdir()
    (output_dir / "checkpoints").mkdir()
    (output_dir / "checkpoints" / "step_0").mkdir()
    (output_dir / "logs").mkdir()

    validate_output_dir(output_dir, resuming=False, clean=True)

    assert not output_dir.exists()


def test_clean_on_nonexistent_dir_is_noop(tmp_path):
    output_dir = tmp_path / "does_not_exist"
    validate_output_dir(output_dir, resuming=False, clean=True)
    assert not output_dir.exists()


def test_durable_touch_creates_marker_and_parent(tmp_path):
    marker = tmp_path / "missing_parent" / "NCCL_READY"

    durable_touch(marker)

    assert marker.exists()
    assert marker.read_text()


def test_sync_wait_for_path_finds_marker_via_parent_listing(tmp_path):
    marker = tmp_path / "NCCL_READY"
    marker.write_text("ready")

    sync_wait_for_path(marker, interval=0.01, log_interval=1)


def test_wait_for_path_finds_marker_via_parent_listing(tmp_path):
    marker = tmp_path / "NCCL_READY"
    marker.write_text("ready")

    asyncio.run(wait_for_path(marker, interval=0.01, log_interval=1))
