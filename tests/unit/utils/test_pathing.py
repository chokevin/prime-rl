from pathlib import Path

import pytest

from prime_rl.utils.pathing import _path_visible, durable_touch, validate_output_dir


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


def test_durable_touch_fsyncs_new_step_directory_parent(tmp_path, monkeypatch):
    fsynced: list[Path] = []

    def fake_fsync_directory(path: Path) -> None:
        fsynced.append(path)

    monkeypatch.setattr("prime_rl.utils.pathing._fsync_directory", fake_fsync_directory)

    broadcasts_dir = tmp_path / "broadcasts"
    broadcasts_dir.mkdir()
    marker = broadcasts_dir / "step_1" / "STABLE"

    durable_touch(marker)

    assert marker.exists()
    assert marker.parent in fsynced
    assert broadcasts_dir in fsynced


def test_path_visible_scans_ancestor_entries_when_exists_is_stale(tmp_path, monkeypatch):
    marker = tmp_path / "broadcasts" / "step_1" / "STABLE"
    marker.parent.mkdir(parents=True)
    marker.write_text("ready")
    original_exists = Path.exists

    def stale_exists(path: Path) -> bool:
        if path in {marker, marker.parent, marker.parent.parent}:
            return False
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", stale_exists)

    assert _path_visible(marker)
