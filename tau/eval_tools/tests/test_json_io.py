import hashlib
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

import tau.eval_tools.json_io as json_io_module
from tau.eval_tools.json_io import load_json_with_sha256, promote_file_noreplace, write_json_exclusive


def test_generic_file_promotion_streams_large_evidence_without_retaining_bytes(tmp_path):
    staging = tmp_path / "inference.attempt-pod.log"
    final = tmp_path / "inference.log"
    raw = b"x" * (3 * 1024 * 1024 + 17)
    staging.write_bytes(raw)

    snapshot = promote_file_noreplace(staging, final)

    assert snapshot.digest == hashlib.sha256(raw).hexdigest()
    assert snapshot.size == len(raw)
    assert not hasattr(snapshot, "raw")
    assert final.read_bytes() == raw


@pytest.mark.parametrize("phase", ["during-write", "before-install"])
def test_exclusive_json_interruption_leaves_no_partial_final_and_retry_succeeds(
    tmp_path,
    monkeypatch,
    phase,
):
    path = tmp_path / "evidence.json"
    payload = {"attempt": 1, "status": "success"}
    original_write_all = json_io_module._write_all
    original_before_install = json_io_module._before_json_install

    if phase == "during-write":

        def interrupt_write(descriptor, raw):
            original_write_all(descriptor, raw[:7])
            raise RuntimeError("interrupted during write")

        monkeypatch.setattr(json_io_module, "_write_all", interrupt_write)
    else:

        def interrupt_before_install(_path):
            raise RuntimeError("interrupted before install")

        monkeypatch.setattr(json_io_module, "_before_json_install", interrupt_before_install)

    with pytest.raises(RuntimeError, match="interrupted"):
        write_json_exclusive(path, payload)

    assert not path.exists()
    stages = list(tmp_path.glob(".evidence.json.stage-*"))
    assert len(stages) == 1

    monkeypatch.setattr(json_io_module, "_write_all", original_write_all)
    monkeypatch.setattr(json_io_module, "_before_json_install", original_before_install)
    write_json_exclusive(path, payload)
    assert load_json_with_sha256(path)[0] == payload
    assert stages[0].exists()


def test_exclusive_json_interruption_after_install_leaves_verified_immutable_final(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    payload = {"attempt": 1, "status": "success"}

    def interrupt_after_install(_path):
        raise RuntimeError("interrupted after install")

    monkeypatch.setattr(json_io_module, "_after_json_install", interrupt_after_install)
    with pytest.raises(RuntimeError, match="interrupted after install"):
        write_json_exclusive(path, payload)

    assert load_json_with_sha256(path)[0] == payload
    with pytest.raises(FileExistsError):
        write_json_exclusive(path, {"attempt": 2})


def test_exclusive_json_concurrent_writers_install_one_complete_final(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    barrier = Barrier(2)

    def wait_before_install(_path):
        barrier.wait(timeout=10)

    monkeypatch.setattr(json_io_module, "_before_json_install", wait_before_install)
    payloads = [{"writer": 1}, {"writer": 2}]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(write_json_exclusive, path, payload) for payload in payloads]

    outcomes = []
    for future in futures:
        try:
            future.result()
        except FileExistsError:
            outcomes.append("lost")
        else:
            outcomes.append("won")
    assert sorted(outcomes) == ["lost", "won"]
    assert load_json_with_sha256(path)[0] in payloads
    assert len(list(tmp_path.glob(".evidence.json.stage-*"))) == 1


def test_exclusive_json_partial_existing_final_is_immutable(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_bytes(b'{"status":"part')

    with pytest.raises(FileExistsError):
        write_json_exclusive(path, {"status": "success"})

    assert path.read_bytes() == b'{"status":"part'
    assert not list(tmp_path.glob(".evidence.json.stage-*"))


def test_exclusive_json_preserves_suspicious_stage_collision(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    suspicious = tmp_path / ".evidence.json.stage-collision"
    suspicious.write_text("do not remove")
    tokens = iter(["collision", "owned"])
    monkeypatch.setattr(json_io_module.secrets, "token_hex", lambda _size: next(tokens))

    write_json_exclusive(path, {"status": "success"})

    assert suspicious.read_text() == "do not remove"
    assert load_json_with_sha256(path)[0] == {"status": "success"}


def test_exclusive_json_closes_staged_writer_before_promotion(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    original_open = json_io_module.os.open
    original_close = json_io_module.os.close
    original_rename = json_io_module.rename_entry_noreplace
    staged_writers = set()
    closed_descriptors = set()

    def tracked_open(name, flags, mode=0o777, *, dir_fd=None):
        descriptor = original_open(name, flags, mode, dir_fd=dir_fd)
        if str(name).startswith(".evidence.json.stage-") and flags & json_io_module.os.O_WRONLY:
            staged_writers.add(descriptor)
        return descriptor

    def tracked_close(descriptor):
        if descriptor in staged_writers:
            closed_descriptors.add(descriptor)
        original_close(descriptor)

    def assert_closed_before_rename(directory_descriptor, source_name, destination_name):
        assert staged_writers
        assert staged_writers <= closed_descriptors
        original_rename(directory_descriptor, source_name, destination_name)

    monkeypatch.setattr(
        json_io_module.os,
        "supports_dir_fd",
        json_io_module.os.supports_dir_fd | {tracked_open},
    )
    monkeypatch.setattr(json_io_module.os, "open", tracked_open)
    monkeypatch.setattr(json_io_module.os, "close", tracked_close)
    monkeypatch.setattr(json_io_module, "rename_entry_noreplace", assert_closed_before_rename)

    write_json_exclusive(path, {"status": "success"})

    assert load_json_with_sha256(path)[0] == {"status": "success"}


def test_exclusive_json_rejects_stage_swap_without_installing_replacement(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    displaced = tmp_path / "displaced-owned-stage"
    replacement = b'{"replacement":true}\n'

    def swap_stage(stage):
        stage.rename(displaced)
        stage.write_bytes(replacement)

    monkeypatch.setattr(json_io_module, "_before_json_install", swap_stage)
    with pytest.raises(RuntimeError, match="staging changed"):
        write_json_exclusive(path, {"owned": True})

    assert not path.exists()
    assert displaced.exists()
    assert displaced.read_bytes() != replacement
    stages = list(tmp_path.glob(".evidence.json.stage-*"))
    assert len(stages) == 1
    assert stages[0].read_bytes() == replacement


def test_exclusive_json_quarantines_stage_swap_after_validation_and_retry_succeeds(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "evidence.json"
    displaced = tmp_path / "displaced-validated-stage"
    replacement = b'{"replacement":true}\n'
    payload = {"owned": True}
    original_hook = json_io_module._after_json_stage_validation

    def swap_after_validation(stage):
        stage.rename(displaced)
        stage.write_bytes(replacement)

    monkeypatch.setattr(json_io_module, "_after_json_stage_validation", swap_after_validation)
    with pytest.raises(RuntimeError, match="changed during installation"):
        write_json_exclusive(path, payload)

    assert not path.exists()
    assert displaced.read_bytes() != replacement
    quarantines = list(tmp_path.glob(".evidence.json.quarantine-*"))
    assert len(quarantines) == 1
    assert quarantines[0].read_bytes() == replacement

    monkeypatch.setattr(json_io_module, "_after_json_stage_validation", original_hook)
    write_json_exclusive(path, payload)
    assert load_json_with_sha256(path)[0] == payload
    assert quarantines[0].read_bytes() == replacement


def test_exclusive_json_quarantines_final_swap_before_reload_and_retry_succeeds(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "evidence.json"
    displaced = tmp_path / "displaced-installed-final"
    replacement = b'{"replacement":true}\n'
    payload = {"owned": True}
    original_hook = json_io_module._after_json_install

    def swap_before_reload(final):
        final.rename(displaced)
        final.write_bytes(replacement)

    monkeypatch.setattr(json_io_module, "_after_json_install", swap_before_reload)
    with pytest.raises(RuntimeError, match="changed during installation"):
        write_json_exclusive(path, payload)

    assert not path.exists()
    assert load_json_with_sha256(displaced)[0] == payload
    quarantines = list(tmp_path.glob(".evidence.json.quarantine-*"))
    assert len(quarantines) == 1
    assert quarantines[0].read_bytes() == replacement

    monkeypatch.setattr(json_io_module, "_after_json_install", original_hook)
    write_json_exclusive(path, payload)
    assert load_json_with_sha256(path)[0] == payload
    assert quarantines[0].read_bytes() == replacement


def test_exclusive_json_existing_final_is_never_quarantined(tmp_path):
    path = tmp_path / "evidence.json"
    write_json_exclusive(path, {"immutable": True})
    original = path.read_bytes()

    with pytest.raises(FileExistsError):
        write_json_exclusive(path, {"replacement": True})

    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".evidence.json.quarantine-*"))
