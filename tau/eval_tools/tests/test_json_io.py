from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

import tau.eval_tools.json_io as json_io_module
from tau.eval_tools.json_io import load_json_with_sha256, write_json_exclusive


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
