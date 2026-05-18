import pytest

pytest.importorskip("torch")

from prime_rl.utils import utils


def test_clean_exit_destroys_distributed_process_group(monkeypatch):
    calls = []

    monkeypatch.delenv("PRIME_RL_SKIP_DIST_DESTROY", raising=False)
    monkeypatch.setattr(utils.wandb, "finish", lambda *args, **kwargs: None)
    monkeypatch.setattr(utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(utils.dist, "destroy_process_group", lambda: calls.append("destroyed"))

    @utils.clean_exit
    def run():
        return "done"

    assert run() == "done"
    assert calls == ["destroyed"]


def test_clean_exit_can_leave_distributed_process_group_to_owner(monkeypatch):
    calls = []

    monkeypatch.setenv("PRIME_RL_SKIP_DIST_DESTROY", "1")
    monkeypatch.setattr(utils.wandb, "finish", lambda *args, **kwargs: None)
    monkeypatch.setattr(utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(utils.dist, "destroy_process_group", lambda: calls.append("destroyed"))

    @utils.clean_exit
    def run():
        return "done"

    assert run() == "done"
    assert calls == []
