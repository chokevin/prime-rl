import importlib.util
import sys
from types import ModuleType

import pytest


def _load_patches(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    source_spec = importlib.util.find_spec("prime_rl.inference.patches")
    assert source_spec is not None
    assert source_spec.origin is not None
    spec = importlib.util.spec_from_file_location("_prime_rl_inference_patches_test", source_spec.origin)
    assert spec is not None
    assert spec.loader is not None
    patches = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patches)
    return patches


def _install_fake_vllm(monkeypatch: pytest.MonkeyPatch, *, include_pause_complete: bool) -> type:
    vllm = ModuleType("vllm")
    vllm.__version__ = "0.20.2"
    vllm.__path__ = []
    monkeypatch.setitem(sys.modules, "vllm", vllm)

    for module_name in ("vllm.v1", "vllm.v1.core", "vllm.v1.core.sched"):
        module = ModuleType(module_name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, module_name, module)

    config = ModuleType("vllm.config")

    class ParallelConfig:
        @staticmethod
        def has_unfinished_dp(dp_group, local_unfinished):
            return local_unfinished

    config.ParallelConfig = ParallelConfig
    monkeypatch.setitem(sys.modules, "vllm.config", config)

    interface = ModuleType("vllm.v1.core.sched.interface")

    class PauseState:
        UNPAUSED = "unpaused"

    interface.PauseState = PauseState
    monkeypatch.setitem(sys.modules, "vllm.v1.core.sched.interface", interface)

    engine = ModuleType("vllm.v1.engine")

    class EngineCoreOutputs:
        def __init__(self, start_wave=None):
            self.start_wave = start_wave

    class EngineCoreRequestType:
        START_DP_WAVE = "start_dp_wave"

    engine.EngineCoreOutputs = EngineCoreOutputs
    engine.EngineCoreRequestType = EngineCoreRequestType
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", engine)

    core = ModuleType("vllm.v1.engine.core")

    class EngineCore:
        def add_request(self, request, request_wave=0):
            return None

    class EngineCoreProc:
        def _handle_client_request(self, request_type, request):
            return None

        def resume_scheduler(self):
            return None

    if include_pause_complete:

        def _pause_complete(self):
            return True

        EngineCoreProc._pause_complete = _pause_complete

    class DPEngineCoreProc:
        def _has_global_unfinished_reqs(self, local_unfinished):
            return local_unfinished

    core.EngineCore = EngineCore
    core.EngineCoreProc = EngineCoreProc
    core.DPEngineCoreProc = DPEngineCoreProc
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", core)

    request = ModuleType("vllm.v1.request")

    class Request:
        pass

    request.Request = Request
    monkeypatch.setitem(sys.modules, "vllm.v1.request", request)
    return DPEngineCoreProc


def test_dp_pause_resume_patch_fails_clearly_for_old_vllm_shape(monkeypatch):
    patches = _load_patches(monkeypatch)
    _install_fake_vllm(monkeypatch, include_pause_complete=False)

    with pytest.raises(RuntimeError, match="vLLM >=0.21.0.*EngineCoreProc\\._pause_complete"):
        patches.monkey_patch_dp_engine_core_pause_resume_deadlock()


def test_dp_pause_resume_patch_applies_when_expected_vllm_apis_exist(monkeypatch):
    patches = _load_patches(monkeypatch)
    dp_engine_core_proc = _install_fake_vllm(monkeypatch, include_pause_complete=True)

    patches.monkey_patch_dp_engine_core_pause_resume_deadlock()

    assert dp_engine_core_proc._has_global_unfinished_reqs.__name__ == "_patched_has_global_unfinished_reqs"
