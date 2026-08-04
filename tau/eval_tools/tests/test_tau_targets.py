"""Target-level regressions over the checked-in `tau/*.yaml` files.

These are distinct from the per-mode Python contract tests in `test_output_paths.py`:
those prove `output_paths.py`/`run-prime-rl.sh` reject a bad or missing recovery
identity env at runtime; these prove the *checked-in target files themselves* cannot
drift apart or silently omit part of that same contract, which is what actually caused
the live CPU-preflight blocker (the preflight target simply never set these vars).
"""

from pathlib import Path

import yaml

from tau.eval_tools.f12_recovery import F12_RECOVERY_ENVIRONMENT, f12_recovery_environment_variable_names

TAU_DIR = Path(__file__).parents[2]
H200_RECOVERY_TARGET = TAU_DIR / "f12-eval-post-recovery.yaml"
CPU_PREFLIGHT_TARGET = TAU_DIR / "f12-eval-post-recovery-preflight.yaml"


def _target_env(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text())["runtime"]["env"]


def test_cpu_preflight_and_h200_recovery_targets_share_identical_recovery_identity_env():
    h200_env = _target_env(H200_RECOVERY_TARGET)
    preflight_env = _target_env(CPU_PREFLIGHT_TARGET)

    recovery_names = f12_recovery_environment_variable_names()
    assert recovery_names, "expected at least one recovery identity env var name"

    for name in recovery_names:
        assert name in h200_env, f"{H200_RECOVERY_TARGET.name} is missing {name}"
        assert name in preflight_env, f"{CPU_PREFLIGHT_TARGET.name} is missing {name}"
        assert preflight_env[name] == h200_env[name], (
            f"{name} differs between {H200_RECOVERY_TARGET.name} ({h200_env[name]!r}) "
            f"and {CPU_PREFLIGHT_TARGET.name} ({preflight_env[name]!r})"
        )

    # Belt-and-suspenders: the same set, derived directly from F12_RECOVERY_ENVIRONMENT,
    # so a future key added only to f12_recovery.py's dict is also caught here.
    expected_names = {f"PRIME_RL_RECOVERY_{name.upper()}" for name in F12_RECOVERY_ENVIRONMENT}
    assert expected_names == set(recovery_names)
    assert expected_names <= h200_env.keys()
    assert expected_names <= preflight_env.keys()


def test_recovery_target_env_values_match_f12_recovery_environment_constants():
    # Cross-YAML parity alone would still pass if both targets were edited to the same
    # *wrong* value, so also compare each target's values directly against the Python
    # F12_RECOVERY_ENVIRONMENT constants -- the actual values validate_f12_recovery_environment
    # checks against at runtime.
    for target in (H200_RECOVERY_TARGET, CPU_PREFLIGHT_TARGET):
        target_env = _target_env(target)
        for name, expected_value in F12_RECOVERY_ENVIRONMENT.items():
            env_name = f"PRIME_RL_RECOVERY_{name.upper()}"
            actual_value = target_env[env_name]
            assert isinstance(actual_value, str), (
                f"{target.name}:{env_name} must be a YAML string, got {actual_value!r}"
            )
            assert actual_value == expected_value, (
                f"{target.name}:{env_name} is {actual_value!r}, expected {expected_value!r}"
            )


def test_both_recovery_targets_pin_the_same_runtime_source_and_fresh_output_generation():
    h200_env = _target_env(H200_RECOVERY_TARGET)
    preflight_env = _target_env(CPU_PREFLIGHT_TARGET)

    assert h200_env["PRIME_RL_REPO_SHA"] == preflight_env["PRIME_RL_REPO_SHA"]

    runtime_source = h200_env["PRIME_RL_REPO_SHA"]
    # The runtime source that produced the live blocker must never be reused as the
    # output generation for the repin that fixes it.
    assert runtime_source != "2594a4c0bdd1ba8ccc754e12205f4b0201bfef86"
    assert runtime_source not in h200_env["PRIME_RL_LORA_ADAPTER_PATH"]

    h200_output = yaml.safe_load(H200_RECOVERY_TARGET.read_text())["storage"]["output"]
    preflight_output = yaml.safe_load(CPU_PREFLIGHT_TARGET.read_text())["storage"]["output"]
    assert runtime_source in h200_output
    assert runtime_source in preflight_output
    assert h200_env["PRIME_RL_RECOVERY_PREFLIGHT_PATH"].startswith(
        f"/data/pretraining-data/prime-rl-math-7b-h200/generations/{runtime_source}/"
    )
    assert preflight_env["PRIME_RL_RECOVERY_PREFLIGHT_OUTPUT_PATH"] == h200_env["PRIME_RL_RECOVERY_PREFLIGHT_PATH"]


def test_h200_recovery_target_keeps_fail_closed_preflight_digest_placeholder():
    h200_env = _target_env(H200_RECOVERY_TARGET)
    assert h200_env["PRIME_RL_RECOVERY_PREFLIGHT_SHA256"] == "0" * 64
