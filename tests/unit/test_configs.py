from pathlib import Path
from typing import Annotated, Literal

import pytest
import tomli_w
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from pydantic_config import ConfigFileError

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.configs.shared import TransportConfig
from prime_rl.configs.trainer import ModelConfig as TrainerModelConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.utils.config import BaseConfig, cli

# All config config classes
CONFIG_CLASSES = [
    RLConfig,
    TrainerConfig,
    SFTConfig,
    OrchestratorConfig,
    InferenceConfig,
]


def get_config_files() -> list[Path]:
    """Any TOML file inside `configs/` or `examples/`"""
    config_files = list(Path("configs").rglob("*.toml"))
    example_files = list(Path("examples").rglob("*.toml"))

    return config_files + example_files


@pytest.mark.parametrize("config_file", get_config_files(), ids=lambda x: x.as_posix())
def test_load_configs(config_file: Path):
    """Tests that all config files can be loaded by at least one config class."""
    could_parse = []
    for config_cls in CONFIG_CLASSES:
        try:
            cli(config_cls, args=["@", config_file.as_posix()])
            could_parse.append(True)
        except (ValidationError, ConfigFileError, SystemExit):
            could_parse.append(False)
    assert any(could_parse), f"No config class could be parsed from {config_file}"


class NestedConfig(BaseConfig):
    lr: float = 1e-4
    weight_decay: float = 0.01
    name: str = "default"


class VariantA(BaseModel):
    type: Literal["a"] = "a"
    alpha: float = 0.1
    shared: int = 1


class VariantB(BaseModel):
    type: Literal["b"] = "b"
    beta: float = 0.2
    shared: int = 1


VariantType = Annotated[VariantA | VariantB, Field(discriminator="type")]


class DummyConfig(BaseConfig):
    name: str = "experiment"
    seed: int = 42
    nested: NestedConfig = NestedConfig()
    variant: VariantType = VariantA()


def write_toml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def test_defaults():
    """All defaults are applied when no TOML or CLI args are given."""
    config = cli(DummyConfig, args=[])
    assert config.name == "experiment"
    assert config.seed == 42
    assert config.nested.lr == 1e-4
    assert config.nested.weight_decay == 0.01
    assert config.variant.type == "a"
    assert config.variant.alpha == 0.1


def test_toml_partial_nested_override(tmp_path):
    """Partially overriding a nested model preserves unset field defaults."""
    write_toml(tmp_path / "cfg.toml", {"nested": {"lr": 3e-4}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.nested.lr == 3e-4
    assert config.nested.weight_decay == 0.01
    assert config.nested.name == "default"


def test_toml_discriminated_union_default_type(tmp_path):
    """Overriding a discriminated union field without 'type' uses the default variant."""
    write_toml(tmp_path / "cfg.toml", {"variant": {"alpha": 0.9}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.variant.type == "a"
    assert config.variant.alpha == 0.9
    assert config.variant.shared == 1


def test_toml_discriminated_union_switch_variant(tmp_path):
    """Providing an explicit 'type' switches to that variant."""
    write_toml(tmp_path / "cfg.toml", {"variant": {"type": "b"}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.variant.type == "b"
    assert config.variant.beta == 0.2


def test_toml_discriminated_union_override_switch_variant(tmp_path):
    """Providing an explicit 'type' overrides the default variant."""
    write_toml(tmp_path / "cfg.toml", {"variant": {"type": "b", "beta": 0.5}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.variant.type == "b"
    assert config.variant.beta == 0.5


def test_cli_overrides_defaults():
    """CLI args override defaults."""
    config = cli(DummyConfig, args=["--name", "my-run", "--seed", "7"])
    assert config.name == "my-run"
    assert config.seed == 7
    assert config.nested.lr == 1e-4


def test_toml_overrides_defaults(tmp_path):
    """TOML overrides defaults."""
    write_toml(tmp_path / "cfg.toml", {"name": "my-run", "seed": 7, "nested": {"lr": 3e-4}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml")])
    assert config.name == "my-run"
    assert config.seed == 7
    assert config.nested.lr == 3e-4


def test_cli_overrides_toml(tmp_path):
    """CLI args override TOML."""
    write_toml(tmp_path / "cfg.toml", {"seed": 1, "nested": {"lr": 3e-4}})
    config = cli(DummyConfig, args=["@", str(tmp_path / "cfg.toml"), "--seed", "99", "--nested.lr", "5e-5"])
    assert config.seed == 99
    assert config.nested.lr == 5e-5
    # TOML value not overridden by CLI should still be applied (not reverted to class default)
    assert config.nested.weight_decay == 0.01


def test_removed_fused_lm_head_chunk_size_field_is_rejected():
    with pytest.raises(ValidationError, match="fused_lm_head_chunk_size"):
        TrainerModelConfig.model_validate({"fused_lm_head_chunk_size": "auto"})


def test_selective_activation_checkpointing_requires_custom_impl():
    with pytest.raises(ValidationError, match="Selective activation checkpointing requires model.impl='custom'"):
        TrainerModelConfig.model_validate({"impl": "hf", "ac": {"mode": "selective"}})


def test_ray_runtime_config_requires_ray_transport():
    with pytest.raises(ValidationError, match="rollout_transport.type = 'ray'"):
        cli(
            RLConfig,
            args=[
                "@",
                "examples/reverse_text/rl.toml",
                "--experimental.ray.enabled",
            ],
        )


def test_ray_runtime_config_parses_with_ray_transport():
    config = cli(
        RLConfig,
        args=[
            "@",
            "examples/reverse_text/rl.toml",
            "--experimental.ray.enabled",
            "--experimental.ray.namespace",
            "test",
            "--trainer.rollout-transport.type",
            "ray",
            "--orchestrator.rollout-transport.type",
            "ray",
        ],
    )
    assert config.experimental.ray.enabled
    assert config.experimental.ray.namespace == "test"
    assert config.experimental.ray.placement_strategy == "STRICT_PACK"
    assert config.experimental.ray.trainer_backend == "tasks"
    assert config.experimental.ray.inference_backend == "prime_vllm"
    assert config.trainer.rollout_transport.type == "ray"
    assert config.orchestrator.rollout_transport.type == "ray"


def test_ray_runtime_config_parses_prime_vllm_inference_backend():
    config = cli(
        RLConfig,
        args=[
            "@",
            "examples/reverse_text/rl.toml",
            "--experimental.ray.enabled",
            "--experimental.ray.inference-backend",
            "prime_vllm",
            "--trainer.rollout-transport.type",
            "ray",
            "--orchestrator.rollout-transport.type",
            "ray",
        ],
    )
    assert config.experimental.ray.inference_backend == "prime_vllm"


def test_ray_runtime_config_parses_ray_train_backend():
    config = cli(
        RLConfig,
        args=[
            "@",
            "examples/reverse_text/rl.toml",
            "--experimental.ray.enabled",
            "--experimental.ray.trainer-backend",
            "ray_train",
            "--experimental.ray.train-run-name",
            "test-run",
            "--experimental.ray.train-storage-path",
            "/tmp/ray-train",
            "--trainer.rollout-transport.type",
            "ray",
            "--orchestrator.rollout-transport.type",
            "ray",
        ],
    )
    assert config.experimental.ray.trainer_backend == "ray_train"
    assert config.experimental.ray.train_run_name == "test-run"
    assert config.experimental.ray.train_storage_path == "/tmp/ray-train"


def test_ray_runtime_config_requires_matching_actor_name():
    with pytest.raises(ValidationError, match="same actor_name"):
        cli(
            RLConfig,
            args=[
                "@",
                "examples/reverse_text/rl.toml",
                "--experimental.ray.enabled",
                "--trainer.rollout-transport.type",
                "ray",
                "--trainer.rollout-transport.actor-name",
                "trainer-transport",
                "--orchestrator.rollout-transport.type",
                "ray",
                "--orchestrator.rollout-transport.actor-name",
                "orchestrator-transport",
            ],
        )


def test_ray_runtime_config_requires_matching_namespace():
    with pytest.raises(ValidationError, match="same namespace"):
        cli(
            RLConfig,
            args=[
                "@",
                "examples/reverse_text/rl.toml",
                "--experimental.ray.enabled",
                "--trainer.rollout-transport.type",
                "ray",
                "--trainer.rollout-transport.namespace",
                "trainer",
                "--orchestrator.rollout-transport.type",
                "ray",
                "--orchestrator.rollout-transport.namespace",
                "orchestrator",
            ],
        )


def test_ray_transport_config_parses():
    transport = TypeAdapter(TransportConfig).validate_python(
        {"type": "ray", "address": "auto", "namespace": "test", "actor_name": "transport"}
    )
    assert transport.type == "ray"
    assert transport.address == "auto"
    assert transport.namespace == "test"
    assert transport.actor_name == "transport"
