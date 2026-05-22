from pathlib import Path
from typing import Annotated, Literal

import pytest
import tomli_w
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from pydantic_config import ConfigFileError

from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.orchestrator import OrchestratorConfig, TeacherModelConfig
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.configs.shared import TransportConfig
from prime_rl.configs.trainer import ModelConfig as TrainerModelConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.ray.native import (
    _monitor_roles,
    _orchestrator_with_ray_inference_endpoint,
    _orchestrator_with_ray_teacher_endpoint,
)
from prime_rl.transport.ray import _RayTransportStore
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
    """Any TOML file inside `configs/` or `examples/` (skips the configs/private/ submodule)."""
    private = Path("configs/private")
    config_files = [p for p in Path("configs").rglob("*.toml") if private not in p.parents]
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


def test_orchestrator_request_picker_config_defaults_to_direct():
    config = OrchestratorConfig.model_validate({"use_renderer": False})
    assert config.experimental.request_picker.type == "direct"


def test_orchestrator_external_request_picker_config():
    config = OrchestratorConfig.model_validate(
        {
            "use_renderer": False,
            "experimental": {
                "request_picker": {
                    "type": "external",
                    "adapter_url": "http://picker.local/pick",
                    "timeout": 0.25,
                }
            },
        }
    )
    assert config.experimental.request_picker.type == "external"
    assert config.experimental.request_picker.adapter_url == "http://picker.local/pick"
    assert config.experimental.request_picker.timeout == 0.25


def test_orchestrator_prime_aware_request_picker_config():
    config = OrchestratorConfig.model_validate(
        {
            "use_renderer": False,
            "experimental": {
                "request_picker": {
                    "type": "prime_aware",
                    "inflight_slack": 1,
                    "waiting_weight": 2.0,
                    "decode_deficit_weight": 0.5,
                    "history_penalty_cap": 2.0,
                    "wave_minimax_size": 16,
                }
            },
        }
    )
    assert config.experimental.request_picker.type == "prime_aware"
    assert config.experimental.request_picker.inflight_slack == 1
    assert config.experimental.request_picker.waiting_weight == 2.0
    assert config.experimental.request_picker.decode_deficit_weight == 0.5
    assert config.experimental.request_picker.history_penalty_cap == 2.0
    assert config.experimental.request_picker.wave_minimax_size == 16


def test_nccl_async_level_above_one_requires_explicit_opt_in():
    with pytest.raises(ValidationError, match="allow_async_level_gt_1"):
        TrainerConfig.model_validate(
            {
                "max_async_level": 2,
                "weight_broadcast": {"type": "nccl"},
            }
        )

    with pytest.raises(ValidationError, match="allow_async_level_gt_1"):
        OrchestratorConfig.model_validate(
            {
                "use_renderer": False,
                "max_async_level": 2,
                "weight_broadcast": {"type": "nccl"},
            }
        )


def test_nccl_async_level_above_one_opt_in_requires_non_strict_and_off_policy_capacity():
    trainer = TrainerConfig.model_validate(
        {
            "max_async_level": 2,
            "weight_broadcast": {"type": "nccl", "allow_async_level_gt_1": True},
        }
    )
    assert trainer.weight_broadcast.type == "nccl"
    assert trainer.weight_broadcast.allow_async_level_gt_1

    orchestrator = OrchestratorConfig.model_validate(
        {
            "use_renderer": False,
            "max_async_level": 2,
            "max_off_policy_steps": 2,
            "strict_async_level": False,
            "weight_broadcast": {"type": "nccl", "allow_async_level_gt_1": True},
        }
    )
    assert orchestrator.weight_broadcast.type == "nccl"
    assert orchestrator.weight_broadcast.allow_async_level_gt_1

    with pytest.raises(ValidationError, match="strict_async_level=false"):
        OrchestratorConfig.model_validate(
            {
                "use_renderer": False,
                "max_async_level": 2,
                "strict_async_level": True,
                "weight_broadcast": {"type": "nccl", "allow_async_level_gt_1": True},
            }
        )

    with pytest.raises(ValidationError, match="max_async_level must be <= max_off_policy_steps"):
        OrchestratorConfig.model_validate(
            {
                "use_renderer": False,
                "max_async_level": 5,
                "max_off_policy_steps": 4,
                "weight_broadcast": {"type": "nccl", "allow_async_level_gt_1": True},
            }
        )


def test_shared_nccl_async_level_above_one_propagates_opt_in_and_guards_orchestrator():
    config = RLConfig.model_validate(
        {
            "max_async_level": 2,
            "weight_broadcast": {"type": "nccl", "allow_async_level_gt_1": True},
            "trainer": {},
            "orchestrator": {
                "use_renderer": False,
                "max_off_policy_steps": 2,
                "strict_async_level": False,
            },
        }
    )
    assert config.trainer.max_async_level == 2
    assert config.trainer.weight_broadcast.type == "nccl"
    assert config.trainer.weight_broadcast.allow_async_level_gt_1
    assert config.orchestrator.max_async_level == 2
    assert config.orchestrator.weight_broadcast.type == "nccl"
    assert config.orchestrator.weight_broadcast.allow_async_level_gt_1

    with pytest.raises(ValidationError, match="allow_async_level_gt_1"):
        RLConfig.model_validate(
            {
                "max_async_level": 2,
                "weight_broadcast": {"type": "nccl"},
                "trainer": {},
                "orchestrator": {
                    "use_renderer": False,
                    "max_off_policy_steps": 2,
                    "strict_async_level": False,
                },
            }
        )

    with pytest.raises(ValidationError, match="strict_async_level=false"):
        RLConfig.model_validate(
            {
                "max_async_level": 2,
                "weight_broadcast": {"type": "nccl", "allow_async_level_gt_1": True},
                "trainer": {},
                "orchestrator": {
                    "use_renderer": False,
                    "max_off_policy_steps": 2,
                    "strict_async_level": True,
                },
            }
        )


def test_nccl_final_step_async_level_uses_opt_in_with_default_max_async_level():
    config = RLConfig.model_validate(
        {
            "max_steps": 10,
            "max_async_level": 1,
            "weight_broadcast": {
                "type": "nccl",
                "allow_async_level_gt_1": True,
                "final_step_async_level": 2,
            },
            "trainer": {},
            "orchestrator": {
                "use_renderer": False,
                "max_off_policy_steps": 4,
                "strict_async_level": False,
            },
        }
    )
    assert config.trainer.max_async_level == 1
    assert config.trainer.weight_broadcast.final_step_async_level == 2
    assert config.orchestrator.max_async_level == 1
    assert config.orchestrator.weight_broadcast.final_step_async_level == 2

    with pytest.raises(ValidationError, match="allow_async_level_gt_1"):
        RLConfig.model_validate(
            {
                "max_steps": 10,
                "max_async_level": 1,
                "weight_broadcast": {"type": "nccl", "final_step_async_level": 2},
                "trainer": {},
                "orchestrator": {
                    "use_renderer": False,
                    "max_off_policy_steps": 4,
                    "strict_async_level": False,
                },
            }
        )

    with pytest.raises(ValidationError, match="strict_async_level=false"):
        RLConfig.model_validate(
            {
                "max_steps": 10,
                "max_async_level": 1,
                "weight_broadcast": {
                    "type": "nccl",
                    "allow_async_level_gt_1": True,
                    "final_step_async_level": 2,
                },
                "trainer": {},
                "orchestrator": {
                    "use_renderer": False,
                    "max_off_policy_steps": 4,
                    "strict_async_level": True,
                },
            }
        )

    with pytest.raises(ValidationError, match="max_async_level must be <= max_off_policy_steps"):
        RLConfig.model_validate(
            {
                "max_steps": 10,
                "max_async_level": 1,
                "weight_broadcast": {
                    "type": "nccl",
                    "allow_async_level_gt_1": True,
                    "final_step_async_level": 5,
                },
                "trainer": {},
                "orchestrator": {
                    "use_renderer": False,
                    "max_off_policy_steps": 4,
                    "strict_async_level": False,
                },
            }
        )


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


def test_orchestrator_vlm_configs_must_disable_renderer():
    with pytest.raises(ValidationError, match="orchestrator.use_renderer is not supported for VLMs"):
        OrchestratorConfig.model_validate(
            {
                "model": {
                    "vlm": {
                        "vision_encoder_attr": "model.visual",
                        "language_model_attr": "model.language_model",
                    }
                }
            }
        )

    config = OrchestratorConfig.model_validate(
        {
            "model": {
                "vlm": {
                    "vision_encoder_attr": "model.visual",
                    "language_model_attr": "model.language_model",
                }
            },
            "use_renderer": False,
        }
    )

    assert config.use_renderer is False


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


def test_ray_runtime_config_parses_ray_cluster_deployment(tmp_path):
    write_toml(
        tmp_path / "ray_cluster.toml",
        {
            "deployment": {
                "type": "ray_cluster",
                "gpus_per_node": 1,
                "num_train_gpus": 1,
                "num_infer_gpus": 1,
            },
            "experimental": {
                "ray": {
                    "enabled": True,
                    "address": "auto",
                    "namespace": "test",
                    "placement_strategy": "SPREAD",
                    "trainer_backend": "ray_train",
                    "inference_backend": "prime_vllm",
                }
            },
            "trainer": {"rollout_transport": {"type": "ray", "address": "auto", "namespace": "test"}},
            "orchestrator": {"rollout_transport": {"type": "ray", "address": "auto", "namespace": "test"}},
        },
    )

    config = cli(
        RLConfig,
        args=[
            "@",
            "examples/reverse_text/rl.toml",
            "@",
            str(tmp_path / "ray_cluster.toml"),
        ],
    )

    assert config.deployment.type == "ray_cluster"
    assert config.deployment.num_train_gpus == 1
    assert config.deployment.num_infer_gpus == 1
    assert config.experimental.ray.enabled
    assert config.experimental.ray.placement_strategy == "SPREAD"
    assert config.experimental.ray.trainer_backend == "ray_train"


def test_ray_cluster_deployment_can_be_selected_before_ray_cli_flags(tmp_path):
    write_toml(
        tmp_path / "ray_cluster.toml",
        {
            "deployment": {
                "type": "ray_cluster",
                "num_train_gpus": 1,
                "num_infer_gpus": 1,
            },
            "trainer": {"rollout_transport": {"type": "ray"}},
            "orchestrator": {"rollout_transport": {"type": "ray"}},
        },
    )

    config = cli(
        RLConfig,
        args=[
            "@",
            "examples/reverse_text/rl.toml",
            "@",
            str(tmp_path / "ray_cluster.toml"),
            "--experimental.ray.enabled",
        ],
    )
    assert config.deployment.type == "ray_cluster"
    assert config.experimental.ray.enabled


def test_ray_runtime_rejects_slurm_multinode_deployment(tmp_path):
    write_toml(
        tmp_path / "ray_multinode.toml",
        {
            "deployment": {
                "type": "multi_node",
                "num_train_nodes": 1,
                "num_infer_nodes": 1,
            },
            "experimental": {"ray": {"enabled": True}},
            "trainer": {"rollout_transport": {"type": "ray"}},
            "orchestrator": {"rollout_transport": {"type": "ray"}},
        },
    )

    with pytest.raises(ConfigFileError, match="Use deployment.type = 'ray_cluster'"):
        cli(
            RLConfig,
            args=[
                "@",
                "examples/reverse_text/rl.toml",
                "@",
                str(tmp_path / "ray_multinode.toml"),
            ],
        )


def test_ray_cluster_inference_bundle_must_fit_one_node(tmp_path):
    write_toml(
        tmp_path / "ray_cluster.toml",
        {
            "deployment": {
                "type": "ray_cluster",
                "gpus_per_node": 1,
                "num_train_gpus": 1,
                "num_infer_gpus": 2,
            },
            "experimental": {"ray": {"enabled": True}},
            "trainer": {"rollout_transport": {"type": "ray"}},
            "orchestrator": {"rollout_transport": {"type": "ray"}},
        },
    )

    with pytest.raises(ConfigFileError, match="Prime-vLLM inference is scheduled as one Ray task"):
        cli(
            RLConfig,
            args=[
                "@",
                "examples/reverse_text/rl.toml",
                "@",
                str(tmp_path / "ray_cluster.toml"),
            ],
        )


def test_ray_cluster_num_teacher_gpus_auto_configures_teacher_model(tmp_path):
    write_toml(
        tmp_path / "ray_cluster_teacher.toml",
        {
            "deployment": {
                "type": "ray_cluster",
                "num_train_gpus": 1,
                "num_infer_gpus": 1,
                "num_teacher_gpus": 1,
            },
            "experimental": {"ray": {"enabled": True}},
            "trainer": {
                "rollout_transport": {"type": "ray"},
                "loss": {"teacher_tau": 1.0},
            },
            "orchestrator": {"rollout_transport": {"type": "ray"}},
        },
    )

    config = cli(
        RLConfig,
        args=[
            "@",
            "examples/reverse_text/rl.toml",
            "@",
            str(tmp_path / "ray_cluster_teacher.toml"),
        ],
    )

    assert config.deployment.type == "ray_cluster"
    assert config.teacher_inference is not None
    assert config.teacher_inference.server.port == config.inference.server.port + 1
    assert config.orchestrator.teacher_model is not None
    assert config.orchestrator.teacher_model.client.base_url == ["http://localhost:8001/v1"]
    assert config.orchestrator.teacher_model.model.name == config.teacher_inference.model.name


def test_ray_cluster_auto_setup_orchestrator_num_train_workers(tmp_path):
    """Multi-GPU trainer in ray_cluster auto-sets orchestrator.num_train_workers."""
    write_toml(
        tmp_path / "ray_cluster.toml",
        {
            "deployment": {
                "type": "ray_cluster",
                "gpus_per_node": 8,
                "num_train_gpus": 8,
                "num_infer_gpus": 1,
            },
            "experimental": {"ray": {"enabled": True}},
            "trainer": {"rollout_transport": {"type": "ray"}},
            "orchestrator": {"rollout_transport": {"type": "ray"}},
        },
    )

    config = cli(
        RLConfig,
        args=[
            "@",
            "examples/reverse_text/rl.toml",
            "@",
            str(tmp_path / "ray_cluster.toml"),
        ],
    )

    assert config.deployment.type == "ray_cluster"
    assert config.orchestrator.num_train_workers == 8


def test_ray_cluster_auto_setup_inference_dp_from_num_infer_gpus(tmp_path):
    """num_infer_gpus / tp drives inference.parallel.dp and api_server_count."""
    write_toml(
        tmp_path / "ray_cluster.toml",
        {
            "deployment": {
                "type": "ray_cluster",
                "gpus_per_node": 8,
                "num_train_gpus": 1,
                "num_infer_gpus": 4,
            },
            "experimental": {"ray": {"enabled": True}},
            "trainer": {"rollout_transport": {"type": "ray"}},
            "orchestrator": {"rollout_transport": {"type": "ray"}},
            "inference": {"parallel": {"tp": 2}},
        },
    )

    config = cli(
        RLConfig,
        args=[
            "@",
            "examples/reverse_text/rl.toml",
            "@",
            str(tmp_path / "ray_cluster.toml"),
        ],
    )

    assert config.deployment.type == "ray_cluster"
    assert config.inference.parallel.tp == 2
    assert config.inference.parallel.dp == 2  # 4 / 2
    assert config.inference.api_server_count >= 2


def test_ray_cluster_auto_setup_dp_replicate_for_multi_node_trainer(tmp_path):
    """num_train_gpus > gpus_per_node defaults trainer.model.dp_replicate to HSDP."""
    write_toml(
        tmp_path / "ray_cluster.toml",
        {
            "deployment": {
                "type": "ray_cluster",
                "gpus_per_node": 8,
                "num_train_gpus": 16,
                "num_infer_gpus": 8,
            },
            "experimental": {"ray": {"enabled": True}},
            "trainer": {"rollout_transport": {"type": "ray"}},
            "orchestrator": {"rollout_transport": {"type": "ray"}},
            "inference": {"parallel": {"tp": 8}},
        },
    )

    config = cli(
        RLConfig,
        args=[
            "@",
            "examples/reverse_text/rl.toml",
            "@",
            str(tmp_path / "ray_cluster.toml"),
        ],
    )

    assert config.deployment.type == "ray_cluster"
    assert config.trainer.model.dp_replicate == 2  # 16 / 8


def test_ray_cluster_user_dp_replicate_is_preserved(tmp_path):
    """Explicit trainer.model.dp_replicate is not clobbered by ray_cluster auto-setup."""
    write_toml(
        tmp_path / "ray_cluster.toml",
        {
            "deployment": {
                "type": "ray_cluster",
                "gpus_per_node": 8,
                "num_train_gpus": 16,
                "num_infer_gpus": 8,
            },
            "experimental": {"ray": {"enabled": True}},
            "trainer": {
                "rollout_transport": {"type": "ray"},
                "model": {"dp_replicate": 4},
            },
            "orchestrator": {"rollout_transport": {"type": "ray"}},
            "inference": {"parallel": {"tp": 8}},
        },
    )

    config = cli(
        RLConfig,
        args=[
            "@",
            "examples/reverse_text/rl.toml",
            "@",
            str(tmp_path / "ray_cluster.toml"),
        ],
    )

    assert config.trainer.model.dp_replicate == 4


def test_ray_cluster_auto_setup_dp_replicate_skipped_for_single_node_trainer(tmp_path):
    """num_train_gpus <= gpus_per_node leaves trainer.model.dp_replicate at the default."""
    write_toml(
        tmp_path / "ray_cluster.toml",
        {
            "deployment": {
                "type": "ray_cluster",
                "gpus_per_node": 8,
                "num_train_gpus": 8,
                "num_infer_gpus": 1,
            },
            "experimental": {"ray": {"enabled": True}},
            "trainer": {"rollout_transport": {"type": "ray"}},
            "orchestrator": {"rollout_transport": {"type": "ray"}},
        },
    )

    config = cli(
        RLConfig,
        args=[
            "@",
            "examples/reverse_text/rl.toml",
            "@",
            str(tmp_path / "ray_cluster.toml"),
        ],
    )

    assert config.trainer.model.dp_replicate == 1


def test_ray_cluster_16gpu_example_config_parses():
    """The shipped 16-GPU example config parses and auto-setup runs end to end."""
    config = cli(
        RLConfig,
        args=[
            "@",
            "examples/reverse_text/rl.toml",
            "@",
            "k8s/raycluster/rl-16gpu-example.toml",
        ],
    )

    assert config.deployment.type == "ray_cluster"
    assert config.deployment.num_train_gpus == 8
    assert config.deployment.num_infer_gpus == 8
    assert config.deployment.gpus_per_node == 8
    assert config.experimental.ray.enabled
    assert config.inference is not None
    assert config.inference.parallel.tp == 8
    assert config.inference.parallel.dp == 1
    assert config.orchestrator.num_train_workers == 8
    assert config.trainer.model.dp_replicate == 1


def test_ray_runtime_config_parses_runtime_env(tmp_path):
    config = cli(
        RLConfig,
        args=[
            "@",
            "examples/reverse_text/rl.toml",
            "--experimental.ray.enabled",
            "--experimental.ray.address",
            "ray-head.ray.svc.cluster.local:6379",
            "--experimental.ray.runtime-env.working-dir",
            tmp_path.as_posix(),
            "--experimental.ray.runtime-env.env-vars",
            '{"PYTHONPATH": "/repo/src:/repo/packages/prime-rl-configs/src"}',
            "--trainer.rollout-transport.type",
            "ray",
            "--trainer.rollout-transport.address",
            "ray-head.ray.svc.cluster.local:6379",
            "--orchestrator.rollout-transport.type",
            "ray",
            "--orchestrator.rollout-transport.address",
            "ray-head.ray.svc.cluster.local:6379",
        ],
    )
    assert config.experimental.ray.address == "ray-head.ray.svc.cluster.local:6379"
    assert config.experimental.ray.runtime_env.working_dir == tmp_path.as_posix()
    assert config.experimental.ray.runtime_env.env_vars["PYTHONPATH"] == "/repo/src:/repo/packages/prime-rl-configs/src"
    assert config.trainer.rollout_transport.address == "ray-head.ray.svc.cluster.local:6379"
    assert config.orchestrator.rollout_transport.address == "ray-head.ray.svc.cluster.local:6379"


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


def test_ray_native_rewrites_local_inference_client_urls():
    orchestrator = OrchestratorConfig()
    inference = InferenceConfig()
    inference.server.port = 8123
    orchestrator.client.admin_base_url = ["http://127.0.0.1:8000/v1"]

    rewritten = _orchestrator_with_ray_inference_endpoint(orchestrator, inference, "10.0.4.184")

    assert rewritten.client.base_url == ["http://10.0.4.184:8123/v1"]
    assert rewritten.client.admin_base_url == ["http://10.0.4.184:8123/v1"]
    assert orchestrator.client.base_url == ["http://localhost:8000/v1"]
    assert orchestrator.client.admin_base_url == ["http://127.0.0.1:8000/v1"]


def test_ray_native_preserves_external_inference_client_urls():
    orchestrator = OrchestratorConfig()
    inference = InferenceConfig()
    orchestrator.client.base_url = ["http://inference.ray.svc.cluster.local:8000/v1"]

    rewritten = _orchestrator_with_ray_inference_endpoint(orchestrator, inference, "10.0.4.184")

    assert rewritten.client.base_url == ["http://inference.ray.svc.cluster.local:8000/v1"]


def test_ray_native_rewrites_local_teacher_inference_client_urls():
    orchestrator = OrchestratorConfig()
    orchestrator.teacher_model = TeacherModelConfig()
    orchestrator.teacher_model.client.base_url = ["http://0.0.0.0:8001/v1"]
    teacher_inference = InferenceConfig()
    teacher_inference.server.port = 8124

    rewritten = _orchestrator_with_ray_teacher_endpoint(orchestrator, teacher_inference, "10.0.9.162")

    assert rewritten.teacher_model.client.base_url == ["http://10.0.9.162:8124/v1"]
    assert orchestrator.teacher_model.client.base_url == ["http://0.0.0.0:8001/v1"]


def test_ray_transport_config_reclaim_stale_actor_defaults_false():
    transport = TypeAdapter(TransportConfig).validate_python({"type": "ray"})
    assert transport.type == "ray"
    assert transport.reclaim_stale_actor is False


def test_ray_transport_store_per_rank_micro_batch_cap():
    """Regression: max_queued_items applies per data_rank, not globally."""
    store = _RayTransportStore(max_queued_items=2)

    store.put_micro_batch(data_rank=0, step=0, payload=b"a")
    store.put_micro_batch(data_rank=0, step=1, payload=b"b")
    store.put_micro_batch(data_rank=1, step=0, payload=b"c")
    store.put_micro_batch(data_rank=1, step=1, payload=b"d")

    with pytest.raises(RuntimeError, match="data_rank=0"):
        store.put_micro_batch(data_rank=0, step=2, payload=b"e")

    assert store.pop_micro_batch(data_rank=1, step=0) == b"c"
    store.put_micro_batch(data_rank=1, step=2, payload=b"f")


def test_ray_transport_store_per_sender_training_batch_cap():
    """Regression: training-batch cap is per sender_id, not globally."""
    store = _RayTransportStore(max_queued_items=2)

    store.put_training_batch("sender-a", b"a0")
    store.put_training_batch("sender-a", b"a1")
    store.put_training_batch("sender-b", b"b0")
    store.put_training_batch("sender-b", b"b1")

    with pytest.raises(RuntimeError, match="sender-a"):
        store.put_training_batch("sender-a", b"a2")


class _FakeRay:
    """Minimal stub used to drive _monitor_roles without a live Ray cluster."""

    def __init__(self, ready_order: list[object], failure_map: dict[object, BaseException] | None = None) -> None:
        self._ready_order = list(ready_order)
        self._failures = failure_map or {}
        self.cancelled: list[object] = []

    def wait(self, refs, num_returns: int = 1, timeout: float = 0.0):
        ref_set = set(refs)
        for ref in list(self._ready_order):
            if ref in ref_set:
                self._ready_order.remove(ref)
                return [ref], [r for r in refs if r is not ref]
        return [], list(refs)

    def get(self, ref):
        if ref in self._failures:
            raise self._failures[ref]
        return None

    def cancel(self, ref, force: bool = False) -> None:
        self.cancelled.append(ref)


def test_monitor_roles_surfaces_real_exception_from_failed_role():
    """When a role's Ray task raises, _monitor_roles must chain that exception so
    operators see the real cause (e.g. vLLM stack trace) and not just a launcher message."""
    inference_ref = object()
    orchestrator_ref = object()
    real_failure = RuntimeError("vLLM engine OOM")
    refs = {
        inference_ref: ("inference", Path("/tmp/inference.log")),
        orchestrator_ref: ("orchestrator", Path("/tmp/orchestrator.log")),
    }
    fake = _FakeRay(ready_order=[inference_ref], failure_map={inference_ref: real_failure})

    with pytest.raises(RuntimeError, match="Ray-native role inference failed") as excinfo:
        _monitor_roles(fake, refs, critical_names={"orchestrator"}, poll_interval_seconds=0.0)

    assert excinfo.value.__cause__ is real_failure
    assert orchestrator_ref in fake.cancelled


def test_monitor_roles_raises_when_long_running_role_returns_cleanly():
    """When a non-critical role returns cleanly before training finishes (e.g. inference
    exits 0 unexpectedly), _monitor_roles must raise and cancel remaining roles."""
    inference_ref = object()
    orchestrator_ref = object()
    refs = {
        inference_ref: ("inference", Path("/tmp/inference.log")),
        orchestrator_ref: ("orchestrator", Path("/tmp/orchestrator.log")),
    }
    fake = _FakeRay(ready_order=[inference_ref])

    with pytest.raises(RuntimeError, match="exited before training finished"):
        _monitor_roles(fake, refs, critical_names={"orchestrator"}, poll_interval_seconds=0.0)

    assert orchestrator_ref in fake.cancelled


def test_monitor_roles_cancels_non_critical_after_critical_completes():
    """When all critical roles finish, _monitor_roles must cancel remaining non-critical
    roles instead of waiting on them forever."""
    orchestrator_ref = object()
    inference_ref = object()
    refs = {
        orchestrator_ref: ("orchestrator", Path("/tmp/orchestrator.log")),
        inference_ref: ("inference", Path("/tmp/inference.log")),
    }
    fake = _FakeRay(ready_order=[orchestrator_ref])

    _monitor_roles(fake, refs, critical_names={"orchestrator"}, poll_interval_seconds=0.0)

    assert inference_ref in fake.cancelled
    assert orchestrator_ref not in fake.cancelled


def test_orchestrator_renderer_auto_rejects_unmapped_model():
    """use_renderer=True with renderer.name='auto' must reject models not in MODEL_RENDERER_MAP."""
    with pytest.raises(ValidationError, match="silently fall back to DefaultRenderer"):
        OrchestratorConfig.model_validate({"model": {"name": "not-a-real-org/not-a-real-model"}})


def test_orchestrator_renderer_auto_accepts_mapped_model():
    """The default Qwen model is in MODEL_RENDERER_MAP and should validate cleanly."""
    config = OrchestratorConfig.model_validate({"model": {"name": "Qwen/Qwen3-0.6B"}})
    assert config.use_renderer is True
    assert config.renderer.name == "auto"


def test_orchestrator_explicit_renderer_skips_unmapped_check():
    """Explicit renderer.name bypasses the auto-resolution check — user opted in."""
    config = OrchestratorConfig.model_validate(
        {
            "model": {"name": "not-a-real-org/not-a-real-model"},
            "renderer": {"name": "qwen3"},
        }
    )
    assert config.renderer.name == "qwen3"


def test_orchestrator_use_renderer_false_skips_unmapped_check():
    """use_renderer=False means the renderer client isn't used, so MODEL_RENDERER_MAP doesn't apply."""
    config = OrchestratorConfig.model_validate(
        {
            "model": {"name": "not-a-real-org/not-a-real-model"},
            "use_renderer": False,
        }
    )
    assert config.use_renderer is False


def test_orchestrator_explicit_default_renderer_with_unmapped_model():
    """renderer.name='default' is an explicit opt-in to DefaultRenderer and must pass."""
    config = OrchestratorConfig.model_validate(
        {
            "model": {"name": "not-a-real-org/not-a-real-model"},
            "renderer": {"name": "default", "tool_parser": "qwen3"},
        }
    )
    assert config.renderer.name == "default"
    assert config.renderer.tool_parser == "qwen3"
