import json

import pytest
from pydantic import ValidationError
from pydantic_config import ConfigFileError

from prime_rl.configs.rl import RLConfig
from prime_rl.launchers.rayjob import (
    INFERENCE_TOML,
    ORCHESTRATOR_TOML,
    RAY_DRIVER_NAME,
    ROLE_SCRIPT_NAME,
    TRAINER_TOML,
    render_rayjob_yaml,
)
from prime_rl.utils.config import cli

RAYJOB_CONFIG = """
output_dir = "outputs/test-rayjob"
dry_run = true
max_steps = 2
seq_len = 512

[deployment]
type = "multi_node"
num_train_nodes = 1
num_infer_nodes = 1
gpus_per_node = 8

[rayjob]
job_name = "test-prime-rl"
namespace = "ray"
image = "example.com/prime-rl-ray:test"
shared_pvc_name = "shared-data"
inference_startup_grace_seconds = 0

[weight_broadcast]
type = "nccl"

[model]
name = "Qwen/Qwen3-0.6B"

[trainer]

[orchestrator]
batch_size = 2
rollouts_per_example = 1

[[orchestrator.train.env]]
id = "reverse-text"

[inference.parallel]
tp = 2
dp = 4
"""


def load_config(tmp_path, text: str = RAYJOB_CONFIG) -> RLConfig:
    config_path = tmp_path / "rl.toml"
    config_path.write_text(text)
    return cli(RLConfig, args=["@", config_path.as_posix()])


def parse_manifest_stream(rendered: str) -> list[dict]:
    return [json.loads(doc) for doc in rendered.split("\n---\n")]


def test_multinode_allows_rayjob_without_slurm(tmp_path):
    config = load_config(tmp_path)

    assert config.slurm is None
    assert config.rayjob is not None
    assert config.rayjob.job_name == "test-prime-rl"


def test_multinode_requires_one_launcher(tmp_path):
    without_launcher = RAYJOB_CONFIG.replace(
        """
[rayjob]
job_name = "test-prime-rl"
namespace = "ray"
image = "example.com/prime-rl-ray:test"
shared_pvc_name = "shared-data"
inference_startup_grace_seconds = 0
""",
        "",
    )

    with pytest.raises((ConfigFileError, ValidationError), match="SLURM or RayJob"):
        load_config(tmp_path, without_launcher)


def test_multinode_rejects_slurm_and_rayjob_together(tmp_path):
    both_launchers = RAYJOB_CONFIG.replace(
        "[rayjob]",
        """
[slurm]
job_name = "test-prime-rl"

[rayjob]
""",
    )

    with pytest.raises((ConfigFileError, ValidationError), match="only one multi-node launcher"):
        load_config(tmp_path, both_launchers)


def test_rayjob_manifest_renders_runtime_contract(tmp_path):
    config = load_config(tmp_path)
    rendered = render_rayjob_yaml(
        config,
        {
            TRAINER_TOML: "trainer = true\n",
            ORCHESTRATOR_TOML: "orchestrator = true\n",
            INFERENCE_TOML: "inference = true\n",
        },
    )

    config_map, rayjob = parse_manifest_stream(rendered)
    assert config_map["kind"] == "ConfigMap"
    assert config_map["metadata"]["name"] == "test-prime-rl-runtime"
    assert config_map["data"][TRAINER_TOML] == "trainer = true\n"
    assert ROLE_SCRIPT_NAME in config_map["data"]
    assert RAY_DRIVER_NAME in config_map["data"]

    assert rayjob["kind"] == "RayJob"
    assert rayjob["metadata"]["annotations"]["prime-rl.ai/runtime-status"] == "rayjob-role-entrypoints-rendered"
    assert rayjob["spec"]["entrypoint"] == "python /prime-rl/runtime/prime-rl-ray-driver.py"

    cluster = rayjob["spec"]["rayClusterSpec"]
    assert cluster["headGroupSpec"]["template"]["spec"]["containers"][0]["image"] == "example.com/prime-rl-ray:test"
    worker_groups = {group["groupName"]: group for group in cluster["workerGroupSpecs"]}
    assert set(worker_groups) == {"prime-trainer", "prime-inference"}
    assert worker_groups["prime-trainer"]["rayStartParams"]["resources"] == '\'{"prime_trainer":1}\''
    assert worker_groups["prime-inference"]["rayStartParams"]["resources"] == '\'{"prime_inference":1}\''

    trainer_container = worker_groups["prime-trainer"]["template"]["spec"]["containers"][0]
    assert trainer_container["resources"]["limits"]["nvidia.com/gpu"] == 8
    env = {item["name"]: item["value"] for item in trainer_container["env"]}
    assert env["PRIME_RL_NUM_TRAIN_NODES"] == "1"
    assert env["PRIME_RL_NUM_INFER_NODES"] == "1"
    assert env["PRIME_RL_INFERENCE_TP"] == "2"
    assert env["PRIME_RL_WEIGHT_BROADCAST_TYPE"] == "nccl"
