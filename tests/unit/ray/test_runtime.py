from prime_rl.configs.rl import RLConfig
from prime_rl.ray.runtime import inference_bundles


def test_h200_inference_bundles_for_mixed_compute():
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {},
            "inference": {"parallel": {"tp": 8, "dp": 2}},
            "deployment": {
                "type": "multi_node",
                "num_train_nodes": 2,
                "num_infer_nodes": 2,
                "gpus_per_node": 8,
            },
            "execution": {
                "type": "ray",
                "trainer": {"accelerator_type": "A100"},
                "inference": {"accelerator_type": "H200"},
            },
            "weight_broadcast": {"type": "filesystem"},
        }
    )

    assert inference_bundles(config) == [
        {"CPU": 1.0, "GPU": 8, "accelerator_type:H200": 0.001},
        {"CPU": 1.0, "GPU": 8, "accelerator_type:H200": 0.001},
    ]
