import torch

from prime_rl.trainer.rl.broadcast.nccl import filter_state_dict_by_layers


def test_filter_state_dict_by_layers_groups_in_single_pass_order():
    state_dict = {
        "model.embed_tokens.weight": torch.empty(1),
        "model.layers.1.mlp.down_proj.weight": torch.empty(2),
        "model.layers.0.self_attn.q_proj.weight": torch.empty(3),
        "model.layers.not_a_number.weight": torch.empty(4),
        "model.norm.weight": torch.empty(5),
    }

    groups = filter_state_dict_by_layers(state_dict, num_layers=2, layer_prefix="model.layers.")

    assert [(layer_id, list(group)) for layer_id, group in groups] == [
        (
            -1,
            [
                "model.embed_tokens.weight",
                "model.layers.not_a_number.weight",
                "model.norm.weight",
            ],
        ),
        (0, ["model.layers.0.self_attn.q_proj.weight"]),
        (1, ["model.layers.1.mlp.down_proj.weight"]),
    ]
