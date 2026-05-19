from prime_rl.ray.train import _get_ray_train_local_topology


class FakeRayTrainContext:
    def __init__(self, local_rank: int, local_world_size: int):
        self.local_rank = local_rank
        self.local_world_size = local_world_size

    def get_local_rank(self) -> int:
        return self.local_rank

    def get_local_world_size(self) -> int:
        return self.local_world_size


def test_ray_train_local_topology_uses_ray_context_when_all_gpus_visible(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

    assert _get_ray_train_local_topology(FakeRayTrainContext(3, 8)) == (3, 8)


def test_ray_train_local_topology_handles_single_visible_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")

    assert _get_ray_train_local_topology(FakeRayTrainContext(3, 8)) == (0, 1)
