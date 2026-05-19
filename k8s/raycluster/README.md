# Prime-RL on RayCluster (Kubernetes)

This directory is a **Kubernetes-native** starting point for running Prime-RL on a
RayCluster managed by [KubeRay](https://github.com/ray-project/kuberay). It is the
recommended path for new Kubernetes deployments; the legacy
StatefulSet/Helm chart under `../prime-rl/` is the SLURM-shaped path and is kept for
backwards compatibility.

The manifests use a simple, portable shape:

- CPU Ray head (no GPUs) for placement, GCS, the dashboard, and the job submission API.
- GPU worker group sized via the `ray.io/v1` `replicas` field; each worker exposes
  `num-gpus = 1` to Ray and reserves one `nvidia.com/gpu`.
- A single shared `ReadWriteMany` PVC mounted at `/shared` on the head and every
  worker for the Prime-RL checkout, model and dataset caches, checkpoints, and
  the filesystem weight-broadcast directory.
- A separate `batch/v1` Job that runs from the Ray head and submits the `rl`
  launcher with `RAY_ADDRESS=auto`.

Cluster-specific labels, priority classes, image-pull secrets, and resource
claim systems are intentionally not included. Add them as overlays for your
environment.

## Prerequisites

- A Kubernetes cluster you can `kubectl apply` to.
- [KubeRay operator](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started.html)
  installed in the cluster.
- GPU-capable worker nodes with the
  [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/getting-started.html)
  (or equivalent) so containers can request `nvidia.com/gpu`.
- A `ReadWriteMany` storage class available, with a `PersistentVolumeClaim` named
  `prime-rl-shared` already bound. Both the Ray head and the GPU workers mount it
  at `/shared`.
- A container image that has Prime-RL's runtime dependencies installed and Ray
  on the Python `PATH`. The example assumes `rayproject/ray:2.40.0-py312` for
  first-touch testing; in practice you will want a Prime-RL image built from
  the repo's `Dockerfile.cuda` plus `uv sync --all-extras`.

## Files

- `raycluster.yaml` — KubeRay `RayCluster` with a CPU head group and a GPU worker
  group. Edit `replicas`, `nodeSelector`, `image`, and `claimName` for your
  cluster.
- `raycluster-16gpu.yaml` — same shape, sized for a 16-GPU run: two GPU worker
  pods, eight GPUs each, with `num-gpus = 8` per worker so an 8-way TP or FSDP
  role fits on one pod.
- `rl-launch-job.yaml` — `batch/v1` `Job` that runs on the CPU head pool, clones
  Prime-RL into a per-attempt directory under `/shared/checkouts/`, and submits
  the `rl` launcher to the RayCluster via `ray job submit`.
  Set `CONFIG_PATH` to choose a config file relative to the checkout, such as
  `k8s/raycluster/rl-16gpu-example.toml`.
- `rl-example.toml` — minimal `ray_cluster`-shaped Prime-RL config that mirrors
  a small run: one trainer GPU, one inference GPU, `experimental.ray.enabled`,
  Ray Train trainer backend, Prime-vLLM inference backend, and Ray rollout
  transport. Adapt model, data, and GPU counts for your experiment.
- `rl-16gpu-example.toml` — 16-GPU split: `num_train_gpus = 8`,
  `num_infer_gpus = 8`, `inference.parallel.tp = 8`. Pairs with
  `raycluster-16gpu.yaml`.

## Launch

```bash
# 1. Bring up the RayCluster.
kubectl apply -f k8s/raycluster/raycluster.yaml

# 2. Wait for head and workers to be Ready.
kubectl get rayclusters -n prime-rl -w

# 3. Submit the Prime-RL launcher from the head node.
kubectl apply -f k8s/raycluster/rl-launch-job.yaml

# 4. Tail logs of the submitted Ray job (the Job pod streams them).
kubectl logs -f -n prime-rl job/prime-rl-launch
```

The launch Job writes resolved configs and per-role logs under
`/shared/outputs/<run-id>/<attempt-id>/`:

```
/shared/outputs/<run-id>/<attempt-id>/
  configs/{inference,orchestrator,trainer}.toml
  logs/{inference.log,orchestrator.log,trainer/rank_0.log}
  checkpoints/step_*/...
  weights/step_*/STABLE
  run_default/broadcasts/step_*/STABLE
  final_summary.json
```

A successful run logs `Ray-native RL training finished!` from the launcher.

## Why this shape

- **CPU head, GPU workers.** The head only needs to host GCS, dashboard, and the
  job submission API. Reserving GPUs on the head wastes hardware and forces
  every Prime-RL role onto the head pool. With a CPU head, `num_train_gpus`,
  `num_infer_gpus`, and `num_teacher_gpus` are spread across the worker group via
  Ray placement.
- **`num-gpus = 1` per worker.** Prime-RL's `ray_cluster` deployment shape models
  *logical* per-role GPU counts. One GPU per worker keeps Ray's view of the
  cluster and Kubernetes' view aligned and lets `placement_strategy = "SPREAD"`
  put trainer, primary inference, and teacher inference on distinct nodes.
- **Shared PVC at `/shared`.** Prime-RL's HF-compatible filesystem weight
  broadcast and checkpoint flow expect the trainer's write directory to be
  visible to every inference task. A single `ReadWriteMany` PVC keeps the
  contract identical to single-node and SLURM runs.
- **Job submission via `RAY_ADDRESS=auto` from the head.** A non-Ray driver pod
  outside the RayCluster that calls `ray.init(address="<head-svc>:6379")` can
  connect to GCS but has no local raylet, so worker placement fails. Running the
  launcher from a head-affinitized Job avoids that footgun.
- **Ray Train GPU visibility.** The launcher sets
  `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` for the Ray job runtime so
  Ray Train's distributed bootstrap can map local ranks to physical GPUs before
  Prime-RL enters its trainer loop.

## Scaling to 16 GPUs

To scale up, make sure each single Prime-vLLM role fits on one Ray worker pod.
Trainer workers may span worker pods when using the Ray Train backend.

For a 16-GPU run split as 8 train + 8 infer:

```bash
kubectl apply -f k8s/raycluster/raycluster-16gpu.yaml   # 2 workers x 8 GPU
# set CONFIG_PATH in rl-launch-job.yaml to k8s/raycluster/rl-16gpu-example.toml
kubectl apply -f k8s/raycluster/rl-launch-job.yaml
```

`rl-16gpu-example.toml` sets `num_train_gpus = 8`, `num_infer_gpus = 8`, and
`inference.parallel.tp = 8`. The `ray_cluster` validators derive the matching
orchestrator and inference worker counts.

Other splits work as long as each Prime-vLLM task has enough GPUs on one worker
pod and the RayCluster has enough total GPUs for all roles.

## See also

- `docs/ray.md` — Ray-native architecture and config reference.
- `docs/kubernetes.md` — Kubernetes paths and decision matrix.
- `../prime-rl/` — legacy SLURM-shaped Helm chart (StatefulSets per role).
