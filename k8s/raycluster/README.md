# Prime-RL on RayCluster (Kubernetes)

This directory is a **Kubernetes-native** starting point for running Prime-RL on a
RayCluster managed by [KubeRay](https://github.com/ray-project/kuberay). It is the
recommended path for new Kubernetes deployments of this fork; the legacy
StatefulSet/Helm chart under `../prime-rl/` is the SLURM-shaped path and is kept for
backwards compatibility.

The shape here matches the multi-node A100 RayCluster run that was used to validate
the Ray-native runtime end to end:

- CPU Ray head (no GPUs) for placement, GCS, the dashboard, and the job submission API.
- GPU worker group sized via the `ray.io/v1` `replicas` field; each worker exposes
  `num-gpus = 1` to Ray and reserves one `nvidia.com/gpu`.
- A single shared `ReadWriteMany` PVC mounted at `/shared` on the head and every
  worker for the Prime-RL checkout, model and dataset caches, checkpoints, and
  the filesystem weight-broadcast directory.
- A separate `batch/v1` Job that runs from the Ray head and submits the `rl`
  launcher with `RAY_ADDRESS=auto`.

Vendor-specific labels (Rune, DRA, Kueue, AKS) are intentionally not present here.
Add cluster-specific node selectors, priority classes, image-pull secrets, or DRA
resource claims as overlays for your environment.

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
  the fork into `/shared/checkouts/prime-rl` if absent, and submits the `rl`
  launcher to the RayCluster via `ray job submit --address auto`.
- `rl-example.toml` — minimal `ray_cluster`-shaped Prime-RL config that mirrors
  the validated reverse-text run: one trainer GPU, one inference GPU,
  `experimental.ray.enabled`, `ray_train` trainer backend, `prime_vllm` inference
  backend, Ray rollout transport. Adapt model/dataset/`gpus_per_node` for your
  experiment.
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
`/shared/outputs/<run-id>/` exactly like the validated A100 run:

```
/shared/outputs/<run-id>/
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

## Scaling to 16 GPUs

The two-GPU validated run uses one Ray worker per role. To scale up, the
RayCluster needs enough GPUs to host each role's bundle on a single Ray worker
node — Prime-vLLM inference is one Ray task and must fit on one pod.

For a 16-GPU run split as 8 train + 8 infer:

```bash
kubectl apply -f k8s/raycluster/raycluster-16gpu.yaml   # 2 workers x 8 GPU
# edit rl-launch-job.yaml to submit rl-16gpu-example.toml
kubectl apply -f k8s/raycluster/rl-launch-job.yaml
```

`rl-16gpu-example.toml` sets `num_train_gpus = 8`, `num_infer_gpus = 8`, and
`inference.parallel.tp = 8`. The `ray_cluster` auto-setup validators then derive
`orchestrator.num_train_workers`, `inference.parallel.dp`, and
`inference.api_server_count`. If you opt into NCCL weight broadcast, the
trainer's `inference_world_size` is also set automatically.

Trainer FSDP shape on the 8 train GPUs defaults to pure FSDP (dp_replicate = 1,
dp_shard = 8). For runs where `num_train_gpus > gpus_per_node`, the validators
auto-set `trainer.model.dp_replicate = num_train_gpus // gpus_per_node` so each
FSDP island stays on intra-node NVLINK. Override by setting
`trainer.model.dp_replicate` explicitly in your config.

Other splits work as long as no single role exceeds `gpus_per_node`. Examples:

- `num_train_gpus = 14`, `num_infer_gpus = 2` — distillation-heavy with small
  inference; needs at least one worker that holds the trainer plus an inference
  worker, so two 8-GPU pods are the minimum.
- `num_train_gpus = 8`, `num_infer_gpus = 4`, `num_teacher_gpus = 4` — three
  roles, three 8-GPU worker pods.

Validation status: the two-GPU A100 and three-GPU same-node H200 runs are
confirmed end to end. `num_train_gpus > 1` via Ray Train and `num_infer_gpus > 1`
via Prime-vLLM TP exercise code paths that have not yet been run at this scale
on RayCluster. The config and manifests are in place; the next milestone is an
end-to-end 16-GPU run logged in the PR description.

## H200 / FP8 cluster notes

Validated on A100 multi-node. H200 multi-node RayCluster was attempted but blocked
by **node-pool networking** (kubelet logs/exec 504s and pod DNS failures), not by
Prime-RL. When running on H200 you may also need:

- `VLLM_DEEP_GEMM_WARMUP=skip` in the launch Job's `env:` if vLLM enters Hopper
  FP8 DeepGEMM warmup paths.
- A pre-staged Hugging Face cache on `/shared` if the H200 node pool cannot
  reach `huggingface.co` directly.

See `docs/ray.md` and the PR description for the full H200 validation summary.

## See also

- [`aks.md`](./aks.md) — Azure Kubernetes Service (AKS) overlay: GPU pool
  selectors, DRA or `nvidia.com/gpu` resource shape, ACR images, Azure
  Files/Blob CSI for `RWX` storage, and the H200 caveats from the validation
  runs.
- `docs/ray.md` — Ray-native architecture and config reference.
- `docs/kubernetes.md` — Kubernetes paths and decision matrix.
- `../prime-rl/` — legacy SLURM-shaped Helm chart (StatefulSets per role).
