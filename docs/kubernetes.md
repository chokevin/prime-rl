# Kubernetes

PRIME-RL on Kubernetes has two paths in this fork. New deployments should use the
**Ray-native RayCluster** path. The **legacy StatefulSet Helm chart** is kept for
backwards compatibility with the original SLURM-shaped topology.

## Which Kubernetes path should I use?

| | Ray-native RayCluster (recommended) | Legacy StatefulSet Helm chart |
|---|---|---|
| Shape | KubeRay `RayCluster` + Ray placement | One `StatefulSet` per Prime-RL role |
| Multi-node placement | Ray placement groups, logical `num_train_gpus` / `num_infer_gpus` / `num_teacher_gpus` | Stable DNS per pod, manual `torchrun` rendezvous |
| Lifecycle supervision | Ray driver fails the whole run on any role crash | Per-pod `kubectl` health; restart semantics per StatefulSet |
| Cross-node inference URL | Auto-rewritten by the Ray-native runtime | Resolved via the pod DNS env vars on each role |
| Weight broadcast | Shared `RWX` PVC + Prime-vLLM `/update_weights` | Shared `RWX` PVC + Prime-vLLM `/update_weights` |
| Teacher inference | Driven by `deployment.num_teacher_gpus` | Manual extra `inference` StatefulSet |
| Maturity | Validated end-to-end on multi-node A100; single-node validated on H200 | Long-standing chart used by the upstream `kubernetes.md` flow |
| When to choose | New k8s deployments, especially when you want Ray Train, Ray transport, and logical resource modeling | Clusters that already mirror SLURM topology, or workflows already invested in the chart |

If you do not have a strong reason to stay on the StatefulSet chart, prefer the
RayCluster path. The rest of this document covers both, starting with the
recommended path.

---

## Ray-native RayCluster (recommended)

### Quick start

The example manifests live under [`k8s/raycluster/`](../k8s/raycluster/):

```bash
# 1. Bring up the RayCluster (CPU head + GPU workers).
kubectl apply -f k8s/raycluster/raycluster.yaml

# 2. Submit the Prime-RL launcher from the head pool.
kubectl apply -f k8s/raycluster/rl-launch-job.yaml

# 3. Tail the launcher.
kubectl logs -f -n prime-rl job/prime-rl-launch
```

See [`k8s/raycluster/README.md`](../k8s/raycluster/README.md) for prerequisites
(KubeRay operator, GPU operator, shared `ReadWriteMany` PVC) and for the full
file layout, and see [`docs/ray.md`](ray.md) for the Ray-native architecture and
the `experimental.ray.*` config reference.

### Why this shape

- **CPU head, GPU workers.** Keeps GPUs reserved for Prime-RL roles instead of
  GCS/dashboard housekeeping, and lets `placement_strategy = "SPREAD"` put
  trainer, primary Prime-vLLM, and teacher Prime-vLLM on distinct nodes.
- **Logical role GPU counts.** `deployment.type = "ray_cluster"` with
  `num_train_gpus`, `num_infer_gpus`, and `num_teacher_gpus` describes what
  Prime-RL needs; Ray places the work on whatever RayCluster workers can satisfy
  the placement group. No `torchrun` rendezvous to hand-roll.
- **Single shared PVC.** Filesystem weight broadcast, checkpoints, dataset and
  HF caches, and the Prime-RL checkout all live under `/shared`. This matches
  the contract used by single-node and SLURM runs, so the rest of the trainer
  code is unchanged.
- **Submit from the head, not from outside.** The launcher Job runs on the same
  pool as the Ray head so `RAY_ADDRESS=auto` resolves via a local raylet. A
  plain Kubernetes pod that hits the GCS port directly can connect but cannot
  place workers.

### Validation status

- Multi-node A100 RayCluster: validated end to end (trainer step completed,
  checkpoint/broadcast/final weights produced, custom Prime-vLLM routes served,
  `Ray-native RL training finished!` logged).
- Same-node H200: Prime-RL runtime path validated with trainer + primary
  Prime-vLLM + teacher Prime-vLLM on three H200 GPUs.
- Multi-node H200 RayCluster: **not yet validated**. Blocked by node-pool
  networking on the available H200 pool (kubelet logs/exec 504s and pod DNS
  failures); not a Prime-RL bug. Fix the node networking first, then the same
  `k8s/raycluster/` manifests should apply.

### Cloud-specific notes

For Azure Kubernetes Service (AKS) — GPU node pool selectors, DRA vs
`nvidia.com/gpu` resource shape, ACR images, Azure Files / Azure Blob CSI for
`RWX` storage, optional Kueue, and H200/DeepGEMM caveats from the validation
runs — see [`k8s/raycluster/aks.md`](../k8s/raycluster/aks.md).

---

## Legacy StatefulSet Helm chart

The original Kubernetes path uses a process-role Helm chart: `StatefulSet`s
provide stable names and storage for the same trainer, inference, and
orchestrator roles that the `rl` entrypoint launches locally.

Use this path when your cluster already mirrors SLURM topology, or when you
have existing workflows around the chart that you do not want to rewrite. For
new deployments prefer the Ray-native path above.

## Prerequisites

- Kubernetes cluster with GPU nodes
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/getting-started.html) installed
- [Helm 3.x](https://helm.sh/docs/intro/install/) installed
- Storage class that supports `ReadWriteMany` (e.g., NFS, CephFS, or cloud provider storage)

### Verify Prerequisites

```bash
# Check Helm installation
helm version

# Check GPU operator
kubectl get pods -n gpu-operator

# Check available storage classes
kubectl get storageclass
```

## Quick Start

### 1. Deploy

```bash
# Deploy with a release name
helm install my-exp ./k8s/prime-rl -f ./k8s/prime-rl/examples/reverse-text.yaml

# Or with defaults (no example-specific config)
helm install my-exp ./k8s/prime-rl --set trainer.replicas=3 --set inference.replicas=2
```

### 2. Verify deployment

```bash
# Check pod status
kubectl get pods -l app.kubernetes.io/instance=my-exp

# Should show 3 pods:
# my-exp-orchestrator-0
# my-exp-inference-0
# my-exp-trainer-0
```

### 3. Run training

```bash
# Exec into trainer
kubectl exec -it my-exp-trainer-0 -- bash

# Inside the pod, run training
cd /data
uv run trainer @ /app/examples/reverse_text/configs/train.toml
```

### 4. Monitor progress

```bash
# Get logs
kubectl logs my-exp-trainer-0

# Follow logs in real-time
kubectl logs -f my-exp-trainer-0
```

## Available Examples

The chart includes pre-configured values for each example:

### reverse-text (Small - 1 GPU)

```bash
helm install my-exp ./k8s/prime-rl -f ./k8s/prime-rl/examples/reverse-text.yaml
```

- Model: Qwen3-0.6B
- GPUs: 1 per component
- Runs on consumer GPUs (RTX 3090/4090)
- **Note:** You can use any release name - the chart automatically configures service URLs

## Configuration

### Storage Configuration

By default, the chart creates a 1TB PVC with NFS storage. To customize:

```yaml
# custom-values.yaml
storage:
  storageClassName: my-storage-class
  size: 500Gi
```

Deploy with custom storage:

```bash
helm install my-release ./k8s/prime-rl -f custom-values.yaml
```

### GPU Configuration

Adjust GPU count per component:

```yaml
# custom-gpu.yaml
inference:
  gpu:
    count: 4  # Use 4 GPUs for inference

trainer:
  gpu:
    count: 2  # Use 2 GPUs for training
```

### Resource Limits

Customize memory and CPU:

```yaml
# custom-resources.yaml
trainer:
  resources:
    requests:
      memory: "64Gi"
      cpu: "16"
    limits:
      memory: "128Gi"
      cpu: "32"
```

### Secrets (Optional)

For W&B and HuggingFace authentication:

```bash
# Create secret
kubectl create secret generic prime-rl-secrets \
  --from-literal=wandb-api-key=YOUR_WANDB_KEY \
  --from-literal=hf-token=YOUR_HF_TOKEN

# Enable in values
helm install my-release ./k8s/prime-rl \
  --set config.secrets.enabled=true \
  --set config.secrets.name=prime-rl-secrets
```

## Common Operations

### Deploy a new experiment

```bash
# With example config
helm install my-exp ./k8s/prime-rl -f ./k8s/prime-rl/examples/reverse-text.yaml

# With custom settings
helm install my-exp ./k8s/prime-rl --set trainer.replicas=10 --set inference.replicas=5
```

### Exec into pods

```bash
# Exec into trainer-0
kubectl exec -it my-exp-trainer-0 -- bash

# Exec into specific trainer pod
kubectl exec -it my-exp-trainer-3 -- bash

# Exec into inference
kubectl exec -it my-exp-inference-0 -- bash
```

### View logs

```bash
# Get logs from trainer-0
kubectl logs my-exp-trainer-0

# Follow logs in real-time
kubectl logs -f my-exp-trainer-2

# Get logs from all trainers
kubectl logs -l app.kubernetes.io/instance=my-exp,role=trainer
```

### List all pods

```bash
# List pods for specific experiment
kubectl get pods -l app.kubernetes.io/instance=my-exp

# List all prime-rl pods
kubectl get pods -l app=prime-rl
```

## Architecture

### Components

The chart deploys three main components (all using StatefulSets):

1. **Orchestrator** (StatefulSet) - Coordinates training workflow
   - Always 1 replica: `prime-rl-orchestrator-0`
   - No GPU required
   - Communicates with trainer and inference

2. **Inference** (StatefulSet) - Runs vLLM inference server
   - Scalable replicas with stable pod names: `prime-rl-inference-0`, `prime-rl-inference-1`, ...
   - Each pod gets predictable DNS: `prime-rl-inference-0.prime-rl-inference-headless.default.svc.cluster.local`
   - Requires GPU(s)
   - Serves model predictions

3. **Trainer** (StatefulSet) - Runs SFT or RL training
   - Scalable replicas with stable pod names: `prime-rl-trainer-0`, `prime-rl-trainer-1`, ...
   - Each pod gets predictable DNS: `prime-rl-trainer-0.prime-rl-trainer-headless.default.svc.cluster.local`
   - Requires GPU(s)
   - Updates model weights on shared storage

**Why StatefulSets for all components?**

- **Consistent naming**: All pods have predictable names (`orchestrator-0`, `trainer-0`, `trainer-1`, ...)
- **Stable networking**: Each pod gets its own DNS hostname via headless service
- **Required for distributed training**: PyTorch/vLLM need to discover peers by stable hostname
- **Clean naming**: No random pod suffixes, easier to identify and debug

### Shared Storage

All components mount the same PVC at `/data` for:

- Model checkpoint sharing
- Training data
- Experiment outputs

This is **required** for coordinating weight updates between trainer and inference.

## Environment Variables

Each pod has these K8s environment variables set:

- `$POD_NAME` - Full pod name (e.g., `my-exp-trainer-3`)
- `$POD_IP` - Pod IP address
- `$STATEFUL_REPLICAS` - Total number of replicas for that component
- `$HEADLESS_SERVICE` - DNS name for peer discovery (e.g., `my-exp-trainer-headless.default.svc.cluster.local`)
- `$INFERENCE_URL` - Full URL to the first inference pod (available in orchestrator and trainer pods)

For distributed training, extract the rank from the pod name:

```bash
# Extract ordinal from pod name
RANK=$(echo $POD_NAME | grep -o '[0-9]*$')  # e.g., "my-exp-trainer-3" -> "3"

# Use in torchrun
torchrun \
  --nnodes=$STATEFUL_REPLICAS \
  --node-rank=$RANK \
  --nproc-per-node=8 \
  --rdzv-endpoint=my-exp-trainer-0.$HEADLESS_SERVICE:29501 \
  src/prime_rl/trainer/sft/train.py @ configs/train.toml
```

## Troubleshooting

### Can't access shared storage

Verify PVC is bound:

```bash
kubectl get pvc prime-rl-shared-data
# STATUS should be "Bound"
```

Check mount inside pod:

```bash
kubectl exec -it prime-rl-trainer-xxx -- df -h /data
```

### Pod stuck in Pending

Check if GPU resources are available:

```bash
kubectl describe pod my-exp-trainer-0
```

Look for events like `Insufficient nvidia.com/gpu`.

### Inference server not responding

Check if the inference pod is ready:

```bash
kubectl get pods -l role=inference
kubectl logs my-exp-inference-0
```

## Uninstalling

```bash
# Remove the Helm release
helm uninstall my-exp

# Delete PVC (data will be lost!)
kubectl delete pvc prime-rl-shared-data
```
