# Kubernetes deployment

This fork ships two Kubernetes paths. New deployments should use the Ray-native
RayCluster path under [`raycluster/`](./raycluster/). The legacy StatefulSet
Helm chart under [`prime-rl/`](./prime-rl/) is the SLURM-shaped topology kept
for backwards compatibility.

See [`docs/kubernetes.md`](../docs/kubernetes.md) for the decision matrix and
[`docs/ray.md`](../docs/ray.md) for the Ray-native architecture reference.

## Ray-native RayCluster (recommended)

KubeRay `RayCluster` with a CPU head, a GPU worker group, and a launch Job that
submits the Prime-RL `rl` entrypoint from the head pool. Mirrors the validated
multi-node A100 run.

```bash
# 1. Bring up the cluster (edit namespace, image, nodeSelectors, PVC first).
kubectl apply -f raycluster/raycluster.yaml

# 2. Submit the Prime-RL launcher from the head pool.
kubectl apply -f raycluster/rl-launch-job.yaml

# 3. Follow the launcher.
kubectl logs -f -n prime-rl job/prime-rl-launch
```

See [`raycluster/README.md`](./raycluster/README.md) for prerequisites
(KubeRay operator, NVIDIA GPU Operator, shared `ReadWriteMany` PVC) and for the
H200 / FP8 cluster notes.

## Legacy StatefulSet Helm chart

Process-role Helm chart with one `StatefulSet` per Prime-RL role. Use this only
when your cluster already mirrors SLURM topology or you have existing workflows
around the chart.

```bash
# Deploy with the reverse-text example
helm install my-exp ./prime-rl -f ./prime-rl/examples/reverse-text.yaml

# Verify deployment
kubectl get pods -l app.kubernetes.io/instance=my-exp

# Exec into trainer and run training
kubectl exec -it my-exp-trainer-0 -- bash
cd /data && uv run trainer @ /app/examples/reverse_text/configs/train.toml
```

## Prerequisites

- Kubernetes cluster with GPU nodes
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/getting-started.html) installed
- [Helm 3.x](https://helm.sh/docs/intro/install/) installed
- Storage class that supports `ReadWriteMany` (e.g., NFS, CephFS)

## Chart Structure

```
prime-rl/
├── Chart.yaml
├── values.yaml           # Default configuration
├── examples/
│   └── reverse-text.yaml # Example values for reverse-text
└── templates/
    ├── deployment.yaml   # StatefulSets for orchestrator, inference, trainer
    ├── service.yaml      # Headless services for pod discovery
    └── pvc.yaml          # Shared storage
```

## Configuration

See [values.yaml](./prime-rl/values.yaml) for all available options. Common overrides:

```bash
# Custom GPU allocation
helm install my-exp ./prime-rl \
  --set inference.gpu.count=4 \
  --set trainer.gpu.count=2

# With secrets for W&B/HF
kubectl create secret generic prime-rl-secrets \
  --from-literal=wandb-api-key=YOUR_KEY \
  --from-literal=hf-token=YOUR_TOKEN

helm install my-exp ./prime-rl \
  --set config.secrets.enabled=true \
  --set config.secrets.name=prime-rl-secrets
```

## Uninstalling

```bash
helm uninstall my-exp
kubectl delete pvc prime-rl-shared-data  # Warning: deletes data!
```

## Learn More

- [Full Kubernetes documentation](https://docs.primeintellect.ai/prime-rl/kubernetes) - Architecture, configuration, distributed training
- [Deployment guide](https://docs.primeintellect.ai/prime-rl/deployment) - Non-Kubernetes deployments
- [Troubleshooting](https://docs.primeintellect.ai/prime-rl/troubleshooting) - Common issues
