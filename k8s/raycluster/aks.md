# Prime-RL on AKS

This guide overlays Azure Kubernetes Service (AKS) specifics onto the
vendor-neutral RayCluster example in this directory. The base manifests
(`raycluster.yaml`, `rl-launch-job.yaml`, `rl-example.toml`) stay portable;
this page documents the AKS-specific knobs that the validated multi-node
A100 and same-node H200 Prime-RL runs actually used.

The guide is descriptive of what worked, not prescriptive about the only way
to run on AKS. Substitute your own ACR, storage, and node-pool conventions
where appropriate.

## AKS prerequisites

- An AKS cluster with at least one **GPU node pool** (A100 or H200) and one
  **CPU node pool** for the Ray head and the launch Job.
- A GPU runtime on the GPU pool. Two paths are common:
  - [AKS GPU image and operator](https://learn.microsoft.com/en-us/azure/aks/gpu-cluster)
    (`nvidia.com/gpu` resource limits, the same pattern as the base example).
  - [Dynamic Resource Allocation (DRA)](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/)
    with a `DeviceClass` and `ResourceClaimTemplate` named for your GPU class.
    This is what the validated A100 and H200 runs used.
- The [KubeRay operator](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started.html)
  installed in the cluster.
- An [Azure Container Registry](https://learn.microsoft.com/en-us/azure/container-registry/)
  the cluster's kubelet can pull from, or an image-pull secret in the
  Prime-RL namespace.
- An `RWX`-capable Azure storage class. Two patterns:
  - [Azure Files CSI](https://learn.microsoft.com/en-us/azure/aks/azure-csi-files-storage-provision)
    with `azurefile-csi-premium` (`accessModes: ["ReadWriteMany"]`).
  - [Azure Blob CSI / BlobFuse](https://learn.microsoft.com/en-us/azure/aks/azure-blob-csi)
    with `azureblob-fuse-premium` (what the validated A100 run used: PVC
    `blob-training` mounted at `/shared`).
- A `PersistentVolumeClaim` from that storage class, bound and mounted at
  `/shared` on the head and every worker. The Prime-RL checkout, HF cache,
  dataset cache, checkpoints, weight broadcasts, and Ray Train storage all
  live there.

Optional, depending on your cluster:

- [Kueue](https://kueue.sigs.k8s.io/) for batch admission control with priority
  classes. Not required to run Prime-RL; the validated runs used Kueue via a
  workspace harness, but the manifests in this directory work without it.
- [Azure Workload Identity / Managed Identity](https://learn.microsoft.com/en-us/azure/aks/workload-identity-overview)
  for ACR pulls and Azure storage access without static secrets.

## AKS overlay on `raycluster.yaml`

Start from `raycluster.yaml` in this directory and apply these AKS-specific
changes. The validated topology was: CPU head on the CPU agent pool, two
A100 workers (multi-node) or three H200 workers (same-node) on the GPU agent
pool, anti-affinity to put workers on distinct nodes, Azure Blob CSI PVC at
`/shared`.

### Head pool selector

The Ray head does not need GPUs. Pin it to your CPU node pool:

```yaml
spec:
  headGroupSpec:
    template:
      spec:
        nodeSelector:
          agentpool: <your-cpu-agentpool>   # e.g., "cpu"
```

### GPU worker pool selector and tolerations

Pin GPU workers to your GPU node pool and tolerate the standard
`nvidia.com/gpu` taint:

```yaml
spec:
  workerGroupSpecs:
    - template:
        spec:
          nodeSelector:
            agentpool: <your-gpu-agentpool>
            # Optional GPU class label if your fleet has heterogeneous GPUs:
            # accelerator: a100-nvlink-80gb
          tolerations:
            - key: nvidia.com/gpu
              operator: Exists
              effect: NoSchedule
```

### GPU resource shape: DRA or `nvidia.com/gpu`

Pick **one** of these, matching how your AKS GPU pool exposes GPUs.

DRA (what the validated A100/H200 runs used). The worker container declares a
`claims` entry instead of `nvidia.com/gpu: 1`, and the Pod spec declares a
matching `resourceClaims` entry referencing your `ResourceClaimTemplate`:

```yaml
spec:
  workerGroupSpecs:
    - template:
        spec:
          containers:
            - name: ray-worker
              resources:
                claims:
                  - name: gpu
                requests: { cpu: "8",  memory: "64Gi"  }
                limits:   { cpu: "16", memory: "128Gi" }
          resourceClaims:
            - name: gpu
              resourceClaimTemplateName: <your-full-gpu-claim-template>
```

Classic `nvidia.com/gpu` (the shape used in `raycluster.yaml` by default).
No `resourceClaims` field needed:

```yaml
spec:
  workerGroupSpecs:
    - template:
        spec:
          containers:
            - name: ray-worker
              resources:
                requests: { cpu: "8",  memory: "64Gi"  }
                limits:
                  cpu: "16"
                  memory: "128Gi"
                  nvidia.com/gpu: "1"
```

Whichever you pick, the worker's `rayStartParams.num-gpus` must equal the GPU
count the container actually reserves (the example uses `1` per worker so Ray
sees one GPU per node and `placement_strategy = "SPREAD"` can place trainer,
primary Prime-vLLM, and teacher Prime-vLLM on distinct nodes).

### ACR image

Replace the placeholder `rayproject/ray:2.40.0-py312` image with your
Prime-RL Ray image:

```yaml
spec:
  headGroupSpec:
    template:
      spec:
        containers:
          - name: ray-head
            image: <your-acr>.azurecr.io/<repo>/prime-rl-ray@sha256:<digest>
  workerGroupSpecs:
    - template:
        spec:
          containers:
            - name: ray-worker
              image: <your-acr>.azurecr.io/<repo>/prime-rl-ray@sha256:<digest>
```

Both pods need the same image so the Ray driver's runtime env matches the
worker runtime env. Pin by digest (`@sha256:...`) to avoid silent drift.

### Azure RWX PVC at `/shared`

Replace `prime-rl-shared` with the PVC you created from your Azure storage
class:

```yaml
volumes:
  - name: shared
    persistentVolumeClaim:
      claimName: <your-azure-rwx-pvc>    # e.g., "blob-training"
```

Mount it at `/shared` on the head and every worker, exactly as in the base
example. The launcher Job mounts the same PVC.

### Optional: priority class

If you use Kueue or a custom priority class for batch admission, attach it on
the worker template:

```yaml
spec:
  workerGroupSpecs:
    - template:
        spec:
          priorityClassName: <your-train-priority>
```

This is invisible to Prime-RL; it only affects how the cluster schedules the
worker pods.

## AKS overlay on `rl-launch-job.yaml`

The launch Job is `batch/v1`, so it goes through the regular AKS scheduling
path. Three AKS-specific touches:

- Pin the Job pod to the same CPU pool as the Ray head so `RAY_ADDRESS=auto`
  attaches to a local raylet.
- Use the same ACR image as the RayCluster.
- Mount the same Azure RWX PVC at `/shared`.

```yaml
spec:
  template:
    spec:
      nodeSelector:
        agentpool: <your-cpu-agentpool>
      containers:
        - name: launcher
          image: <your-acr>.azurecr.io/<repo>/prime-rl-ray@sha256:<digest>
      volumes:
        - name: shared
          persistentVolumeClaim:
            claimName: <your-azure-rwx-pvc>
```

## Known AKS caveats

- **H200 node pool networking.** The H200 multi-node RayCluster run in this
  PR was blocked before Prime-RL by node-pool networking on the H200 pool
  (kubelet logs/exec returned 504s, H200 pods could not resolve DNS or reach
  `github.com` / `huggingface.co`). Workers never reached `wait-gcs-ready`
  past the connectivity check. This is a cluster-side issue. Until it is
  resolved, RayCluster multi-node validation should be done on A100. The
  same-node H200 fallback validated the Prime-RL runtime path on three H200
  GPUs (trainer + primary Prime-vLLM + teacher Prime-vLLM) once caches were
  staged on `/shared`.
- **vLLM DeepGEMM warmup on H200 / Hopper FP8.** vLLM 0.20.2 enters the
  Hopper FP8 DeepGEMM warmup path on H200 even when the model is not FP8 and
  fails warmup unless `deep_gemm` is installed *and* the runtime accepts it.
  Set `VLLM_DEEP_GEMM_WARMUP=skip` in the launch Job `env:` on H200 pools.
  Harmless on A100.
- **Egress-restricted node pools.** If your AKS node pool restricts outbound
  to `github.com` and `huggingface.co`, pre-stage the Prime-RL fork checkout
  and HF cache on the Azure RWX PVC instead of relying on the launch Job to
  clone and download at run time. Point
  `experimental.ray.runtime_env.working_dir` and the HF cache env vars at
  `/shared`.
- **DRA preview status.** DRA is still moving through Kubernetes betas; AKS
  exposure varies by cluster version. Verify with `kubectl get deviceclass`
  and `kubectl get resourceclaimtemplates` before adopting the DRA path. If
  DRA is not available on your AKS version, use the `nvidia.com/gpu` shape
  instead — the rest of the manifest is unchanged.

## See also

- `README.md` (this directory) — the vendor-neutral launch flow and
  prerequisites that this AKS guide overlays.
- `../../docs/ray.md` — Ray-native architecture and `experimental.ray.*` config
  reference.
- `../../docs/kubernetes.md` — Kubernetes decision matrix (Ray-native vs the
  legacy SLURM-shaped Helm chart).
