# Mixed A100/H200 Ray RL

This directory contains the first supported Ray/KubeRay shape for prime-rl:

- one CPU Ray head;
- two 8-GPU A100 worker pods;
- two 8-GPU H200 worker pods;
- Kueue gang admission for the complete RayJob;
- one immutable prime-rl image on every pod;
- two symmetric 32B recipes that swap accelerator roles.

Set the image, node-selector values, queue name, storage claim, and config path
before submitting:

```bash
kubectl apply -f k8s/raycluster/mixed-a100-h200-rayjob.yaml
```

The default config trains on A100 and serves inference on H200. Change
`CONFIG_PATH` to `k8s/raycluster/32b-h200-train-a100-infer.toml` to reverse the
assignment.

Ray worker groups expose one custom resource unit per GPU under
`accelerator_type:A100` and `accelerator_type:H200`. Both inference placement
and Ray Train request those resources, keeping each role on its configured
homogeneous accelerator family.
