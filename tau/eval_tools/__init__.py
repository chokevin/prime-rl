"""Pure, macOS-runnable utilities backing the frozen train/eval-disjointness proof and
the paired before/after comparison gate (see ../README.md). Deliberately dependency-light
(stdlib + numpy + pydantic, all already resolved by the root `prime-rl` project) so these
can be unit tested without the Linux/CUDA-only training stack.

The scripts under `live/` do the same job against a real model/dataset/inference server
and are not importable or testable outside the GPU container image.
"""
