"""Drive repeated Prime-RL inference weight updates without the rollout loop.

The NCCL mode mirrors the trainer/orchestrator handshake closely enough to debug
one vLLM inference server with multiple local workers:

1. initialize the server-side NCCL receiver with ``/init_broadcaster``;
2. call Prime's normal client-side pause -> update_weights -> resume helper;
3. wait for the helper's ``NCCL_READY`` marker;
4. broadcast checkpoint-format tensors from rank 0; and
5. repeat for the next weight directory.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import Iterable

import httpx
import torch
from safetensors import safe_open
from torch import Tensor

from prime_rl.trainer.weights import get_max_layer_num
from prime_rl.utils.client import update_weights
from prime_rl.utils.logger import setup_logger

NCCL_READY_MARKER = "NCCL_READY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        action="append",
        default=[],
        help="Inference admin base URL. Repeat for multiple independent inference servers.",
    )
    parser.add_argument(
        "--weight-dir",
        action="append",
        type=Path,
        default=[],
        help="Checkpoint/broadcast weight directory to use for an update cycle. Repeat for multiple cycles.",
    )
    parser.add_argument(
        "--weight-dir-template",
        type=str,
        help="Python format string for per-cycle weight directories, for example '/runs/x/broadcasts/step_{step}'.",
    )
    parser.add_argument("--steps", type=int, default=2, help="Number of update cycles to run.")
    parser.add_argument("--start-step", type=int, default=1, help="First step number for --weight-dir-template.")
    parser.add_argument("--backend", choices=("nccl", "filesystem"), default="nccl")
    parser.add_argument("--http-timeout", type=float, default=900.0)
    parser.add_argument("--ready-timeout", type=float, default=300.0)
    parser.add_argument("--layer-prefix", default="model.layers.")
    parser.add_argument("--device", default="cuda:0", help="CUDA device for NCCL sender rank 0.")
    parser.add_argument("--host", default="127.0.0.1", help="NCCL store host used by /init_broadcaster.")
    parser.add_argument("--port", type=int, default=29501, help="NCCL store port used by /init_broadcaster.")
    parser.add_argument("--inference-world-size", type=int, default=8)
    parser.add_argument(
        "--workers-per-server",
        type=int,
        help="Number of inference workers behind each base URL. Defaults to world_size / num_base_urls.",
    )
    parser.add_argument("--rank-offset", type=int, default=0)
    parser.add_argument("--nccl-timeout", type=int, default=1200)
    parser.add_argument("--skip-init-broadcaster", action="store_true")
    return parser.parse_args()


def resolve_weight_dirs(args: argparse.Namespace) -> list[Path]:
    if args.weight_dir_template and args.weight_dir:
        raise ValueError("Use either --weight-dir-template or repeated --weight-dir, not both")
    if args.weight_dir_template:
        return [
            Path(args.weight_dir_template.format(step=step))
            for step in range(args.start_step, args.start_step + args.steps)
        ]
    if len(args.weight_dir) == 1 and args.steps == 1:
        return args.weight_dir
    if len(args.weight_dir) != args.steps:
        raise ValueError("Pass one --weight-dir per step, or use --weight-dir-template")
    return args.weight_dir


def iter_safetensor_keys(weight_dir: Path) -> Iterable[str]:
    for path in sorted(weight_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            yield from f.keys()


def load_key_group(weight_dir: Path, keys: set[str], device: torch.device) -> dict[str, Tensor]:
    state_dict: dict[str, Tensor] = {}
    for path in sorted(weight_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            matching_keys = [key for key in f.keys() if key in keys]
            for key in matching_keys:
                state_dict[key] = f.get_tensor(key).to(device=device, non_blocking=False)
    return state_dict


def iter_layer_key_groups(weight_dir: Path, layer_prefix: str) -> Iterable[tuple[int, set[str]]]:
    all_keys = set(iter_safetensor_keys(weight_dir))
    if not all_keys:
        raise FileNotFoundError(f"No safetensors weights found in {weight_dir}")

    num_layers = get_max_layer_num({key: torch.empty(0) for key in all_keys}, layer_prefix)
    yield -1, {key for key in all_keys if not key.startswith(layer_prefix)}
    for layer_id in range(num_layers):
        prefix = f"{layer_prefix}{layer_id}."
        yield layer_id, {key for key in all_keys if key.startswith(prefix)}


class CheckpointNCCLBroadcaster:
    def __init__(self, host: str, port: int, world_size: int, device: str, timeout: int):
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable

        disable_nccl_p2p_if_unavailable()
        torch_device = torch.device(device)
        if torch_device.type != "cuda":
            raise ValueError("NCCL microbench requires a CUDA device, for example --device cuda:0")
        torch.cuda.set_device(torch_device)
        pg = StatelessProcessGroup.create(host=host, port=port, rank=0, world_size=world_size, store_timeout=timeout)
        self.communicator = PyNcclCommunicator(pg, device=torch_device)
        self.device = torch_device

    @torch.no_grad()
    def broadcast_checkpoint(self, weight_dir: Path, layer_prefix: str) -> None:
        from prime_rl.trainer.rl.broadcast.nccl import broadcast_integer, broadcast_state_dict

        groups = [(layer_id, keys) for layer_id, keys in iter_layer_key_groups(weight_dir, layer_prefix) if keys]
        broadcast_integer(len(groups), self.communicator)
        for group_idx, (layer_id, keys) in enumerate(groups, start=1):
            start = time.perf_counter()
            state_dict = load_key_group(weight_dir, keys, self.device)
            load_elapsed = time.perf_counter() - start
            broadcast_state_dict(state_dict, self.communicator)
            print(
                f"broadcast group {group_idx}/{len(groups)} layer={layer_id} "
                f"tensors={len(keys)} load={load_elapsed:.2f}s total={time.perf_counter() - start:.2f}s",
                flush=True,
            )
            del state_dict


async def init_broadcasters(args: argparse.Namespace, base_urls: list[str]) -> None:
    if args.skip_init_broadcaster:
        return
    workers_per_server = args.workers_per_server or args.inference_world_size // len(base_urls)
    async with httpx.AsyncClient(timeout=args.http_timeout) as client:
        for idx, base_url in enumerate(base_urls):
            rank_offset = args.rank_offset + idx * workers_per_server
            response = await client.post(
                f"{base_url.rstrip('/')}/init_broadcaster",
                json={
                    "host": args.host,
                    "port": args.port,
                    "rank_offset": rank_offset,
                    "inference_world_size": args.inference_world_size,
                    "timeout": args.nccl_timeout,
                    "quantize_in_weight_transfer": False,
                },
            )
            response.raise_for_status()


async def wait_for_ready_marker(weight_dir: Path, update_task: asyncio.Task[None], timeout: float) -> None:
    marker = weight_dir / NCCL_READY_MARKER
    if marker.exists():
        marker.unlink()
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        if marker.exists():
            return
        if update_task.done():
            await update_task
        await asyncio.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for {marker}")


async def run_microbench(args: argparse.Namespace) -> None:
    base_urls = args.base_url or ["http://127.0.0.1:8000"]
    weight_dirs = resolve_weight_dirs(args)

    broadcaster = None
    if args.backend == "nccl":
        broadcaster_task = asyncio.create_task(
            asyncio.to_thread(
                CheckpointNCCLBroadcaster,
                args.host,
                args.port,
                args.inference_world_size + 1,
                args.device,
                args.nccl_timeout,
            )
        )
        await init_broadcasters(args, base_urls)
        broadcaster = await broadcaster_task

    admin_clients = [httpx.AsyncClient(base_url=base_url, timeout=args.http_timeout) for base_url in base_urls]
    try:
        for cycle_idx, weight_dir in enumerate(weight_dirs, start=1):
            start = time.perf_counter()
            print(f"update cycle {cycle_idx}/{len(weight_dirs)} start weight_dir={weight_dir}", flush=True)
            update_task = asyncio.create_task(update_weights(admin_clients, weight_dir))
            if broadcaster is not None:
                await wait_for_ready_marker(weight_dir, update_task, args.ready_timeout)
                await asyncio.to_thread(broadcaster.broadcast_checkpoint, weight_dir, args.layer_prefix)
            await update_task
            print(
                f"update cycle {cycle_idx}/{len(weight_dirs)} complete in {time.perf_counter() - start:.2f}s",
                flush=True,
            )
    finally:
        await asyncio.gather(*(client.aclose() for client in admin_clients))


def main() -> None:
    setup_logger("info")
    asyncio.run(run_microbench(parse_args()))


if __name__ == "__main__":
    main()
