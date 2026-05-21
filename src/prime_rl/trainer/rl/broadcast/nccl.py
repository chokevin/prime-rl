import pickle
import time
from pathlib import Path
from typing import Callable, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from prime_rl.configs.trainer import NCCLWeightBroadcastConfig
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.utils import get_world
from prime_rl.trainer.weights import get_max_layer_num
from prime_rl.utils.logger import get_logger
from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable
from prime_rl.utils.pathing import sync_wait_for_path
from prime_rl.utils.utils import get_broadcast_dir, get_step_path
from prime_rl.utils.vlm import get_layer_prefix

NCCL_READY_MARKER = "NCCL_READY"


LogFn = Callable[[str], None]


def broadcast_integer(integer: int, communicator: PyNcclCommunicator) -> None:
    """Broadcast an integer to a process group using NCCL communicator."""
    integer_tensor = torch.tensor([integer], dtype=torch.long).cuda()
    communicator.broadcast(integer_tensor, src=0)


def broadcast_state_dict(
    state_dict: dict[str, Tensor],
    communicator: PyNcclCommunicator,
    label: str | None = None,
    log_fn: LogFn | None = None,
) -> None:
    """Broadcast a state dict to NCCL process group using the PyNcclCommunicator."""
    start = time.perf_counter()
    prefix = f"{label} " if label else ""

    # Group tensors by dtype
    group_start = time.perf_counter()
    dtype_groups: dict[torch.dtype, list[tuple[str, Tensor]]] = {}
    for key, value in state_dict.items():
        assert not isinstance(value, DTensor), (
            "DTensor is not supported for broadcast, should have been converted to tensor already"
        )
        dtype = value.dtype
        if dtype not in dtype_groups:
            dtype_groups[dtype] = []
        dtype_groups[dtype].append((key, value))
    grouping_elapsed = time.perf_counter() - group_start

    # Build metadata: for each dtype group, store keys and shapes
    metadata_start = time.perf_counter()
    metadata = {}
    for dtype, items in dtype_groups.items():
        metadata[dtype] = [(key, value.shape, value.numel()) for key, value in items]

    # Send metadata
    state = pickle.dumps(metadata)
    metadata_prep_elapsed = time.perf_counter() - metadata_start

    metadata_broadcast_start = time.perf_counter()
    size_tensor = torch.tensor([len(state)], dtype=torch.long).cuda()
    communicator.broadcast(size_tensor, src=0)
    state_tensor = torch.ByteTensor(list(state)).cuda()
    communicator.broadcast(state_tensor, src=0)
    metadata_broadcast_elapsed = time.perf_counter() - metadata_broadcast_start

    if log_fn is not None:
        num_tensors = sum(len(items) for items in dtype_groups.values())
        log_fn(
            f"{prefix}NCCL sender state_dict metadata tensors={num_tensors} "
            f"dtype_groups={len(dtype_groups)} bytes={len(state)} "
            f"group={grouping_elapsed:.2f}s prep={metadata_prep_elapsed:.2f}s "
            f"broadcast={metadata_broadcast_elapsed:.2f}s"
        )

    # Concatenate and broadcast tensors grouped by dtype
    for dtype, items in dtype_groups.items():
        dtype_start = time.perf_counter()
        elements = sum(value.numel() for _, value in items)
        # Flatten all tensors and concatenate
        concat_start = time.perf_counter()
        flat_tensors = [value.flatten() for _, value in items]
        concatenated = torch.cat(flat_tensors)
        concat_elapsed = time.perf_counter() - concat_start

        broadcast_start = time.perf_counter()
        communicator.broadcast(concatenated, src=0)
        broadcast_elapsed = time.perf_counter() - broadcast_start
        if log_fn is not None:
            log_fn(
                f"{prefix}NCCL sender dtype={dtype} tensors={len(items)} elements={elements} "
                f"concat={concat_elapsed:.2f}s broadcast={broadcast_elapsed:.2f}s "
                f"total={time.perf_counter() - dtype_start:.2f}s"
            )
        del concatenated
        # Clean up individual tensors
        for _, value in items:
            del value
    if log_fn is not None:
        log_fn(f"{prefix}NCCL sender state_dict complete in {time.perf_counter() - start:.2f}s")


def filter_state_dict_by_layers(
    state_dict: dict[str, torch.Tensor],
    num_layers: int,
    layer_prefix: str,
    log_fn: LogFn | None = None,
) -> list[tuple[int, dict[str, torch.Tensor]]]:
    """Yield non-layer weights first, then each layer's weights.

    Returns (layer_idx, layer_state_dict) where layer_idx is -1 for the non-layer
    dict and the actual layer index (0, 1, ...) for layer dicts.
    """
    start = time.perf_counter()
    layer_state_dicts = [dict[str, torch.Tensor]() for _ in range(num_layers)]
    non_layer_state_dict: dict[str, torch.Tensor] = {}
    misplaced_layer_tensors = 0
    for key, value in state_dict.items():
        if not key.startswith(layer_prefix):
            non_layer_state_dict[key] = value
            continue

        layer_num_str = key[len(layer_prefix) :].split(".")[0]
        if not layer_num_str.isdigit():
            non_layer_state_dict[key] = value
            misplaced_layer_tensors += 1
            continue

        layer_idx = int(layer_num_str)
        if 0 <= layer_idx < num_layers:
            layer_state_dicts[layer_idx][key] = value
        else:
            non_layer_state_dict[key] = value
            misplaced_layer_tensors += 1

    if log_fn is not None:
        total_tensors = len(non_layer_state_dict) + sum(len(layer_state_dict) for layer_state_dict in layer_state_dicts)
        log_fn(
            f"NCCL sender grouped state_dict tensors={total_tensors} layers={num_layers} "
            f"non_layer_tensors={len(non_layer_state_dict)} "
            f"misplaced_layer_tensors={misplaced_layer_tensors} in {time.perf_counter() - start:.2f}s"
        )

    return [(-1, non_layer_state_dict), *list(enumerate(layer_state_dicts))]


def preprocess_layer_checkpoint(
    model: nn.Module,
    layer_state_dict: dict[str, Tensor],
    layer_idx: int,
) -> dict[str, Tensor]:
    if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(layer_state_dict):
        model.convert_layer_to_hf(layer_state_dict, layer_idx)
        return layer_state_dict

    from transformers.core_model_loading import revert_weight_conversion

    return revert_weight_conversion(model, layer_state_dict)


def preprocess_layer_quantized(
    model: nn.Module,
    layer_state_dict: dict[str, Tensor],
    layer_idx: int,
) -> dict[str, Tensor]:
    if layer_idx < 0:
        return layer_state_dict
    return model.convert_layer_to_vllm_kernel(layer_state_dict, layer_idx, quantize_fp8=True)


class NCCLWeightBroadcastSender:
    def __init__(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        device: int | str | torch.device,
        timeout: int,
        dtype: torch.dtype = torch.bfloat16,
        quantize_in_weight_transfer: bool = False,
    ):
        self.logger = get_logger()
        self.world = get_world()
        self.dtype = dtype
        self.quantize_in_weight_transfer = quantize_in_weight_transfer

        if self.world.is_master:
            disable_nccl_p2p_if_unavailable()
            # Trainer is on rank 0 in process group with all inference GPUs
            pg = StatelessProcessGroup.create(
                host=host, port=port, rank=rank, world_size=world_size, store_timeout=timeout
            )
            self.communicator = PyNcclCommunicator(pg, device=device)
            self.logger.debug("NCCL broadcast initialized on master rank")
        else:
            self.logger.debug("NCCL broadcast initialized on non-master rank (no communicator)")

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        """Broadcast the state dict of a model into the inference pool using NCCL."""
        start = time.perf_counter()
        state_dict = model.state_dict()
        self.logger.info(
            f"NCCL sender collected model.state_dict tensors={len(state_dict)} in {time.perf_counter() - start:.2f}s"
        )

        start = time.perf_counter()
        layer_prefix = get_layer_prefix(model.config)
        num_layers = get_max_layer_num(state_dict, layer_prefix)
        num_state_dict_to_send = num_layers + 1  # we send all layer plus the remaining weights
        self.logger.info(
            f"NCCL sender planned {num_state_dict_to_send} groups "
            f"(layers={num_layers}, layer_prefix={layer_prefix}) in {time.perf_counter() - start:.2f}s"
        )

        if self.world.is_master:
            broadcast_integer(num_state_dict_to_send, self.communicator)

        self.logger.debug(f"Broadcasting {num_state_dict_to_send} layer state dicts")
        preprocess_fn: Callable[[nn.Module, dict[str, Tensor], int], dict[str, Tensor]]
        if self.quantize_in_weight_transfer:
            preprocess_fn = preprocess_layer_quantized
        else:
            preprocess_fn = preprocess_layer_checkpoint

        groups = filter_state_dict_by_layers(state_dict, num_layers, layer_prefix, log_fn=self.logger.info)
        previous_group_complete = time.perf_counter()
        broadcast_start = time.perf_counter()
        for group_idx, (layer_id, layer_state_dict) in enumerate(groups, start=1):
            layer_start = time.perf_counter()
            layer_label = f"layer={layer_id} group={group_idx}/{num_state_dict_to_send}"
            self.logger.info(
                f"NCCL sender {layer_label} prep start tensors={len(layer_state_dict)} "
                f"since_previous_group={layer_start - previous_group_complete:.2f}s"
            )

            phase_start = time.perf_counter()
            layer_state_dict = self._resolve_dtensors(layer_state_dict)
            self.logger.info(f"NCCL sender {layer_label} resolved DTensors in {time.perf_counter() - phase_start:.2f}s")

            if self.world.is_master:
                phase_start = time.perf_counter()
                layer_state_dict = preprocess_fn(model, layer_state_dict, layer_id)
                self.logger.info(f"NCCL sender {layer_label} preprocessed in {time.perf_counter() - phase_start:.2f}s")

                broadcast_state_dict(
                    layer_state_dict,
                    self.communicator,
                    label=f"NCCL sender {layer_label}",
                    log_fn=self.logger.info,
                )
            self.logger.info(f"NCCL sender {layer_label} complete in {time.perf_counter() - layer_start:.2f}s")
            previous_group_complete = time.perf_counter()
        self.logger.info(f"NCCL sender broadcast_weights complete in {time.perf_counter() - broadcast_start:.2f}s")

    def _resolve_dtensors(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        dtensor_count = 0
        materialized_elements = 0
        for key, value in list(state_dict.items()):
            if isinstance(value, DTensor):
                dtensor_count += 1
                materialized_elements += value.numel()
                state_dict[key] = cast(DTensor, value.to(self.dtype)).full_tensor()
        if dtensor_count and torch.cuda.is_available():
            torch.cuda.synchronize()
        if dtensor_count:
            self.logger.info(
                f"NCCL sender materialized DTensors tensors={dtensor_count} elements={materialized_elements}"
            )
        return state_dict


class NCCLWeightBroadcast(WeightBroadcast):
    """Broadcast weights into the inference engine using NCCL."""

    def __init__(
        self,
        output_dir: Path,
        config: NCCLWeightBroadcastConfig,
        device: int | str | torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(output_dir)
        self.logger = get_logger()
        self.world = get_world()
        self.multi_run_manager = get_multi_run_manager()
        self.nccl_broadcast_sender = NCCLWeightBroadcastSender(
            config.host,
            config.port,
            0,
            config.inference_world_size + 1,
            device,
            config.timeout,
            dtype,
            quantize_in_weight_transfer=config.quantize_in_weight_transfer,
        )

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        """Broadcast the state dict of a model into the inference pool using NCCL and notifies the orchestrator."""
        self.logger.debug("Starting broadcasting weights to inference engine via NCCL")
        start_time = time.perf_counter()
        # `_compute_notified_runs` is a pure function of SPMD-replicated state on
        # multi_run_manager, so every trainer rank derives the same list. Only
        # the master touches the filesystem to notify the orchestrator, but all
        # ranks must wait on NCCL_READY before entering the broadcast path:
        # the broadcast preparation (DTensor resolution, quantization) enqueues
        # collectives on non-master ranks, and if those ranks start prep before
        # the orchestrator has paused inference, the collectives sit unmatched
        # until NCCL's watchdog kills the process after 10 min.
        notified_runs = self._compute_notified_runs()
        if self.world.is_master:
            self._notify_orchestrator(notified_runs)
        self._wait_for_nccl_ready(notified_runs)
        self.nccl_broadcast_sender.broadcast_weights(model, step)
        self.logger.debug(f"Weights broadcasted in {time.perf_counter() - start_time:.2f}s")

    def _compute_notified_runs(self) -> list[tuple[int, Path]]:
        """Derive the list of (run_idx, save_dir) pairs that need broadcasting.

        Pure function of `multi_run_manager` state, which is replicated across
        trainer ranks (SPMD). Returns the same list on every rank so master and
        non-master ranks agree on which NCCL_READY markers to wait for.
        """
        notified_runs: list[tuple[int, Path]] = []
        for idx in self.multi_run_manager.used_idxs:
            if not self.multi_run_manager.ready_to_update[idx]:
                continue
            try:
                save_dir = get_step_path(
                    get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                    self.multi_run_manager.progress[idx].step,
                )
                notified_runs.append((idx, save_dir))
            except FileNotFoundError:
                self.logger.warning(f"Run {idx} is deleted, skipping")
            except Exception as e:
                self.logger.error(f"Error resolving broadcast dir for run {idx}: {e}")
        return notified_runs

    def _notify_orchestrator(self, notified_runs: list[tuple[int, Path]]) -> None:
        """Create STABLE markers for each notified run and clear their ready flags.

        Master-only side effects (filesystem writes + state mutation). Called
        after `_compute_notified_runs`; non-master ranks skip this entirely.
        """
        for idx, save_dir in notified_runs:
            try:
                save_dir.mkdir(parents=True, exist_ok=True)
                stable_file = save_dir / "STABLE"
                stable_file.touch()
            except FileNotFoundError:
                self.logger.warning(f"Run {idx} is deleted, skipping")
            except Exception as e:
                self.logger.error(f"Error broadcasting weights for run {idx}: {e}")
            finally:
                self.multi_run_manager.ready_to_update[idx] = False

    def _wait_for_nccl_ready(self, notified_runs: list[tuple[int, Path]]):
        """Wait for inference workers to signal they are ready to receive NCCL broadcast."""
        for idx, save_dir in notified_runs:
            nccl_ready_file = save_dir / NCCL_READY_MARKER
            self.logger.debug(f"Waiting for NCCL_READY marker at {nccl_ready_file}")
            sync_wait_for_path(nccl_ready_file, interval=0.1, log_interval=10)
            self.logger.debug(f"Inference workers ready for NCCL broadcast (run {idx})")
