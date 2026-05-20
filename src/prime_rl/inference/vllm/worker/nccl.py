import pickle
import time
from typing import TYPE_CHECKING, Generator, cast

import torch
from torch.nn import Module
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.weight_transfer import (
    load_weights_checkpoint_layerwise,
    load_weights_kernel,
    update_mla_absorbed_weights,
)
from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object

logger = init_logger("vllm.inference.vllm.worker_nccl")


def receive_integer(communicator: PyNcclCommunicator) -> int:
    """Receive an integer from the trainer master rank using NCCL communicator."""
    integer_tensor = torch.tensor([10], dtype=torch.long).to(communicator.device)
    communicator.broadcast(integer_tensor, src=0)
    return cast(int, integer_tensor.item())


def receive_state_dict(
    communicator: PyNcclCommunicator, label: str | None = None
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Stream tensors in a state dict broadcasted over NCCL."""
    start = time.perf_counter()
    prefix = f"{label} " if label else ""
    logger.info(f"{prefix}receive_state_dict metadata start")
    size_tensor = torch.tensor([10], dtype=torch.long).to(communicator.device)
    communicator.broadcast(size_tensor, src=0)
    state_tensor = torch.empty(cast(int, size_tensor.item()), dtype=torch.uint8).to(communicator.device)
    communicator.broadcast(state_tensor, src=0)

    metadata = pickle.loads(bytes(state_tensor.cpu().numpy()))
    num_tensors = sum(len(tensor_info_list) for tensor_info_list in metadata.values())
    logger.info(
        f"{prefix}receive_state_dict metadata complete in {time.perf_counter() - start:.2f}s "
        f"(dtype_groups={len(metadata)}, tensors={num_tensors})"
    )

    # Receive concatenated tensors per dtype and split them back
    for dtype, tensor_info_list in metadata.items():
        # Receive concatenated tensor for this dtype
        dtype_start = time.perf_counter()
        total_elements = sum(numel for _, _, numel in tensor_info_list)
        concatenated = torch.empty(total_elements, dtype=dtype, device=communicator.device)
        communicator.broadcast(concatenated, src=0)
        broadcast_elapsed = time.perf_counter() - dtype_start

        # Split concatenated tensor back into individual tensors
        split_start = time.perf_counter()
        consumer_elapsed = 0.0
        offset = 0
        for key, shape, numel in tensor_info_list:
            tensor = concatenated[offset : offset + numel].view(shape).clone()
            offset += numel
            yield_start = time.perf_counter()
            try:
                yield key, tensor
            finally:
                consumer_elapsed += time.perf_counter() - yield_start
                del tensor

        logger.info(
            f"{prefix}receive_state_dict dtype={dtype} tensors={len(tensor_info_list)} "
            f"elements={total_elements} broadcast={broadcast_elapsed:.2f}s "
            f"split_and_consume={time.perf_counter() - split_start:.2f}s "
            f"consumer={consumer_elapsed:.2f}s"
        )
        del concatenated
    logger.info(f"{prefix}receive_state_dict complete in {time.perf_counter() - start:.2f}s")


class NCCLWeightBroadcastReceiver:
    def __init__(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        device: int | str | torch.device,
        timeout: int,
    ):
        logger.info(f"Initializing NCCL broadcast receiver ({host}:{port}, rank={rank}, world_size={world_size})")
        disable_nccl_p2p_if_unavailable()

        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size, store_timeout=timeout)
        self.communicator = PyNcclCommunicator(pg, device=device)

    @torch.no_grad()
    def receive_state_dict(self):
        """Receives the state dict of a model from the trainer master rank using NCCL communicator."""
        start = time.perf_counter()
        logger.info("Receiving weights from trainer")
        num_state_dict_to_receive = receive_integer(self.communicator)
        logger.info(f"Receiving {num_state_dict_to_receive} layer state dicts")
        for layer_id in range(num_state_dict_to_receive):
            layer_start = time.perf_counter()
            logger.info(f"Receiving state dict {layer_id + 1}/{num_state_dict_to_receive}")
            for key, value in receive_state_dict(
                self.communicator, label=f"state_dict={layer_id + 1}/{num_state_dict_to_receive}"
            ):
                yield key, value
            logger.info(
                f"Received state dict {layer_id + 1}/{num_state_dict_to_receive} "
                f"in {time.perf_counter() - layer_start:.2f}s"
            )
        logger.info(f"Received all trainer weights in {time.perf_counter() - start:.2f}s")


class NCCLWeightUpdateWorker(Worker):
    """vLLM worker extension for updating weights in-place using NCCL."""

    def init_broadcaster(
        self,
        host: str,
        port: int,
        rank_offset: int,
        inference_world_size: int,
        timeout: int,
        quantize_in_weight_transfer: bool = False,
    ) -> None:
        """Initialize the NCCL broadcast receiver.

        Args:
            rank_offset: Starting GPU offset for this server in the global inference group.
            inference_world_size: Total number of inference GPUs across all servers.
        """
        self.quantize_in_weight_transfer = quantize_in_weight_transfer
        # Use the worker's device index directly as the local rank.
        # The previous dp_group-based computation broke in vLLM v1 multiprocess
        # DP mode where each worker is a separate process with a singleton
        # DP group (rank_in_group is always 0).
        local_rank = self.device.index
        global_rank_inference = rank_offset + local_rank

        logger.info(
            f"Worker [local_rank={local_rank} rank_offset={rank_offset}] "
            f"-> [global_rank={global_rank_inference} inference_world_size={inference_world_size}]"
        )

        self.nccl_broadcast_receiver = NCCLWeightBroadcastReceiver(
            host=host,
            port=port,
            rank=global_rank_inference + 1,  # +1 as the trainer broadcaster is on rank 0
            world_size=inference_world_size + 1,  # +1 as the trainer broadcaster is on rank 0
            device=self.device,
            timeout=timeout,
        )

    def liveness_probe(self) -> None:
        """No-op RPC used by the API server liveness endpoint."""
        return None

    def update_weights_from_path(self, weight_dir: str) -> None:
        """Update weights with the nccl communicator."""
        start = time.perf_counter()
        logger.info(f"NCCL worker update_weights_from_path start (weight_dir={weight_dir})")
        model_runner = self.model_runner
        if hasattr(model_runner.model, "runnable"):
            model = model_runner.model.runnable
        else:
            model = model_runner.model
        assert isinstance(model, Module)

        state_iter = self.nccl_broadcast_receiver.receive_state_dict()
        if self.quantize_in_weight_transfer:
            load_weights_kernel(model, state_iter)
            update_mla_absorbed_weights(model)
            logger.info(f"NCCL worker update_weights_from_path complete in {time.perf_counter() - start:.2f}s")
            return

        load_weights_checkpoint_layerwise(
            model,
            state_iter,
            self.model_runner.model_config,
            self.vllm_config,
        )
        logger.info(f"NCCL worker update_weights_from_path complete in {time.perf_counter() - start:.2f}s")
