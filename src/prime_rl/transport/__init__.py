from importlib import import_module
from pathlib import Path

from prime_rl.configs.shared import TransportConfig
from prime_rl.transport.base import MicroBatchReceiver, MicroBatchSender, TrainingBatchReceiver, TrainingBatchSender
from prime_rl.transport.types import MicroBatch, TrainingBatch, TrainingSample

_BACKEND_EXPORTS = {
    "FileSystemTrainingBatchSender": ("prime_rl.transport.filesystem", "FileSystemTrainingBatchSender"),
    "FileSystemTrainingBatchReceiver": ("prime_rl.transport.filesystem", "FileSystemTrainingBatchReceiver"),
    "FileSystemMicroBatchSender": ("prime_rl.transport.filesystem", "FileSystemMicroBatchSender"),
    "FileSystemMicroBatchReceiver": ("prime_rl.transport.filesystem", "FileSystemMicroBatchReceiver"),
    "ZMQTrainingBatchSender": ("prime_rl.transport.zmq", "ZMQTrainingBatchSender"),
    "ZMQTrainingBatchReceiver": ("prime_rl.transport.zmq", "ZMQTrainingBatchReceiver"),
    "ZMQMicroBatchSender": ("prime_rl.transport.zmq", "ZMQMicroBatchSender"),
    "ZMQMicroBatchReceiver": ("prime_rl.transport.zmq", "ZMQMicroBatchReceiver"),
    "RayTrainingBatchSender": ("prime_rl.transport.ray", "RayTrainingBatchSender"),
    "RayTrainingBatchReceiver": ("prime_rl.transport.ray", "RayTrainingBatchReceiver"),
    "RayMicroBatchSender": ("prime_rl.transport.ray", "RayMicroBatchSender"),
    "RayMicroBatchReceiver": ("prime_rl.transport.ray", "RayMicroBatchReceiver"),
}


def __getattr__(name: str):
    if name not in _BACKEND_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _BACKEND_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def setup_training_batch_sender(output_dir: Path, transport: TransportConfig) -> TrainingBatchSender:
    if transport.type == "filesystem":
        from prime_rl.transport.filesystem import FileSystemTrainingBatchSender

        return FileSystemTrainingBatchSender(output_dir)
    elif transport.type == "zmq":
        from prime_rl.transport.zmq import ZMQTrainingBatchSender

        return ZMQTrainingBatchSender(output_dir, transport)
    elif transport.type == "ray":
        from prime_rl.transport.ray import RayTrainingBatchSender

        return RayTrainingBatchSender(output_dir, transport)
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")


def setup_training_batch_receiver(transport: TransportConfig) -> TrainingBatchReceiver:
    if transport.type == "filesystem":
        from prime_rl.transport.filesystem import FileSystemTrainingBatchReceiver

        return FileSystemTrainingBatchReceiver()
    elif transport.type == "zmq":
        from prime_rl.transport.zmq import ZMQTrainingBatchReceiver

        return ZMQTrainingBatchReceiver(transport)
    elif transport.type == "ray":
        from prime_rl.transport.ray import RayTrainingBatchReceiver

        return RayTrainingBatchReceiver(transport)
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")


def setup_micro_batch_sender(
    output_dir: Path, data_world_size: int, current_step: int, transport: TransportConfig
) -> MicroBatchSender:
    if transport.type == "filesystem":
        from prime_rl.transport.filesystem import FileSystemMicroBatchSender

        return FileSystemMicroBatchSender(output_dir, data_world_size, current_step)
    elif transport.type == "zmq":
        from prime_rl.transport.zmq import ZMQMicroBatchSender

        return ZMQMicroBatchSender(output_dir, data_world_size, current_step, transport)
    elif transport.type == "ray":
        from prime_rl.transport.ray import RayMicroBatchSender

        return RayMicroBatchSender(output_dir, data_world_size, current_step, transport)
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")


def setup_micro_batch_receiver(
    output_dir: Path, data_rank: int, current_step: int, transport: TransportConfig
) -> MicroBatchReceiver:
    if transport.type == "filesystem":
        from prime_rl.transport.filesystem import FileSystemMicroBatchReceiver

        return FileSystemMicroBatchReceiver(output_dir, data_rank, current_step)
    elif transport.type == "zmq":
        from prime_rl.transport.zmq import ZMQMicroBatchReceiver

        return ZMQMicroBatchReceiver(output_dir, data_rank, current_step, transport)
    elif transport.type == "ray":
        from prime_rl.transport.ray import RayMicroBatchReceiver

        return RayMicroBatchReceiver(output_dir, data_rank, current_step, transport)
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")


__all__ = [
    "FileSystemTrainingBatchSender",
    "FileSystemTrainingBatchReceiver",
    "FileSystemMicroBatchSender",
    "FileSystemMicroBatchReceiver",
    "RayTrainingBatchSender",
    "RayTrainingBatchReceiver",
    "RayMicroBatchSender",
    "RayMicroBatchReceiver",
    "MicroBatchReceiver",
    "MicroBatchSender",
    "TrainingSample",
    "TrainingBatch",
    "MicroBatch",
    "setup_training_batch_sender",
    "setup_training_batch_receiver",
    "setup_micro_batch_sender",
    "setup_micro_batch_receiver",
]
