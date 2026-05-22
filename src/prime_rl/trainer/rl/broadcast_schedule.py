def should_broadcast_weights(
    *,
    progress_step: int,
    max_steps: int | None,
    max_async_level: int,
    weight_broadcast_type: str,
) -> bool:
    if progress_step <= 0:
        return False

    if weight_broadcast_type == "filesystem":
        return True

    last_async_level_steps = max_steps is not None and progress_step >= max_steps - max_async_level
    return not last_async_level_steps
