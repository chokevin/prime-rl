def should_broadcast_weights(
    *,
    progress_step: int,
    max_steps: int | None,
    max_async_level: int,
    final_step_async_level: int | None,
    weight_broadcast_type: str,
) -> bool:
    if progress_step <= 0:
        return False

    if weight_broadcast_type == "filesystem":
        return True

    final_async_level = final_step_async_level or max_async_level
    last_async_level_steps = max_steps is not None and progress_step >= max_steps - final_async_level
    return not last_async_level_steps
