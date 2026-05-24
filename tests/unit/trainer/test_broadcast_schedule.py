from prime_rl.trainer.rl.broadcast_schedule import should_broadcast_weights


def test_nccl_skips_step_zero_and_final_async_window():
    decisions = [
        should_broadcast_weights(
            progress_step=step,
            max_steps=10,
            max_async_level=2,
            final_step_async_level=None,
            weight_broadcast_type="nccl",
        )
        for step in range(10)
    ]

    assert decisions == [False, True, True, True, True, True, True, True, False, False]


def test_nccl_broadcasts_indefinite_runs_after_step_zero():
    assert not should_broadcast_weights(
        progress_step=0,
        max_steps=None,
        max_async_level=2,
        final_step_async_level=None,
        weight_broadcast_type="nccl",
    )
    assert should_broadcast_weights(
        progress_step=8,
        max_steps=None,
        max_async_level=2,
        final_step_async_level=None,
        weight_broadcast_type="nccl",
    )


def test_filesystem_keeps_final_broadcasts_for_resume():
    assert should_broadcast_weights(
        progress_step=9,
        max_steps=10,
        max_async_level=2,
        final_step_async_level=None,
        weight_broadcast_type="filesystem",
    )


def test_nccl_final_step_async_level_skips_extra_final_broadcasts():
    decisions = [
        should_broadcast_weights(
            progress_step=step,
            max_steps=10,
            max_async_level=1,
            final_step_async_level=2,
            weight_broadcast_type="nccl",
        )
        for step in range(10)
    ]

    assert decisions == [False, True, True, True, True, True, True, True, False, False]
