import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import verifiers as vf

from prime_rl.configs.shared import (
    LLMD_REQUIRED_WEIGHT_VERSION_HEADER,
    LLMD_REQUIRED_WEIGHT_VERSION_STATE_KEY,
    LLMD_ROLLOUT_ID_HEADER,
    LLMD_ROLLOUT_ID_STATE_KEY,
)
from prime_rl.orchestrator.scheduler import GroupState, InflightRequest, Scheduler
from prime_rl.utils.async_utils import safe_cancel


def make_scheduler() -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.max_async_level = 1
    scheduler.strict_async_level = False
    scheduler.step = 9
    scheduler.ckpt_step = 7
    scheduler.config = SimpleNamespace(output_dir=Path("/tmp/prime-rl-test"))
    scheduler.logger = MagicMock()
    scheduler.checkpoint_ready = asyncio.Event()
    scheduler.checkpoint_ready.set()
    scheduler.lora_name = None
    scheduler.model_name = "test-model"
    scheduler.update_weights_time = 0
    scheduler.wait_for_ckpt_time = 0
    scheduler.inflight_requests = {}
    scheduler.groups = {}
    scheduler.next_rollout_request_id = 0
    scheduler.max_off_policy_steps = 1
    scheduler.cancelled_rollouts_count = 0
    scheduler.policy_update_lock = asyncio.Lock()
    scheduler.inflight_policy_update_task = None
    scheduler.update_policy_task = None
    scheduler.enable_policy_updates = True
    scheduler.rate_limiter = None
    return scheduler


def test_update_off_policy_does_not_increment_interleaved_on_policy_tasks():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_off_policy_steps = 1
        scheduler.cancelled_rollouts_count = 0
        scheduler.logger = MagicMock()

        client = SimpleNamespace(api_base_url="http://test")
        stale_task = asyncio.create_task(asyncio.sleep(60))
        survivor_task = asyncio.create_task(asyncio.sleep(60))
        interleaved_task = None

        scheduler.inflight_requests = {
            stale_task: InflightRequest(off_policy_steps=1, client_config=client, env_name="test", group_id=1),
            survivor_task: InflightRequest(off_policy_steps=0, client_config=client, env_name="test", group_id=2),
        }

        async def drop_group(group_id: int) -> int:
            tasks_to_remove = [
                task for task, info in list(scheduler.inflight_requests.items()) if info.group_id == group_id
            ]
            for task in tasks_to_remove:
                scheduler.inflight_requests.pop(task, None)
                task.cancel()

            await asyncio.sleep(0)

            nonlocal interleaved_task
            if interleaved_task is None:
                interleaved_task = asyncio.create_task(asyncio.sleep(60))
                scheduler.inflight_requests[interleaved_task] = InflightRequest(
                    off_policy_steps=0,
                    client_config=client,
                    env_name="test",
                    group_id=3,
                )
            return len(tasks_to_remove)

        scheduler.drop_group = drop_group

        await scheduler._update_off_policy()

        assert stale_task not in scheduler.inflight_requests
        assert scheduler.inflight_requests[survivor_task].off_policy_steps == 1
        assert interleaved_task is not None
        assert scheduler.inflight_requests[interleaved_task].off_policy_steps == 0
        assert scheduler.cancelled_rollouts_count == 1

        for task in (stale_task, survivor_task, interleaved_task):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.sleep(0)

    asyncio.run(run())


def test_maybe_update_policy_reuses_inflight_update_after_cancellation():
    async def run() -> None:
        scheduler = make_scheduler()
        started = asyncio.Event()
        release = asyncio.Event()
        applied_steps: list[int] = []

        async def update_weights(weight_dir, lora_name=None, step=0) -> None:
            applied_steps.append(step)
            started.set()
            await release.wait()

        scheduler.inference_pool = SimpleNamespace(
            update_weights=update_weights,
            update_model_name=MagicMock(),
        )
        scheduler._update_off_policy = AsyncMock()

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=8),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new=AsyncMock()),
        ):
            first = asyncio.create_task(scheduler.maybe_update_policy())
            await started.wait()
            await safe_cancel(first)

            second = asyncio.create_task(scheduler.maybe_update_policy())
            await asyncio.sleep(0)
            assert applied_steps == [8]

            release.set()
            await second

        assert applied_steps == [8]
        assert scheduler.ckpt_step == 8

    asyncio.run(run())


def test_stop_cancels_inflight_policy_update_task():
    async def run() -> None:
        scheduler = make_scheduler()
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def update_weights(weight_dir, lora_name=None, step=0) -> None:
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        scheduler.inference_pool = SimpleNamespace(
            update_weights=update_weights,
            update_model_name=MagicMock(),
        )
        scheduler._update_off_policy = AsyncMock()

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=8),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new=AsyncMock()),
        ):
            scheduler.update_policy_task = asyncio.create_task(scheduler.maybe_update_policy())
            await started.wait()
            await asyncio.wait_for(scheduler.stop(), timeout=0.2)

        assert cancelled.is_set()
        assert scheduler.update_policy_task is None
        assert scheduler.inflight_policy_update_task is None

    asyncio.run(run())


def test_client_identity_distinguishes_base_url_and_dp_rank():
    client_a = vf.ClientConfig(
        api_base_url="http://worker-a:8000/v1",
        extra_headers={"X-data-parallel-rank": "0"},
    )
    client_b = vf.ClientConfig(
        api_base_url="http://worker-a:8000/v1",
        extra_headers={"X-data-parallel-rank": "1"},
    )

    assert Scheduler._client_identity(client_a) != Scheduler._client_identity(client_b)


def test_llmd_routing_metadata_is_added_only_when_client_requests_state_headers():
    scheduler = make_scheduler()
    example = {"env_name": "test-env", "example_id": "example-123"}
    plain_client = vf.ClientConfig(
        api_base_url="http://worker-a:8000/v1",
        extra_headers_from_state={"X-Session-ID": "example_id"},
    )
    llmd_client = vf.ClientConfig(
        api_base_url="http://llmd-router:8000/v1",
        extra_headers_from_state={
            LLMD_ROLLOUT_ID_HEADER: LLMD_ROLLOUT_ID_STATE_KEY,
            LLMD_REQUIRED_WEIGHT_VERSION_HEADER: LLMD_REQUIRED_WEIGHT_VERSION_STATE_KEY,
        },
    )

    unchanged = scheduler._example_with_llmd_routing_metadata(
        example,
        plain_client,
        group_id=3,
        request_id=4,
    )
    enriched = scheduler._example_with_llmd_routing_metadata(
        example,
        llmd_client,
        group_id=3,
        request_id=4,
    )

    assert unchanged is example
    assert LLMD_ROLLOUT_ID_STATE_KEY not in example
    assert enriched[LLMD_ROLLOUT_ID_STATE_KEY] == "step-9-policy-7-group-3-request-4"
    assert enriched[LLMD_REQUIRED_WEIGHT_VERSION_STATE_KEY] == "7"
    assert enriched["example_id"] == "example-123"


def test_schedule_rollout_passes_llmd_routing_metadata_to_env():
    async def run() -> None:
        scheduler = make_scheduler()
        client = vf.ClientConfig(
            api_base_url="http://llmd-router:8000/v1",
            extra_headers_from_state={
                LLMD_ROLLOUT_ID_HEADER: LLMD_ROLLOUT_ID_STATE_KEY,
                LLMD_REQUIRED_WEIGHT_VERSION_HEADER: LLMD_REQUIRED_WEIGHT_VERSION_STATE_KEY,
            },
        )
        env = SimpleNamespace(
            requires_group_scoring=False,
            run_rollout=AsyncMock(return_value={"error": None, "trajectory": [{"role": "assistant"}]}),
        )
        scheduler.inference_pool = SimpleNamespace(train_clients=[client])
        scheduler.train_envs = {"test-env": env}
        scheduler.groups[0] = GroupState(
            example={"env_name": "test-env", "example_id": "example-123"},
            rollouts_to_schedule=1,
        )

        await scheduler.schedule_rollout(0)
        task = next(iter(scheduler.inflight_requests))
        await task

        call_kwargs = env.run_rollout.call_args.kwargs
        assert call_kwargs["example"][LLMD_ROLLOUT_ID_STATE_KEY] == "step-9-policy-7-group-0-request-0"
        assert call_kwargs["example"][LLMD_REQUIRED_WEIGHT_VERSION_STATE_KEY] == "7"

    asyncio.run(run())
