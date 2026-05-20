import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import verifiers as vf

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


def test_apply_policy_update_cancels_stale_rollouts_before_update_weights():
    async def run() -> None:
        scheduler = make_scheduler()
        stale_task = asyncio.create_task(asyncio.sleep(60))
        survivor_task = asyncio.create_task(asyncio.sleep(60))
        client = SimpleNamespace(api_base_url="http://test", extra_headers={})
        scheduler.groups = {
            1: GroupState(example={"env_name": "test"}, rollouts_to_schedule=0),
            2: GroupState(example={"env_name": "test"}, rollouts_to_schedule=0),
        }
        scheduler.inflight_requests = {
            stale_task: InflightRequest(off_policy_steps=1, client_config=client, env_name="test", group_id=1),
            survivor_task: InflightRequest(off_policy_steps=0, client_config=client, env_name="test", group_id=2),
        }
        observed_at_update = {}

        async def update_weights(weight_dir, lora_name=None, step=0) -> None:
            observed_at_update["stale_present"] = stale_task in scheduler.inflight_requests
            observed_at_update["survivor_steps"] = scheduler.inflight_requests[survivor_task].off_policy_steps

        scheduler.inference_pool = SimpleNamespace(
            update_weights=update_weights,
            update_model_name=MagicMock(),
        )

        with patch("prime_rl.orchestrator.scheduler.wait_for_path", new=AsyncMock()):
            await scheduler._apply_policy_update(8)

        assert observed_at_update == {"stale_present": False, "survivor_steps": 1}
        assert stale_task not in scheduler.inflight_requests
        assert survivor_task in scheduler.inflight_requests
        assert scheduler.ckpt_step == 8

        if not survivor_task.done():
            survivor_task.cancel()
        await asyncio.sleep(0)

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


def test_schedule_rollout_uses_configured_request_picker():
    async def run() -> None:
        client_a = vf.ClientConfig(
            client_idx=0,
            api_base_url="http://worker-a:8000/v1",
            extra_headers={"X-data-parallel-rank": "0"},
        )
        client_b = vf.ClientConfig(
            client_idx=1,
            api_base_url="http://worker-a:8000/v1",
            extra_headers={"X-data-parallel-rank": "1"},
        )
        selected_contexts = []

        class PickSecondClient:
            async def select_client(self, candidates, inflight, context):
                selected_contexts.append(context)
                assert candidates == [client_a, client_b]
                assert inflight[Scheduler._client_identity(client_a)] == 0
                assert inflight[Scheduler._client_identity(client_b)] == 0
                return client_b

        class Env:
            requires_group_scoring = False

            async def run_rollout(self, **kwargs):
                return {"error": None, "trajectory": [{"role": "assistant", "content": "ok"}]}

        scheduler = make_scheduler()
        scheduler.request_picker = PickSecondClient()
        scheduler.inference_pool = SimpleNamespace(train_clients=[client_a, client_b])
        scheduler.train_envs = SimpleNamespace(get=MagicMock(return_value=Env()))
        scheduler.groups = {123: GroupState(example={"env_name": "test-env"}, rollouts_to_schedule=1)}

        await scheduler.schedule_rollout(123)

        assert scheduler.groups[123].pinned_client == client_b
        assert len(scheduler.inflight_requests) == 1
        task, info = next(iter(scheduler.inflight_requests.items()))
        assert info.client_config == client_b
        assert info.env_name == "test-env"
        assert selected_contexts[0].env_name == "test-env"
        assert selected_contexts[0].group_id == 123
        assert selected_contexts[0].model_name == "test-model"
        assert selected_contexts[0].ckpt_step == 7

        await task

    asyncio.run(run())
