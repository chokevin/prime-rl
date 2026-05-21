import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import verifiers as vf

from prime_rl.orchestrator.request_picker import DirectRequestPicker
from prime_rl.orchestrator.scheduler import InflightRequest, Scheduler
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
    scheduler.inflight_rollouts_at_pause = 0
    scheduler.oldest_off_policy_at_pause = 0
    scheduler.inflight_requests = {}
    scheduler.groups = {}
    scheduler.max_off_policy_steps = 1
    scheduler.cancelled_rollouts_count = 0
    scheduler.policy_update_lock = asyncio.Lock()
    scheduler.inflight_policy_update_task = None
    scheduler.update_policy_task = None
    scheduler.enable_policy_updates = True
    scheduler.request_picker = DirectRequestPicker()
    scheduler.metric_values = defaultdict(list)
    scheduler.metric_counts = Counter()
    scheduler.last_update_metrics = {}
    scheduler.total_rollouts_by_env = defaultdict(int)
    scheduler.empty_rollouts_by_env = defaultdict(int)
    scheduler.errored_rollouts_by_env = defaultdict(int)
    scheduler.dropped_groups_by_env = defaultdict(int)
    scheduler.completed_rollouts_by_client = Counter()
    scheduler.cancelled_rollouts_by_client = Counter()
    scheduler.request_wall_seconds_by_client = defaultdict(list)
    scheduler.last_request_wall_seconds_by_client = {}
    scheduler.inference_pool = SimpleNamespace(get_metrics=lambda: {})
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


def test_cancel_inflight_rollouts_records_client_cancellations():
    async def run() -> None:
        scheduler = make_scheduler()
        client = vf.ClientConfig(
            client_idx=3,
            api_base_url="http://worker-a:8000/v1",
            extra_headers={"X-data-parallel-rank": "1"},
        )
        task = asyncio.create_task(asyncio.sleep(60))
        scheduler.inflight_requests[task] = InflightRequest(
            off_policy_steps=0,
            client_config=client,
            env_name="test",
            group_id=1,
            rollout_count=2,
        )

        await scheduler.cancel_inflight_rollouts()

        assert scheduler.cancelled_rollouts_count == 2
        assert scheduler.cancelled_rollouts_by_client[Scheduler._client_identity(client)] == 2
        assert scheduler.metric_counts["scheduler/cancelled_rollouts/client_3_worker_a_8000_dp_1"] == 2
        assert scheduler.inflight_requests == {}

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


def test_get_metrics_logs_instrumentation_payload():
    scheduler = make_scheduler()
    scheduler.inflight_rollouts_at_pause = 4
    scheduler.oldest_off_policy_at_pause = 2
    scheduler.last_update_metrics = {"time/update_ready_marker": 0.25}
    scheduler.metric_values["rollout_request_wall_seconds"].append(1.5)
    scheduler.metric_counts["scheduler/cancelled_rollouts/client_1_worker_a_8000_dp_0"] = 3

    metrics = scheduler.get_metrics()

    assert metrics["rollout_request_wall_seconds"] == 1.5
    assert metrics["time/update_ready_marker"] == 0.25

    message = scheduler.logger.info.call_args.args[0]
    prefix = "Scheduler instrumentation metrics: "
    assert message.startswith(prefix)
    payload = json.loads(message.removeprefix(prefix))
    assert payload["rollout_request_wall_seconds"] == 1.5
    assert payload["scheduler/cancelled_rollouts/client_1_worker_a_8000_dp_0"] == 3
    assert payload["scheduler/inflight_rollouts_at_pause"] == 4
    assert payload["scheduler/oldest_off_policy_at_pause"] == 2
    assert payload["time/update_ready_marker"] == 0.25
