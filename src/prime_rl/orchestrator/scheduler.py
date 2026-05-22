from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import verifiers as vf
from aiolimiter import AsyncLimiter

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.request_picker import (
    CandidateStats,
    ClientIdentity,
    DirectRequestPicker,
    RequestPickContext,
    client_identity,
    client_metric_label,
    endpoint_label_from_url,
    request_picker_long_output_cold_start_ratio,
    select_with_metrics,
    setup_request_picker,
)
from prime_rl.orchestrator.vf_utils import get_completion_len, get_prompt_len, get_seq_len
from prime_rl.utils.async_utils import safe_cancel, safe_cancel_all
from prime_rl.utils.client import InferencePool
from prime_rl.utils.logger import ProgressTracker, get_logger
from prime_rl.utils.utils import (
    get_broadcast_dir,
    get_latest_ckpt_step,
    get_step_path,
    wait_for_path,
)

SCHEDULER_INSTRUMENTATION_PREFIXES = (
    "rollout_",
    "request_picker_",
    "scheduler_refill_gap_seconds",
    "scheduler/completed_rollouts/",
    "scheduler/cancelled_rollouts/",
    "scheduler/client_metrics_",
    "scheduler/inflight_at_selection/",
    "scheduler/off_policy_level_at_completion/",
    "scheduler/selected_client/",
    "time/update_",
)

SCHEDULER_INSTRUMENTATION_KEYS = {
    "scheduler/inflight_rollouts_at_pause",
    "scheduler/oldest_off_policy_at_pause",
    "time/update_weights",
    "time/wait_for_ckpt",
}

PICKER_HISTORY_SIZE = 128


def _example_fingerprint(example: dict) -> str:
    payload = json.dumps(example, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _default_max_completion_tokens(config: OrchestratorConfig) -> int | None:
    train_config = getattr(config, "train", None)
    sampling = getattr(train_config, "sampling", None)
    max_completion_tokens = getattr(sampling, "max_completion_tokens", None)
    return max_completion_tokens if isinstance(max_completion_tokens, int) else None


def _max_completion_tokens_by_env(config: OrchestratorConfig) -> dict[str, int]:
    train_config = getattr(config, "train", None)
    env_configs = getattr(train_config, "env", [])
    max_completion_tokens_by_env: dict[str, int] = {}
    for env_config in env_configs:
        sampling = getattr(env_config, "sampling", None)
        max_completion_tokens = getattr(sampling, "max_completion_tokens", None)
        if not isinstance(max_completion_tokens, int):
            continue
        for env_name in (getattr(env_config, "resolved_name", None), getattr(env_config, "name", None)):
            if isinstance(env_name, str):
                max_completion_tokens_by_env[env_name] = max_completion_tokens
    return max_completion_tokens_by_env


@dataclass
class InflightRequest:
    """Metadata for an in-flight request."""

    off_policy_steps: int
    client_config: vf.ClientConfig
    env_name: str
    group_id: int | None = None
    rollout_count: int = 1
    # Dispatch round the request belongs to (see GroupState.current_round).
    round_id: int = 0
    request_id: int = 0
    request_started_at: float = 0.0
    dispatch_wait_seconds: float = 0.0
    client_inflight_at_selection: int = 0
    example_fingerprint: str | None = None
    predicted_completion_tokens: float | None = None


@dataclass
class GroupState:
    """Tracks the state of a rollout group (one example × N rollouts)."""

    example: dict
    rollouts_to_schedule: int
    completed_rollouts: list[vf.RolloutOutput] = field(default_factory=list)
    pinned_client: vf.ClientConfig | None = None
    # Number of dispatch rounds in which at least one rollout returned errored
    # or empty trajectories. Compared against
    # config.max_error_reschedule_attempts to decide when to drop a
    # permanently-stuck group. Counts rounds, not rollouts: a failed round in
    # an individual-scoring env that happens to dispatch N rollouts at once
    # still only counts as 1.
    failed_attempts: int = 0
    # Round id assigned to newly-dispatched rollouts. Advances after a failure
    # is counted so the resulting reschedule starts a new round.
    current_round: int = 0
    # Highest round already counted as failed; used to dedupe failures from
    # multiple rollouts in the same round.
    last_failed_round: int = -1
    created_at: float = field(default_factory=time.perf_counter)
    first_dispatch_at: float | None = None
    first_completion_at: float | None = None
    completed_request_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seq_tokens: int = 0
    request_wall_seconds: list[float] = field(default_factory=list)
    slowest_request_id: int | None = None
    slowest_request_wall_seconds: float = 0.0
    slowest_request_client: vf.ClientConfig | None = None
    example_fingerprint: str | None = None
    predicted_completion_tokens: float | None = None
    completion_prediction_source: str = "none"


class Scheduler:
    """
    Asynchronously manages scheduling of rollout requests and policy updates.
    Keeps a constant number of rollouts in-flight (continuous batching) and
    updates the policy as soon as it becomes available.

    References:
    - AReal: https://arxiv.org/abs/2505.24298v1
    - PipelineRL: https://arxiv.org/abs/2509.19128v1
    """

    def __init__(
        self,
        train_envs: TrainEnvs,
        inference_pool: InferencePool,
        buffer: Buffer,
        config: OrchestratorConfig,
        max_inflight_rollouts: int,
        max_async_level: int,
        max_off_policy_steps: int,
        strict_async_level: bool,
        tasks_per_minute: int | None,
        enable_policy_updates: bool = True,
        lora_name: str | None = None,
    ):
        self.logger = get_logger()
        if tasks_per_minute is not None:
            self.rate_limiter = AsyncLimiter(max_rate=tasks_per_minute, time_period=60)
        else:
            self.rate_limiter = None
        self.train_envs = train_envs
        self.buffer = buffer
        self.config = config
        self.batch_size = config.batch_size
        self.token_batch_size = config.token_batch_size
        self.rollouts_per_example = config.rollouts_per_example
        self.max_inflight_rollouts = max_inflight_rollouts
        self.max_async_level = max_async_level
        self.max_off_policy_steps = max_off_policy_steps
        self.strict_async_level = strict_async_level
        self.enable_policy_updates = enable_policy_updates
        self.lora_name = lora_name
        self.model_name = self.config.model.name
        self.json_logging = config.log.json_logging

        # Inference pool - used for admin operations (adapter sync) and metrics
        self.inference_pool = inference_pool
        self.request_picker = setup_request_picker(config.experimental.request_picker)

        group_scoring_envs = [env.name for env in train_envs if env.requires_group_scoring]
        if group_scoring_envs:
            self.logger.info(f"Group rollout scoring active for env(s): {', '.join(group_scoring_envs)}")

        # Track in-flight requests: task -> info
        self.inflight_requests: dict[asyncio.Task, InflightRequest] = {}

        # Track in-progress groups while rollouts are generated independently.
        self.next_group_id = 0
        self.next_request_id = 0
        self.groups: dict[int, GroupState] = {}

        self.step, self.ckpt_step = 0, 0
        self.checkpoint_ready = asyncio.Event()
        self.checkpoint_ready.set()
        self.update_weights_time, self.wait_for_ckpt_time = 0, 0
        self.update_policy_task: asyncio.Task | None = None
        self.inflight_policy_update_task: asyncio.Task | None = None
        self.policy_update_lock = asyncio.Lock()
        self.cancelled_rollouts_count = 0
        self.empty_rollouts_by_env: dict[str, int] = defaultdict(int)
        self.errored_rollouts_by_env: dict[str, int] = defaultdict(int)
        self.total_rollouts_by_env: dict[str, int] = defaultdict(int)
        self.dropped_groups_by_env: dict[str, int] = defaultdict(int)
        self.last_batch_generation_time = 0.0
        self.last_update_metrics: dict[str, float] = {}
        self.inflight_rollouts_at_pause = 0
        self.oldest_off_policy_at_pause = 0
        self.metric_values: dict[str, list[float]] = defaultdict(list)
        self.metric_counts: Counter[str] = Counter()
        self.completed_rollouts_by_client: Counter[ClientIdentity] = Counter()
        self.cancelled_rollouts_by_client: Counter[ClientIdentity] = Counter()
        self.request_wall_seconds_by_client: dict[ClientIdentity, deque[float]] = defaultdict(
            lambda: deque(maxlen=PICKER_HISTORY_SIZE)
        )
        self.last_request_wall_seconds_by_client: dict[ClientIdentity, float] = {}
        self.completion_tokens_by_client: dict[ClientIdentity, deque[float]] = defaultdict(
            lambda: deque(maxlen=PICKER_HISTORY_SIZE)
        )
        self.last_completion_tokens_by_client: dict[ClientIdentity, float] = {}
        self.completion_tokens_by_fingerprint: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=PICKER_HISTORY_SIZE)
        )
        self.completion_tokens_by_env: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=PICKER_HISTORY_SIZE))
        self.group_wall_seconds_by_client: dict[ClientIdentity, deque[float]] = defaultdict(
            lambda: deque(maxlen=PICKER_HISTORY_SIZE)
        )
        self.last_group_wall_seconds_by_client: dict[ClientIdentity, float] = {}
        self.group_tail_seconds_by_client: dict[ClientIdentity, deque[float]] = defaultdict(
            lambda: deque(maxlen=PICKER_HISTORY_SIZE)
        )
        self.last_group_tail_seconds_by_client: dict[ClientIdentity, float] = {}
        self.off_policy_steps_by_client: dict[ClientIdentity, deque[float]] = defaultdict(
            lambda: deque(maxlen=PICKER_HISTORY_SIZE)
        )
        self.last_off_policy_steps_by_client: dict[ClientIdentity, float] = {}
        self.default_max_completion_tokens = _default_max_completion_tokens(config)
        self.max_completion_tokens_by_env = _max_completion_tokens_by_env(config)
        self._last_refill_end_time: float | None = None

    @property
    def uses_token_batching(self) -> bool:
        return self.token_batch_size is not None

    @property
    def batch_target(self) -> int:
        if self.uses_token_batching:
            assert self.token_batch_size is not None
            return self.token_batch_size
        assert self.batch_size is not None
        return self.batch_size

    def _effective_max_async_level(self) -> int:
        weight_broadcast = self.config.weight_broadcast
        final_step_async_level = (
            weight_broadcast.final_step_async_level
            if weight_broadcast.type == "nccl" and weight_broadcast.final_step_async_level is not None
            else self.max_async_level
        )
        max_steps = self.config.max_steps
        if max_steps is None or final_step_async_level <= self.max_async_level:
            return self.max_async_level

        last_broadcast_step = max(max_steps - final_step_async_level - 1, 0)
        return max(self.max_async_level, self.step - last_broadcast_step)

    def get_batch_progress_increment(self, rollouts: list[vf.RolloutOutput]) -> int:
        if self.uses_token_batching:
            return sum(get_seq_len(rollout) for rollout in rollouts)
        return len(rollouts)

    def finalize_batch_rollouts(self, rollouts: list[vf.RolloutOutput]) -> list[vf.RolloutOutput]:
        if self.batch_size is None:
            return rollouts
        return rollouts[: self.batch_size]

    async def cancel_inflight_rollouts(self):
        """Cancel all in-flight rollout requests."""
        count = sum(info.rollout_count for info in self.inflight_requests.values())
        for info in self.inflight_requests.values():
            self.cancelled_rollouts_by_client[self._client_identity(info.client_config)] += info.rollout_count
            self._record_client_count("scheduler/cancelled_rollouts", info.client_config, info.rollout_count)
        await safe_cancel_all(list(self.inflight_requests))
        self.inflight_requests.clear()
        self.groups.clear()
        self.cancelled_rollouts_count += count

    @staticmethod
    def _client_identity(c: vf.ClientConfig) -> tuple[str, str | None]:
        return client_identity(c)

    def _record_value(self, name: str, value: float) -> None:
        self.metric_values[name].append(value)

    def _record_count(self, name: str, value: int | float = 1) -> None:
        self.metric_counts[name] += value

    def _record_client_count(self, prefix: str, client: vf.ClientConfig, value: int | float = 1) -> None:
        self._record_count(f"{prefix}/{client_metric_label(client)}", value)

    def _record_client_value(self, prefix: str, client: vf.ClientConfig, value: float) -> None:
        self._record_value(f"{prefix}/{client_metric_label(client)}", value)

    def _record_group_value(self, name: str, group: GroupState, value: float) -> None:
        self._record_value(name, value)
        if group.pinned_client is not None:
            self._record_client_value(name, group.pinned_client, value)

    def _record_contributing_rollouts(
        self,
        group: GroupState,
        rollout_info: InflightRequest,
        valid_rollouts: list[vf.RolloutOutput],
        request_wall_seconds: float,
    ) -> None:
        if not valid_rollouts:
            return

        prompt_tokens = sum(get_prompt_len(rollout) for rollout in valid_rollouts)
        completion_tokens = sum(get_completion_len(rollout) for rollout in valid_rollouts)
        seq_tokens = sum(get_seq_len(rollout) for rollout in valid_rollouts)

        group.completed_request_count += 1
        group.prompt_tokens += prompt_tokens
        group.completion_tokens += completion_tokens
        group.seq_tokens += seq_tokens
        group.request_wall_seconds.append(request_wall_seconds)
        if request_wall_seconds >= group.slowest_request_wall_seconds:
            group.slowest_request_wall_seconds = request_wall_seconds
            group.slowest_request_id = rollout_info.request_id
            group.slowest_request_client = rollout_info.client_config
        if group.example_fingerprint is not None:
            self.completion_tokens_by_fingerprint[group.example_fingerprint].append(float(completion_tokens))
        self.completion_tokens_by_env[rollout_info.env_name].append(float(completion_tokens))
        identity = self._client_identity(rollout_info.client_config)
        self.completion_tokens_by_client[identity].append(float(completion_tokens))
        self.last_completion_tokens_by_client[identity] = float(completion_tokens)

        self._record_value("rollout_request_prompt_tokens", float(prompt_tokens))
        self._record_value("rollout_request_completion_tokens", float(completion_tokens))
        self._record_value("rollout_request_seq_tokens", float(seq_tokens))
        self._record_client_value("rollout_request_prompt_tokens", rollout_info.client_config, float(prompt_tokens))
        self._record_client_value(
            "rollout_request_completion_tokens", rollout_info.client_config, float(completion_tokens)
        )
        self._record_client_value("rollout_request_seq_tokens", rollout_info.client_config, float(seq_tokens))

    def _record_completed_group_attribution(
        self,
        group: GroupState,
        group_wall_seconds: float,
        group_tail_seconds: float,
    ) -> None:
        self._record_group_value("rollout_group_wall_seconds", group, group_wall_seconds)
        self._record_group_value("rollout_group_tail_seconds", group, group_tail_seconds)
        self._record_group_value("rollout_group_prompt_tokens", group, float(group.prompt_tokens))
        self._record_group_value("rollout_group_completion_tokens", group, float(group.completion_tokens))
        self._record_group_value("rollout_group_seq_tokens", group, float(group.seq_tokens))
        self._record_group_value("rollout_group_completed_request_count", group, float(group.completed_request_count))
        self._record_group_value(
            "rollout_group_slowest_request_wall_seconds",
            group,
            group.slowest_request_wall_seconds,
        )
        if group.request_wall_seconds:
            request_wall_spread = max(group.request_wall_seconds) - min(group.request_wall_seconds)
            self._record_group_value("rollout_group_request_wall_spread_seconds", group, request_wall_spread)
        if group.first_dispatch_at is not None:
            self._record_group_value(
                "rollout_group_time_to_first_dispatch_seconds",
                group,
                group.first_dispatch_at - group.created_at,
            )
        if group.first_completion_at is not None:
            self._record_group_value(
                "rollout_group_time_to_first_completion_seconds",
                group,
                group.first_completion_at - group.created_at,
            )
        if group.slowest_request_client is not None:
            self._record_client_count("rollout_group_slowest_request_client", group.slowest_request_client)

    async def _select_least_loaded_client(self) -> vf.ClientConfig:
        """Select the client with the fewest in-flight tasks.

        Uses (api_base_url, dp_rank) as identity rather than client_idx so that
        load tracking survives elastic pool refreshes (which reassign indices).
        """
        clients = self.inference_pool.train_clients
        while not clients:
            await asyncio.sleep(1)
            clients = self.inference_pool.train_clients
        inflight = Counter(self._client_identity(info.client_config) for info in self.inflight_requests.values())
        return min(clients, key=lambda c: inflight[self._client_identity(c)])

    async def _get_train_clients(self) -> list[vf.ClientConfig]:
        clients = self.inference_pool.train_clients
        while not clients:
            await asyncio.sleep(1)
            clients = self.inference_pool.train_clients
        return clients

    def _oldest_inflight_seconds(self) -> float:
        now = time.perf_counter()
        ages = [
            now - info.request_started_at for info in self.inflight_requests.values() if info.request_started_at > 0
        ]
        return max(ages, default=0.0)

    def _candidate_stats(self, clients: list[vf.ClientConfig]) -> dict[ClientIdentity, CandidateStats]:
        endpoint_metrics = self.inference_pool.get_client_metrics()
        endpoint_client_counts = Counter(endpoint_label_from_url(client.api_base_url) for client in clients)
        inflight_predicted_completion_tokens: Counter[ClientIdentity] = Counter()
        for info in self.inflight_requests.values():
            if info.predicted_completion_tokens is None:
                continue
            inflight_predicted_completion_tokens[self._client_identity(info.client_config)] += (
                info.predicted_completion_tokens
            )
        stats: dict[ClientIdentity, CandidateStats] = {}
        for client in clients:
            identity = self._client_identity(client)
            wall_times = self.request_wall_seconds_by_client.get(identity, [])
            completion_tokens = self.completion_tokens_by_client.get(identity, [])
            group_wall_times = self.group_wall_seconds_by_client.get(identity, [])
            tail_times = self.group_tail_seconds_by_client.get(identity, [])
            off_policy_steps = self.off_policy_steps_by_client.get(identity, [])
            endpoint_label = endpoint_label_from_url(client.api_base_url)
            endpoint_metric_snapshot = dict(endpoint_metrics.get(endpoint_label, {}))
            metrics_available = float(bool(endpoint_metric_snapshot))
            endpoint_metric_snapshot["metrics_available"] = metrics_available
            endpoint_metric_snapshot["metrics_scope_dp_rank_precise"] = float(
                metrics_available > 0.0 and endpoint_client_counts[endpoint_label] == 1
            )
            endpoint_metric_snapshot["metrics_scope_base_url_client_count"] = float(
                endpoint_client_counts[endpoint_label]
            )
            stats[identity] = CandidateStats(
                completed_rollouts=self.completed_rollouts_by_client[identity],
                cancelled_rollouts=self.cancelled_rollouts_by_client[identity],
                request_wall_seconds_mean=sum(wall_times) / len(wall_times) if wall_times else None,
                request_wall_seconds_last=self.last_request_wall_seconds_by_client.get(identity),
                completion_tokens_mean=(sum(completion_tokens) / len(completion_tokens) if completion_tokens else None),
                completion_tokens_last=self.last_completion_tokens_by_client.get(identity),
                group_wall_seconds_mean=(sum(group_wall_times) / len(group_wall_times) if group_wall_times else None),
                group_wall_seconds_last=self.last_group_wall_seconds_by_client.get(identity),
                group_tail_seconds_mean=sum(tail_times) / len(tail_times) if tail_times else None,
                group_tail_seconds_last=self.last_group_tail_seconds_by_client.get(identity),
                off_policy_steps_mean=(sum(off_policy_steps) / len(off_policy_steps) if off_policy_steps else None),
                off_policy_steps_last=self.last_off_policy_steps_by_client.get(identity),
                inflight_predicted_completion_tokens=float(inflight_predicted_completion_tokens[identity]),
                endpoint_metrics=endpoint_metric_snapshot,
            )
        return stats

    def _predict_completion_tokens(self, example_fingerprint: str, env_name: str | None) -> tuple[float | None, str]:
        completion_tokens = self.completion_tokens_by_fingerprint.get(example_fingerprint)
        if completion_tokens:
            return sum(completion_tokens) / len(completion_tokens), "fingerprint"

        if env_name is not None:
            env_completion_tokens = self.completion_tokens_by_env.get(env_name)
            if env_completion_tokens:
                return sum(env_completion_tokens) / len(env_completion_tokens), "env"

            max_completion_tokens = self._max_completion_tokens_for_env(env_name)
            cold_start_ratio = request_picker_long_output_cold_start_ratio(self.request_picker)
            if max_completion_tokens is not None and cold_start_ratio > 0:
                return cold_start_ratio * max_completion_tokens, "cold_start"

        return None, "none"

    def _ensure_group_prediction(self, group: GroupState, env_name: str | None = None) -> None:
        if group.example_fingerprint is None:
            group.example_fingerprint = _example_fingerprint(group.example)
        if group.predicted_completion_tokens is None:
            resolved_env_name = env_name or group.example.get("env_name")
            group.predicted_completion_tokens, group.completion_prediction_source = self._predict_completion_tokens(
                group.example_fingerprint, resolved_env_name
            )

    def _max_completion_tokens_for_env(self, env_name: str) -> int | None:
        return self.max_completion_tokens_by_env.get(env_name, self.default_max_completion_tokens)

    async def _select_request_client(
        self, group_id: int, group: GroupState, env_name: str, cache_salt: str
    ) -> tuple[vf.ClientConfig | None, int]:
        clients = await self._get_train_clients()
        inflight = Counter(self._client_identity(info.client_config) for info in self.inflight_requests.values())
        self._ensure_group_prediction(group, env_name)
        context = RequestPickContext(
            env_name=env_name,
            group_id=group_id,
            model_name=self.model_name,
            step=self.step,
            ckpt_step=self.ckpt_step,
            cache_salt=cache_salt,
            group_age_seconds=time.perf_counter() - group.created_at,
            rollouts_to_schedule=group.rollouts_to_schedule,
            completed_rollouts=len(group.completed_rollouts),
            max_off_policy_level=self.max_off_policy_level,
            oldest_inflight_seconds=self._oldest_inflight_seconds(),
            example_fingerprint=group.example_fingerprint,
            predicted_completion_tokens=group.predicted_completion_tokens,
            max_completion_tokens=self._max_completion_tokens_for_env(env_name),
        )
        candidate_stats = self._candidate_stats(clients)

        if isinstance(self.request_picker, DirectRequestPicker):
            start = time.perf_counter()
            client = min(clients, key=lambda c: inflight[self._client_identity(c)])
            latency_seconds = time.perf_counter() - start
            attempts = 1
            selected_inflight = inflight[self._client_identity(client)]
            selected_score = float(selected_inflight)
            selected_score_components = None
            score_component_stats = None
        else:
            result = await select_with_metrics(
                self.request_picker,
                clients,
                inflight,
                context,
                candidate_stats,
            )
            client = result.client
            latency_seconds = result.latency_seconds
            attempts = result.attempts
            selected_inflight = result.selected_inflight
            selected_score = result.selected_score
            selected_score_components = result.selected_score_components
            score_component_stats = result.score_component_stats

        self._record_value("request_picker_latency_seconds", latency_seconds)
        self._record_value("request_picker_selected_inflight", float(selected_inflight))
        self._record_value("request_picker_candidate_count", float(len(clients)))
        self._record_value("request_picker_attempts", float(attempts))
        if selected_score is not None:
            self._record_value("request_picker_selected_score", selected_score)
        if selected_score_components is not None:
            for component, value in selected_score_components.items():
                self._record_value(f"request_picker_selected_score_component/{component}", value)
        if score_component_stats is not None:
            for component_stat, value in score_component_stats.items():
                self._record_value(f"request_picker_score_component/{component_stat}", value)
        selected_metrics = candidate_stats.get(self._client_identity(client), CandidateStats()).endpoint_metrics or {}
        self._record_client_value(
            "scheduler/client_metrics_available",
            client,
            selected_metrics.get("metrics_available", 0.0),
        )
        self._record_client_value(
            "scheduler/client_metrics_dp_rank_precise",
            client,
            selected_metrics.get("metrics_scope_dp_rank_precise", 0.0),
        )
        self._record_client_value(
            "scheduler/client_metrics_base_url_client_count",
            client,
            selected_metrics.get("metrics_scope_base_url_client_count", 0.0),
        )
        self._record_client_count("scheduler/selected_client", client)
        self._record_client_value("scheduler/inflight_at_selection", client, float(selected_inflight))
        self.logger.debug(
            "Selected rollout client "
            f"group_id={group_id} env={env_name} client_idx={client.client_idx} "
            f"base_url={client.api_base_url} dp_rank={client.extra_headers.get('X-data-parallel-rank')} "
            f"inflight={selected_inflight} picker_latency={latency_seconds:.6f}s"
        )
        return client, selected_inflight

    async def drop_group(self, group_id: int) -> int:
        """Drop a group and cancel any remaining in-flight rollouts for it. Returns the number of cancelled rollouts."""
        tasks_to_cancel = []
        rollout_count = 0
        for task, info in list(self.inflight_requests.items()):
            if info.group_id != group_id:
                continue
            self.inflight_requests.pop(task, None)
            tasks_to_cancel.append(task)
            rollout_count += info.rollout_count
            self.cancelled_rollouts_by_client[self._client_identity(info.client_config)] += info.rollout_count
            self._record_client_count("scheduler/cancelled_rollouts", info.client_config, info.rollout_count)
        self.groups.pop(group_id, None)
        await safe_cancel_all(tasks_to_cancel)
        return rollout_count

    async def schedule_rollout(self, group_id: int) -> bool:
        """Asynchronously schedules a rollout request (or a group request for group-scoring envs)."""
        dispatch_start = time.perf_counter()
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        group = self.groups.get(group_id)
        if group is None or group.rollouts_to_schedule <= 0:
            return False

        env_name = group.example["env_name"]
        env = self.train_envs.get(env_name)
        cache_salt = str(self.ckpt_step)
        selected_inflight = 0
        if group.pinned_client is not None:
            client_config = group.pinned_client
        else:
            client_config, selected_inflight = await self._select_request_client(group_id, group, env_name, cache_salt)
            if client_config is None:
                return False
            if group_id not in self.groups:
                return False
            group.pinned_client = client_config

        if group.first_dispatch_at is None:
            group.first_dispatch_at = time.perf_counter()

        request_id = self.next_request_id
        self.next_request_id += 1
        request_started_at = time.perf_counter()
        dispatch_wait_seconds = request_started_at - dispatch_start
        group_age_at_dispatch_seconds = request_started_at - group.created_at
        self._record_value("rollout_dispatch_wait_seconds", dispatch_wait_seconds)
        self._record_client_value("rollout_dispatch_wait_seconds", client_config, dispatch_wait_seconds)
        self._record_value(
            "rollout_request_completion_prediction_available",
            1.0 if group.predicted_completion_tokens is not None else 0.0,
        )
        self._record_value(f"rollout_request_completion_prediction_source/{group.completion_prediction_source}", 1.0)
        if group.predicted_completion_tokens is not None:
            self._record_value("rollout_request_predicted_completion_tokens", group.predicted_completion_tokens)
            self._record_client_value(
                "rollout_request_predicted_completion_tokens", client_config, group.predicted_completion_tokens
            )
        self._record_value("rollout_request_group_age_at_dispatch_seconds", group_age_at_dispatch_seconds)
        self._record_client_value(
            "rollout_request_group_age_at_dispatch_seconds", client_config, group_age_at_dispatch_seconds
        )
        self._record_value("rollout_request_completed_rollouts_at_dispatch", float(len(group.completed_rollouts)))
        self._record_client_value(
            "rollout_request_completed_rollouts_at_dispatch", client_config, float(len(group.completed_rollouts))
        )
        if env.requires_group_scoring:
            rollout_count = group.rollouts_to_schedule
            group.rollouts_to_schedule = 0
            task = asyncio.create_task(
                env.run_group(
                    client=client_config,
                    example=group.example,
                    model_name=self.model_name,
                    rollouts_per_example=rollout_count,
                    cache_salt=cache_salt,
                )
            )
        else:
            rollout_count = 1
            group.rollouts_to_schedule -= 1
            task = asyncio.create_task(
                env.run_rollout(
                    client=client_config,
                    example=group.example,
                    model_name=self.model_name,
                    cache_salt=cache_salt,
                )
            )
        self.inflight_requests[task] = InflightRequest(
            off_policy_steps=0,
            client_config=client_config,
            env_name=env_name,
            group_id=group_id,
            rollout_count=rollout_count,
            round_id=group.current_round,
            request_id=request_id,
            request_started_at=request_started_at,
            dispatch_wait_seconds=dispatch_wait_seconds,
            client_inflight_at_selection=selected_inflight,
            example_fingerprint=group.example_fingerprint,
            predicted_completion_tokens=group.predicted_completion_tokens,
        )
        return True

    @property
    def inflight_rollout_count(self) -> int:
        return sum(info.rollout_count for info in self.inflight_requests.values())

    @property
    def inflight_sample_count(self) -> int:
        pending = sum(g.rollouts_to_schedule for g in self.groups.values())
        return self.inflight_rollout_count + pending

    async def _schedule_next_request(self) -> bool:
        remaining_capacity = self.max_inflight_rollouts - self.inflight_rollout_count

        if remaining_capacity <= 0:
            return False

        for group_id, group in self.groups.items():
            if group.rollouts_to_schedule <= 0:
                continue
            env = self.train_envs.get(group.example["env_name"])
            cost = group.rollouts_to_schedule if env.requires_group_scoring else 1
            if cost <= remaining_capacity:
                if await self.schedule_rollout(group_id=group_id):
                    return True

        if remaining_capacity < self.rollouts_per_example:
            return False

        example = self.buffer.sample_examples(n=1)[0]
        example_fingerprint = _example_fingerprint(example)
        predicted_completion_tokens, completion_prediction_source = self._predict_completion_tokens(
            example_fingerprint, example.get("env_name")
        )
        group_id = self.next_group_id
        self.next_group_id += 1
        self.groups[group_id] = GroupState(
            example=example,
            rollouts_to_schedule=self.rollouts_per_example,
            example_fingerprint=example_fingerprint,
            predicted_completion_tokens=predicted_completion_tokens,
            completion_prediction_source=completion_prediction_source,
        )
        if await self.schedule_rollout(group_id=group_id):
            return True
        self.groups.pop(group_id, None)
        return False

    async def _fill_inflight_requests(self) -> None:
        while await self._schedule_next_request():
            pass

    async def update_policy_loop(self):
        """Continuously checks for new policy checkpoints."""
        while True:
            await self.maybe_update_policy()
            await asyncio.sleep(1)

    def _compute_next_ckpt_step(self) -> int:
        latest_ckpt_step = get_latest_ckpt_step(get_broadcast_dir(self.config.output_dir)) or 0
        effective_max_async_level = self._effective_max_async_level()
        async_away_ckpt_step = max(self.step - effective_max_async_level, 0)
        if self.strict_async_level or effective_max_async_level > self.max_async_level:
            return async_away_ckpt_step
        return max(async_away_ckpt_step, latest_ckpt_step)

    async def _apply_policy_update(self, next_ckpt_step: int) -> None:
        effective_max_async_level = self._effective_max_async_level()
        async_away_ckpt_step = max(self.step - effective_max_async_level, 0)
        if next_ckpt_step == async_away_ckpt_step:
            self.logger.info(
                f"Orchestrator paused: waiting for trainer process to complete checkpoint {next_ckpt_step} "
                f"(>{effective_max_async_level} step(s) ahead). Training is progressing normally."
            )
            self.checkpoint_ready.clear()
            wait_for_ckpt_start_time = time.perf_counter()
            await wait_for_path(get_step_path(get_broadcast_dir(self.config.output_dir), next_ckpt_step) / "STABLE")
            self.wait_for_ckpt_time = time.perf_counter() - wait_for_ckpt_start_time
            self.logger.info(
                f"Orchestrator resumed: checkpoint {next_ckpt_step} ready (after {self.wait_for_ckpt_time:.2f}s)"
            )

        self.logger.debug(
            f"Got new policy with step {next_ckpt_step}. Updating weights and cancelling old rollout requests."
        )

        self.inflight_rollouts_at_pause = self.inflight_rollout_count
        self.oldest_off_policy_at_pause = self.max_off_policy_level
        update_weights_start_time = time.perf_counter()
        weights_path = get_step_path(get_broadcast_dir(self.config.output_dir), next_ckpt_step)
        update_metrics = await self.inference_pool.update_weights(
            weights_path, lora_name=self.lora_name, step=next_ckpt_step
        )
        self.last_update_metrics = update_metrics or {}
        self.update_weights_time = time.perf_counter() - update_weights_start_time
        self.logger.debug(f"Updated weights to step {next_ckpt_step} in {self.update_weights_time:.2f}s")

        self.ckpt_step = next_ckpt_step
        if self.lora_name is not None:
            self.model_name = self.lora_name
            self.inference_pool.update_model_name(self.model_name)

        self.checkpoint_ready.set()
        await self._update_off_policy()

    async def _get_or_start_policy_update_task(self, next_ckpt_step: int) -> asyncio.Task:
        async with self.policy_update_lock:
            task = self.inflight_policy_update_task
            if task is not None and not task.done():
                return task

            task = asyncio.create_task(self._apply_policy_update(next_ckpt_step))
            self.inflight_policy_update_task = task

            def _clear_inflight_policy_update(done_task: asyncio.Task) -> None:
                if self.inflight_policy_update_task is done_task:
                    self.inflight_policy_update_task = None

            task.add_done_callback(_clear_inflight_policy_update)
            return task

    async def maybe_update_policy(self):
        """Updates the policy to the latest available checkpoint. Aborts rollout requests that are older than the max retention steps."""
        if not self.enable_policy_updates:
            self.ckpt_step = self.step
            self.checkpoint_ready.set()
            return

        while True:
            next_ckpt_step = self._compute_next_ckpt_step()
            if next_ckpt_step <= self.ckpt_step:
                return

            task = await self._get_or_start_policy_update_task(next_ckpt_step)
            await asyncio.shield(task)

    async def _update_off_policy(self) -> None:
        stale_group_ids = {
            info.group_id
            for info in self.inflight_requests.values()
            if info.group_id is not None and info.off_policy_steps >= self.max_off_policy_steps
        }
        tasks_to_increment = [
            task
            for task, info in list(self.inflight_requests.items())
            if info.group_id is None or info.group_id not in stale_group_ids
        ]

        counts = await asyncio.gather(*(self.drop_group(gid) for gid in stale_group_ids))
        removed = sum(counts)
        for task in tasks_to_increment:
            info = self.inflight_requests.get(task)
            if info is None:
                continue
            info.off_policy_steps += 1

        self.cancelled_rollouts_count += removed
        if removed:
            self.logger.warning(
                f"Cancelled {removed} old rollout requests (will refill naturally). "
                f"Consider increasing max_off_policy_steps to avoid this."
            )

    async def generate_batch(self, step: int) -> list[vf.RolloutOutput]:
        """Continuously generates a batch of rollouts."""
        self.step = step

        if self.enable_policy_updates:
            # Cancel the previous update policy task to avoid concurrent updates
            if self.update_policy_task is not None:
                await safe_cancel(self.update_policy_task)

            # Manually check the async barrier before starting the step, then re-create the update policy loop
            # This ensures that we respect max_async_level, while still listening for policy updates mid-step
            await self.maybe_update_policy()
            self.update_policy_task = asyncio.create_task(self.update_policy_loop())
        else:
            self.ckpt_step = step
            self.checkpoint_ready.set()

        batch_start_time = time.perf_counter()
        self._last_refill_end_time = None

        self.logger.debug("Starting to generate batch rollouts")

        batch_rollouts: list[vf.RolloutOutput] = []
        batch_progress = 0
        pbar = ProgressTracker(
            total=self.batch_target, desc="Generating rollouts (train)", json_logging=self.json_logging, step=step
        )

        while batch_progress < self.batch_target:
            refill_start = time.perf_counter()
            if self._last_refill_end_time is not None:
                self._record_value("scheduler_refill_gap_seconds", refill_start - self._last_refill_end_time)
            await self._fill_inflight_requests()
            self._last_refill_end_time = time.perf_counter()
            inflight_tasks = list(self.inflight_requests.keys())

            finished_tasks, _ = await asyncio.wait(
                inflight_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            await self.checkpoint_ready.wait()

            for finished_task in finished_tasks:
                if batch_progress >= self.batch_target:
                    break

                rollout_info = self.inflight_requests.pop(finished_task, None)
                if rollout_info is None:
                    continue

                group_id = rollout_info.group_id
                env_name = rollout_info.env_name
                request_wall_seconds = time.perf_counter() - rollout_info.request_started_at
                identity = self._client_identity(rollout_info.client_config)
                self._record_value("rollout_request_wall_seconds", request_wall_seconds)
                self._record_client_value(
                    "rollout_request_wall_seconds", rollout_info.client_config, request_wall_seconds
                )
                self.request_wall_seconds_by_client[identity].append(request_wall_seconds)
                self.last_request_wall_seconds_by_client[identity] = request_wall_seconds
                aggregation_start = time.perf_counter()

                try:
                    group = self.groups.get(group_id)
                    if group is None:
                        self._record_value(
                            "rollout_response_aggregation_seconds", time.perf_counter() - aggregation_start
                        )
                        continue

                    env = self.train_envs.get(env_name)
                    result = finished_task.result()
                    rollouts: list[vf.RolloutOutput] = result if isinstance(result, list) else [result]
                    self.total_rollouts_by_env[env_name] += len(rollouts)

                    # Check for empty/errored rollouts and reschedule
                    valid_rollouts = []
                    has_failures = False
                    last_failure_reason: str | None = None
                    for rollout in rollouts:
                        if rollout["error"] is not None:
                            self.errored_rollouts_by_env[env_name] += 1
                            has_failures = True
                            last_failure_reason = rollout["error"]["error_chain_repr"]
                            self.logger.warning(
                                f"Rollout error in group {group_id} ({env_name}), re-scheduling "
                                f"({len(group.completed_rollouts)}/{self.rollouts_per_example} complete): "
                                f"{last_failure_reason}"
                            )
                        elif len(rollout["trajectory"]) == 0:
                            self.empty_rollouts_by_env[env_name] += 1
                            has_failures = True
                            last_failure_reason = "empty trajectory"
                            self.logger.warning(
                                f"Empty trajectory in group {group_id} ({env_name}), re-scheduling "
                                f"({len(group.completed_rollouts)}/{self.rollouts_per_example} complete)"
                            )
                        else:
                            rollout["env_name"] = env_name
                            valid_rollouts.append(rollout)
                    if valid_rollouts and group.first_completion_at is None:
                        group.first_completion_at = time.perf_counter()

                    if has_failures:
                        # Dedupe failures within the same dispatch round: an
                        # individual-scoring env dispatches N rollouts at once,
                        # so a single failed round can produce up to N failed
                        # tasks. We only count the round once.
                        if rollout_info.round_id > group.last_failed_round:
                            group.failed_attempts += 1
                            group.last_failed_round = rollout_info.round_id
                            group.current_round = rollout_info.round_id + 1
                        max_attempts = self.config.max_error_reschedule_attempts
                        if max_attempts is not None and group.failed_attempts >= max_attempts:
                            # Permanently-stuck group: drop it from this step and let the
                            # rest of the batch proceed. Avoids a single bad example (e.g.
                            # an agent rollout whose sandbox poll keeps timing out)
                            # blocking step progress forever.
                            self.dropped_groups_by_env[env_name] += 1
                            self.logger.warning(
                                f"Dropping group {group_id} ({env_name}) after {group.failed_attempts} "
                                f"failed dispatch rounds ({len(group.completed_rollouts)}/{self.rollouts_per_example} "
                                f"complete). Last failure: {last_failure_reason}. Set "
                                f"orchestrator.max_error_reschedule_attempts higher (or to None) "
                                f"to retry more aggressively."
                            )
                            await self.drop_group(group_id)
                            self._record_value(
                                "rollout_response_aggregation_seconds", time.perf_counter() - aggregation_start
                            )
                            continue

                    if has_failures and env.requires_group_scoring:
                        # Group scoring requires all rollouts — discard partial results, reschedule full group
                        group.completed_rollouts.clear()
                        group.rollouts_to_schedule = self.rollouts_per_example
                        self._record_value(
                            "rollout_response_aggregation_seconds", time.perf_counter() - aggregation_start
                        )
                        continue

                    # For individual scoring, reschedule only the failed ones
                    self._record_contributing_rollouts(
                        group,
                        rollout_info,
                        valid_rollouts,
                        request_wall_seconds,
                    )
                    group.rollouts_to_schedule += len(rollouts) - len(valid_rollouts)
                    group.completed_rollouts.extend(valid_rollouts)
                    if len(group.completed_rollouts) < self.rollouts_per_example:
                        self.completed_rollouts_by_client[identity] += len(valid_rollouts)
                        self._record_client_count(
                            "scheduler/completed_rollouts", rollout_info.client_config, len(valid_rollouts)
                        )
                        self._record_client_value(
                            "scheduler/off_policy_level_at_completion",
                            rollout_info.client_config,
                            float(rollout_info.off_policy_steps),
                        )
                        self.off_policy_steps_by_client[identity].append(float(rollout_info.off_policy_steps))
                        self.last_off_policy_steps_by_client[identity] = float(rollout_info.off_policy_steps)
                        self._record_value(
                            "rollout_response_aggregation_seconds", time.perf_counter() - aggregation_start
                        )
                        continue
                    group_completed_at = time.perf_counter()
                    group_wall_seconds = group_completed_at - group.created_at
                    self.group_wall_seconds_by_client[identity].append(group_wall_seconds)
                    self.last_group_wall_seconds_by_client[identity] = group_wall_seconds
                    if group.first_completion_at is not None:
                        group_tail_seconds = group_completed_at - group.first_completion_at
                    else:
                        group_tail_seconds = 0.0
                    self.group_tail_seconds_by_client[identity].append(group_tail_seconds)
                    self.last_group_tail_seconds_by_client[identity] = group_tail_seconds
                    completed_group = self.groups.pop(group_id)
                    self._record_completed_group_attribution(
                        completed_group,
                        group_wall_seconds,
                        group_tail_seconds,
                    )
                    completed_rollouts = completed_group.completed_rollouts
                    self.completed_rollouts_by_client[identity] += len(valid_rollouts)
                    self._record_client_count(
                        "scheduler/completed_rollouts", rollout_info.client_config, len(valid_rollouts)
                    )
                    self._record_client_value(
                        "scheduler/off_policy_level_at_completion",
                        rollout_info.client_config,
                        float(rollout_info.off_policy_steps),
                    )
                    self.off_policy_steps_by_client[identity].append(float(rollout_info.off_policy_steps))
                    self.last_off_policy_steps_by_client[identity] = float(rollout_info.off_policy_steps)

                except asyncio.CancelledError:
                    if group_id is not None:
                        await self.drop_group(group_id)
                    self._record_value("rollout_response_aggregation_seconds", time.perf_counter() - aggregation_start)
                    continue
                except Exception as e:
                    self.logger.warning(f"Rollout failed: {e}")
                    if group_id is not None:
                        await self.drop_group(group_id)
                    self._record_value("rollout_response_aggregation_seconds", time.perf_counter() - aggregation_start)
                    continue

                self.buffer.update(completed_rollouts)
                accepted_rollouts = self.buffer.sample_rollouts(n=self.rollouts_per_example)
                self._record_value("rollout_response_aggregation_seconds", time.perf_counter() - aggregation_start)

                batch_rollouts.extend(accepted_rollouts)
                progress_increment = self.get_batch_progress_increment(accepted_rollouts)
                batch_progress += progress_increment
                pbar.update(progress_increment)

        refill_start = time.perf_counter()
        if self._last_refill_end_time is not None:
            self._record_value("scheduler_refill_gap_seconds", refill_start - self._last_refill_end_time)
        await self._fill_inflight_requests()
        self._last_refill_end_time = time.perf_counter()

        batch_rollouts = self.finalize_batch_rollouts(batch_rollouts)
        pbar.close()
        self.last_batch_generation_time = time.perf_counter() - batch_start_time
        return batch_rollouts

    async def stop(self) -> None:
        await self.cancel_inflight_rollouts()
        if self.update_policy_task is not None:
            await safe_cancel(self.update_policy_task)
            self.update_policy_task = None
        if self.inflight_policy_update_task is not None:
            await safe_cancel(self.inflight_policy_update_task)
            self.inflight_policy_update_task = None
        await self.request_picker.aclose()

    @property
    def max_off_policy_level(self) -> int:
        steps = [info.off_policy_steps for info in self.inflight_requests.values()]
        if not steps:
            return 0
        return max(steps)

    @property
    def mean_off_policy_level(self) -> float:
        steps = [info.off_policy_steps for info in self.inflight_requests.values()]
        if not steps:
            return 0
        return sum(steps) / len(steps)

    @property
    def async_level(self) -> int:
        return self.step - self.ckpt_step

    def _consume_recorded_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for name, values in self.metric_values.items():
            if not values:
                continue
            metrics[name] = sum(values) / len(values)
            metrics[f"{name}/max"] = max(values)
            metrics[f"{name}/min"] = min(values)
            metrics[f"{name}/count"] = len(values)
        for name, value in self.metric_counts.items():
            metrics[name] = value
        self.metric_values.clear()
        self.metric_counts.clear()
        return metrics

    def get_metrics(self) -> dict[str, float]:
        total_rollouts = sum(self.total_rollouts_by_env.values())
        metrics = {
            "time/wait_for_ckpt": self.wait_for_ckpt_time,
            "time/update_weights": self.update_weights_time,
            "scheduler/inflight_rollouts_at_pause": self.inflight_rollouts_at_pause,
            "scheduler/oldest_off_policy_at_pause": self.oldest_off_policy_at_pause,
            "scheduler/async_level": self.async_level,
            "scheduler/inflight_rollouts": self.inflight_rollout_count,
            "scheduler/inflight_samples": self.inflight_sample_count,
            "scheduler/cancelled_rollouts": self.cancelled_rollouts_count,
            "empty_rollouts/all": sum(self.empty_rollouts_by_env.values()) / max(total_rollouts, 1),
            "errored_rollouts/all": sum(self.errored_rollouts_by_env.values()) / max(total_rollouts, 1),
            "dropped_groups/all": sum(self.dropped_groups_by_env.values()),
            "off_policy_level/all/max": self.max_off_policy_level,
            "off_policy_level/all/mean": self.mean_off_policy_level,
        }
        for env_name in self.total_rollouts_by_env:
            env_total = max(self.total_rollouts_by_env[env_name], 1)
            metrics[f"empty_rollouts/{env_name}"] = self.empty_rollouts_by_env.get(env_name, 0) / env_total
            metrics[f"errored_rollouts/{env_name}"] = self.errored_rollouts_by_env.get(env_name, 0) / env_total
        for env_name, count in self.dropped_groups_by_env.items():
            metrics[f"dropped_groups/{env_name}"] = count
        by_env: dict[str, list[int]] = {}
        for info in self.inflight_requests.values():
            by_env.setdefault(info.env_name, []).append(info.off_policy_steps)
        for env_name, steps in by_env.items():
            metrics[f"off_policy_level/{env_name}/max"] = max(steps)
            metrics[f"off_policy_level/{env_name}/mean"] = sum(steps) / len(steps)
        metrics.update(self.last_update_metrics)
        metrics.update(self._consume_recorded_metrics())
        self.cancelled_rollouts_count = 0
        self.empty_rollouts_by_env.clear()
        self.errored_rollouts_by_env.clear()
        self.total_rollouts_by_env.clear()
        self.dropped_groups_by_env.clear()
        self.last_update_metrics = {}
        self.completed_rollouts_by_client.clear()
        self.cancelled_rollouts_by_client.clear()

        # Add inference pool metrics (e.g. elastic pool server counts)
        metrics.update(self.inference_pool.get_metrics())

        self._log_instrumentation_metrics(metrics)

        return metrics

    def _log_instrumentation_metrics(self, metrics: dict[str, float]) -> None:
        instrumentation_metrics = {
            name: value
            for name, value in metrics.items()
            if name in SCHEDULER_INSTRUMENTATION_KEYS
            or any(name.startswith(prefix) for prefix in SCHEDULER_INSTRUMENTATION_PREFIXES)
        }
        if instrumentation_metrics:
            self.logger.info(
                "Scheduler instrumentation metrics: "
                + json.dumps(instrumentation_metrics, sort_keys=True, separators=(",", ":"))
            )
