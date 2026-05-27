from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from urllib.parse import urlparse

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
    request_picker_wave_minimax_size,
    request_picker_wave_overhang_limit,
    request_picker_wave_overhang_start_progress,
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

SUM_ENDPOINT_METRICS = {
    "completed_requests_per_s",
    "decode_throughput_tps",
    "num_requests_running",
    "num_requests_waiting",
    "prefill_throughput_tps",
}
MAX_ENDPOINT_METRICS = {
    "gpu_cache_usage_perc",
    "gpu_prefix_cache_hit_rate",
}


def endpoint_host_label_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or (parsed.netloc or parsed.path).split(":")[0]
    return host.replace(".", "_").replace("-", "_")


def merge_endpoint_metric_snapshots(snapshots: list[dict[str, float]]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for snapshot in snapshots:
        for name, value in snapshot.items():
            if not isinstance(value, (int, float)):
                continue
            if name in MAX_ENDPOINT_METRICS:
                merged[name] = max(merged.get(name, 0.0), float(value))
            elif name in SUM_ENDPOINT_METRICS:
                merged[name] = merged.get(name, 0.0) + float(value)
            else:
                merged[name] = float(value)
    return merged


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
    dispatch_wave_id: int = 0
    refill_wave_id: int = 0
    example_fingerprint: str | None = None
    predicted_completion_tokens: float | None = None
    completion_prediction_source: str = "none"


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
        student_inference: InferencePool,
        teacher_inference: InferencePool | None,
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
        self.rollouts_per_example = config.group_size
        self.max_inflight_rollouts = max_inflight_rollouts
        self.max_async_level = max_async_level
        self.max_off_policy_steps = max_off_policy_steps
        self.strict_async_level = strict_async_level
        self.enable_policy_updates = enable_policy_updates
        self.lora_name = lora_name
        self.student_inference = student_inference
        self.teacher_inference = teacher_inference
        if config.training_mode == "sft":
            assert teacher_inference is not None
            self.rollout_inference = teacher_inference
        else:
            self.rollout_inference = student_inference
        self.model_name = self.rollout_inference.model_name
        self.json_logging = config.log.json_logging

        self.request_picker = setup_request_picker(config.experimental.request_picker)

        group_scoring_envs = [env.name for env in train_envs if env.requires_group_scoring]
        if group_scoring_envs:
            self.logger.info(f"Group rollout scoring active for env(s): {', '.join(group_scoring_envs)}")

        # Track in-flight requests: task -> info
        self.inflight_requests: dict[asyncio.Task, InflightRequest] = {}

        # Track in-progress groups while rollouts are generated independently.
        self.next_group_id = 0
        self.next_request_id = 0
        self.next_dispatch_wave_id = 0
        self.next_refill_wave_id = 0
        self.current_refill_wave_id = 0
        self.step_dispatched_requests = 0
        self.step_completed_requests = 0
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
        max_async_level = getattr(self, "max_async_level", 1)
        weight_broadcast = getattr(self.config, "weight_broadcast", None)
        if weight_broadcast is None:
            return max_async_level
        final_step_async_level = (
            weight_broadcast.final_step_async_level
            if weight_broadcast.type == "nccl" and weight_broadcast.final_step_async_level is not None
            else max_async_level
        )
        max_steps = getattr(self.config, "max_steps", None)
        if max_steps is None or final_step_async_level <= max_async_level:
            return max_async_level

        last_broadcast_step = max(max_steps - final_step_async_level - 1, 0)
        return max(max_async_level, self.step - last_broadcast_step)

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
        if not hasattr(self, "metric_values"):
            self.metric_values = defaultdict(list)
        self.metric_values[name].append(value)

    def _record_count(self, name: str, value: int | float = 1) -> None:
        if not hasattr(self, "metric_counts"):
            self.metric_counts = Counter()
        self.metric_counts[name] += value

    def _record_client_count(self, prefix: str, client: vf.ClientConfig, value: int | float = 1) -> None:
        self._record_count(f"{prefix}/{client_metric_label(client)}", value)

    def _record_client_value(self, prefix: str, client: vf.ClientConfig, value: float) -> None:
        self._record_value(f"{prefix}/{client_metric_label(client)}", value)

    def _record_group_value(self, name: str, group: GroupState, value: float) -> None:
        self._record_value(name, value)
        if group.pinned_client is not None:
            self._record_client_value(name, group.pinned_client, value)

    def _log_replay_event(self, event: str, payload: dict) -> None:
        payload = {"event": event, "schema_version": 1, **payload}
        self.logger.info("Scheduler replay event: " + json.dumps(payload, sort_keys=True, separators=(",", ":")))

    def _client_replay_entry(
        self,
        client: vf.ClientConfig,
        inflight: Counter[ClientIdentity],
        candidate_stats: dict[ClientIdentity, CandidateStats],
        candidate_scores: dict[ClientIdentity, float] | None,
        candidate_score_components: dict[ClientIdentity, dict[str, float]] | None,
        selected_client: vf.ClientConfig,
        predicted_completion_tokens: float | None,
        completion_prediction_source: str,
    ) -> dict:
        identity = self._client_identity(client)
        stats = candidate_stats.get(identity, CandidateStats())
        metrics = stats.endpoint_metrics or {}
        return {
            "client_idx": client.client_idx,
            "client_label": client_metric_label(client),
            "dp_rank": client.extra_headers.get("X-data-parallel-rank"),
            "endpoint_label": endpoint_label_from_url(client.api_base_url),
            "selected": self._client_identity(client) == self._client_identity(selected_client),
            "running_count": inflight[identity],
            "score": (candidate_scores or {}).get(identity),
            "score_components": (candidate_score_components or {}).get(identity),
            "predicted_completion_tokens": predicted_completion_tokens,
            "completion_prediction_source": completion_prediction_source,
            "inflight_predicted_completion_tokens": stats.inflight_predicted_completion_tokens,
            "completion_tokens_mean": stats.completion_tokens_mean,
            "completion_tokens_last": stats.completion_tokens_last,
            "request_wall_seconds_mean": stats.request_wall_seconds_mean,
            "request_wall_seconds_last": stats.request_wall_seconds_last,
            "group_wall_seconds_mean": stats.group_wall_seconds_mean,
            "group_wall_seconds_last": stats.group_wall_seconds_last,
            "decode_throughput_tps": metrics.get("decode_throughput_tps"),
            "completed_requests_per_s": metrics.get("completed_requests_per_s"),
            "num_requests_running": metrics.get("num_requests_running"),
            "num_requests_waiting": metrics.get("num_requests_waiting"),
            "gpu_cache_usage_perc": metrics.get("gpu_cache_usage_perc"),
            "metrics_available": metrics.get("metrics_available", 0.0),
            "metrics_scope_endpoint_exact": metrics.get("metrics_scope_endpoint_exact", 0.0),
            "metrics_scope_host_match_count": metrics.get("metrics_scope_host_match_count", 0.0),
            "metrics_scope_dp_rank_precise": metrics.get("metrics_scope_dp_rank_precise", 0.0),
            "metrics_scope_base_url_client_count": metrics.get("metrics_scope_base_url_client_count", 0.0),
        }

    def _log_dispatch_replay_event(
        self,
        *,
        request_id: int,
        dispatch_wave_id: int,
        refill_wave_id: int,
        group_id: int,
        group: GroupState,
        env_name: str,
        client: vf.ClientConfig,
        clients: list[vf.ClientConfig],
        inflight: Counter[ClientIdentity],
        candidate_stats: dict[ClientIdentity, CandidateStats],
        selected_inflight: int,
        selected_score: float | None,
        selected_score_components: dict[str, float] | None,
        candidate_scores: dict[ClientIdentity, float] | None,
        candidate_score_components: dict[ClientIdentity, dict[str, float]] | None,
    ) -> None:
        self._log_replay_event(
            "dispatch",
            {
                "step": self.step,
                "ckpt_step": self.ckpt_step,
                "request_id": request_id,
                "group_id": group_id,
                "dispatch_wave_id": dispatch_wave_id,
                "refill_wave_id": refill_wave_id,
                "env_name": env_name,
                "example_fingerprint": group.example_fingerprint,
                "rollouts_to_schedule": group.rollouts_to_schedule,
                "completed_rollouts": len(group.completed_rollouts),
                "selected_client": client_metric_label(client),
                "selected_client_idx": client.client_idx,
                "selected_dp_rank": client.extra_headers.get("X-data-parallel-rank"),
                "selected_inflight": selected_inflight,
                "selected_score": selected_score,
                "selected_score_components": selected_score_components,
                "predicted_completion_tokens": group.predicted_completion_tokens,
                "completion_prediction_source": group.completion_prediction_source,
                "max_completion_tokens": self._max_completion_tokens_for_env(env_name),
                "candidates": [
                    self._client_replay_entry(
                        candidate,
                        inflight,
                        candidate_stats,
                        candidate_scores,
                        candidate_score_components,
                        client,
                        group.predicted_completion_tokens,
                        group.completion_prediction_source,
                    )
                    for candidate in clients
                ],
            },
        )

    def _log_completion_replay_event(
        self,
        *,
        rollout_info: InflightRequest,
        group: GroupState,
        valid_rollouts: list[vf.RolloutOutput],
        request_wall_seconds: float,
        group_closed: bool,
        group_wall_seconds: float | None = None,
        group_tail_seconds: float | None = None,
    ) -> None:
        self._log_replay_event(
            "completion",
            {
                "step": self.step,
                "ckpt_step": self.ckpt_step,
                "request_id": rollout_info.request_id,
                "group_id": rollout_info.group_id,
                "dispatch_wave_id": rollout_info.dispatch_wave_id,
                "refill_wave_id": rollout_info.refill_wave_id,
                "env_name": rollout_info.env_name,
                "selected_client": client_metric_label(rollout_info.client_config),
                "selected_client_idx": rollout_info.client_config.client_idx,
                "selected_dp_rank": rollout_info.client_config.extra_headers.get("X-data-parallel-rank"),
                "request_wall_seconds": request_wall_seconds,
                "actual_prompt_tokens": float(sum(get_prompt_len(rollout) for rollout in valid_rollouts)),
                "actual_completion_tokens": float(sum(get_completion_len(rollout) for rollout in valid_rollouts)),
                "actual_seq_tokens": float(sum(get_seq_len(rollout) for rollout in valid_rollouts)),
                "predicted_completion_tokens": rollout_info.predicted_completion_tokens,
                "completion_prediction_source": rollout_info.completion_prediction_source,
                "group_closed": group_closed,
                "group_wall_seconds": group_wall_seconds,
                "group_tail_seconds": group_tail_seconds,
                "group_completed_request_count": group.completed_request_count,
                "group_completion_tokens": group.completion_tokens,
                "group_slowest_request_id": group.slowest_request_id,
                "group_slowest_request_wall_seconds": group.slowest_request_wall_seconds,
                "group_slowest_request_client": (
                    client_metric_label(group.slowest_request_client)
                    if group.slowest_request_client is not None
                    else None
                ),
            },
        )

    def _group_predicted_completion_load(self, group: GroupState) -> float:
        if group.predicted_completion_tokens is not None:
            return group.predicted_completion_tokens
        max_completion_tokens = self._max_completion_tokens_for_env(group.example["env_name"])
        return float(max_completion_tokens or 1)

    def _endpoint_metrics_for_client(
        self,
        client: vf.ClientConfig,
        endpoint_metrics: dict[str, dict[str, float]],
    ) -> tuple[dict[str, float], float, float]:
        endpoint_label = endpoint_label_from_url(client.api_base_url)
        exact = endpoint_metrics.get(endpoint_label)
        if exact:
            return dict(exact), 1.0, 1.0

        host_label = endpoint_host_label_from_url(client.api_base_url)
        host_matches = [
            snapshot
            for label, snapshot in endpoint_metrics.items()
            if label == host_label or label.startswith(f"{host_label}_")
        ]
        if not host_matches:
            return {}, 0.0, 0.0
        return merge_endpoint_metric_snapshots(host_matches), 0.0, float(len(host_matches))

    def _wave_minimax_throughput_penalty(
        self,
        stats: CandidateStats,
        all_stats: list[CandidateStats],
        request_load: float,
    ) -> float:
        metrics = stats.endpoint_metrics or {}
        if not metrics:
            return 0.0

        def rate_deficit(metric_name: str) -> float:
            rate = metrics.get(metric_name)
            if rate is None:
                return 0.0
            max_rate = max(
                ((candidate.endpoint_metrics or {}).get(metric_name, 0.0) for candidate in all_stats), default=0.0
            )
            if max_rate <= 0:
                return 0.0
            return max(max_rate - rate, 0.0) / max_rate

        def decode_guardrail_deficit() -> float:
            ratio = getattr(self.request_picker, "decode_guardrail_ratio", 0.0)
            if not isinstance(ratio, (int, float)):
                return 0.0
            rate = metrics.get("decode_throughput_tps")
            if rate is None:
                return 0.0
            max_rate = max(
                ((candidate.endpoint_metrics or {}).get("decode_throughput_tps", 0.0) for candidate in all_stats),
                default=0.0,
            )
            if max_rate <= 0:
                return 0.0
            tolerated_rate = max_rate * max(1.0 - float(ratio), 0.0)
            if rate >= tolerated_rate:
                return 0.0
            return (tolerated_rate - rate) / max_rate

        decode_penalty = getattr(self.request_picker, "decode_guardrail_penalty", 0.0)
        completed_rps_weight = getattr(self.request_picker, "completed_rps_deficit_weight", 0.0)
        if not isinstance(decode_penalty, (int, float)):
            decode_penalty = 0.0
        if not isinstance(completed_rps_weight, (int, float)):
            completed_rps_weight = 0.0

        penalty = float(decode_penalty) * decode_guardrail_deficit()
        penalty += float(completed_rps_weight) * rate_deficit("completed_requests_per_s")
        return request_load * penalty

    def _wave_minimax_prime_proxy_penalty(
        self,
        stats: CandidateStats,
        all_stats: list[CandidateStats],
        request_load: float,
    ) -> float:
        completed_rps_weight = getattr(self.request_picker, "completed_rps_deficit_weight", 0.0)
        decode_penalty = getattr(self.request_picker, "decode_guardrail_penalty", 0.0)
        if not isinstance(completed_rps_weight, (int, float)):
            completed_rps_weight = 0.0
        if not isinstance(decode_penalty, (int, float)):
            decode_penalty = 0.0

        def latency_excess(value: float | None, values: list[float | None]) -> float:
            observed = [item for item in values if item is not None and item >= 0]
            if value is None or len(observed) < 2:
                return 0.0
            reference = min(observed)
            if value <= reference:
                return 0.0
            return (value - reference) / max(sum(observed) / len(observed), 1.0)

        def inverse_rate_deficit(value: float, values: list[float]) -> float:
            if value <= 0 or len(values) < 2:
                return 0.0
            best = max(values)
            if best <= 0:
                return 0.0
            return max(best - value, 0.0) / best

        def mean_or_latest(mean: float | None, latest: float | None) -> float | None:
            return latest if latest is not None else mean

        completed_counts = [float(candidate.completed_rollouts) for candidate in all_stats]
        completed_rate_penalty = inverse_rate_deficit(float(stats.completed_rollouts), completed_counts)
        wall_penalty = latency_excess(
            mean_or_latest(stats.request_wall_seconds_mean, stats.request_wall_seconds_last),
            [
                mean_or_latest(candidate.request_wall_seconds_mean, candidate.request_wall_seconds_last)
                for candidate in all_stats
            ],
        )
        group_wall_penalty = latency_excess(
            mean_or_latest(stats.group_wall_seconds_mean, stats.group_wall_seconds_last),
            [
                mean_or_latest(candidate.group_wall_seconds_mean, candidate.group_wall_seconds_last)
                for candidate in all_stats
            ],
        )
        completion_length_penalty = latency_excess(
            mean_or_latest(stats.completion_tokens_mean, stats.completion_tokens_last),
            [
                mean_or_latest(candidate.completion_tokens_mean, candidate.completion_tokens_last)
                for candidate in all_stats
            ],
        )

        proxy_penalty = float(completed_rps_weight) * completed_rate_penalty
        proxy_penalty += float(decode_penalty) * (wall_penalty + group_wall_penalty + completion_length_penalty)
        return request_load * proxy_penalty

    def _new_group_from_example(self, example: dict) -> tuple[int, GroupState]:
        example_fingerprint = _example_fingerprint(example)
        predicted_completion_tokens, completion_prediction_source = self._predict_completion_tokens(
            example_fingerprint, example.get("env_name")
        )
        group_id = self.next_group_id
        self.next_group_id += 1
        group = GroupState(
            example=example,
            rollouts_to_schedule=self.rollouts_per_example,
            example_fingerprint=example_fingerprint,
            predicted_completion_tokens=predicted_completion_tokens,
            completion_prediction_source=completion_prediction_source,
        )
        self.groups[group_id] = group
        return group_id, group

    def _candidate_wave_groups(self, remaining_capacity: int, wave_size: int) -> list[int]:
        group_ids: list[int] = []
        remaining = remaining_capacity
        for group_id, group in self.groups.items():
            if len(group_ids) >= wave_size or group.rollouts_to_schedule <= 0:
                continue
            env = self.train_envs.get(group.example["env_name"])
            cost = group.rollouts_to_schedule if env.requires_group_scoring else 1
            if cost <= remaining:
                group_ids.append(group_id)
                remaining -= cost

        while len(group_ids) < wave_size and remaining >= self.rollouts_per_example:
            example = self.buffer.sample_examples(n=1)[0]
            group_id, _ = self._new_group_from_example(example)
            group_ids.append(group_id)
            remaining -= self.rollouts_per_example

        return group_ids

    async def _assign_wave_minimax_clients(self, group_ids: list[int]) -> dict[int, vf.ClientConfig]:
        clients = await self._get_train_clients()
        inflight = Counter(self._client_identity(info.client_config) for info in self.inflight_requests.values())
        candidate_stats = self._candidate_stats(clients)
        projected_load = {
            self._client_identity(client): candidate_stats[
                self._client_identity(client)
            ].inflight_predicted_completion_tokens
            for client in clients
        }
        assigned_counts: Counter[ClientIdentity] = Counter()
        assignments: dict[int, vf.ClientConfig] = {}
        assignment_penalties: dict[int, float] = {}
        stats_list = list(candidate_stats.values())

        sorted_group_ids = sorted(
            group_ids,
            key=lambda group_id: self._group_predicted_completion_load(self.groups[group_id]),
            reverse=True,
        )
        for group_id in sorted_group_ids:
            group = self.groups[group_id]
            load = self._group_predicted_completion_load(group)
            client = min(
                clients,
                key=lambda candidate: (
                    projected_load[self._client_identity(candidate)]
                    + load
                    + self._wave_minimax_throughput_penalty(
                        candidate_stats[self._client_identity(candidate)],
                        stats_list,
                        load,
                    )
                    + self._wave_minimax_prime_proxy_penalty(
                        candidate_stats[self._client_identity(candidate)],
                        stats_list,
                        load,
                    ),
                    inflight[self._client_identity(candidate)] + assigned_counts[self._client_identity(candidate)],
                    candidate.client_idx,
                ),
            )
            identity = self._client_identity(client)
            assignment_penalties[group_id] = self._wave_minimax_throughput_penalty(
                candidate_stats[identity],
                stats_list,
                load,
            ) + self._wave_minimax_prime_proxy_penalty(
                candidate_stats[identity],
                stats_list,
                load,
            )
            projected_load[identity] += load
            assigned_counts[identity] += 1
            assignments[group_id] = client

        self._record_value("request_picker_wave_minimax_group_count", float(len(group_ids)))
        if group_ids:
            self._record_value(
                "request_picker_wave_minimax_predicted_completion_tokens",
                sum(self._group_predicted_completion_load(self.groups[group_id]) for group_id in group_ids),
            )
        self._log_replay_event(
            "wave_assignment",
            {
                "step": self.step,
                "ckpt_step": self.ckpt_step,
                "refill_wave_id": self.current_refill_wave_id,
                "group_ids": group_ids,
                "assignments": [
                    {
                        "group_id": group_id,
                        "client_label": client_metric_label(assignments[group_id]),
                        "client_idx": assignments[group_id].client_idx,
                        "dp_rank": assignments[group_id].extra_headers.get("X-data-parallel-rank"),
                        "predicted_completion_tokens": self.groups[group_id].predicted_completion_tokens,
                        "completion_prediction_source": self.groups[group_id].completion_prediction_source,
                        "throughput_penalty": assignment_penalties[group_id],
                    }
                    for group_id in sorted_group_ids
                ],
            },
        )
        return assignments

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
        clients = self.rollout_inference.train_clients
        while not clients:
            await asyncio.sleep(1)
            clients = self.rollout_inference.train_clients
        inflight = Counter(self._client_identity(info.client_config) for info in self.inflight_requests.values())
        return min(clients, key=lambda c: inflight[self._client_identity(c)])

    async def _get_train_clients(self) -> list[vf.ClientConfig]:
        clients = self.rollout_inference.train_clients
        while not clients:
            await asyncio.sleep(1)
            clients = self.rollout_inference.train_clients
        return clients

    def _oldest_inflight_seconds(self) -> float:
        now = time.perf_counter()
        ages = [
            now - info.request_started_at for info in self.inflight_requests.values() if info.request_started_at > 0
        ]
        return max(ages, default=0.0)

    def _candidate_stats(self, clients: list[vf.ClientConfig]) -> dict[ClientIdentity, CandidateStats]:
        get_client_metrics = getattr(self.rollout_inference, "get_client_metrics", None)
        endpoint_metrics = get_client_metrics() if get_client_metrics is not None else {}
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
            wall_times = getattr(self, "request_wall_seconds_by_client", {}).get(identity, [])
            completion_tokens = getattr(self, "completion_tokens_by_client", {}).get(identity, [])
            group_wall_times = getattr(self, "group_wall_seconds_by_client", {}).get(identity, [])
            tail_times = getattr(self, "group_tail_seconds_by_client", {}).get(identity, [])
            off_policy_steps = getattr(self, "off_policy_steps_by_client", {}).get(identity, [])
            endpoint_label = endpoint_label_from_url(client.api_base_url)
            endpoint_metric_snapshot, endpoint_exact_match, host_match_count = self._endpoint_metrics_for_client(
                client, endpoint_metrics
            )
            metrics_available = float(bool(endpoint_metric_snapshot))
            endpoint_metric_snapshot["metrics_available"] = metrics_available
            endpoint_metric_snapshot["metrics_scope_endpoint_exact"] = endpoint_exact_match
            endpoint_metric_snapshot["metrics_scope_host_match_count"] = host_match_count
            endpoint_metric_snapshot["metrics_scope_dp_rank_precise"] = float(
                metrics_available > 0.0 and endpoint_exact_match > 0.0 and endpoint_client_counts[endpoint_label] == 1
            )
            endpoint_metric_snapshot["metrics_scope_base_url_client_count"] = float(
                endpoint_client_counts[endpoint_label]
            )
            stats[identity] = CandidateStats(
                completed_rollouts=getattr(self, "completed_rollouts_by_client", Counter())[identity],
                cancelled_rollouts=getattr(self, "cancelled_rollouts_by_client", Counter())[identity],
                request_wall_seconds_mean=sum(wall_times) / len(wall_times) if wall_times else None,
                request_wall_seconds_last=getattr(self, "last_request_wall_seconds_by_client", {}).get(identity),
                completion_tokens_mean=(sum(completion_tokens) / len(completion_tokens) if completion_tokens else None),
                completion_tokens_last=getattr(self, "last_completion_tokens_by_client", {}).get(identity),
                group_wall_seconds_mean=(sum(group_wall_times) / len(group_wall_times) if group_wall_times else None),
                group_wall_seconds_last=getattr(self, "last_group_wall_seconds_by_client", {}).get(identity),
                group_tail_seconds_mean=sum(tail_times) / len(tail_times) if tail_times else None,
                group_tail_seconds_last=getattr(self, "last_group_tail_seconds_by_client", {}).get(identity),
                off_policy_steps_mean=(sum(off_policy_steps) / len(off_policy_steps) if off_policy_steps else None),
                off_policy_steps_last=getattr(self, "last_off_policy_steps_by_client", {}).get(identity),
                inflight_predicted_completion_tokens=float(inflight_predicted_completion_tokens[identity]),
                endpoint_metrics=endpoint_metric_snapshot,
            )
        return stats

    def _predict_completion_tokens(self, example_fingerprint: str, env_name: str | None) -> tuple[float | None, str]:
        completion_tokens = getattr(self, "completion_tokens_by_fingerprint", {}).get(example_fingerprint)
        if completion_tokens:
            return sum(completion_tokens) / len(completion_tokens), "fingerprint"

        if env_name is not None:
            env_completion_tokens = getattr(self, "completion_tokens_by_env", {}).get(env_name)
            if env_completion_tokens:
                return sum(env_completion_tokens) / len(env_completion_tokens), "env"

            max_completion_tokens = self._max_completion_tokens_for_env(env_name)
            request_picker = getattr(self, "request_picker", DirectRequestPicker())
            cold_start_ratio = request_picker_long_output_cold_start_ratio(request_picker)
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
        return getattr(self, "max_completion_tokens_by_env", {}).get(
            env_name, getattr(self, "default_max_completion_tokens", None)
        )

    async def _select_request_client(
        self,
        group_id: int,
        group: GroupState,
        env_name: str,
        cache_salt: str,
        request_id: int,
        dispatch_wave_id: int,
        refill_wave_id: int,
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

        request_picker = getattr(self, "request_picker", DirectRequestPicker())
        if isinstance(request_picker, DirectRequestPicker):
            start = time.perf_counter()
            client = min(clients, key=lambda c: inflight[self._client_identity(c)])
            latency_seconds = time.perf_counter() - start
            attempts = 1
            selected_inflight = inflight[self._client_identity(client)]
            selected_score = float(selected_inflight)
            selected_score_components = None
            score_component_stats = None
            candidate_scores = None
            candidate_score_components = None
        else:
            result = await select_with_metrics(
                request_picker,
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
            candidate_scores = result.candidate_scores
            candidate_score_components = result.candidate_score_components

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
        self._log_dispatch_replay_event(
            request_id=request_id,
            dispatch_wave_id=dispatch_wave_id,
            refill_wave_id=refill_wave_id,
            group_id=group_id,
            group=group,
            env_name=env_name,
            client=client,
            clients=clients,
            inflight=inflight,
            candidate_stats=candidate_stats,
            selected_inflight=selected_inflight,
            selected_score=selected_score,
            selected_score_components=selected_score_components,
            candidate_scores=candidate_scores,
            candidate_score_components=candidate_score_components,
        )
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
        request_id = getattr(self, "next_request_id", 0)
        dispatch_wave_id = getattr(self, "next_dispatch_wave_id", 0)
        refill_wave_id = getattr(self, "current_refill_wave_id", 0)
        selected_inflight = 0
        if group.pinned_client is not None:
            client_config = group.pinned_client
            clients = await self._get_train_clients()
            inflight = Counter(self._client_identity(info.client_config) for info in self.inflight_requests.values())
            selected_inflight = inflight[self._client_identity(client_config)]
            self._ensure_group_prediction(group, env_name)
            self._log_dispatch_replay_event(
                request_id=request_id,
                dispatch_wave_id=dispatch_wave_id,
                refill_wave_id=refill_wave_id,
                group_id=group_id,
                group=group,
                env_name=env_name,
                client=client_config,
                clients=clients,
                inflight=inflight,
                candidate_stats=self._candidate_stats(clients),
                selected_inflight=selected_inflight,
                selected_score=float(selected_inflight),
                selected_score_components=None,
                candidate_scores=None,
                candidate_score_components=None,
            )
        else:
            client_config, selected_inflight = await self._select_request_client(
                group_id,
                group,
                env_name,
                cache_salt,
                request_id,
                dispatch_wave_id,
                refill_wave_id,
            )
            if client_config is None:
                return False
            if group_id not in self.groups:
                return False
            group.pinned_client = client_config
        self.next_request_id = request_id + 1
        self.next_dispatch_wave_id = dispatch_wave_id + 1

        if group.first_dispatch_at is None:
            group.first_dispatch_at = time.perf_counter()

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
            dispatch_wave_id=dispatch_wave_id,
            refill_wave_id=refill_wave_id,
            example_fingerprint=group.example_fingerprint,
            predicted_completion_tokens=group.predicted_completion_tokens,
            completion_prediction_source=group.completion_prediction_source,
        )
        self.step_dispatched_requests = getattr(self, "step_dispatched_requests", 0) + 1
        self._record_value("request_picker_step_dispatched_requests", float(self.step_dispatched_requests))
        self._record_value("request_picker_step_dispatch_overhang", float(getattr(self, "step_dispatch_overhang", 0)))
        return True

    @property
    def inflight_rollout_count(self) -> int:
        return sum(info.rollout_count for info in self.inflight_requests.values())

    @property
    def inflight_sample_count(self) -> int:
        pending = sum(g.rollouts_to_schedule for g in self.groups.values())
        return self.inflight_rollout_count + pending

    @property
    def step_dispatch_overhang(self) -> int:
        return max(self.step_dispatched_requests - self.step_completed_requests, 0)

    def _effective_wave_minimax_size(self, wave_size: int, batch_progress: int | None) -> int:
        limit = request_picker_wave_overhang_limit(self.request_picker)
        if limit <= 0:
            return wave_size

        progress_fraction = 0.0
        if batch_progress is not None and self.batch_target > 0:
            progress_fraction = min(max(batch_progress / self.batch_target, 0.0), 1.0)

        overhang = self.step_dispatch_overhang
        self._record_value("request_picker_wave_overhang", float(overhang))
        self._record_value("request_picker_wave_overhang_limit", float(limit))
        self._record_value("request_picker_wave_overhang_progress", progress_fraction)

        start_progress = request_picker_wave_overhang_start_progress(self.request_picker)
        if progress_fraction < start_progress:
            self._record_value("request_picker_wave_effective_size", float(wave_size))
            return wave_size

        remaining = limit - overhang
        if remaining <= 0:
            self._record_value("request_picker_wave_overhang_limited", 1.0)
            self._record_value("request_picker_wave_effective_size", 0.0)
            return 0

        effective_size = min(wave_size, remaining)
        self._record_value("request_picker_wave_overhang_limited", 1.0 if effective_size < wave_size else 0.0)
        self._record_value("request_picker_wave_effective_size", float(effective_size))
        return effective_size

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

        group_id, _ = self._new_group_from_example(self.buffer.sample_examples(n=1)[0])
        if await self.schedule_rollout(group_id=group_id):
            return True
        self.groups.pop(group_id, None)
        return False

    async def _fill_inflight_requests(self, batch_progress: int | None = None) -> None:
        self.current_refill_wave_id = self.next_refill_wave_id
        self.next_refill_wave_id += 1
        wave_size = request_picker_wave_minimax_size(self.request_picker)
        if wave_size:
            await self._fill_inflight_requests_wave_minimax(wave_size, batch_progress=batch_progress)
            return
        while await self._schedule_next_request():
            pass

    async def _fill_inflight_requests_wave_minimax(self, wave_size: int, batch_progress: int | None = None) -> None:
        while True:
            remaining_capacity = self.max_inflight_rollouts - self.inflight_rollout_count
            if remaining_capacity <= 0:
                return
            effective_wave_size = self._effective_wave_minimax_size(wave_size, batch_progress)
            if effective_wave_size <= 0:
                return
            group_ids = self._candidate_wave_groups(remaining_capacity, effective_wave_size)
            if not group_ids:
                return
            assignments = await self._assign_wave_minimax_clients(group_ids)
            scheduled_any = False
            for group_id in group_ids:
                group = self.groups.get(group_id)
                if group is None:
                    continue
                group.pinned_client = assignments[group_id]
                if await self.schedule_rollout(group_id=group_id):
                    scheduled_any = True
                elif group.rollouts_to_schedule > 0 and not group.completed_rollouts:
                    self.groups.pop(group_id, None)
            if not scheduled_any:
                return

    async def update_policy_loop(self):
        """Continuously checks for new policy checkpoints."""
        while True:
            await self.maybe_update_policy()
            await asyncio.sleep(1)

    def _compute_next_ckpt_step(self) -> int:
        latest_ckpt_step = get_latest_ckpt_step(get_broadcast_dir(self.config.output_dir)) or 0
        effective_max_async_level = self._effective_max_async_level()
        async_away_ckpt_step = max(self.step - effective_max_async_level, 0)
        max_async_level = getattr(self, "max_async_level", 1)
        if getattr(self, "strict_async_level", False) or effective_max_async_level > max_async_level:
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
        update_metrics = await self.student_inference.update_weights(
            weights_path, lora_name=self.lora_name, step=next_ckpt_step
        )
        self.last_update_metrics = update_metrics or {}
        self.update_weights_time = time.perf_counter() - update_weights_start_time
        self.logger.debug(f"Updated weights to step {next_ckpt_step} in {self.update_weights_time:.2f}s")

        self.ckpt_step = next_ckpt_step
        if self.lora_name is not None:
            self.student_inference.update_model_name(self.lora_name)
            if self.rollout_inference is self.student_inference:
                self.model_name = self.lora_name

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
        if not getattr(self, "enable_policy_updates", True):
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
        self.step_dispatched_requests = 0
        self.step_completed_requests = 0

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
            await self._fill_inflight_requests(batch_progress=batch_progress)
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
                        self.step_completed_requests += 1
                        self._record_value(
                            "request_picker_step_completed_requests", float(self.step_completed_requests)
                        )
                        self._record_value("request_picker_step_dispatch_overhang", float(self.step_dispatch_overhang))
                        self._log_completion_replay_event(
                            rollout_info=rollout_info,
                            group=group,
                            valid_rollouts=valid_rollouts,
                            request_wall_seconds=request_wall_seconds,
                            group_closed=False,
                        )
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
                    self.step_completed_requests += 1
                    self._record_value("request_picker_step_completed_requests", float(self.step_completed_requests))
                    self._record_value("request_picker_step_dispatch_overhang", float(self.step_dispatch_overhang))
                    self._log_completion_replay_event(
                        rollout_info=rollout_info,
                        group=completed_group,
                        valid_rollouts=valid_rollouts,
                        request_wall_seconds=request_wall_seconds,
                        group_closed=True,
                        group_wall_seconds=group_wall_seconds,
                        group_tail_seconds=group_tail_seconds,
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
        await self._fill_inflight_requests(batch_progress=batch_progress)
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
        request_picker = getattr(self, "request_picker", None)
        if request_picker is not None:
            await request_picker.aclose()

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
        metrics.update(self.rollout_inference.get_metrics())

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
