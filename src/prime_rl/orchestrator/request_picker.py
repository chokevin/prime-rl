from __future__ import annotations

import asyncio
import statistics
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Protocol
from urllib.parse import urlparse

import httpx
import verifiers as vf

from prime_rl.utils.logger import get_logger

ClientIdentity = tuple[str, str | None]


def client_identity(client: vf.ClientConfig) -> ClientIdentity:
    """Stable rollout client identity across elastic client rebuilds."""
    return (
        client.api_base_url,
        client.extra_headers.get("X-data-parallel-rank"),
    )


def endpoint_label_from_url(url: str) -> str:
    parsed = urlparse(url)
    netloc = parsed.netloc or parsed.path
    return netloc.replace(":", "_").replace(".", "_").replace("-", "_")


def client_metric_label(client: vf.ClientConfig) -> str:
    dp_rank = client.extra_headers.get("X-data-parallel-rank", "none")
    endpoint = endpoint_label_from_url(client.api_base_url)
    return f"client_{client.client_idx}_{endpoint}_dp_{dp_rank}"


@dataclass(frozen=True)
class RequestPickContext:
    env_name: str
    group_id: int
    model_name: str
    step: int
    ckpt_step: int
    cache_salt: str
    group_age_seconds: float
    rollouts_to_schedule: int
    completed_rollouts: int
    max_off_policy_level: int
    oldest_inflight_seconds: float


@dataclass(frozen=True)
class CandidateStats:
    completed_rollouts: int = 0
    cancelled_rollouts: int = 0
    request_wall_seconds_mean: float | None = None
    request_wall_seconds_last: float | None = None
    group_wall_seconds_mean: float | None = None
    group_wall_seconds_last: float | None = None
    group_tail_seconds_mean: float | None = None
    group_tail_seconds_last: float | None = None
    off_policy_steps_mean: float | None = None
    off_policy_steps_last: float | None = None
    endpoint_metrics: dict[str, float] | None = None


@dataclass(frozen=True)
class RequestPickResult:
    client: vf.ClientConfig
    latency_seconds: float
    attempts: int
    candidate_count: int
    selected_inflight: int
    selected_score: float | None = None
    selected_score_components: dict[str, float] | None = None


class RequestPicker(Protocol):
    async def select_client(
        self,
        candidates: list[vf.ClientConfig],
        inflight: Mapping[ClientIdentity, int],
        context: RequestPickContext,
        candidate_stats: Mapping[ClientIdentity, CandidateStats],
    ) -> vf.ClientConfig: ...

    async def aclose(self) -> None: ...


class DirectRequestPicker:
    """Sentinel for the default path: use Scheduler's existing direct selection."""

    async def select_client(
        self,
        candidates: list[vf.ClientConfig],
        inflight: Mapping[ClientIdentity, int],
        context: RequestPickContext,
        candidate_stats: Mapping[ClientIdentity, CandidateStats],
    ) -> vf.ClientConfig:
        raise RuntimeError("DirectRequestPicker is a scheduler sentinel and should not be called")

    async def aclose(self) -> None:
        return None


class LeastLoadedRequestPicker:
    """No-op in-process picker that reproduces the direct least-loaded policy."""

    def __init__(self):
        self.last_score: float | None = None

    async def select_client(
        self,
        candidates: list[vf.ClientConfig],
        inflight: Mapping[ClientIdentity, int],
        context: RequestPickContext,
        candidate_stats: Mapping[ClientIdentity, CandidateStats],
    ) -> vf.ClientConfig:
        client = min(candidates, key=lambda c: inflight.get(client_identity(c), 0))
        self.last_score = float(inflight.get(client_identity(client), 0))
        return client

    async def aclose(self) -> None:
        return None


class PrimeAwareRequestPicker:
    """In-process picker that keeps Prime's fast path but avoids observed stragglers."""

    def __init__(
        self,
        inflight_slack: int = 2,
        inflight_weight: float = 1.0,
        waiting_weight: float = 1.0,
        running_weight: float = 0.25,
        request_wall_weight: float = 1.0,
        group_wall_weight: float = 3.0,
        group_tail_weight: float = 1.0,
        off_policy_weight: float = 0.25,
        cancelled_weight: float = 0.25,
        decode_deficit_weight: float = 2.0,
        completed_rps_deficit_weight: float = 0.0,
        cache_usage_weight: float = 0.25,
        history_penalty_cap: float = 4.0,
        group_tail_pressure_weight: float = 0.0,
        group_tail_pressure_threshold_seconds: float = 60.0,
        waiting_backpressure_threshold: float | None = None,
        waiting_backpressure_penalty: float = 0.0,
        decode_guardrail_ratio: float = 0.0,
        decode_guardrail_penalty: float = 0.0,
        max_inflight_per_client: int | None = None,
    ):
        self.inflight_slack = inflight_slack
        self.inflight_weight = inflight_weight
        self.waiting_weight = waiting_weight
        self.running_weight = running_weight
        self.request_wall_weight = request_wall_weight
        self.group_wall_weight = group_wall_weight
        self.group_tail_weight = group_tail_weight
        self.off_policy_weight = off_policy_weight
        self.cancelled_weight = cancelled_weight
        self.decode_deficit_weight = decode_deficit_weight
        self.completed_rps_deficit_weight = completed_rps_deficit_weight
        self.cache_usage_weight = cache_usage_weight
        self.history_penalty_cap = history_penalty_cap
        self.group_tail_pressure_weight = group_tail_pressure_weight
        self.group_tail_pressure_threshold_seconds = group_tail_pressure_threshold_seconds
        self.waiting_backpressure_threshold = waiting_backpressure_threshold
        self.waiting_backpressure_penalty = waiting_backpressure_penalty
        self.decode_guardrail_ratio = decode_guardrail_ratio
        self.decode_guardrail_penalty = decode_guardrail_penalty
        self.max_inflight_per_client = max_inflight_per_client
        self.last_score: float | None = None
        self.last_score_components: dict[str, float] | None = None

    async def select_client(
        self,
        candidates: list[vf.ClientConfig],
        inflight: Mapping[ClientIdentity, int],
        context: RequestPickContext,
        candidate_stats: Mapping[ClientIdentity, CandidateStats],
    ) -> vf.ClientConfig:
        candidate_inflight = {
            client_identity(client): inflight.get(client_identity(client), 0) for client in candidates
        }
        eligible_candidates = _filter_by_inflight_cap(candidates, candidate_inflight, self.max_inflight_per_client)
        if eligible_candidates:
            candidates = eligible_candidates
        min_inflight = min(candidate_inflight.values(), default=0)
        balanced_candidates = [
            client
            for client in candidates
            if candidate_inflight[client_identity(client)] <= min_inflight + self.inflight_slack
        ]
        score_components = {
            client_identity(client): self._score_components(
                client, inflight, context, candidate_stats, balanced_candidates
            )
            for client in balanced_candidates
        }
        scores = {identity: sum(components.values()) for identity, components in score_components.items()}
        client = min(
            balanced_candidates,
            key=lambda c: (scores[client_identity(c)], candidate_inflight[client_identity(c)], c.client_idx),
        )
        selected_identity = client_identity(client)
        self.last_score = scores[selected_identity]
        self.last_score_components = score_components[selected_identity]
        return client

    def _score_components(
        self,
        client: vf.ClientConfig,
        inflight: Mapping[ClientIdentity, int],
        context: RequestPickContext,
        candidate_stats: Mapping[ClientIdentity, CandidateStats],
        candidates: Iterable[vf.ClientConfig],
    ) -> dict[str, float]:
        identity = client_identity(client)
        stats = candidate_stats.get(identity, CandidateStats())
        metrics = stats.endpoint_metrics or {}
        candidate_stats_subset = [
            candidate_stats.get(client_identity(candidate), CandidateStats()) for candidate in candidates
        ]
        history_penalty = (
            self.request_wall_weight
            * _latency_excess(
                _mean_or_latest(stats.request_wall_seconds_mean, stats.request_wall_seconds_last),
                (
                    _mean_or_latest(candidate.request_wall_seconds_mean, candidate.request_wall_seconds_last)
                    for candidate in candidate_stats_subset
                ),
            )
            + self.group_wall_weight
            * _latency_excess(
                _mean_or_latest(stats.group_wall_seconds_mean, stats.group_wall_seconds_last),
                (
                    _mean_or_latest(candidate.group_wall_seconds_mean, candidate.group_wall_seconds_last)
                    for candidate in candidate_stats_subset
                ),
            )
            + self.group_tail_weight
            * _latency_excess(
                _mean_or_latest(stats.group_tail_seconds_mean, stats.group_tail_seconds_last),
                (
                    _mean_or_latest(candidate.group_tail_seconds_mean, candidate.group_tail_seconds_last)
                    for candidate in candidate_stats_subset
                ),
            )
        )
        return {
            "inflight": self.inflight_weight * inflight.get(identity, 0),
            "waiting": self.waiting_weight * metrics.get("num_requests_waiting", 0.0),
            "running": self.running_weight * metrics.get("num_requests_running", 0.0),
            "history": min(history_penalty, self.history_penalty_cap),
            "group_tail_pressure": self.group_tail_pressure_weight
            * _group_tail_pressure(
                context,
                stats,
                candidate_stats_subset,
                self.group_tail_pressure_threshold_seconds,
            ),
            "off_policy": self.off_policy_weight
            * _latest_or_mean(stats.off_policy_steps_last, stats.off_policy_steps_mean),
            "cancelled": self.cancelled_weight * stats.cancelled_rollouts,
            "decode_deficit": self.decode_deficit_weight
            * _rate_deficit("decode_throughput_tps", stats, candidate_stats.values()),
            "completed_rps_deficit": self.completed_rps_deficit_weight
            * _rate_deficit("completed_requests_per_s", stats, candidate_stats.values()),
            "cache_usage": self.cache_usage_weight * metrics.get("gpu_cache_usage_perc", 0.0),
            "waiting_backpressure": self.waiting_backpressure_penalty
            * _threshold_excess(metrics.get("num_requests_waiting", 0.0), self.waiting_backpressure_threshold),
            "decode_guardrail": self.decode_guardrail_penalty
            * _decode_guardrail_deficit(
                "decode_throughput_tps", stats, candidate_stats.values(), self.decode_guardrail_ratio
            ),
        }

    async def aclose(self) -> None:
        return None


class ExternalRequestPicker:
    """Adapter boundary for Prime-aware pickers outside the generation data path."""

    def __init__(
        self,
        adapter_url: str,
        timeout: float,
        max_attempts: int = 3,
        retry_backoff: float = 0.05,
        client: httpx.AsyncClient | None = None,
    ):
        self.adapter_url = adapter_url
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.retry_backoff = retry_backoff
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None
        self.last_attempts = 0

    async def select_client(
        self,
        candidates: list[vf.ClientConfig],
        inflight: Mapping[ClientIdentity, int],
        context: RequestPickContext,
        candidate_stats: Mapping[ClientIdentity, CandidateStats],
    ) -> vf.ClientConfig:
        payload = {
            "request": asdict(context),
            "candidates": [
                {
                    "client_idx": client.client_idx,
                    "api_base_url": client.api_base_url,
                    "dp_rank": client.extra_headers.get("X-data-parallel-rank"),
                    "inflight": inflight.get(client_identity(client), 0),
                    **asdict(candidate_stats.get(client_identity(client), CandidateStats())),
                }
                for client in candidates
            ],
        }
        decision = await self._post_with_retries(payload, context)
        return _match_decision(candidates, decision)

    async def _post_with_retries(self, payload: dict, context: RequestPickContext) -> dict:
        logger = get_logger()
        last_error: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            self.last_attempts = attempt
            try:
                return await self._post_decision(payload)
            except Exception as e:
                if not _is_retryable_picker_error(e):
                    logger.error(
                        f"External request picker call failed for group_id={context.group_id} "
                        f"with non-retryable error: {e!r}"
                    )
                    raise
                if attempt >= self.max_attempts:
                    logger.error(
                        f"External request picker call exhausted retry budget for group_id={context.group_id} "
                        f"after {self.max_attempts} attempt(s): {e!r}"
                    )
                    raise
                last_error = e
                delay = self.retry_backoff * (2 ** (attempt - 1))
                logger.warning(
                    f"External request picker call failed for group_id={context.group_id} "
                    f"(attempt {attempt}/{self.max_attempts}, retrying in {delay:.2f}s): {e!r}"
                )
                await asyncio.sleep(delay)
        assert last_error is not None
        raise last_error

    async def _post_decision(self, payload: dict) -> dict:
        response = await self._client.post(self.adapter_url, json=payload)
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _match_decision(candidates: list[vf.ClientConfig], decision: dict) -> vf.ClientConfig:
    if "client_idx" in decision:
        for client in candidates:
            if client.client_idx == decision["client_idx"]:
                return client
        raise ValueError(f"External request picker returned unknown client_idx={decision['client_idx']!r}")

    if "api_base_url" not in decision:
        raise ValueError("External request picker decision must include client_idx or api_base_url")

    decision_identity = (decision["api_base_url"], decision.get("dp_rank"))
    for client in candidates:
        if client_identity(client) == decision_identity:
            return client
    raise ValueError(f"External request picker returned unknown client identity={decision_identity!r}")


def _is_retryable_picker_error(exception: BaseException) -> bool:
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code >= 500
    return isinstance(exception, (httpx.TimeoutException, httpx.TransportError))


def _latest_or_mean(latest: float | None, mean: float | None) -> float:
    if latest is not None:
        return latest
    if mean is not None:
        return mean
    return 0.0


def _mean_or_latest(mean: float | None, latest: float | None) -> float | None:
    if mean is not None:
        return mean
    return latest


def _latency_excess(value: float | None, all_values: Iterable[float | None]) -> float:
    if value is None:
        return 0.0
    observed_values = [candidate for candidate in all_values if candidate is not None and candidate > 0]
    if len(observed_values) < 2:
        return 0.0

    reference = statistics.median(observed_values)
    if reference <= 0:
        return 0.0
    return max(value - reference, 0.0) / reference


def _group_tail_pressure(
    context: RequestPickContext,
    stats: CandidateStats,
    all_stats: Iterable[CandidateStats],
    threshold_seconds: float,
) -> float:
    tail_excess = _latency_excess(
        _mean_or_latest(stats.group_tail_seconds_mean, stats.group_tail_seconds_last),
        (
            _mean_or_latest(candidate.group_tail_seconds_mean, candidate.group_tail_seconds_last)
            for candidate in all_stats
        ),
    )
    if tail_excess <= 0:
        return 0.0

    group_progress = context.completed_rollouts / max(context.completed_rollouts + context.rollouts_to_schedule, 1)
    if threshold_seconds <= 0:
        age_pressure = 1.0
    else:
        age_pressure = max(context.oldest_inflight_seconds - threshold_seconds, 0.0) / threshold_seconds
    return tail_excess * (1.0 + group_progress + age_pressure)


def _rate_deficit(metric_name: str, stats: CandidateStats, all_stats: Iterable[CandidateStats]) -> float:
    rate = (stats.endpoint_metrics or {}).get(metric_name)
    if rate is None:
        return 0.0

    max_rate = max(
        ((candidate.endpoint_metrics or {}).get(metric_name, 0.0) for candidate in all_stats),
        default=0.0,
    )
    if max_rate <= 0:
        return 0.0
    return max(max_rate - rate, 0.0) / max_rate


def _threshold_excess(value: float, threshold: float | None) -> float:
    if threshold is None or value <= threshold:
        return 0.0
    return (value - threshold) / max(threshold, 1.0)


def _decode_guardrail_deficit(
    metric_name: str,
    stats: CandidateStats,
    all_stats: Iterable[CandidateStats],
    tolerance_ratio: float,
) -> float:
    rate = (stats.endpoint_metrics or {}).get(metric_name)
    if rate is None:
        return 0.0

    max_rate = max(
        ((candidate.endpoint_metrics or {}).get(metric_name, 0.0) for candidate in all_stats),
        default=0.0,
    )
    if max_rate <= 0:
        return 0.0

    tolerated_rate = max_rate * max(1.0 - tolerance_ratio, 0.0)
    if rate >= tolerated_rate:
        return 0.0
    return (tolerated_rate - rate) / max_rate


def setup_request_picker(config) -> RequestPicker:
    if config.type == "direct":
        return DirectRequestPicker()
    if config.type == "least_loaded":
        return LeastLoadedRequestPicker()
    if config.type == "prime_aware":
        return PrimeAwareRequestPicker(
            inflight_slack=config.inflight_slack,
            inflight_weight=config.inflight_weight,
            waiting_weight=config.waiting_weight,
            running_weight=config.running_weight,
            request_wall_weight=config.request_wall_weight,
            group_wall_weight=config.group_wall_weight,
            group_tail_weight=config.group_tail_weight,
            off_policy_weight=config.off_policy_weight,
            cancelled_weight=config.cancelled_weight,
            decode_deficit_weight=config.decode_deficit_weight,
            completed_rps_deficit_weight=config.completed_rps_deficit_weight,
            cache_usage_weight=config.cache_usage_weight,
            history_penalty_cap=config.history_penalty_cap,
            group_tail_pressure_weight=config.group_tail_pressure_weight,
            group_tail_pressure_threshold_seconds=config.group_tail_pressure_threshold_seconds,
            waiting_backpressure_threshold=config.waiting_backpressure_threshold,
            waiting_backpressure_penalty=config.waiting_backpressure_penalty,
            decode_guardrail_ratio=config.decode_guardrail_ratio,
            decode_guardrail_penalty=config.decode_guardrail_penalty,
            max_inflight_per_client=config.max_inflight_per_client,
        )
    if config.type == "external":
        return ExternalRequestPicker(
            adapter_url=config.adapter_url,
            timeout=config.timeout,
            max_attempts=config.max_attempts,
            retry_backoff=config.retry_backoff,
        )
    raise ValueError(f"Unsupported request picker type: {config.type}")


async def select_with_metrics(
    picker: RequestPicker,
    candidates: list[vf.ClientConfig],
    inflight: Mapping[ClientIdentity, int],
    context: RequestPickContext,
    candidate_stats: Mapping[ClientIdentity, CandidateStats],
) -> RequestPickResult:
    start = time.perf_counter()
    client = await picker.select_client(candidates, inflight, context, candidate_stats)
    latency = time.perf_counter() - start
    attempts = getattr(picker, "last_attempts", 1)
    selected_score = getattr(picker, "last_score", None)
    selected_score_components = getattr(picker, "last_score_components", None)
    return RequestPickResult(
        client=client,
        latency_seconds=latency,
        attempts=attempts,
        candidate_count=len(candidates),
        selected_inflight=inflight.get(client_identity(client), 0),
        selected_score=float(selected_score) if isinstance(selected_score, (int, float)) else None,
        selected_score_components=selected_score_components if isinstance(selected_score_components, dict) else None,
    )


def request_picker_inflight_cap(picker: RequestPicker) -> int | None:
    cap = getattr(picker, "max_inflight_per_client", None)
    return cap if isinstance(cap, int) else None


def has_request_picker_capacity(
    picker: RequestPicker,
    candidates: list[vf.ClientConfig],
    inflight: Mapping[ClientIdentity, int],
) -> bool:
    cap = request_picker_inflight_cap(picker)
    if cap is None:
        return True
    return any(inflight.get(client_identity(client), 0) < cap for client in candidates)


def _filter_by_inflight_cap(
    candidates: list[vf.ClientConfig],
    candidate_inflight: Mapping[ClientIdentity, int],
    cap: int | None,
) -> list[vf.ClientConfig]:
    if cap is None:
        return candidates
    return [client for client in candidates if candidate_inflight.get(client_identity(client), 0) < cap]
