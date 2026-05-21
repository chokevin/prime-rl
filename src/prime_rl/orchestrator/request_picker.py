from __future__ import annotations

import asyncio
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
        inflight_weight: float = 1.0,
        waiting_weight: float = 1.0,
        running_weight: float = 0.25,
        request_wall_weight: float = 0.5,
        group_tail_weight: float = 0.5,
        off_policy_weight: float = 0.25,
        cancelled_weight: float = 0.25,
        decode_deficit_weight: float = 1.0,
        completed_rps_deficit_weight: float = 1.0,
        cache_usage_weight: float = 0.25,
    ):
        self.inflight_weight = inflight_weight
        self.waiting_weight = waiting_weight
        self.running_weight = running_weight
        self.request_wall_weight = request_wall_weight
        self.group_tail_weight = group_tail_weight
        self.off_policy_weight = off_policy_weight
        self.cancelled_weight = cancelled_weight
        self.decode_deficit_weight = decode_deficit_weight
        self.completed_rps_deficit_weight = completed_rps_deficit_weight
        self.cache_usage_weight = cache_usage_weight
        self.last_score: float | None = None

    async def select_client(
        self,
        candidates: list[vf.ClientConfig],
        inflight: Mapping[ClientIdentity, int],
        context: RequestPickContext,
        candidate_stats: Mapping[ClientIdentity, CandidateStats],
    ) -> vf.ClientConfig:
        scores = {client_identity(client): self._score(client, inflight, candidate_stats) for client in candidates}
        client = min(candidates, key=lambda c: scores[client_identity(c)])
        self.last_score = scores[client_identity(client)]
        return client

    def _score(
        self,
        client: vf.ClientConfig,
        inflight: Mapping[ClientIdentity, int],
        candidate_stats: Mapping[ClientIdentity, CandidateStats],
    ) -> float:
        identity = client_identity(client)
        stats = candidate_stats.get(identity, CandidateStats())
        metrics = stats.endpoint_metrics or {}
        return (
            self.inflight_weight * inflight.get(identity, 0)
            + self.waiting_weight * metrics.get("num_requests_waiting", 0.0)
            + self.running_weight * metrics.get("num_requests_running", 0.0)
            + self.request_wall_weight
            * _latest_or_mean(stats.request_wall_seconds_last, stats.request_wall_seconds_mean)
            + self.group_tail_weight * _latest_or_mean(stats.group_tail_seconds_last, stats.group_tail_seconds_mean)
            + self.off_policy_weight * _latest_or_mean(stats.off_policy_steps_last, stats.off_policy_steps_mean)
            + self.cancelled_weight * stats.cancelled_rollouts
            + self.decode_deficit_weight * _rate_deficit("decode_throughput_tps", stats, candidate_stats.values())
            + self.completed_rps_deficit_weight
            * _rate_deficit("completed_requests_per_s", stats, candidate_stats.values())
            + self.cache_usage_weight * metrics.get("gpu_cache_usage_perc", 0.0)
        )

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


def setup_request_picker(config) -> RequestPicker:
    if config.type == "direct":
        return DirectRequestPicker()
    if config.type == "least_loaded":
        return LeastLoadedRequestPicker()
    if config.type == "prime_aware":
        return PrimeAwareRequestPicker(
            inflight_weight=config.inflight_weight,
            waiting_weight=config.waiting_weight,
            running_weight=config.running_weight,
            request_wall_weight=config.request_wall_weight,
            group_tail_weight=config.group_tail_weight,
            off_policy_weight=config.off_policy_weight,
            cancelled_weight=config.cancelled_weight,
            decode_deficit_weight=config.decode_deficit_weight,
            completed_rps_deficit_weight=config.completed_rps_deficit_weight,
            cache_usage_weight=config.cache_usage_weight,
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
    return RequestPickResult(
        client=client,
        latency_seconds=latency,
        attempts=attempts,
        candidate_count=len(candidates),
        selected_inflight=inflight.get(client_identity(client), 0),
        selected_score=float(selected_score) if isinstance(selected_score, (int, float)) else None,
    )
