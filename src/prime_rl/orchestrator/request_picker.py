from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Protocol

import httpx
import verifiers as vf

from prime_rl.utils.utils import get_logger

ClientIdentity = tuple[str, str | None]


def client_identity(client: vf.ClientConfig) -> ClientIdentity:
    """Stable rollout client identity across elastic client rebuilds."""
    return (
        client.api_base_url,
        client.extra_headers.get("X-data-parallel-rank"),
    )


@dataclass(frozen=True)
class RequestPickContext:
    env_name: str
    group_id: int
    model_name: str
    ckpt_step: int
    cache_salt: str


class RequestPicker(Protocol):
    async def select_client(
        self,
        candidates: list[vf.ClientConfig],
        inflight: Mapping[ClientIdentity, int],
        context: RequestPickContext,
    ) -> vf.ClientConfig: ...


class LeastLoadedRequestPicker:
    async def select_client(
        self,
        candidates: list[vf.ClientConfig],
        inflight: Mapping[ClientIdentity, int],
        context: RequestPickContext,
    ) -> vf.ClientConfig:
        return min(candidates, key=lambda c: inflight.get(client_identity(c), 0))


class ExternalRequestPicker:
    """Adapter boundary for EPP-style pickers that are not in the HTTP data path.

    The adapter receives Prime's logical rollout clients and returns one of
    them. It deliberately does not proxy generation traffic or admin endpoints.
    """

    def __init__(self, adapter_url: str, timeout: float, max_attempts: int = 3, retry_backoff: float = 0.05):
        self.adapter_url = adapter_url
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.retry_backoff = retry_backoff

    async def select_client(
        self,
        candidates: list[vf.ClientConfig],
        inflight: Mapping[ClientIdentity, int],
        context: RequestPickContext,
    ) -> vf.ClientConfig:
        payload = {
            "request": asdict(context),
            "candidates": [
                {
                    "client_idx": client.client_idx,
                    "api_base_url": client.api_base_url,
                    "dp_rank": client.extra_headers.get("X-data-parallel-rank"),
                    "inflight": inflight.get(client_identity(client), 0),
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
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.adapter_url, json=payload)
            response.raise_for_status()
            return response.json()


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


def setup_request_picker(config) -> RequestPicker:
    if config.type == "least_loaded":
        return LeastLoadedRequestPicker()
    if config.type == "external":
        return ExternalRequestPicker(
            adapter_url=config.adapter_url,
            timeout=config.timeout,
            max_attempts=config.max_attempts,
            retry_backoff=config.retry_backoff,
        )
    raise ValueError(f"Unsupported request picker type: {config.type}")
