from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Protocol

import httpx
import verifiers as vf

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

    def __init__(self, adapter_url: str, timeout: float):
        self.adapter_url = adapter_url
        self.timeout = timeout

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
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.adapter_url, json=payload)
            response.raise_for_status()
        decision = response.json()
        return _match_decision(candidates, decision)


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


def setup_request_picker(config) -> RequestPicker:
    if config.type == "least_loaded":
        return LeastLoadedRequestPicker()
    if config.type == "external":
        return ExternalRequestPicker(adapter_url=config.adapter_url, timeout=config.timeout)
    raise ValueError(f"Unsupported request picker type: {config.type}")
