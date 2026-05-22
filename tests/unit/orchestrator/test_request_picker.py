from __future__ import annotations

import asyncio

import verifiers as vf

from prime_rl.orchestrator.request_picker import (
    CandidateStats,
    ExternalRequestPicker,
    LeastLoadedRequestPicker,
    PrimeAwareRequestPicker,
    RequestPickContext,
    client_identity,
)


def _client(idx: int, base_url: str, dp_rank: str | None = None) -> vf.ClientConfig:
    headers = {}
    if dp_rank is not None:
        headers["X-data-parallel-rank"] = dp_rank
    return vf.ClientConfig(client_idx=idx, api_base_url=base_url, extra_headers=headers)


def _context() -> RequestPickContext:
    return RequestPickContext(
        env_name="math",
        group_id=7,
        model_name="test-model",
        step=3,
        ckpt_step=2,
        cache_salt="2",
        group_age_seconds=1.25,
        rollouts_to_schedule=1,
        completed_rollouts=0,
        max_off_policy_level=2,
        oldest_inflight_seconds=4.5,
    )


def test_least_loaded_request_picker_matches_direct_scheduler_policy():
    async def run() -> None:
        clients = [
            _client(0, "http://worker-a:8000/v1", "0"),
            _client(1, "http://worker-a:8000/v1", "1"),
            _client(2, "http://worker-b:8000/v1", "0"),
        ]
        inflight = {
            client_identity(clients[0]): 4,
            client_identity(clients[1]): 1,
            client_identity(clients[2]): 3,
        }

        direct = min(clients, key=lambda c: inflight[client_identity(c)])
        picked = await LeastLoadedRequestPicker().select_client(clients, inflight, _context(), {})

        assert picked == direct
        assert picked.client_idx == 1

    asyncio.run(run())


def test_external_request_picker_reuses_client_and_sends_prime_aware_fields():
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"client_idx": 1}

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def post(self, url: str, json: dict) -> FakeResponse:
            self.calls.append((url, json))
            return FakeResponse()

    async def run() -> None:
        fake_client = FakeClient()
        picker = ExternalRequestPicker(
            adapter_url="http://picker.local/pick",
            timeout=1.0,
            client=fake_client,
        )
        clients = [
            _client(0, "http://worker-a:8000/v1", "0"),
            _client(1, "http://worker-b:8000/v1", "0"),
        ]
        stats = {
            client_identity(clients[1]): CandidateStats(
                completed_rollouts=5,
                cancelled_rollouts=1,
                request_wall_seconds_mean=3.25,
                request_wall_seconds_last=4.0,
                group_wall_seconds_mean=5.0,
                group_wall_seconds_last=6.0,
                group_tail_seconds_mean=1.5,
                group_tail_seconds_last=2.0,
                off_policy_steps_mean=1.0,
                off_policy_steps_last=2.0,
                endpoint_metrics={"decode_throughput_tps": 123.0, "num_requests_waiting": 2.0},
            )
        }

        first = await picker.select_client(clients, {}, _context(), stats)
        second = await picker.select_client(clients, {}, _context(), stats)

        assert first.client_idx == 1
        assert second.client_idx == 1
        assert len(fake_client.calls) == 2
        assert fake_client.calls[0][0] == "http://picker.local/pick"
        payload = fake_client.calls[0][1]
        assert payload["request"]["group_id"] == 7
        assert payload["request"]["max_off_policy_level"] == 2
        assert payload["candidates"][1]["completed_rollouts"] == 5
        assert payload["candidates"][1]["group_wall_seconds_last"] == 6.0
        assert payload["candidates"][1]["group_tail_seconds_last"] == 2.0
        assert payload["candidates"][1]["off_policy_steps_last"] == 2.0
        assert payload["candidates"][1]["endpoint_metrics"]["decode_throughput_tps"] == 123.0

    asyncio.run(run())


def test_prime_aware_request_picker_keeps_near_least_loaded_balance():
    async def run() -> None:
        clients = [
            _client(0, "http://worker-a:8000/v1", "0"),
            _client(1, "http://worker-a:8000/v1", "1"),
        ]
        stats = {
            client_identity(clients[0]): CandidateStats(
                request_wall_seconds_mean=120.0,
                group_wall_seconds_mean=120.0,
            ),
            client_identity(clients[1]): CandidateStats(
                request_wall_seconds_mean=1.0,
                group_wall_seconds_mean=1.0,
            ),
        }

        picked = await PrimeAwareRequestPicker(inflight_slack=1).select_client(
            clients,
            {
                client_identity(clients[0]): 0,
                client_identity(clients[1]): 5,
            },
            _context(),
            stats,
        )

        assert picked.client_idx == 0

    asyncio.run(run())


def test_prime_aware_request_picker_uses_group_wall_when_balanced():
    async def run() -> None:
        clients = [
            _client(0, "http://worker-a:8000/v1", "0"),
            _client(1, "http://worker-a:8000/v1", "1"),
        ]
        stats = {
            client_identity(clients[0]): CandidateStats(group_wall_seconds_mean=120.0),
            client_identity(clients[1]): CandidateStats(group_wall_seconds_mean=30.0),
        }

        picked = await PrimeAwareRequestPicker().select_client(
            clients,
            {
                client_identity(clients[0]): 2,
                client_identity(clients[1]): 2,
            },
            _context(),
            stats,
        )

        assert picked.client_idx == 1

    asyncio.run(run())


def test_prime_aware_request_picker_matches_least_loaded_without_signals():
    async def run() -> None:
        clients = [
            _client(0, "http://worker-a:8000/v1", "0"),
            _client(1, "http://worker-b:8000/v1", "0"),
            _client(2, "http://worker-c:8000/v1", "0"),
        ]
        inflight = {
            client_identity(clients[0]): 2,
            client_identity(clients[1]): 0,
            client_identity(clients[2]): 1,
        }

        picked = await PrimeAwareRequestPicker().select_client(clients, inflight, _context(), {})

        assert picked.client_idx == 1

    asyncio.run(run())


def test_prime_aware_request_picker_avoids_queued_straggler():
    async def run() -> None:
        clients = [
            _client(0, "http://worker-a:8000/v1", "0"),
            _client(1, "http://worker-b:8000/v1", "0"),
        ]
        stats = {
            client_identity(clients[0]): CandidateStats(
                request_wall_seconds_last=6.0,
                group_tail_seconds_last=4.0,
                off_policy_steps_last=3.0,
                endpoint_metrics={
                    "num_requests_waiting": 8.0,
                    "num_requests_running": 2.0,
                    "decode_throughput_tps": 250.0,
                    "completed_requests_per_s": 0.05,
                },
            ),
            client_identity(clients[1]): CandidateStats(
                request_wall_seconds_last=1.0,
                group_tail_seconds_last=0.25,
                endpoint_metrics={
                    "num_requests_waiting": 0.0,
                    "num_requests_running": 1.0,
                    "decode_throughput_tps": 500.0,
                    "completed_requests_per_s": 0.10,
                },
            ),
        }

        picked = await PrimeAwareRequestPicker().select_client(
            clients,
            {
                client_identity(clients[0]): 0,
                client_identity(clients[1]): 1,
            },
            _context(),
            stats,
        )

        assert picked.client_idx == 1

    asyncio.run(run())
