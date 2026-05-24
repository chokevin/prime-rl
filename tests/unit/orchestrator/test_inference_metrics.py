import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector


class FakeMetricsClient:
    def __init__(self, base_url: str, text: str | None = None, error: Exception | None = None):
        self.base_url = base_url
        self.text = text
        self.error = error

    async def get(self, *args, **kwargs):
        if self.error is not None:
            raise self.error
        return SimpleNamespace(text=self.text, raise_for_status=lambda: None)


def test_inference_metrics_warns_when_some_endpoints_fail():
    async def run() -> None:
        collector = InferenceMetricsCollector(
            [
                FakeMetricsClient("http://worker-a:8100", "vllm:num_requests_running 2\n"),
                FakeMetricsClient("http://worker-b:8100", error=RuntimeError("boom")),
            ]
        )
        collector.logger = MagicMock()

        with patch("prime_rl.orchestrator.inference_metrics.wandb.log"):
            await collector._collect_and_log()

        collector.logger.warning.assert_called_once_with(
            "Inference metrics unavailable (1/2 inference /metrics endpoint(s) did not respond); "
            "request picker throughput/cache signals will be absent."
        )

    asyncio.run(run())


def test_inference_metrics_logs_vllm_latency_histograms_per_endpoint_and_aggregate():
    async def run() -> None:
        first_scrape = "\n".join(
            [
                "vllm:e2e_request_latency_seconds_sum 10",
                "vllm:e2e_request_latency_seconds_count 2",
                "vllm:inter_token_latency_seconds_sum 1",
                "vllm:inter_token_latency_seconds_count 2",
                "vllm:time_to_first_token_seconds_sum 3",
                "vllm:time_to_first_token_seconds_count 2",
                "vllm:nixl_xfer_time_seconds_sum 4",
                "vllm:nixl_xfer_time_seconds_count 2",
            ]
        )
        second_scrape = "\n".join(
            [
                "vllm:e2e_request_latency_seconds_sum 16",
                "vllm:e2e_request_latency_seconds_count 4",
                "vllm:inter_token_latency_seconds_sum 2",
                "vllm:inter_token_latency_seconds_count 4",
                "vllm:time_to_first_token_seconds_sum 5",
                "vllm:time_to_first_token_seconds_count 4",
                "vllm:nixl_xfer_time_seconds_sum 7",
                "vllm:nixl_xfer_time_seconds_count 4",
            ]
        )
        client = FakeMetricsClient("http://worker-a:8100", first_scrape)
        metric_sink = MagicMock()
        collector = InferenceMetricsCollector([client], metric_sink=metric_sink)

        with patch("prime_rl.orchestrator.inference_metrics.wandb.log") as wandb_log:
            await collector._collect_and_log()
            client.text = second_scrape
            await collector._collect_and_log()

        logged_metrics = wandb_log.call_args.args[0]
        assert logged_metrics["inference/e2e_request_latency_seconds_avg_ms"] == 3000.0
        assert logged_metrics["inference/inter_token_latency_seconds_avg_ms"] == 500.0
        assert logged_metrics["inference/time_to_first_token_seconds_avg_ms"] == 1000.0
        assert logged_metrics["inference/nixl_xfer_time_seconds_avg_ms"] == 1500.0
        assert logged_metrics["inference/server/worker_a_8100/e2e_request_latency_seconds_avg_ms"] == 3000.0
        assert logged_metrics["inference/server/worker_a_8100/inter_token_latency_seconds_avg_ms"] == 500.0
        assert logged_metrics["inference/server/worker_a_8100/time_to_first_token_seconds_avg_ms"] == 1000.0
        assert logged_metrics["inference/skew/e2e_request_latency_seconds_avg_ms/max"] == 3000.0
        metric_sink.assert_called_with(
            {
                "worker_a_8100": {
                    "e2e_request_latency_seconds_avg_ms": 3000.0,
                    "inter_token_latency_seconds_avg_ms": 500.0,
                    "time_to_first_token_seconds_avg_ms": 1000.0,
                    "nixl_xfer_time_seconds_avg_ms": 1500.0,
                }
            }
        )

    asyncio.run(run())
