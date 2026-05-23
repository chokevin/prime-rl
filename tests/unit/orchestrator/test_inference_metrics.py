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
