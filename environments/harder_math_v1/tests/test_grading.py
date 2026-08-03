import asyncio
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from harder_math_v1.taskset import (
    HarderMathData,
    HarderMathTask,
    HarderMathTaskConfig,
)


@pytest.fixture
def task() -> HarderMathTask:
    return HarderMathTask(
        HarderMathData(
            prompt="Find one half.",
            question="Find one half.",
            answer="\\frac{1}{2}",
            record_id="fixture@aaaaaaaa:algebra:test:0000000000000000",
            content_sha256="0" * 64,
            prompt_sha256="2" * 64,
            source_sha256="1" * 64,
            source="fixture",
            revision="a" * 40,
            upstream_config="algebra",
            upstream_split="test",
            partition="eval",
            tier="base",
            level="Level 1",
        ),
        HarderMathTaskConfig(math_verify_timeout=5),
    )


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("Reasoning. \\boxed{0.5}", 1.0),
        ("Reasoning. \\boxed{2}", 0.0),
        ("Reasoning. \\boxed{", 0.0),
        ("<think>unfinished reasoning \\boxed{0.5}", 0.0),
    ],
)
def test_deterministic_grader_behaviors(
    task: HarderMathTask,
    reply: str,
    expected: float,
) -> None:
    trace = SimpleNamespace(last_reply=reply)
    assert asyncio.run(task.correct(trace)) == expected


def test_task_config_rejects_judges_and_invalid_timeout() -> None:
    with pytest.raises(ValueError, match="does not allow judge plugins"):
        HarderMathTaskConfig(judges=[vf.ReferenceJudgeConfig()])
    with pytest.raises(ValueError, match="math_verify_timeout must be positive"):
        HarderMathTaskConfig(math_verify_timeout=0)
