import random

import harder_math_v1
import pytest
import verifiers.v1 as vf
from harder_math_v1.catalog import (
    build_catalog,
    normalize_math_row,
    select_records,
    validate_partition_disjointness,
)
from harder_math_v1.taskset import HarderMathConfig, HarderMathTaskConfig, _to_task


def test_stable_ids_order_and_indices_under_input_shuffle(source_manifest) -> None:
    source = source_manifest.sources[0]
    rows = [
        {
            "problem": f"Problem {index}",
            "level": "Level 5",
            "type": "Algebra",
            "solution": f"\\boxed{{{index}}}",
        }
        for index in range(8)
    ]

    def normalize(input_rows):
        return build_catalog(
            normalize_math_row(
                row,
                source=source,
                config="algebra",
                upstream_split="train",
                partition="train",
            )
            for row in input_rows
        )

    expected = normalize(rows)
    shuffled = list(rows)
    random.Random(42).shuffle(shuffled)
    actual = normalize(shuffled)

    assert [record.record_id for record in actual] == [record.record_id for record in expected]
    assert select_records(actual, partition="train", tier="hard") == actual
    tasks = [_to_task(record, idx, HarderMathTaskConfig()) for idx, record in enumerate(actual)]
    assert [task.data.idx for task in tasks] == list(range(len(actual)))
    assert [task.data.record_id for task in tasks] == [record.record_id for record in expected]


def test_duplicate_content_and_partition_overlap_fail_loudly(source_manifest) -> None:
    source = source_manifest.sources[0]
    row = {
        "problem": "Same problem",
        "level": "Level 5",
        "type": "Algebra",
        "solution": "\\boxed{42}",
    }
    train = normalize_math_row(
        row,
        source=source,
        config="algebra",
        upstream_split="train",
        partition="train",
    )
    eval_record = normalize_math_row(
        row,
        source=source,
        config="algebra",
        upstream_split="test",
        partition="eval",
    )

    with pytest.raises(ValueError, match="duplicate content hash"):
        build_catalog([train, eval_record])
    with pytest.raises(ValueError, match="train/eval content hash overlap"):
        validate_partition_disjointness([train], [eval_record])


def test_same_prompt_with_different_gold_fails_prompt_disjointness(source_manifest) -> None:
    source = source_manifest.sources[0]
    train = normalize_math_row(
        {
            "problem": "Same prompt, disputed answer",
            "level": "Level 5",
            "type": "Algebra",
            "solution": "\\boxed{41}",
        },
        source=source,
        config="algebra",
        upstream_split="train",
        partition="train",
    )
    eval_record = normalize_math_row(
        {
            "problem": "Same prompt, disputed answer",
            "level": "Level 5",
            "type": "Algebra",
            "solution": "\\boxed{42}",
        },
        source=source,
        config="algebra",
        upstream_split="test",
        partition="eval",
    )

    assert train.content_sha256 != eval_record.content_sha256
    assert train.prompt_sha256 == eval_record.prompt_sha256
    with pytest.raises(ValueError, match="duplicate prompt hash"):
        build_catalog([train, eval_record])
    with pytest.raises(ValueError, match="train/eval prompt hash overlap"):
        validate_partition_disjointness([train], [eval_record])


def test_plugin_export_and_config_narrowing() -> None:
    assert harder_math_v1.__all__ == ["HarderMathTaskset"]
    assert issubclass(harder_math_v1.HarderMathTaskset, vf.Taskset)
    assert vf.taskset_config_type("harder-math-v1") is HarderMathConfig

    config = vf.SingleAgentEnvConfig.model_validate(
        {
            "taskset": {
                "id": "harder-math-v1",
                "partition": "eval",
                "tier": "hard",
            },
            "agent": {
                "harness": {"id": "null"},
                "runtime": {"type": "subprocess"},
            },
        }
    )

    assert isinstance(config.taskset, HarderMathConfig)
    assert config.taskset.partition == "eval"
    assert config.taskset.tier == "hard"
    assert config.agent.harness is not None
    assert config.agent.harness.id == "null"
    assert isinstance(config.agent.runtime, vf.SubprocessConfig)

    with pytest.raises(ValueError, match="does not allow judge"):
        HarderMathTaskConfig(judges=[vf.ReferenceJudgeConfig()])
