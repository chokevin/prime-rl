from __future__ import annotations

import argparse
import os
import re
import stat
from pathlib import Path

from tau.eval_tools.json_io import write_json_exclusive
from tau.eval_tools.manifest import (
    GIT_SHA_RE,
    FileManifest,
    FrozenEvalManifest,
    ModelSnapshot,
    make_tree_immutable,
    materialize_regular_snapshot,
    validate_file_manifest,
    validate_manifest_contract,
    validate_model_materialization,
    validate_training_materialization,
    validate_tree_immutable,
)

RUN_ROOT_RE = re.compile(r"^prime-rl-run-[A-Za-z0-9]+$")


def validate_run_root(run_root: Path) -> Path:
    run_root = Path(os.path.abspath(run_root))
    canonical_tmp = Path("/tmp").resolve(strict=True)
    if run_root.parent.resolve(strict=True) != canonical_tmp or not RUN_ROOT_RE.fullmatch(run_root.name):
        raise ValueError(f"private run root must be a direct /tmp/prime-rl-run-<nonce> child: {run_root}")
    if run_root.is_symlink() or run_root.resolve(strict=True) != run_root:
        raise ValueError(f"private run root is not canonical: {run_root}")
    metadata = run_root.stat()
    if metadata.st_uid != os.getuid():
        raise ValueError(f"private run root is not owned by uid {os.getuid()}: {run_root}")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ValueError(f"private run root mode must be 0700: {run_root}")
    return run_root


def _seal_tree(
    run_root: Path,
    path: Path,
    expected: FileManifest,
    *,
    kind: str,
    revision: str,
) -> None:
    validate_file_manifest(path, expected)
    make_tree_immutable(path)
    validate_tree_immutable(path, expected)
    marker = run_root / f"{kind}.complete.json"
    write_json_exclusive(
        marker,
        {
            "schema_version": 1,
            "kind": kind,
            "path": str(path),
            "revision": revision,
            "aggregate_sha256": expected.aggregate_sha256,
        },
    )
    marker.chmod(0o444)


def materialize_model(
    model: ModelSnapshot,
    run_root: Path,
) -> Path:
    from huggingface_hub import model_info, snapshot_download

    run_root = validate_run_root(run_root)
    info = model_info(model.name, revision=model.revision)
    if info.sha != model.revision:
        raise RuntimeError(f"huggingface_hub resolved {model.name}@{model.revision} to {info.sha!r}")
    cache_dir = run_root / "model-cache"
    downloaded = Path(
        snapshot_download(
            repo_id=model.name,
            revision=model.revision,
            cache_dir=cache_dir,
        )
    )
    if downloaded.name != model.revision:
        raise RuntimeError(f"downloaded model path {downloaded} is not exact revision {model.revision}")
    destination = run_root / "model"
    actual = materialize_regular_snapshot(downloaded, run_root, destination)
    if actual != model.file_manifest:
        raise ValueError(
            "job-private model contents do not match the frozen model manifest "
            f"(actual={actual.aggregate_sha256}, expected={model.file_manifest.aggregate_sha256})"
        )
    _seal_tree(run_root, destination, model.file_manifest, kind="model", revision=model.revision)
    return validate_model_materialization(model, destination, run_root)


def materialize_model_for_freeze(
    model_name: str,
    model_revision: str,
    run_root: Path,
) -> tuple[ModelSnapshot, Path]:
    from huggingface_hub import model_info, snapshot_download

    if not GIT_SHA_RE.fullmatch(model_revision):
        raise ValueError("model revision must be an exact 40-character lowercase commit SHA")
    run_root = validate_run_root(run_root)
    info = model_info(model_name, revision=model_revision)
    if info.sha != model_revision:
        raise RuntimeError(f"huggingface_hub resolved {model_name}@{model_revision} to {info.sha!r}")
    downloaded = Path(
        snapshot_download(
            repo_id=model_name,
            revision=model_revision,
            cache_dir=run_root / "model-cache",
        )
    )
    if downloaded.name != model_revision:
        raise RuntimeError(f"downloaded model path {downloaded} is not exact revision {model_revision}")
    destination = run_root / "model"
    file_manifest = materialize_regular_snapshot(downloaded, run_root, destination)
    model = ModelSnapshot(name=model_name, revision=model_revision, file_manifest=file_manifest)
    _seal_tree(run_root, destination, file_manifest, kind="model", revision=model_revision)
    return model, validate_model_materialization(model, destination, run_root)


def materialize_training_dataset(
    manifest: FrozenEvalManifest,
    run_root: Path,
) -> Path:
    from huggingface_hub import dataset_info, snapshot_download

    run_root = validate_run_root(run_root)
    taskset = manifest.train_taskset
    info = dataset_info(taskset.dataset_name, revision=taskset.dataset_revision)
    if info.sha != taskset.dataset_revision:
        raise RuntimeError(
            f"huggingface_hub resolved {taskset.dataset_name}@{taskset.dataset_revision} to {info.sha!r}"
        )
    downloaded = Path(
        snapshot_download(
            repo_id=taskset.dataset_name,
            repo_type="dataset",
            revision=taskset.dataset_revision,
            cache_dir=run_root / "dataset-cache",
        )
    )
    if downloaded.name != taskset.dataset_revision:
        raise RuntimeError(
            f"downloaded training dataset path {downloaded} is not exact revision {taskset.dataset_revision}"
        )
    destination = run_root / "training-dataset"
    actual = materialize_regular_snapshot(downloaded, run_root, destination)
    if actual != manifest.training_data.file_manifest:
        raise ValueError(
            "job-private training dataset files do not match the frozen identity "
            f"(actual={actual.aggregate_sha256}, expected={manifest.training_data.file_manifest.aggregate_sha256})"
        )
    _seal_tree(
        run_root,
        destination,
        manifest.training_data.file_manifest,
        kind="training-dataset",
        revision=taskset.dataset_revision,
    )
    return validate_training_materialization(manifest, destination, run_root)


def materialize_training_dataset_for_freeze(
    dataset_name: str,
    dataset_revision: str,
    run_root: Path,
) -> tuple[Path, FileManifest]:
    from huggingface_hub import dataset_info, snapshot_download

    if not GIT_SHA_RE.fullmatch(dataset_revision):
        raise ValueError("training dataset revision must be an exact 40-character lowercase commit SHA")
    run_root = validate_run_root(run_root)
    info = dataset_info(dataset_name, revision=dataset_revision)
    if info.sha != dataset_revision:
        raise RuntimeError(f"huggingface_hub resolved {dataset_name}@{dataset_revision} to {info.sha!r}")
    downloaded = Path(
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            revision=dataset_revision,
            cache_dir=run_root / "dataset-cache",
        )
    )
    if downloaded.name != dataset_revision:
        raise RuntimeError(f"downloaded training dataset path {downloaded} is not exact revision {dataset_revision}")
    destination = run_root / "training-dataset"
    file_manifest = materialize_regular_snapshot(downloaded, run_root, destination)
    _seal_tree(
        run_root,
        destination,
        file_manifest,
        kind="training-dataset",
        revision=dataset_revision,
    )
    return destination, file_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--require-finalized", action="store_true")
    args = parser.parse_args(argv)

    manifest = FrozenEvalManifest.load(Path(args.manifest))
    validate_manifest_contract(
        manifest,
        expected_source_revision=manifest.source_revision,
        expected_verifiers_revision=manifest.verifiers_revision,
        expected_tasksets_revision=manifest.eval_taskset.taskset_revision,
        expected_model_name=manifest.model.name,
        expected_model_revision=manifest.model.revision,
        require_finalized=args.require_finalized,
    )
    print(materialize_model(manifest.model, Path(args.run_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
