from __future__ import annotations

import argparse
import hashlib
import os
import re
import secrets
import shutil
import signal
import subprocess
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from tau.eval_tools.artifacts import (
    SOURCE_CONFIG_REL,
    AdapterPublicationState,
    TrainingCompletionAttestation,
    TrainingPreflight,
    _cancellation_scope,
    _check_cancelled,
    _quarantine_owned_staging,
    _write_json_staged_noreplace,
    attempt_paths,
    publish_training_result,
    select_final_adapter,
    write_training_preflight,
)
from tau.eval_tools.json_io import fsync_directory, load_json_with_sha256, write_json_exclusive
from tau.eval_tools.live.private_materialization_live import (
    materialize_model,
    materialize_training_dataset,
    validate_run_root,
)
from tau.eval_tools.live.run_frozen_eval_live import _reload_and_verify_examples
from tau.eval_tools.live.validate_training_data_live import (
    _reload_training_records,
    resolve_effective_rl_config,
    validate_resolved_rl_config,
)
from tau.eval_tools.manifest import (
    FrozenEvalManifest,
    build_file_manifest,
    validate_manifest_contract,
    validate_model_materialization,
    validate_training_prompt_hashes,
)


@dataclass(frozen=True)
class RLProcessResult:
    argv: tuple[str, ...]
    executable: str
    pid: int
    started_at: str
    ended_at: str
    return_code: int
    cancelled_signal: int | None


PRIVATE_RUN_ROOT_RE = re.compile(r"^prime-rl-run-[0-9a-f]{32}$")


class SupervisorCancelled(InterruptedError):
    def __init__(self, signum: int):
        self.signum = signum
        self.exit_code = 128 + signum
        super().__init__(f"training transaction cancelled by signal {signum}")


@dataclass
class _CancellationState:
    signum: int | None = None
    process: object | None = None
    adapter_publication: AdapterPublicationState = field(default_factory=AdapterPublicationState)

    def handle_signal(self, signum: int, _frame: object) -> None:
        if self.signum is None:
            self.signum = signum
        process = self.process
        if process is not None and process.poll() is None:
            try:
                _signal_process_group(process.pid, signum)
            except ProcessLookupError:
                pass

    def check(self) -> None:
        if self.signum is not None:
            raise SupervisorCancelled(self.signum)

    def forward_pending(self) -> None:
        if self.signum is not None and self.process is not None and self.process.poll() is None:
            try:
                _signal_process_group(self.process.pid, self.signum)
            except ProcessLookupError:
                pass


@contextmanager
def _supervisor_cancellation() -> Iterator[_CancellationState]:
    state = _CancellationState()
    previous_handlers = {
        signum: signal.signal(signum, state.handle_signal) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        with _cancellation_scope(state.check, state.adapter_publication):
            yield state
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def generate_attempt_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{secrets.token_hex(8)}"


def _signal_process_group(pid: int, signum: int) -> None:
    os.killpg(pid, signum)


def _run_rl_process(argv: Sequence[str], cancellation: _CancellationState) -> RLProcessResult:
    argv = tuple(argv)
    executable = shutil.which(argv[0])
    if executable is None:
        raise FileNotFoundError(f"trusted RL executable is unavailable: {argv[0]}")
    executable = str(Path(executable).resolve(strict=True))
    started_at = datetime.now(timezone.utc).isoformat()
    cancellation.check()
    process = None
    try:
        process = subprocess.Popen(argv, start_new_session=True)
        cancellation.process = process
        cancellation.forward_pending()
        return_code = process.wait()
    finally:
        cancellation.process = None
    if process is None:
        raise RuntimeError("RL process was not started")
    ended_at = datetime.now(timezone.utc).isoformat()
    return RLProcessResult(
        argv=argv,
        executable=executable,
        pid=process.pid,
        started_at=started_at,
        ended_at=ended_at,
        return_code=return_code,
        cancelled_signal=cancellation.signum,
    )


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    fsync_directory(path.parent)


def create_attempt_directory(output_dir: Path, attempt_id: str) -> Path:
    output_dir = Path(output_dir).resolve(strict=True)
    attempts_dir = output_dir / "attempts"
    attempts_dir.mkdir(mode=0o755, exist_ok=True)
    if attempts_dir.is_symlink() or attempts_dir.resolve(strict=True).parent != output_dir:
        raise ValueError("attempts directory must be a canonical child of the training output")
    paths = attempt_paths(output_dir, attempt_id, require_existing=False)
    paths.directory.mkdir(mode=0o755, exist_ok=False)
    paths.run_output.mkdir(mode=0o755)
    fsync_directory(paths.directory)
    fsync_directory(attempts_dir)
    return paths.directory


def create_private_run_root() -> Path:
    tmp_root = Path("/tmp").resolve(strict=True)
    for _ in range(8):
        run_root = tmp_root / f"prime-rl-run-{secrets.token_hex(16)}"
        try:
            run_root.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return validate_run_root(run_root)
    raise FileExistsError("could not allocate a fresh private training run root")


def remove_private_run_root(run_root: Path) -> None:
    run_root = Path(run_root)
    if run_root.is_symlink():
        raise ValueError("refusing to remove a symlinked private run root")
    canonical = run_root.resolve(strict=True)
    if canonical.parent != Path("/tmp").resolve(strict=True) or not PRIVATE_RUN_ROOT_RE.fullmatch(canonical.name):
        raise ValueError(f"refusing to remove unsafe private run root: {canonical}")
    for current, directories, files in os.walk(canonical):
        current_path = Path(current)
        for name in [*directories, *files]:
            if (current_path / name).is_symlink():
                raise ValueError(f"refusing to remove private run root containing symlink: {current_path / name}")
        current_path.chmod(0o700)
    shutil.rmtree(canonical)


def prepare_training_attempt(
    *,
    manifest: FrozenEvalManifest,
    source_config_path: Path,
    artifact_output_dir: Path,
    run_root: Path,
    attempt_id: str,
) -> TrainingPreflight:
    _check_cancelled()
    if manifest.rl_config is None:
        raise ValueError("training requires a finalized RL config identity")
    artifact_output_dir = Path(artifact_output_dir).resolve(strict=True)
    paths = attempt_paths(artifact_output_dir, attempt_id, require_existing=True)
    run_root = validate_run_root(run_root)
    model_path = materialize_model(manifest.model, run_root)
    _check_cancelled()
    validate_model_materialization(manifest.model, model_path, run_root)
    dataset_path = materialize_training_dataset(manifest, run_root)
    _check_cancelled()
    _reload_and_verify_examples(manifest)
    records = _reload_training_records(manifest, dataset_path)
    validate_training_prompt_hashes(manifest, records)
    _check_cancelled()
    private_config_path = run_root / "resolved-train.toml"
    _, resolved_bytes, config_identity = resolve_effective_rl_config(
        source_config_path,
        manifest,
        source_config_rel=SOURCE_CONFIG_REL,
        model_path=model_path,
        dataset_path=dataset_path,
        output_dir=paths.run_output,
        logical_output_dir=Path(manifest.rl_config.output_dir),
        max_steps=manifest.rl_config.max_steps,
    )
    if config_identity != manifest.rl_config:
        raise ValueError("attempt resolved config identity does not match frozen manifest")
    _check_cancelled()
    _write_bytes_exclusive(private_config_path, resolved_bytes)
    private_config_path.chmod(0o444)
    _check_cancelled()
    _write_bytes_exclusive(paths.resolved_config, resolved_bytes)
    paths.resolved_config.chmod(0o444)
    _check_cancelled()
    preflight = write_training_preflight(
        output_path=paths.preflight,
        attempt_id=attempt_id,
        manifest=manifest,
        artifact_output_dir=artifact_output_dir,
        attempt_output_dir=paths.run_output,
        run_root=run_root,
        model_path=model_path,
        dataset_path=dataset_path,
        private_config_path=private_config_path,
        resolved_config_path=paths.resolved_config,
    )
    _check_cancelled()
    validate_training_prompt_hashes(manifest, _reload_training_records(manifest, dataset_path))
    marker = run_root / "training-inputs-complete.json"
    write_json_exclusive(
        marker,
        {
            "schema_version": 1,
            "status": "verified",
            "attempt_id": attempt_id,
            "manifest_identity_hash": manifest.identity_hash(),
            "resolved_config_sha256": preflight.resolved_config_sha256,
        },
    )
    _check_cancelled()
    marker.chmod(0o444)
    fsync_directory(run_root)
    run_root.chmod(0o555)
    fsync_directory(run_root.parent)
    return preflight


def _write_completion_from_process(
    *,
    manifest: FrozenEvalManifest,
    preflight: TrainingPreflight,
    process_result: RLProcessResult,
    expected_step: int,
    expected_rank: int,
) -> TrainingCompletionAttestation:
    _check_cancelled()
    paths = attempt_paths(preflight.artifact_output_dir, preflight.attempt_id, require_existing=True)
    expected_argv = ("uv", "run", "--no-sync", "rl", "@", preflight.private_config_path)
    if process_result.argv != expected_argv:
        raise ValueError("supervisor did not execute the exact trusted RL command")
    if Path(process_result.executable).name != "uv" or not Path(process_result.executable).is_absolute():
        raise ValueError("supervisor did not bind the resolved uv executable")
    if process_result.pid <= 1:
        raise ValueError("supervisor observed an invalid RL PID")
    if process_result.cancelled_signal is not None:
        raise InterruptedError("cancelled RL process cannot produce a completion attestation")
    if process_result.return_code != 0:
        raise RuntimeError(f"rl process exited with status {process_result.return_code}")
    if datetime.fromisoformat(process_result.ended_at) < datetime.fromisoformat(process_result.started_at):
        raise ValueError("RL process end time precedes start time")
    preflight_payload, preflight_sha256 = load_json_with_sha256(paths.preflight)
    loaded_preflight = TrainingPreflight.model_validate(preflight_payload)
    if loaded_preflight != preflight:
        raise ValueError("preflight changed while RL was running")
    _check_cancelled()
    _, resolved_bytes, config_identity = validate_resolved_rl_config(
        paths.resolved_config,
        manifest,
        source_config_rel=SOURCE_CONFIG_REL,
        model_path=Path(preflight.model_path),
        dataset_path=Path(preflight.dataset_path),
        output_dir=paths.run_output,
        logical_output_dir=Path(manifest.rl_config.output_dir),
        max_steps=manifest.rl_config.max_steps,
    )
    _, private_bytes, private_identity = validate_resolved_rl_config(
        Path(preflight.private_config_path),
        manifest,
        source_config_rel=SOURCE_CONFIG_REL,
        model_path=Path(preflight.model_path),
        dataset_path=Path(preflight.dataset_path),
        output_dir=paths.run_output,
        logical_output_dir=Path(manifest.rl_config.output_dir),
        max_steps=manifest.rl_config.max_steps,
    )
    if config_identity != manifest.rl_config or config_identity != preflight.rl_config:
        raise ValueError("post-process config identity does not match frozen/preflight identity")
    if (
        private_identity != config_identity
        or private_bytes != resolved_bytes
        or hashlib.sha256(resolved_bytes).hexdigest() != preflight.resolved_config_sha256
    ):
        raise ValueError("post-process private/durable config bytes or identity changed")
    _check_cancelled()
    validate_model_materialization(
        manifest.model,
        Path(preflight.model_path),
        Path(preflight.run_root),
    )
    validate_training_prompt_hashes(
        manifest,
        _reload_training_records(manifest, Path(preflight.dataset_path)),
    )
    _check_cancelled()
    _, source_adapter, _ = select_final_adapter(
        paths.run_output / "weights",
        expected_step=expected_step,
        expected_rank=expected_rank,
    )
    _check_cancelled()
    stable_marker = source_adapter.parent / "STABLE"
    attestation = TrainingCompletionAttestation(
        attempt_id=preflight.attempt_id,
        manifest_identity_hash=manifest.identity_hash(),
        preflight_sha256=preflight_sha256,
        resolved_config_sha256=preflight.resolved_config_sha256,
        rl_pid=process_result.pid,
        started_at=process_result.started_at,
        ended_at=process_result.ended_at,
        return_code=0,
        command_argv=list(process_result.argv),
        executable=process_result.executable,
        rl_config=config_identity,
        run_root=preflight.run_root,
        private_config_path=preflight.private_config_path,
        resolved_config_path=preflight.resolved_config_path,
        artifact_output_dir=preflight.artifact_output_dir,
        attempt_output_dir=preflight.attempt_output_dir,
        source_step=expected_step,
        stable_marker_path=str(stable_marker),
        stable_marker_sha256=hashlib.sha256(stable_marker.read_bytes()).hexdigest(),
        source_adapter_path=str(source_adapter),
        source_adapter_files=build_file_manifest(source_adapter),
    )
    _write_json_staged_noreplace(
        final_path=paths.completion,
        staging_path=paths.completion_staging,
        payload=attestation.model_dump(),
    )
    return attestation


def _cleanup_cancelled_attempt(preflight: TrainingPreflight, adapter_state: AdapterPublicationState) -> None:
    paths = attempt_paths(preflight.artifact_output_dir, preflight.attempt_id, require_existing=True)
    result_path = Path(preflight.artifact_output_dir) / "training-result.json"
    if not adapter_state.installed_by_invocation and not adapter_state.ownership_transferred:
        if os.path.lexists(result_path):
            raise RuntimeError("refusing to clean cancellation state after a final training result exists")
        _quarantine_owned_staging(paths.publication, expected_name="publication.json")
        _quarantine_owned_staging(paths.completion, expected_name="completion.json")
    _quarantine_owned_staging(
        paths.completion_staging,
        expected_name=".completion.json.stage",
    )
    _quarantine_owned_staging(paths.publication_staging, expected_name=".publication.stage")


def _cleanup_recovery_staging(output_dir: Path, attempt_id: str) -> None:
    paths = attempt_paths(output_dir, attempt_id, require_existing=True)
    _quarantine_owned_staging(
        paths.completion_staging,
        expected_name=".completion.json.stage",
    )
    _quarantine_owned_staging(paths.publication_staging, expected_name=".publication.stage")


def _cleanup_publication_staging(output_dir: Path, attempt_id: str) -> None:
    paths = attempt_paths(output_dir, attempt_id, require_existing=True)
    _quarantine_owned_staging(paths.publication_staging, expected_name=".publication.stage")


def supervise_prepared_attempt(
    *,
    manifest: FrozenEvalManifest,
    preflight: TrainingPreflight,
    expected_rank: int = 16,
) -> object:
    with _supervisor_cancellation() as cancellation:
        try:
            return _supervise_prepared_attempt(
                manifest=manifest,
                preflight=preflight,
                cancellation=cancellation,
                expected_rank=expected_rank,
            )
        except SupervisorCancelled as error:
            try:
                _cleanup_cancelled_attempt(preflight, cancellation.adapter_publication)
            except Exception as cleanup_error:
                error.add_note(f"cancelled-attempt cleanup failed: {cleanup_error}")
                raise error from cleanup_error
            raise
        except Exception:
            _cleanup_publication_staging(preflight.artifact_output_dir, preflight.attempt_id)
            raise


def _supervise_prepared_attempt(
    *,
    manifest: FrozenEvalManifest,
    preflight: TrainingPreflight,
    cancellation: _CancellationState,
    expected_rank: int,
) -> object:
    cancellation.check()
    if manifest.rl_config is None:
        raise ValueError("training supervision requires the frozen RL config identity")
    if Path(preflight.artifact_output_dir, "training-result.json").exists():
        raise FileExistsError("training-result.json already exists; refusing to rerun completed training")
    paths = attempt_paths(preflight.artifact_output_dir, preflight.attempt_id, require_existing=True)
    private_config = Path(preflight.private_config_path)
    run_root = Path(preflight.run_root).resolve(strict=True)
    if private_config.is_symlink() or private_config.resolve(strict=True) != run_root / "resolved-train.toml":
        raise ValueError("RL launch config is not the fixed private non-symlink config")
    _, private_bytes, private_identity = validate_resolved_rl_config(
        private_config,
        manifest,
        source_config_rel=SOURCE_CONFIG_REL,
        model_path=Path(preflight.model_path),
        dataset_path=Path(preflight.dataset_path),
        output_dir=paths.run_output,
        logical_output_dir=Path(manifest.rl_config.output_dir),
        max_steps=manifest.rl_config.max_steps,
    )
    cancellation.check()
    _, durable_bytes, durable_identity = validate_resolved_rl_config(
        paths.resolved_config,
        manifest,
        source_config_rel=SOURCE_CONFIG_REL,
        model_path=Path(preflight.model_path),
        dataset_path=Path(preflight.dataset_path),
        output_dir=paths.run_output,
        logical_output_dir=Path(manifest.rl_config.output_dir),
        max_steps=manifest.rl_config.max_steps,
    )
    if (
        private_identity != manifest.rl_config
        or private_identity != preflight.rl_config
        or durable_identity != private_identity
        or private_bytes != durable_bytes
        or hashlib.sha256(private_bytes).hexdigest() != preflight.resolved_config_sha256
    ):
        raise ValueError("private RL launch config does not match durable preflight/config identity")
    cancellation.check()
    command = ("uv", "run", "--no-sync", "rl", "@", preflight.private_config_path)
    process_result = _run_rl_process(command, cancellation)
    cancellation.check()
    if process_result.return_code != 0:
        raise RuntimeError(
            f"rl process {process_result.pid} exited with status {process_result.return_code}; "
            "no completion attestation or result was written"
        )
    metrics_path = Path(preflight.attempt_output_dir) / "metrics.jsonl"
    if metrics_path.is_symlink() or not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        raise FileNotFoundError("successful RL process did not produce the configured attempt metrics.jsonl")
    cancellation.check()
    _write_completion_from_process(
        manifest=manifest,
        preflight=preflight,
        process_result=process_result,
        expected_step=manifest.rl_config.max_steps,
        expected_rank=expected_rank,
    )
    cancellation.check()
    result = publish_training_result(
        output_dir=Path(preflight.artifact_output_dir),
        manifest=manifest,
        attempt_id=preflight.attempt_id,
        expected_step=manifest.rl_config.max_steps,
        expected_rank=expected_rank,
    )
    cancellation.check()
    return result


def run_training_attempt(
    *,
    manifest_path: Path,
    source_config_path: Path,
    artifact_output_dir: Path,
) -> object:
    with _supervisor_cancellation() as cancellation:
        artifact_output_dir = Path(artifact_output_dir).resolve(strict=True)
        cancellation.check()
        if (artifact_output_dir / "training-result.json").exists():
            raise FileExistsError("training-result.json already exists; training is already complete")
        manifest = FrozenEvalManifest.load(manifest_path)
        cancellation.check()
        if manifest.state != "finalized" or manifest.rl_config is None:
            raise ValueError("training supervisor requires a finalized manifest")
        validate_manifest_contract(
            manifest,
            expected_source_revision=manifest.source_revision,
            expected_verifiers_revision=manifest.verifiers_revision,
            expected_tasksets_revision=manifest.eval_taskset.taskset_revision,
            expected_model_name=manifest.model.name,
            expected_model_revision=manifest.model.revision,
            require_finalized=True,
        )
        cancellation.check()
        attempt_id = generate_attempt_id()
        create_attempt_directory(artifact_output_dir, attempt_id)
        print(f"[training-supervisor] attempt_id={attempt_id}", flush=True)
        cancellation.check()
        run_root = create_private_run_root()
        preflight = None
        try:
            cancellation.check()
            preflight = prepare_training_attempt(
                manifest=manifest,
                source_config_path=source_config_path,
                artifact_output_dir=artifact_output_dir,
                run_root=run_root,
                attempt_id=attempt_id,
            )
            cancellation.check()
            return _supervise_prepared_attempt(
                manifest=manifest,
                preflight=preflight,
                cancellation=cancellation,
                expected_rank=16,
            )
        except SupervisorCancelled as error:
            try:
                if preflight is not None:
                    _cleanup_cancelled_attempt(preflight, cancellation.adapter_publication)
                else:
                    paths = attempt_paths(artifact_output_dir, attempt_id, require_existing=True)
                    _quarantine_owned_staging(
                        paths.completion_staging,
                        expected_name=".completion.json.stage",
                    )
                    _quarantine_owned_staging(paths.publication_staging, expected_name=".publication.stage")
            except Exception as cleanup_error:
                error.add_note(f"cancelled-attempt cleanup failed: {cleanup_error}")
                raise error from cleanup_error
            raise
        except Exception:
            if preflight is not None:
                _cleanup_publication_staging(artifact_output_dir, attempt_id)
            raise
        finally:
            if os.path.lexists(run_root):
                remove_private_run_root(run_root)


def recover_publish(
    *,
    manifest_path: Path,
    artifact_output_dir: Path,
    attempt_id: str,
) -> object:
    with _supervisor_cancellation() as cancellation:
        try:
            cancellation.check()
            manifest = FrozenEvalManifest.load(manifest_path)
            cancellation.check()
            if manifest.state != "finalized" or manifest.rl_config is None:
                raise ValueError("publication recovery requires a finalized manifest")
            return publish_training_result(
                output_dir=artifact_output_dir,
                manifest=manifest,
                attempt_id=attempt_id,
                expected_step=manifest.rl_config.max_steps,
                expected_rank=16,
                recovery=True,
            )
        except SupervisorCancelled as error:
            try:
                _cleanup_recovery_staging(artifact_output_dir, attempt_id)
            except Exception as cleanup_error:
                error.add_note(f"recovery staging cleanup failed: {cleanup_error}")
                raise error from cleanup_error
            raise
        except Exception:
            _cleanup_publication_staging(artifact_output_dir, attempt_id)
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--output-dir", required=True)
    recover_parser = subparsers.add_parser("recover-publish")
    recover_parser.add_argument("--manifest", required=True)
    recover_parser.add_argument("--output-dir", required=True)
    recover_parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            run_training_attempt(
                manifest_path=Path(args.manifest),
                source_config_path=Path(args.config),
                artifact_output_dir=Path(args.output_dir),
            )
        else:
            recover_publish(
                manifest_path=Path(args.manifest),
                artifact_output_dir=Path(args.output_dir),
                attempt_id=args.attempt_id,
            )
    except SupervisorCancelled as error:
        print(f"[training-supervisor] {error}", flush=True)
        return error.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
