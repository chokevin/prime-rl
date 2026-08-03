"""Validate and stage the checked-in Tau image pin.

Every checked-in ``tau/*.yaml`` target carries the fully-qualified public digest from
``tau/image.pin.json``. This helper rejects any target whose ``runtime.image`` drifts
from that pin, then copies the validated targets and their entrypoint script into
``tau/.rendered/``. Validation happens for every template before any output is written,
so a partial re-pin cannot leave a mixed rendered directory.

Usage:
    python3 tau/render_image.py                 # render every tau/*.yaml -> tau/.rendered/
    python3 tau/render_image.py --check-only     # validate the pin file only, write nothing
    python3 tau/render_image.py --pin other.json --out-dir /path

Rendered files are written under `tau/.rendered/` (gitignored -- see `.gitignore`) and are
what `tau run validate --config ...` / `tau run --config ... --dry-run=client` /
`tau run --config ...` must be pointed at. Use the rendered paths so every operation is
preceded by pin validation and uses the mirrored entrypoint beside the rendered config.

`entrypoint:` paths resolve relative to *the config file's own directory* (confirmed via
`tau run --dry-run=client`), so this script also mirrors `tau/scripts/` into the output
directory -- a rendered `tau/.rendered/<target>.yaml`'s `entrypoint: scripts/run-prime-
rl.sh` needs `tau/.rendered/scripts/run-prime-rl.sh` to exist, not just the original.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
IMAGE_LINE_RE = re.compile(r"^  image: (.+)$", re.MULTILINE)
# A short (7-12 char) or full (40 char) hex commit SHA; upstream tags this repo observes
# use short (9-char) SHAs (e.g. "bbb90a1b4"), so accept either length.
SOURCE_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
FORBIDDEN_DIGESTS = {"latest", ""}


class PinError(ValueError):
    """Raised when the image pin or a checked-in target is invalid."""


def load_pin(pin_path: Path) -> dict:
    if not pin_path.exists():
        raise PinError(
            f"{pin_path} does not exist -- resolve and record a real image digest first (see tau/README.md's 'Image and overlay strategy')."
        )
    try:
        pin = json.loads(pin_path.read_text())
    except json.JSONDecodeError as exc:
        raise PinError(f"{pin_path} is not valid JSON: {exc}") from exc

    for field in ("image_repo", "digest", "source_commit"):
        value = pin.get(field)
        if not value or not isinstance(value, str):
            raise PinError(f"{pin_path} is missing a non-empty string '{field}' field.")

    digest = pin["digest"]
    if digest.strip().lower() in FORBIDDEN_DIGESTS or not DIGEST_RE.match(digest):
        raise PinError(
            f"{pin_path}'s digest {digest!r} is not a real 'sha256:<64-hex>' digest "
            "(refusing a placeholder, empty, or 'latest' value)."
        )

    source_commit = pin["source_commit"]
    if source_commit.strip().lower() in FORBIDDEN_DIGESTS or not SOURCE_COMMIT_RE.match(source_commit):
        raise PinError(
            f"{pin_path}'s source_commit {source_commit!r} is not a plausible git commit "
            "SHA (refusing a placeholder or empty value)."
        )

    return pin


def pinned_image_ref(pin: dict) -> str:
    return f"{pin['image_repo']}@{pin['digest']}"


def load_validated_template(src: Path, image_ref: str) -> str:
    """Return a template only when its checked-in image exactly matches the pin."""
    text = src.read_text()
    image_lines = IMAGE_LINE_RE.findall(text)
    if image_lines != [image_ref]:
        raise PinError(f"{src} must contain exactly one runtime.image equal to {image_ref!r}; got {image_lines!r}")
    return text


def sync_scripts_dir(tau_dir: Path, out_dir: Path) -> bool:
    """Mirror `tau/scripts/` into the rendered output directory. `entrypoint:` paths
    resolve relative to *the config file's own directory* (confirmed empirically via
    `tau run --dry-run=client` — see tau/README.md), so a rendered
    `tau/.rendered/<target>.yaml` referencing `entrypoint: scripts/run-prime-rl.sh`
    needs `tau/.rendered/scripts/run-prime-rl.sh` to actually exist, not just
    `tau/scripts/run-prime-rl.sh`."""
    src = tau_dir / "scripts"
    if not src.is_dir():
        return False
    dst = out_dir / "scripts"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pin", type=Path, default=Path("tau/image.pin.json"), help="Path to the image pin JSON.")
    parser.add_argument(
        "--tau-dir", type=Path, default=Path("tau"), help="Directory containing the tau/*.yaml templates."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write rendered files (default: <tau-dir>/.rendered).",
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Validate the pin file only; write no rendered files."
    )
    args = parser.parse_args(argv)

    try:
        pin = load_pin(args.pin)
    except PinError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    image_ref = pinned_image_ref(pin)
    print(f"ok: pin file valid -- image={image_ref} source_commit={pin['source_commit']}")
    if args.check_only:
        return 0

    out_dir = args.out_dir or (args.tau_dir / ".rendered")
    templates = sorted(args.tau_dir.glob("*.yaml"))
    if not templates:
        print(f"error: no *.yaml templates found under {args.tau_dir}", file=sys.stderr)
        return 1
    try:
        rendered = {template: load_validated_template(template, image_ref) for template in templates}
    except PinError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for template, text in rendered.items():
        dst = out_dir / template.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)
        print(f"ok: validated and staged {template} -> {dst}")

    if sync_scripts_dir(args.tau_dir, out_dir):
        print(
            f"ok: mirrored {args.tau_dir / 'scripts'} -> {out_dir / 'scripts'} (entrypoint paths resolve relative to the rendered config's own directory)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
