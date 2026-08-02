"""Render helper for the Tau image pin: substitutes the placeholder in every checked-in
`tau/*.yaml` template's `runtime.image` field with the fully-qualified, real digest
recorded in `tau/image.pin.json`.

Why this exists (not just hardcoding the digest into the YAML): the checked-in templates
must never contain a real or fake digest by themselves (`latest` or a placeholder that
*looks* like a digest is exactly what a careless copy-paste would produce and exactly
what silently rots). Making the digest live in exactly one place (`image.pin.json`) and
requiring an explicit render step before `tau run validate`/`tau run --config ... --dry-
run=client` means:

  - re-pinning to a new digest is a one-line JSON edit, not N find-and-replace edits
    across every target file;
  - this script itself refuses to render (nonzero exit, no output written) if the pin
    file is missing, malformed, or contains a placeholder/`latest` value -- so a
    half-finished re-pin can't silently produce a runnable-looking config.

Empirically (tested against this exact sentinel via `tau run --dry-run=client`), Tau's
client-side validate/dry-run does *not* reject an unrendered `runtime.image` string --
it passes the literal sentinel straight through into the rendered `batch/v1 Job`. The
real safety net is one step later: `RENDER_REQUIRED__see_tau/render_image.py` is not a
resolvable image reference, so an accidental un-rendered submit fails loudly at
image-pull time (`ErrImagePull`/`ImagePullBackOff`), never silently runs against
whatever `runtime.image` last happened to resolve to. Always render first and point at
`tau/.rendered/*.yaml` so this never has to be relied on.

Usage:
    python3 tau/render_image.py                 # render every tau/*.yaml -> tau/.rendered/
    python3 tau/render_image.py --check-only     # validate the pin file only, write nothing
    python3 tau/render_image.py --pin other.json --out-dir /path

Rendered files are written under `tau/.rendered/` (gitignored -- see `.gitignore`) and are
what `tau run validate --config ...` / `tau run --config ... --dry-run=client` /
`tau run --config ...` must be pointed at. Never invoke `tau run <target>` by its bare
positional name for these targets -- that resolves the *unrendered* `tau/<target>.yaml`
template, which still carries the sentinel.

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

PLACEHOLDER = "RENDER_REQUIRED__see_tau/render_image.py"
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
# A short (7-12 char) or full (40 char) hex commit SHA; upstream tags this repo observes
# use short (9-char) SHAs (e.g. "bbb90a1b4"), so accept either length.
SOURCE_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
FORBIDDEN_DIGESTS = {"latest", "", PLACEHOLDER}


class PinError(ValueError):
    """Raised when `image.pin.json` is missing, malformed, or carries a placeholder."""


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


def render_file(src: Path, dst: Path, image_ref: str) -> bool:
    """Substitute the placeholder in `src`, writing to `dst`. Returns True if the
    placeholder was found (and substituted), False if `src` had nothing to render
    (e.g. a target file with no runtime.image field yet, or already-rendered input)."""
    text = src.read_text()
    if PLACEHOLDER not in text:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text.replace(PLACEHOLDER, image_ref))
    return True


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
    rendered_any = False
    for template in sorted(args.tau_dir.glob("*.yaml")):
        dst = out_dir / template.name
        if render_file(template, dst, image_ref):
            print(f"ok: rendered {template} -> {dst}")
            rendered_any = True
    if not rendered_any:
        print(
            f"warning: no *.yaml under {args.tau_dir} contained the {PLACEHOLDER!r} sentinel -- nothing rendered.",
            file=sys.stderr,
        )
        return 0

    if sync_scripts_dir(args.tau_dir, out_dir):
        print(
            f"ok: mirrored {args.tau_dir / 'scripts'} -> {out_dir / 'scripts'} (entrypoint paths resolve relative to the rendered config's own directory)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
