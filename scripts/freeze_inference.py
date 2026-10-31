"""Snapshot inference code and declared assets; never run a model or score a bank."""

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
ASSET_LIMIT = 16 * 1024**2


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root, ignore_python_cache=False):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not supported in a frozen snapshot: {path}")
        relative = path.relative_to(root)
        cache = "__pycache__" in relative.parts or path.suffix == ".pyc"
        if ignore_python_cache and relative.parts[0] != "checkpoint" and cache:
            continue
        if path.is_file():
            result[relative.as_posix()] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    return result


def verify_snapshot(snapshot):
    snapshot = Path(snapshot).resolve()
    manifest = json.loads((snapshot / "freeze.json").read_text(encoding="utf-8"))
    actual = inventory(snapshot, ignore_python_cache=True)
    actual.pop("freeze.json", None)
    if actual != manifest["files"]:
        raise ValueError("Frozen files changed, disappeared, or were added")
    method = Path(manifest["method"])
    if method.is_absolute() or ".." in method.parts or method.as_posix() not in actual:
        raise ValueError("Frozen method must be a recorded relative path")
    if manifest["checkpoint"] not in (None, "checkpoint"):
        raise ValueError("Invalid frozen checkpoint path")
    counted = sum(v["bytes"] for k, v in actual.items() if k.startswith("checkpoint/"))
    embedded = manifest["declared_embedded_asset_bytes"]
    if type(embedded) is not int or embedded < 0:
        raise ValueError("Invalid additional asset declaration")
    if counted + embedded != manifest["declared_inference_asset_bytes"] or counted + embedded > ASSET_LIMIT:
        raise ValueError("Inference asset inventory is inconsistent or exceeds 16 MiB")
    return manifest


def freeze(root, method, checkpoint, output, final_selection=False, embedded_asset_bytes=0):
    root, method, output = Path(root).resolve(), Path(method).resolve(), Path(output).resolve()
    if not method.is_relative_to(root / "methods") or method.suffix != ".py":
        raise ValueError("Place the method and its source helpers in methods/")
    if type(embedded_asset_bytes) is not int or embedded_asset_bytes < 0:
        raise ValueError("Additional asset bytes must be a nonnegative integer")
    release = json.loads((root / "code/PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    for relative, expected in release["files"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"Official release file changed: {relative}")
    if output.is_relative_to(root / "methods"):
        raise ValueError("Write the snapshot outside methods/")
    method_files = inventory(root / "methods", ignore_python_cache=True)
    if any(not name.endswith(".py") for name in method_files):
        raise ValueError("Move non-source method assets into the checkpoint directory so they are counted")
    checkpoint = Path(checkpoint).resolve() if checkpoint else None
    if checkpoint and not checkpoint.is_dir():
        raise ValueError("Checkpoint must be a directory containing all inference assets")
    if checkpoint and (output.is_relative_to(checkpoint) or checkpoint.is_relative_to(output)):
        raise ValueError("Snapshot and checkpoint directories must not contain each other")
    asset_files = inventory(checkpoint) if checkpoint else {}
    if checkpoint and not asset_files:
        raise ValueError("The supplied checkpoint directory is empty")
    asset_bytes = sum(v["bytes"] for v in asset_files.values()) + embedded_asset_bytes
    if asset_bytes > ASSET_LIMIT:
        raise ValueError("Declared uncompressed inference assets exceed 16 MiB")
    output.mkdir(parents=True, exist_ok=False)
    source_names = [f"methods/{name}" for name in method_files]
    source_names += ["code/common.py", "code/predict.py", "code/score_predictions.py", "code/requirements.txt",
                     "scripts/freeze_inference.py", "scripts/evaluate_frozen.py"]
    source_names += [p.name for p in root.glob("requirements*.txt") if p.is_file()]
    for name in sorted(set(source_names)):
        source = root / name
        if source.is_symlink():
            raise ValueError(f"Source symlink not supported: {name}")
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in asset_files:
        target = output / "checkpoint" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkpoint / name, target)
    manifest = {"format_version": 1,
                "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                "purpose": "inference_snapshot_not_a_training_or_performance_result",
                "method": method.relative_to(root).as_posix(),
                "checkpoint": "checkpoint" if checkpoint else None,
                "selection_frozen_before_test": bool(final_selection),
                "declared_embedded_asset_bytes": embedded_asset_bytes,
                "declared_inference_asset_bytes": asset_bytes,
                "asset_limit_bytes": ASSET_LIMIT, "files": inventory(output, ignore_python_cache=True),
                "official_release_sha256": release["files"],
                "asset_review_note": "Keep all non-source inference assets in checkpoint. Checkpoint contents are counted automatically. Inspect all inference source and declare embedded assets; hashes cannot prove that declaration complete.",
                "training_reproduction_note": "This freezes inference only; training code, search records, report and disclosures remain separate deliverables."}
    (output / "freeze.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    verify_snapshot(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", type=Path, default=ROOT / "methods/selector.py")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-selection", action="store_true", help="Mark settings selected before the first final test")
    parser.add_argument("--embedded-asset-bytes", type=int, default=0, help="Manually audited assets embedded in source; all other assets must be in checkpoint")
    args = parser.parse_args()
    result = freeze(ROOT, args.method, args.checkpoint, args.output, args.final_selection, args.embedded_asset_bytes)
    print(json.dumps({k: v for k, v in result.items() if k not in ("files", "official_release_sha256")}, indent=2))


if __name__ == "__main__":
    main()
