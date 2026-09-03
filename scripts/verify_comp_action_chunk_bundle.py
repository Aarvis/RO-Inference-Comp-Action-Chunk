from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OPENPI_MODULE = ROOT / "OpenPI_Module"
if str(OPENPI_MODULE) not in sys.path:
    sys.path.insert(0, str(OPENPI_MODULE))

from origami_comp_action_chunk_runtime import load_origami_comp_action_chunk_bundle  # noqa: E402
from OpenPI_Module.openpi_runtime import load_openpi_runtime_config  # noqa: E402


@dataclass(frozen=True)
class CheckResult:
    label: str
    path: Path
    status: str = "ok"


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _ensure_inside_bundle(path: Path, *, bundle_root: Path, label: str) -> None:
    if not _inside(path, bundle_root):
        raise FileNotFoundError(f"{label} must live inside the model bundle: {path}")


def _check_file(
    path: Path,
    *,
    label: str,
    bundle_root: Path,
    allow_missing: bool,
    enforce_inside: bool = True,
) -> CheckResult:
    resolved = path.resolve()
    if enforce_inside:
        _ensure_inside_bundle(resolved, bundle_root=bundle_root, label=label)
    if not resolved.is_file():
        if allow_missing:
            return CheckResult(label=label, path=resolved, status="missing")
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return CheckResult(label=label, path=resolved)


def _check_dir(
    path: Path,
    *,
    label: str,
    bundle_root: Path,
    allow_missing: bool,
    enforce_inside: bool = True,
) -> CheckResult:
    resolved = path.resolve()
    if enforce_inside:
        _ensure_inside_bundle(resolved, bundle_root=bundle_root, label=label)
    if not resolved.is_dir():
        if allow_missing:
            return CheckResult(label=label, path=resolved, status="missing")
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return CheckResult(label=label, path=resolved)


def _read_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping in {path}, got {type(payload)!r}")
    return payload


def _iter_dino_weight_names(path: Path) -> Iterable[str]:
    for name in ("model.safetensors", "pytorch_model.bin"):
        yield name


def verify_bundle(
    bundle_root_or_config: Path,
    *,
    allow_missing_model_files: bool,
    allow_external_code: bool,
    require_dataset_replay: bool,
) -> list[CheckResult]:
    bundle_config = bundle_root_or_config
    if bundle_config.is_dir():
        bundle_root = bundle_config.resolve()
        bundle_config = bundle_root / "bundle.yaml"
    else:
        bundle_config = bundle_config.resolve()
        bundle_root = bundle_config.parent

    config = load_origami_comp_action_chunk_bundle(bundle_config)
    if config.runtime.camera_image_size != 224:
        raise ValueError(f"runtime.camera_image_size must be 224, got {config.runtime.camera_image_size}")
    if config.server.action_dim != 65:
        raise ValueError(f"server.action_dim must be 65, got {config.server.action_dim}")
    if config.server.action_horizon != 10:
        raise ValueError(f"server.action_horizon must be 10, got {config.server.action_horizon}")
    if config.server.execution_mode not in {"sync", "async"}:
        raise ValueError("server.execution_mode must be sync or async")
    if not config.runtime.dino_local_files_only:
        raise ValueError("runtime.dino_local_files_only must be true for offline inference")
    if config.runtime.dino_trust_remote_code:
        raise ValueError("runtime.dino_trust_remote_code must be false for offline inference")

    results = [
        _check_file(bundle_config, label="Bundle config", bundle_root=bundle_root, allow_missing=False),
        _check_file(config.paths.ooi_config, label="OOI runtime config", bundle_root=bundle_root, allow_missing=False),
        _check_file(
            config.paths.checkpoint_planner_config,
            label="Checkpoint planner runtime config",
            bundle_root=bundle_root,
            allow_missing=False,
        ),
        _check_file(
            config.paths.openpi_runtime_config,
            label="OpenPI runtime config",
            bundle_root=bundle_root,
            allow_missing=False,
        ),
        _check_file(
            config.paths.ooi_checkpoint,
            label="OOI checkpoint",
            bundle_root=bundle_root,
            allow_missing=allow_missing_model_files,
        ),
        _check_file(
            config.paths.checkpoint_planner_checkpoint,
            label="Checkpoint planner checkpoint",
            bundle_root=bundle_root,
            allow_missing=allow_missing_model_files,
        ),
    ]

    planner_manifest_root = config.paths.checkpoint_planner_manifest_root
    if planner_manifest_root is None:
        planner_payload = _read_yaml(config.paths.checkpoint_planner_config)
        planner_paths = planner_payload.get("paths", {})
        if not isinstance(planner_paths, dict):
            planner_paths = {}
        raw_manifest = planner_paths.get("manifest_root") or planner_paths.get("manifest_path")
        if raw_manifest in (None, ""):
            raise ValueError("Checkpoint planner config must define paths.manifest_root or bundle paths.checkpoint_planner_manifest_root")
        planner_manifest_root = Path(str(raw_manifest))
        if not planner_manifest_root.is_absolute():
            planner_manifest_root = (config.paths.checkpoint_planner_config.parent / planner_manifest_root).resolve()

    if planner_manifest_root.suffix:
        results.append(
            _check_file(
                planner_manifest_root,
                label="Checkpoint planner manifest",
                bundle_root=bundle_root,
                allow_missing=allow_missing_model_files,
            )
        )
    else:
        results.extend(
            [
                _check_dir(
                    planner_manifest_root,
                    label="Checkpoint planner manifest root",
                    bundle_root=bundle_root,
                    allow_missing=allow_missing_model_files,
                ),
                _check_file(
                    planner_manifest_root / "manifest.json",
                    label="Checkpoint planner manifest",
                    bundle_root=bundle_root,
                    allow_missing=allow_missing_model_files,
                ),
            ]
        )

    dino_model = config.paths.dino_model_name_or_path
    if not dino_model:
        raise ValueError("paths.dino_model_name_or_path must point to bundled DINO weights")
    dino_dir = Path(str(dino_model))
    results.append(
        _check_dir(
            dino_dir,
            label="DINO model directory",
            bundle_root=bundle_root,
            allow_missing=allow_missing_model_files,
        )
    )
    results.extend(
        [
            _check_file(
                dino_dir / "config.json",
                label="DINO config.json",
                bundle_root=bundle_root,
                allow_missing=allow_missing_model_files,
            ),
            _check_file(
                dino_dir / "preprocessor_config.json",
                label="DINO preprocessor_config.json",
                bundle_root=bundle_root,
                allow_missing=allow_missing_model_files,
            ),
        ]
    )
    if not any((dino_dir / name).is_file() for name in _iter_dino_weight_names(dino_dir)):
        if allow_missing_model_files:
            results.append(CheckResult(label="DINO model weights", path=dino_dir, status="missing"))
        else:
            raise FileNotFoundError(f"DINO weights not found under {dino_dir}")

    openpi_config = load_openpi_runtime_config(config.paths.openpi_runtime_config)
    if openpi_config.action_horizon != config.server.action_horizon:
        raise ValueError(
            f"OpenPI action_horizon={openpi_config.action_horizon} does not match server.action_horizon={config.server.action_horizon}"
        )
    if openpi_config.action_dim != 65 or openpi_config.state_dim != 65 or openpi_config.tactile_dim != 60:
        raise ValueError("OpenPI runtime config must use action_dim=65, state_dim=65, tactile_dim=60")
    if openpi_config.image_size != 224 or openpi_config.tactile_image_size != 224:
        raise ValueError("OpenPI runtime config must use image_size=224 and tactile_image_size=224")

    results.extend(
        [
            _check_dir(
                openpi_config.checkpoint_dir,
                label="OpenPI checkpoint directory",
                bundle_root=bundle_root,
                allow_missing=allow_missing_model_files,
            ),
            _check_dir(
                openpi_config.checkpoint_dir / "params",
                label="OpenPI params directory",
                bundle_root=bundle_root,
                allow_missing=allow_missing_model_files,
            ),
            _check_dir(
                openpi_config.checkpoint_dir / "assets",
                label="OpenPI assets directory",
                bundle_root=bundle_root,
                allow_missing=allow_missing_model_files,
            ),
            _check_file(
                openpi_config.tokenizer_model_path,
                label="PaliGemma tokenizer model",
                bundle_root=bundle_root,
                allow_missing=allow_missing_model_files,
            ),
        ]
    )
    if openpi_config.asset_id not in (None, ""):
        asset_dir = openpi_config.checkpoint_dir / "assets" / str(openpi_config.asset_id)
        results.extend(
            [
                _check_dir(
                    asset_dir,
                    label="OpenPI asset directory",
                    bundle_root=bundle_root,
                    allow_missing=allow_missing_model_files,
                ),
                _check_file(
                    asset_dir / "norm_stats.json",
                    label="OpenPI norm_stats.json",
                    bundle_root=bundle_root,
                    allow_missing=allow_missing_model_files,
                ),
            ]
        )

    if openpi_config.openpi_source_root is not None:
        results.append(
            _check_dir(
                openpi_config.openpi_source_root,
                label="OpenPI source root",
                bundle_root=bundle_root,
                allow_missing=allow_missing_model_files,
                enforce_inside=not allow_external_code,
            )
        )
        results.append(
            _check_dir(
                openpi_config.openpi_source_root / "openpi",
                label="OpenPI python package",
                bundle_root=bundle_root,
                allow_missing=allow_missing_model_files,
                enforce_inside=not allow_external_code,
            )
        )
        results.append(
            _check_dir(
                openpi_config.openpi_source_root / "openpi_client",
                label="OpenPI client python package",
                bundle_root=bundle_root,
                allow_missing=allow_missing_model_files,
                enforce_inside=not allow_external_code,
            )
        )
        results.append(
            _check_dir(
                openpi_config.openpi_source_root / "future_latent_predictor",
                label="OpenPI future-latent helper package",
                bundle_root=bundle_root,
                allow_missing=allow_missing_model_files,
                enforce_inside=not allow_external_code,
            )
        )

    if require_dataset_replay:
        results.append(
            _check_dir(
                config.paths.dataset_root,
                label="Dataset replay root",
                bundle_root=bundle_root,
                allow_missing=allow_missing_model_files,
            )
        )
        results.append(
            _check_dir(
                config.paths.dataset_root / "episodes",
                label="Dataset replay episodes directory",
                bundle_root=bundle_root,
                allow_missing=allow_missing_model_files,
            )
        )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the self-contained comp-action-chunk model bundle.")
    parser.add_argument(
        "--bundle-root",
        default=str(ROOT / "model_bundle"),
        help="Bundle directory or bundle.yaml path.",
    )
    parser.add_argument(
        "--allow-missing-model-files",
        action="store_true",
        help="Validate layout and relative paths before heavyweight weights are copied.",
    )
    parser.add_argument(
        "--allow-external-code",
        action="store_true",
        help="Allow paths.openpi_source_root to point outside the bundle.",
    )
    parser.add_argument(
        "--require-dataset-replay",
        action="store_true",
        help="Also require bundled dataset_replay/episodes for offline replay smoke tests.",
    )
    args = parser.parse_args()

    bundle_root = Path(args.bundle_root).resolve()
    results = verify_bundle(
        bundle_root,
        allow_missing_model_files=bool(args.allow_missing_model_files),
        allow_external_code=bool(args.allow_external_code),
        require_dataset_replay=bool(args.require_dataset_replay),
    )

    print("Comp-action-chunk model bundle verification")
    print(f"  bundle_root: {bundle_root}")
    for result in results:
        print(f"  {result.status:7s}: {result.label} -> {result.path}")
    missing = [result for result in results if result.status == "missing"]
    if missing and not args.allow_missing_model_files:
        return 1
    if missing:
        print(f"  missing : {len(missing)} heavyweight/model paths still need to be populated")
    else:
        print("  status  : ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
