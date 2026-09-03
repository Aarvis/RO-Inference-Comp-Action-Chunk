from __future__ import annotations

import argparse
import dataclasses
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OPENPI_MODULE = ROOT / "OpenPI_Module"
if str(OPENPI_MODULE) not in sys.path:
    sys.path.insert(0, str(OPENPI_MODULE))

from origami_comp_action_chunk_runtime import load_origami_comp_action_chunk_config  # noqa: E402
from origami_comp_action_chunk_runtime.config import load_yaml  # noqa: E402
from OpenPI_Module.openpi_runtime import load_openpi_runtime_config  # noqa: E402


DEFAULT_SOURCE_CONFIG = ROOT / "configs" / "dataset_replay.yaml"
DEFAULT_BUNDLE_ROOT = ROOT / "model_bundle"
VENDORED_OPENPI_SOURCE_ROOT = ROOT / "vendor" / "openpi" / "src"


def _path_to_text(path: Path) -> str:
    return path.as_posix()


def _dataclass_to_yamlable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _dataclass_to_yamlable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return _path_to_text(value)
    if isinstance(value, tuple):
        return [_dataclass_to_yamlable(item) for item in value]
    if isinstance(value, list):
        return [_dataclass_to_yamlable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _dataclass_to_yamlable(item) for key, item in value.items()}
    return value


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _copy_file(src: Path, dst: Path, *, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst, target_is_directory=False)
    else:
        raise ValueError(f"Unsupported copy mode: {mode}")


def _copy_tree(src: Path, dst: Path, *, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copytree(src, dst)
    elif mode == "hardlink":
        shutil.copytree(src, dst, copy_function=os.link)
    elif mode == "symlink":
        os.symlink(src, dst, target_is_directory=True)
    else:
        raise ValueError(f"Unsupported copy mode: {mode}")


def _materialize(src: Path | None, dst: Path, *, label: str, mode: str, overwrite: bool) -> str:
    if src in (None, ""):
        return f"skip     : {label} has no source"
    source = Path(src).resolve()
    dst = dst.resolve()
    if source == dst:
        return f"skip     : {label} already points at {dst}"
    if not source.exists():
        return f"missing  : {label} source not found -> {source}"
    if mode == "none":
        return f"planned  : {label} -> {dst}"
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return f"exists   : {label} -> {dst}"
        _remove_existing(dst)
    if source.is_dir():
        _copy_tree(source, dst, mode=mode)
    else:
        _copy_file(source, dst, mode=mode)
    return f"{mode:8s}: {label} -> {dst}"


def _resolve_config_path(raw_path: str | Path | None, *, config_dir: Path) -> Path | None:
    if raw_path in (None, ""):
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    return path


def _rewrite_ooi_config(source_config: Path, bundle_config: Path, dino_dir_name: str) -> None:
    payload = load_yaml(source_config)
    paths = dict(payload.get("paths", {}))
    runtime = dict(payload.get("runtime", {}))
    paths["checkpoint"] = "../weights/ooi/best.pt"
    paths["output_dir"] = "../outputs/ooi_smoke_test"
    runtime["dino_model_name_or_path"] = f"../weights/dino/{dino_dir_name}"
    runtime["model_cache_dir"] = "../weights/dino"
    runtime["local_files_only"] = True
    runtime["trust_remote_code"] = False
    runtime["shared_dino"] = True
    payload["paths"] = paths
    payload["runtime"] = runtime
    _write_yaml(bundle_config, payload)


def _rewrite_planner_config(source_config: Path, bundle_config: Path) -> None:
    payload = load_yaml(source_config)
    paths = dict(payload.get("paths", {}))
    paths["manifest_root"] = "../weights/checkpoint_planner/manifest"
    payload["paths"] = paths
    _write_yaml(bundle_config, payload)


def _rewrite_openpi_config(source_config: Path, bundle_config: Path, checkpoint_name: str) -> None:
    payload = load_yaml(source_config)
    paths = dict(payload.get("paths", {}))
    paths["checkpoint_dir"] = f"../weights/openpi/{checkpoint_name}"
    paths["tokenizer_model_path"] = "../assets/paligemma_tokenizer.model"
    paths.pop("openpi_source_root", None)
    payload["paths"] = paths
    _write_yaml(bundle_config, payload)


def _find_openpi_project_root(openpi_source_root: Path | None) -> Path | None:
    if openpi_source_root is None:
        return None
    source_root = openpi_source_root.resolve()
    if (source_root / "openpi").is_dir():
        return source_root.parent
    return source_root


def _find_openpi_client_source(openpi_source_root: Path | None) -> Path | None:
    if openpi_source_root is not None and (openpi_source_root / "openpi_client").is_dir():
        return openpi_source_root / "openpi_client"
    project_root = _find_openpi_project_root(openpi_source_root)
    if project_root is None:
        return None
    candidates = (
        project_root / "packages" / "openpi-client" / "src" / "openpi_client",
        project_root.parent / "packages" / "openpi-client" / "src" / "openpi_client",
    )
    return next((path for path in candidates if path.is_dir()), None)


def _find_future_latent_source(openpi_source_root: Path | None) -> Path | None:
    if openpi_source_root is not None and (openpi_source_root / "future_latent_predictor").is_dir():
        return openpi_source_root / "future_latent_predictor"
    project_root = _find_openpi_project_root(openpi_source_root)
    if project_root is None:
        return None
    candidates = (
        project_root / "future_latent_predictor",
        project_root.parent / "future_latent_predictor",
    )
    return next((path for path in candidates if path.is_dir()), None)


def _infer_openpi_source_root(configured_root: Path | None) -> Path | None:
    if configured_root is not None:
        return configured_root
    candidates = (
        ROOT.parent / "openpi" / "src",
        VENDORED_OPENPI_SOURCE_ROOT,
    )
    return next((path.resolve() for path in candidates if (path / "openpi").is_dir()), None)


def prepare_bundle(
    *,
    source_config_path: Path,
    bundle_root: Path,
    copy_mode: str,
    overwrite: bool,
    include_openpi_source: bool,
) -> list[str]:
    config = load_origami_comp_action_chunk_config(source_config_path)
    openpi_config = load_openpi_runtime_config(config.paths.openpi_runtime_config)
    ooi_payload = load_yaml(config.paths.ooi_config)
    ooi_paths = ooi_payload.get("paths", {})
    if not isinstance(ooi_paths, dict):
        ooi_paths = {}
    ooi_checkpoint_source = config.paths.ooi_checkpoint or _resolve_config_path(
        ooi_paths.get("checkpoint"),
        config_dir=config.paths.ooi_config.parent,
    )

    dino_source = Path(str(config.paths.dino_model_name_or_path)).resolve() if config.paths.dino_model_name_or_path else None
    dino_dir_name = dino_source.name if dino_source is not None else "dinov3-vits16plus-pretrain-lvd1689m"
    openpi_checkpoint_name = openpi_config.checkpoint_dir.name

    bundle_root.mkdir(parents=True, exist_ok=True)
    configs_dir = bundle_root / "configs"
    _rewrite_ooi_config(config.paths.ooi_config, configs_dir / "ooi_zoom_runtime.yaml", dino_dir_name)
    _rewrite_planner_config(config.paths.checkpoint_planner_config, configs_dir / "checkpoint_planner_runtime_model.yaml")
    _rewrite_openpi_config(
        config.paths.openpi_runtime_config,
        configs_dir / "openpi_comp_action_chunk_runtime.yaml",
        openpi_checkpoint_name,
    )

    bundle_payload = {
        "bundle": {
            "format": "origami-comp-action-chunk-bundle-v1",
            "inference_kit": "origami-inference-kit-async",
            "policy_name": str(config.server.policy_name),
        },
        "paths": {
            "dataset_root": "dataset_replay",
            "output_dir": "outputs/dataset_replay",
            "ooi_config": "configs/ooi_zoom_runtime.yaml",
            "ooi_checkpoint": "weights/ooi/best.pt",
            "checkpoint_planner_config": "configs/checkpoint_planner_runtime_model.yaml",
            "checkpoint_planner_checkpoint": "weights/checkpoint_planner/checkpoint.pt",
            "checkpoint_planner_manifest_root": "weights/checkpoint_planner/manifest",
            "openpi_runtime_config": "configs/openpi_comp_action_chunk_runtime.yaml",
            "dino_model_name_or_path": f"./weights/dino/{dino_dir_name}",
            "dino_model_cache_dir": "weights/dino",
        },
        "runtime": _dataclass_to_yamlable(config.runtime),
        "server": _dataclass_to_yamlable(config.server),
        "dataset_replay": _dataclass_to_yamlable(config.dataset_replay),
    }
    bundle_payload["runtime"]["dino_local_files_only"] = True
    bundle_payload["runtime"]["dino_trust_remote_code"] = False
    _write_yaml(bundle_root / "bundle.yaml", bundle_payload)

    messages = [
        f"config   : bundle.yaml -> {bundle_root / 'bundle.yaml'}",
        f"config   : OOI -> {configs_dir / 'ooi_zoom_runtime.yaml'}",
        f"config   : checkpoint planner -> {configs_dir / 'checkpoint_planner_runtime_model.yaml'}",
        f"config   : OpenPI -> {configs_dir / 'openpi_comp_action_chunk_runtime.yaml'}",
    ]

    messages.append(
        _materialize(
            ooi_checkpoint_source,
            bundle_root / "weights" / "ooi" / "best.pt",
            label="OOI checkpoint",
            mode=copy_mode,
            overwrite=overwrite,
        )
    )
    messages.append(
        _materialize(
            config.paths.checkpoint_planner_checkpoint,
            bundle_root / "weights" / "checkpoint_planner" / "checkpoint.pt",
            label="Checkpoint planner checkpoint",
            mode=copy_mode,
            overwrite=overwrite,
        )
    )
    messages.append(
        _materialize(
            config.paths.checkpoint_planner_manifest_root,
            bundle_root / "weights" / "checkpoint_planner" / "manifest",
            label="Checkpoint planner manifest",
            mode=copy_mode,
            overwrite=overwrite,
        )
    )
    messages.append(
        _materialize(
            dino_source,
            bundle_root / "weights" / "dino" / dino_dir_name,
            label="DINO model directory",
            mode=copy_mode,
            overwrite=overwrite,
        )
    )
    messages.append(
        _materialize(
            openpi_config.checkpoint_dir,
            bundle_root / "weights" / "openpi" / openpi_checkpoint_name,
            label="OpenPI checkpoint directory",
            mode=copy_mode,
            overwrite=overwrite,
        )
    )
    messages.append(
        _materialize(
            openpi_config.tokenizer_model_path,
            bundle_root / "assets" / "paligemma_tokenizer.model",
            label="PaliGemma tokenizer",
            mode=copy_mode,
            overwrite=overwrite,
        )
    )
    if include_openpi_source:
        openpi_source_root = _infer_openpi_source_root(
            openpi_config.vendored_source_root or openpi_config.openpi_source_root
        )
        vendored_source_root = VENDORED_OPENPI_SOURCE_ROOT
        messages.append(
            _materialize(
                openpi_source_root,
                vendored_source_root,
                label="Project-vendored OpenPI source root",
                mode=copy_mode,
                overwrite=overwrite,
            )
        )
        messages.append(
            _materialize(
                _find_openpi_client_source(openpi_source_root),
                vendored_source_root / "openpi_client",
                label="Project-vendored OpenPI client package",
                mode=copy_mode,
                overwrite=overwrite,
            )
        )
        messages.append(
            _materialize(
                _find_future_latent_source(openpi_source_root),
                vendored_source_root / "future_latent_predictor",
                label="Project-vendored OpenPI future-latent helper package",
                mode=copy_mode,
                overwrite=overwrite,
            )
        )

    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a one-root model bundle for comp-action-chunk inference.")
    parser.add_argument("--source-config", default=str(DEFAULT_SOURCE_CONFIG), help="Existing runtime config to bundle.")
    parser.add_argument("--bundle-root", default=str(DEFAULT_BUNDLE_ROOT), help="Output bundle directory.")
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink", "symlink", "none"),
        default="copy",
        help="How to materialize large files. Use none for a dry-run layout update.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing destination files/directories.")
    parser.add_argument(
        "--include-openpi-source",
        action="store_true",
        help="Copy/link OpenPI source into project vendor/openpi/src. It is not stored in model_bundle.",
    )
    args = parser.parse_args()

    messages = prepare_bundle(
        source_config_path=Path(args.source_config).resolve(),
        bundle_root=Path(args.bundle_root).resolve(),
        copy_mode=str(args.copy_mode),
        overwrite=bool(args.overwrite),
        include_openpi_source=bool(args.include_openpi_source),
    )
    print("Prepared comp-action-chunk model bundle")
    for message in messages:
        print(f"  {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
