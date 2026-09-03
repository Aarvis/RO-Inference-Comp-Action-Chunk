from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from pathlib import Path

from origami_comp_action_chunk_runtime import CompActionChunkPipeline, load_runtime_config
from origami_comp_action_chunk_runtime.server import OrigamiCompActionChunkZenohServer


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "configs" / "dataset_replay.yaml"
DEFAULT_BUNDLE_ROOT = ROOT / "model_bundle"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the integrated comp-action-chunk origami policy over origami-zenoh-v1."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Runtime YAML config. Ignored when --bundle-root is provided.",
    )
    parser.add_argument(
        "--bundle-root",
        default=os.environ.get("ORIGAMI_MODEL_BUNDLE"),
        help="Model bundle directory or bundle.yaml path. Defaults to ORIGAMI_MODEL_BUNDLE.",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("ORIGAMI_ZENOH_ENDPOINT"),
        help="Zenoh endpoint tcp/<host>:<port>, or ORIGAMI_ZENOH_ENDPOINT.",
    )
    parser.add_argument(
        "--session-id",
        default=os.environ.get("ORIGAMI_SESSION_ID"),
        help="Assigned session ID, or ORIGAMI_SESSION_ID.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("sync", "async"),
        default=os.environ.get("EXECUTION_MODE"),
        help="Metadata execution_mode. Defaults to config.server.execution_mode or EXECUTION_MODE.",
    )
    parser.add_argument("--action-horizon", type=int, default=None, help="Override metadata/action horizon.")
    parser.add_argument("--device", default=None, help="Override runtime.device, for example cuda:0 or cpu.")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default=None, help="Override runtime.precision.")
    parser.add_argument(
        "--jax-mem-fraction",
        type=float,
        default=None,
        help="Override OpenPI runtime XLA_PYTHON_CLIENT_MEM_FRACTION before JAX import.",
    )
    parser.add_argument(
        "--warmup-inferences",
        type=int,
        default=None,
        help="Number of startup warmup inferences. Defaults to config.server.warmup_inferences.",
    )
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    return parser


def _select_bundle_root(raw_bundle_root: str | None) -> str | None:
    if raw_bundle_root not in (None, ""):
        return raw_bundle_root
    if (DEFAULT_BUNDLE_ROOT / "bundle.yaml").is_file():
        return str(DEFAULT_BUNDLE_ROOT)
    return None


def main() -> int:
    args = build_argument_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    if not args.endpoint:
        raise SystemExit("--endpoint or ORIGAMI_ZENOH_ENDPOINT is required")
    if not args.session_id:
        raise SystemExit("--session-id or ORIGAMI_SESSION_ID is required")

    config = load_runtime_config(
        config_path=args.config,
        bundle_root=_select_bundle_root(args.bundle_root),
    )

    runtime = config.runtime
    if args.device is not None:
        runtime = dataclasses.replace(runtime, device=str(args.device))
    if args.precision is not None:
        runtime = dataclasses.replace(runtime, precision=str(args.precision))

    server = config.server
    if args.execution_mode is not None:
        server = dataclasses.replace(server, execution_mode=str(args.execution_mode))
    if args.action_horizon is not None:
        server = dataclasses.replace(server, action_horizon=int(args.action_horizon))

    if args.jax_mem_fraction is not None:
        openpi_payload_path = config.paths.openpi_runtime_config
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.jax_mem_fraction)
        logging.info(
            "Overriding XLA_PYTHON_CLIENT_MEM_FRACTION=%s for OpenPI config=%s",
            args.jax_mem_fraction,
            openpi_payload_path,
        )

    config = dataclasses.replace(config, runtime=runtime, server=server)
    pipeline = CompActionChunkPipeline(config)
    warmup_count = int(config.server.warmup_inferences if args.warmup_inferences is None else args.warmup_inferences)
    pipeline.warmup(warmup_count, prompt=config.dataset_replay.prompt)

    zenoh_server = OrigamiCompActionChunkZenohServer(
        pipeline,
        config,
        endpoint=str(args.endpoint),
        session_id=str(args.session_id),
        action_horizon=config.server.action_horizon,
        execution_mode=config.server.execution_mode,
    )
    zenoh_server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
