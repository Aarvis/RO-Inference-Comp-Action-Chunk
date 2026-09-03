#!/usr/bin/env bash
set -euo pipefail

mode="${1:-serve}"
if [ "$#" -gt 0 ]; then
  shift
fi

case "${mode}" in
  serve|dataset-replay|verify-bundle|bash|sh|python|python3)
    ;;
  *)
    echo "[submission][ERROR] unknown mode: ${mode}" >&2
    echo "Usage: <image> [serve|dataset-replay|verify-bundle|bash|sh|python] [args...]" >&2
    exit 2
    ;;
esac

if [ "${mode}" = "bash" ] || [ "${mode}" = "sh" ] || [ "${mode}" = "python" ] || [ "${mode}" = "python3" ]; then
  exec "${mode}" "$@"
fi

bundle_root="${ORIGAMI_MODEL_BUNDLE:-/app/RO-Inference-Comp-Action-Chunk/model_bundle}"
execution_mode="${EXECUTION_MODE:-async}"
warmup_inferences="${ORIGAMI_WARMUP_INFERENCES:-1}"
jax_mem_fraction="${ORIGAMI_JAX_MEM_FRACTION:-${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.60}}"

case "${execution_mode}" in
  sync|async)
    ;;
  *)
    echo "[submission][ERROR] EXECUTION_MODE must be sync or async, got: ${execution_mode}" >&2
    exit 2
    ;;
esac

export HOME="${HOME:-/tmp/origami-home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/origami-cache}"
export HF_HOME="${HF_HOME:-/tmp/origami-hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"
export TORCH_HOME="${TORCH_HOME:-/tmp/origami-torch}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/origami-jax-cache}"
export TMPDIR="${TMPDIR:-/tmp/origami-tmp}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${jax_mem_fraction}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

mkdir -p \
  "${HOME}" \
  "${XDG_CACHE_HOME}" \
  "${HF_HOME}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_LEROBOT_HOME}" \
  "${TORCH_HOME}" \
  "${JAX_COMPILATION_CACHE_DIR}" \
  "${TMPDIR}"

cd /app/RO-Inference-Comp-Action-Chunk

case "${mode}" in
  verify-bundle)
    echo "[submission] mode=verify-bundle bundle=${bundle_root}" >&2
    exec python scripts/verify_comp_action_chunk_bundle.py --bundle-root "${bundle_root}" "$@"
    ;;
  dataset-replay)
    echo "[submission] mode=dataset-replay bundle=${bundle_root}" >&2
    exec python run_dataset_replay.py --bundle-root "${bundle_root}" "$@"
    ;;
  serve)
    : "${ORIGAMI_ZENOH_ENDPOINT:?ORIGAMI_ZENOH_ENDPOINT is required}"
    : "${ORIGAMI_SESSION_ID:?ORIGAMI_SESSION_ID is required}"
    echo "[submission] mode=serve transport=origami-zenoh-v1 execution_mode=${execution_mode} endpoint=${ORIGAMI_ZENOH_ENDPOINT} session=${ORIGAMI_SESSION_ID} bundle=${bundle_root}" >&2
    python scripts/verify_comp_action_chunk_bundle.py --bundle-root "${bundle_root}"
    exec python serve_origami_comp_action_chunk_policy.py \
      --bundle-root "${bundle_root}" \
      --endpoint "${ORIGAMI_ZENOH_ENDPOINT}" \
      --session-id "${ORIGAMI_SESSION_ID}" \
      --execution-mode "${execution_mode}" \
      --jax-mem-fraction "${jax_mem_fraction}" \
      --warmup-inferences "${warmup_inferences}" \
      "$@"
    ;;
esac
