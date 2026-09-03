# RO-Inference-Comp-Action-Chunk

Integrated inference package for the `pi05_origami_comp_action_chunk` policy.

The runtime path is:

1. Shared DINOv3 ViT-S/16+ patch tokens for `head_left`, `wrist_left`, and `wrist_right`.
2. OOI crop prediction from the shared camera tokens.
3. Checkpoint-planner inference from camera tokens, optional OOI tokens, `state_65d`, and `tactile_60d`.
4. OpenPI inference with planner prefix features, FTP tactile prefix inputs, `state_65d`, `tactile_60d`, and the prompt.
5. Output `float32[10,65]` absolute joint-position chunks in radians.

The public server accepts the latest `inferencekit_09_02_2026` `origami-zenoh-v1` observation schema, including optional `observation/image/tactile_raw`.

## Bundle Layout

`model_bundle/bundle.yaml` is the single source of runtime paths. The Docker image is built around this folder.

OpenPI runtime code is vendored in the project at:

```text
vendor/openpi/src/
  openpi/...
  openpi_client/...
  future_latent_predictor/...
```

This source must match the OpenPI checkout used for the final
`pi05_origami_comp_action_chunk` training run.

Required files:

```text
model_bundle/
  bundle.yaml
  assets/
    paligemma_tokenizer.model
  configs/
    ooi_zoom_runtime.yaml
    checkpoint_planner_runtime_model.yaml
    openpi_comp_action_chunk_runtime.yaml
  weights/
    dino/dinov3-vits16plus-pretrain-lvd1689m/
      config.json
      model.safetensors
      preprocessor_config.json
    ooi/best.pt
    checkpoint_planner/checkpoint.pt
    checkpoint_planner/manifest/manifest.json
    openpi/pi05_origami_comp_action_chunk/
      params/
      assets/competition_paper_reprocessed_origami_comp_action_chunk/norm_stats.json
  dataset_replay/
    episodes/...   # optional small replay set baked into the image
```

`model_bundle/configs/openpi_comp_action_chunk_runtime.yaml` contains the
runtime `openpi.config_name: pi05_origami_comp_action_chunk` plus a snapshot of
the training-critical OpenPI model, data, weight-loader, and optimizer config
values used for packaging checks.

## Prepare Bundle

Edit `configs/dataset_replay.yaml` first if the source checkpoints are at different paths. Then materialize the bundle:

```powershell
cd E:\Robot-Origami-Challenge\RO-Inference-Comp-Action-Chunk

python .\scripts\prepare_comp_action_chunk_bundle.py `
  --source-config .\configs\dataset_replay.yaml `
  --bundle-root .\model_bundle `
  --copy-mode copy `
  --overwrite
```

Use `--copy-mode hardlink` only when all source files are on the same filesystem and you want to avoid duplicating large files locally. Do not use `--copy-mode symlink` for the final Docker build, because the image must contain real files.

If the script reports missing sources, copy them manually into the expected `model_bundle/` locations above, or fix the source paths in `configs/dataset_replay.yaml` and rerun the script.

To refresh the project-vendored OpenPI source from
`E:\Robot-Origami-Challenge\openpi`, run:

```powershell
python .\scripts\prepare_comp_action_chunk_bundle.py `
  --source-config .\configs\dataset_replay.yaml `
  --bundle-root .\model_bundle `
  --copy-mode copy `
  --include-openpi-source `
  --overwrite
```

That command updates `vendor/openpi/src`; it does not add code under
`model_bundle/`.

## Add Replay Episodes

For internal replay smoke tests inside Docker, copy a small raw dataset subset into:

```text
model_bundle/dataset_replay/episodes/<episode_uid>/
  arrays/state_65d.npy
  arrays/action_65d.npy
  arrays/tactile_60d.npy
  arrays/timestamps.npy       # optional
  arrays/frame_index.npy      # optional
  videos/head_left.mp4
  videos/wrist_left.mp4
  videos/wrist_right.mp4
  videos/tactile_deform.mp4
  videos/tactile_raw.mp4      # optional, required for deform_plus_raw replay
```

Dataset replay runs three tactile modes by default:

- `deform_only`
- `deform_plus_raw`
- `mixed_50`

The replay log prints action error, training-style OpenPI `loss_base_flow`, raw tactile availability, planner availability, OOI target presence, and per-stage timings.

Replay also shows live `tqdm` progress bars for:

- replay planning by episode;
- each tactile mode's total sample count;
- each tactile mode's episode count;
- each active episode's frame/sample progress.

Server inference logs one INFO line for metadata/reset requests and, by default, one INFO line per inference request with request ID, count, action shape, raw tactile status, planner/OOI status, selected checkpoint, and inference timings. To reduce inference log volume, edit:

```yaml
server:
  log_inference_requests: true
  log_inference_every_n: 10
```

## Verify Before Docker

Run layout verification before building:

```powershell
cd E:\Robot-Origami-Challenge\RO-Inference-Comp-Action-Chunk
python .\scripts\verify_comp_action_chunk_bundle.py --bundle-root .\model_bundle
```

If the replay dataset should also be baked into the image:

```powershell
python .\scripts\verify_comp_action_chunk_bundle.py `
  --bundle-root .\model_bundle `
  --require-dataset-replay
```

During setup only, this command is useful for checking path rewrites before heavy weights are copied:

```powershell
python .\scripts\verify_comp_action_chunk_bundle.py `
  --bundle-root .\model_bundle `
  --allow-missing-model-files
```

## Test Replay Before Docker

Run a small replay locally before building the image:

```powershell
cd E:\Robot-Origami-Challenge\RO-Inference-Comp-Action-Chunk

$env:XLA_PYTHON_CLIENT_PREALLOCATE="false"
$env:XLA_PYTHON_CLIENT_MEM_FRACTION="0.60"
$env:XLA_PYTHON_CLIENT_ALLOCATOR="platform"

python .\run_dataset_replay.py `
  --bundle-root .\model_bundle `
  --max-episodes 1 `
  --max-total-samples 30 `
  --max-loss-samples 10 `
  --loss-every-n-samples 5 `
  --output-dir .\outputs\dataset_replay_smoke
```

For a dry count without model loading:

```powershell
python .\run_dataset_replay.py --bundle-root .\model_bundle --dry-run
```

## Test Server Before Docker

Start a local Zenoh router using the latest inference kit instructions, then start this policy server:

```powershell
cd E:\Robot-Origami-Challenge\RO-Inference-Comp-Action-Chunk

$env:ORIGAMI_ZENOH_ENDPOINT="tcp/127.0.0.1:17447"
$env:ORIGAMI_SESSION_ID="local-contract-test"
$env:XLA_PYTHON_CLIENT_PREALLOCATE="false"
$env:XLA_PYTHON_CLIENT_MEM_FRACTION="0.60"
$env:XLA_PYTHON_CLIENT_ALLOCATOR="platform"

python .\serve_origami_comp_action_chunk_policy.py `
  --bundle-root .\model_bundle `
  --execution-mode async `
  --warmup-inferences 1
```

In another terminal, validate against the public checker:

```powershell
cd E:\Robot-Origami-Challenge\inferencekit_09_02_2026\sharpa_north_ces_lite_sdk-main

uv run --no-sync python .\examples\check_zenoh_policy.py `
  --endpoint tcp/127.0.0.1:17447 `
  --session-id local-contract-test `
  --timeout 180 `
  --requests 3 `
  --expected-horizon 10
```

## Build Docker Image

Build from this directory as the Docker context:

```powershell
cd E:\Robot-Origami-Challenge\RO-Inference-Comp-Action-Chunk

docker build `
  --build-arg EXECUTION_MODE=async `
  -f .\docker\submission-zenoh-bundled.Dockerfile `
  -t ro-inference-comp-action-chunk:async `
  .
```

The build context must contain the populated `model_bundle/`. The final image should not rely on host mounts, host Python packages, Hugging Face cache, or internet access at runtime.

## Test Inside Docker

Verify the bundled assets inside the built image:

```bash
docker run --rm --gpus all ro-inference-comp-action-chunk:async verify-bundle
```

Run a small internal dataset replay from the baked-in replay episodes:

```bash
docker run --rm --gpus all \
  --shm-size 8g \
  -e ORIGAMI_JAX_MEM_FRACTION=0.60 \
  ro-inference-comp-action-chunk:async \
  dataset-replay \
  --max-episodes 1 \
  --max-total-samples 30 \
  --max-loss-samples 10 \
  --loss-every-n-samples 5 \
  --output-dir /tmp/dataset_replay_smoke
```

The dataset replay mode does not require `ORIGAMI_ZENOH_ENDPOINT` or `ORIGAMI_SESSION_ID`.

## Test Docker Zenoh Contract

Use the router image pinned by the public inference kit:

```bash
ROUTER_IMAGE='eclipse/zenoh@sha256:157965d71e0bfd0a044d76a985ff0e5c306ad3968929168fb9678cd2a7fec23f'
IMAGE='ro-inference-comp-action-chunk:async'
SESSION='local-contract-test'

docker network create origami-contract-test

docker run -d --name origami-contract-router \
  --network origami-contract-test \
  -p 127.0.0.1:17447:7447 \
  "$ROUTER_IMAGE" \
  -l tcp/0.0.0.0:7447 \
  --no-multicast-scouting \
  --cfg 'transport/shared_memory/enabled:false'

docker run -d --name origami-contract-policy \
  --network origami-contract-test \
  --gpus all \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges=true \
  --user 65532:65532 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=4g \
  --tmpfs /run:rw,noexec,nosuid,nodev,size=64m \
  --shm-size 8g \
  --memory 32g \
  --cpus 8 \
  --pids-limit 512 \
  -e ORIGAMI_ZENOH_ENDPOINT=tcp/origami-contract-router:7447 \
  -e ORIGAMI_SESSION_ID="$SESSION" \
  -e ORIGAMI_JAX_MEM_FRACTION=0.60 \
  "$IMAGE" \
  serve
```

Watch startup:

```bash
docker logs -f origami-contract-policy
```

Run the validator:

```bash
cd /path/to/inferencekit_09_02_2026/sharpa_north_ces_lite_sdk-main

uv run --no-sync python examples/check_zenoh_policy.py \
  --endpoint tcp/127.0.0.1:17447 \
  --session-id "$SESSION" \
  --timeout 180 \
  --requests 3 \
  --expected-horizon 10
```

Clean up:

```bash
docker rm -f origami-contract-policy origami-contract-router
docker network rm origami-contract-test
```

## Entrypoint Modes

The image entrypoint supports:

```bash
docker run --rm --gpus all ro-inference-comp-action-chunk:async serve
docker run --rm --gpus all ro-inference-comp-action-chunk:async verify-bundle
docker run --rm --gpus all ro-inference-comp-action-chunk:async dataset-replay --dry-run
docker run --rm -it ro-inference-comp-action-chunk:async bash
```

Serving requires only:

```text
ORIGAMI_ZENOH_ENDPOINT
ORIGAMI_SESSION_ID
```

Useful optional env vars:

```text
EXECUTION_MODE=async
ORIGAMI_MODEL_BUNDLE=/app/RO-Inference-Comp-Action-Chunk/model_bundle
ORIGAMI_WARMUP_INFERENCES=1
ORIGAMI_JAX_MEM_FRACTION=0.60
XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

## Final Export

After the Docker validator passes:

```bash
IMAGE='ro-inference-comp-action-chunk:async'
ARCHIVE='ro-inference-comp-action-chunk.tar'

docker save -o "$ARCHIVE" "$IMAGE"
zstd -T0 -19 "$ARCHIVE"
sha256sum "${ARCHIVE}.zst" > "${ARCHIVE}.zst.sha256"
zstd -t "${ARCHIVE}.zst"
```

Keep the image tag or digest, archive, checksum, expected horizon `10`, and the runtime resource assumptions used during validation.
