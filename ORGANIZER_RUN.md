# RO-Inference-Comp-Action-Chunk Organizer Run Guide

This guide is for running the submitted Docker image from a compressed
`tar.zst` archive. The image contains the model bundle and runs the
`origami-zenoh-v1` policy server.

Expected policy output:

```text
actions: float32[10, 65]
```

The action chunk contains 10 absolute 65D joint-position commands.

## 1. Requirements

The host should have:

- Docker with NVIDIA GPU support.
- NVIDIA driver visible from Docker.
- `zstd` installed for decompressing the submitted image archive.
- The official inference-kit checker, if contract validation is required.

Quick GPU check:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

## 2. Load The Image

Assume the submitted archive is:

```text
ro-inference-comp-action-chunk.tar.zst
```

Load it into Docker:

```bash
zstd -dc ro-inference-comp-action-chunk.tar.zst | docker load
```

Confirm the image exists:

```bash
docker images | grep ro-inference-comp-action-chunk
```

Expected image tag:

```text
ro-inference-comp-action-chunk:async
```

If the loaded image has a different tag, set `IMAGE` below to that exact tag.

## 3. Start Zenoh Router

Use the inference-kit router image:

```bash
ROUTER_IMAGE='eclipse/zenoh@sha256:157965d71e0bfd0a044d76a985ff0e5c306ad3968929168fb9678cd2a7fec23f'
IMAGE='ro-inference-comp-action-chunk:async'
SESSION='local-contract-test'

docker network inspect origami-contract-test >/dev/null 2>&1 || \
  docker network create origami-contract-test

docker rm -f origami-contract-router >/dev/null 2>&1 || true

docker run -d --name origami-contract-router \
  --network origami-contract-test \
  -p 127.0.0.1:17447:7447 \
  "$ROUTER_IMAGE" \
  -l tcp/0.0.0.0:7447 \
  --no-multicast-scouting \
  --cfg 'transport/shared_memory/enabled:false'
```

Check router status:

```bash
docker ps --filter name=origami-contract-router
ss -ltnp | grep ':17447'
```

## 4. Start Policy Server

Run the submitted policy image on the same Docker network:

```bash
docker rm -f origami-contract-policy >/dev/null 2>&1 || true

docker run -d --name origami-contract-policy \
  --network origami-contract-test \
  --gpus all \
  --shm-size 8g \
  -e ORIGAMI_ZENOH_ENDPOINT='tcp/origami-contract-router:7447' \
  -e ORIGAMI_SESSION_ID="$SESSION" \
  -e EXECUTION_MODE='async' \
  -e ORIGAMI_WARMUP_INFERENCES='10' \
  -e ORIGAMI_JAX_MEM_FRACTION='0.60' \
  -e XLA_PYTHON_CLIENT_PREALLOCATE='false' \
  -e XLA_PYTHON_CLIENT_ALLOCATOR='platform' \
  "$IMAGE" \
  serve
```

Watch startup:

```bash
docker logs -f origami-contract-policy
```

The server is ready when logs contain:

```text
READY transport=origami-zenoh-v1
```

Startup includes model loading, bundle verification, and 10 warmup inferences.
The server prints `READY` only after warmup is complete. After that, it is
ready to receive observations through Zenoh.

## 5. Validate Contract

From the inference-kit directory:

```bash
cd /path/to/inferencekit_09_02_2026/sharpa_north_ces_lite_sdk-main

uv run --no-sync python ./examples/check_zenoh_policy.py \
  --endpoint tcp/127.0.0.1:17447 \
  --session-id "$SESSION" \
  --timeout 300 \
  --requests 300 \
  --expected-horizon 10
```

Expected result:

```text
metadata: PASS
reset: PASS
infer 1/300: PASS
...
infer 300/300: PASS
```

The checker sends dummy all-zero optional `tactile_raw` images for most
requests and omits `tactile_raw` on the final request. The policy accepts both.

## 6. Cleanup

Stop and remove test containers:

```bash
docker rm -f origami-contract-policy origami-contract-router
docker network rm origami-contract-test
```

## 7. Runtime Notes

The policy server uses:

```text
ORIGAMI_ZENOH_ENDPOINT
ORIGAMI_SESSION_ID
EXECUTION_MODE=async
ORIGAMI_WARMUP_INFERENCES=10
ORIGAMI_JAX_MEM_FRACTION=0.60
XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

The image is intended to run offline after `docker load`; all model weights,
normalization stats, tokenizer assets, DINO files, OOI checkpoint, checkpoint
planner checkpoint, and OpenPI parameters are expected to be baked into the
image.
