# Comp Action Chunk Model Bundle

This folder is the single runtime asset root for `RO-Inference-Comp-Action-Chunk`.
`bundle.yaml` uses paths relative to this folder so the bundle can be moved into
a Docker image without rewriting module configs.

Expected layout:

```text
model_bundle/
  bundle.yaml
  assets/
    paligemma_tokenizer.model
  code/
    openpi/src/openpi/...
    openpi/src/openpi_client/...
    openpi/src/future_latent_predictor/...
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
      assets/
  dataset_replay/
    episodes/...   # optional small replay set for Docker smoke tests
```

Populate this layout with `scripts/prepare_comp_action_chunk_bundle.py`, then
check it with `scripts/verify_comp_action_chunk_bundle.py`.

The `future_latent_predictor` package is packaged even though this config does
not use future latents, because the OpenPI policy module imports it during
startup.
