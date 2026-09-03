download paligemma tokenizer

mkdir -p /home/ubuntu/workspace/RO-Inference-Comp-Action-Chunk/model_bundle/assets

curl -L \
  -o /home/ubuntu/workspace/RO-Inference-Comp-Action-Chunk/model_bundle/assets/paligemma_tokenizer.model \
  https://storage.googleapis.com/big_vision/paligemma_tokenizer.model

hf download facebook/dinov3-vits16plus-pretrain-lvd1689m \
  --local-dir /home/ubuntu/RO-Inference-Comp-Action-Chunk/model_bundle/weights/dino/dinov3-vits16plus-pretrain-lvd1689m


nvidia-smi

docker --version

which nvidia-ctk || echo "NVIDIA Container Toolkit NOT installed"

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

docker info | grep -i runtime

sudo docker run --rm --gpus all ubuntu nvidia-smi