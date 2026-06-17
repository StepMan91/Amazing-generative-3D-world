#!/usr/bin/env base
set -euo pipefail

# Run the NGC PyTorch Container with GPU mapping, exposing Gradio port 7860
# Mounts the workspace to /workspace/lyra and maps the huggingface cache
docker run --gpus all -d -it --name lyra2_dev \
  -v "/home/sparka/Generative 3D environment from picture:/workspace/lyra" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 7860:7860 \
  nvcr.io/nvidia/pytorch:25.01-py3

echo "Docker container lyra2_dev started successfully."
