#!/usr/bin/env bash
set -euo pipefail

echo "=== 1. Installing System Dependencies ==="
apt-get update && apt-get install -y \
  libaio-dev \
  libopenexr-dev \
  ninja-build \
  build-essential \
  cargo \
  libgl1 \
  libglib2.0-0 \
  libx11-dev \
  libxext-dev \
  ffmpeg \
  openexr

echo "=== 2. Setting up Compiler Environment Variables ==="
export USE_NINJA=1
export TORCH_CUDA_ARCH_LIST="12.0"
export FLASH_ATTN_CUDA_ARCHS="120"
export MAX_JOBS=4 # Prevent OOM

# Setup include/library paths
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
export CUDA_HOME=/usr/local/cuda
export CPATH="$CUDA_HOME/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include:${CPATH:-}"
export LD_LIBRARY_PATH="/usr/local/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "=== 3. Checking pre-installed packages ==="
if python3 -c "import flash_attn" &>/dev/null; then
  echo "flash_attn is already installed. Skipping compilation."
else
  echo "flash_attn not found. Compiling from source..."
  pip install --no-build-isolation --no-binary :all: flash-attn==2.6.3
fi

if python3 -c "import transformer_engine" &>/dev/null; then
  echo "transformer_engine is already installed. Skipping installation."
else
  echo "transformer_engine not found. Installing..."
  pip install --no-build-isolation "transformer_engine[pytorch]"
  ln -sf "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart" || true
fi

echo "=== 4. Installing Python dependencies ==="
cd /workspace/lyra/Lyra-2
pip install decord2 OpenEXR trove-classifiers
sed -i '/decord==/d' requirements.txt
sed -i -E '/[Oo]pen[Ee][Xx][Rr]==/d' requirements.txt
pip install --no-deps -r requirements.txt

echo "=== 5. Installing MoGe ==="
pip install "git+https://github.com/microsoft/MoGe.git"

echo "=== 6. Building Vendored CUDA Extensions ==="
echo "Building vipe..."
USE_SYSTEM_EIGEN=0 pip install --no-build-isolation -e 'lyra_2/_src/inference/vipe'

echo "Building depth_anything_3[gs]..."
# Note: This compiles gsplat with Blackwell capability 10.0
pip install --no-build-isolation -e 'lyra_2/_src/inference/depth_anything_3[gs]'

echo "=== Lyra 2.0 Compilation and Installation Complete ==="
