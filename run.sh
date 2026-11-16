#!/usr/bin/env bash
# Usage:
#   ./run.sh build   -> build the image (slow the first time: compiles habitat-sim)
#   ./run.sh check   -> verify torch/CUDA/habitat import inside the container
#   ./run.sh shell   -> interactive shell with the repo mounted
#
# Expected layout on the host:
#   ./Neural-SLAM/        <- your clone of the repo (edit code here, on the host)
#   ./Neural-SLAM/data/   <- Gibson scenes + pointnav dataset (habitat-api layout)
set -euo pipefail

IMAGE=ans-env
REPO_DIR="$(pwd)/Neural-SLAM"

case "${1:-shell}" in
  build)
    docker build -t "$IMAGE" .
    ;;
  check)
    docker run --rm --shm-size=8g "$IMAGE" python -c "
import torch
print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0), '| capability:', torch.cuda.get_device_capability(0))
    x = torch.randn(8, 8).cuda(); print('cuda matmul ok:', (x @ x).sum().item() is not None)
import habitat_sim, habitat
print('habitat_sim + habitat import ok')
"
    ;;
  shell)
    docker run --rm -it --shm-size=8g \
      -v "$REPO_DIR":/workspace/Neural-SLAM \
      "$IMAGE" bash
    ;;
  *)
    echo "usage: $0 {build|check|shell}"; exit 1
    ;;
esac
