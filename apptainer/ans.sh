#!/bin/bash
# Run a command inside the Neural-SLAM container, from the repo root, with the GPU.
#   apptainer/ans.sh python main.py -n1 --auto_gpu_config 0 --split val
# Use inside a Slurm job that has a FULL GPU (--gpus=h100:1); MIG slices can't do EGL rendering.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module load apptainer >/dev/null 2>&1
# --nv also binds the host's glvnd libEGL.so.1 (needs glibc 2.33; image has 2.27).
# Prefer the image's own glvnd dispatch libs; NVIDIA vendor libs still come from /.singularity.d/libs.
exec apptainer exec --nv --cleanenv -B "$REPO" --pwd "$REPO" \
    --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/.singularity.d/libs \
    "${ANS_SIF:-$REPO/neural-slam.sif}" "$@"
