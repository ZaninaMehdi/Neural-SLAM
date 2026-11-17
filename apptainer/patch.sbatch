#!/bin/bash
#SBATCH --account=rrg-lajoiepy
#SBATCH --job-name=ans-patch
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
module load apptainer
REPO=/project/6114327/yichen/Neural-SLAM
export APPTAINER_CACHEDIR=$SLURM_TMPDIR/cache APPTAINER_TMPDIR=$SLURM_TMPDIR/tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
cd "$SLURM_TMPDIR"
cp "$REPO/neural-slam.sif" .
cp "$REPO/apptainer/patch-egl.def" .
apptainer build --fakeroot neural-slam-patched.sif patch-egl.def
cp neural-slam-patched.sif "$REPO/neural-slam.sif.new" && mv "$REPO/neural-slam.sif.new" "$REPO/neural-slam.sif"
ls -lh "$REPO/neural-slam.sif"
