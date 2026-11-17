#!/bin/bash
#SBATCH --account=rrg-lajoiepy
#SBATCH --job-name=ans-build
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
# Builds neural-slam.sif in node-local storage, then copies the single SIF file to the repo dir.
set -euo pipefail
module load apptainer
REPO=/project/6114327/yichen/Neural-SLAM
export APPTAINER_CACHEDIR=$SLURM_TMPDIR/cache APPTAINER_TMPDIR=$SLURM_TMPDIR/tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
cd "$SLURM_TMPDIR"
apptainer build --fakeroot \
    neural-slam.sif "$REPO/apptainer/neural-slam.def"
cp neural-slam.sif "$REPO/neural-slam.sif"
ls -lh "$REPO/neural-slam.sif"
