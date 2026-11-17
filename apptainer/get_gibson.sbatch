#!/bin/bash
#SBATCH --account=rrg-lajoiepy
#SBATCH --job-name=ans-gibson
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
REPO=/project/6114327/yichen/Neural-SLAM
Z=$HOME/scratch/ans_dl/gibson_habitat_trainval.zip
curl -sSfL --retry 5 -C - "https://dl.fbaipublicfiles.com/habitat/data/scene_datasets/gibson_habitat_trainval.zip" -o "$Z"
ls -lh "$Z"
mkdir -p "$SLURM_TMPDIR/g" && unzip -q "$Z" -d "$SLURM_TMPDIR/g"
# flatten any nesting so .glb/.navmesh sit directly in data/scene_datasets/gibson
src=$(dirname "$(find "$SLURM_TMPDIR/g" -name '*.glb' | head -1)")
mkdir -p "$REPO/data/scene_datasets/gibson"
cp -r "$src"/. "$REPO/data/scene_datasets/gibson/"
echo "glb: $(ls $REPO/data/scene_datasets/gibson/*.glb | wc -l)  navmesh: $(ls $REPO/data/scene_datasets/gibson/*.navmesh | wc -l)"
echo GIBSON_DONE
