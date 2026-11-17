#!/bin/bash
#SBATCH --account=rrg-lajoiepy
#SBATCH --job-name=ans-smoke
#SBATCH --time=00:30:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
# Checks GPU torch + habitat-sim EGL rendering, then (if Gibson is present) runs the upstream sanity command.
set -eo pipefail
cd /project/6114327/yichen/Neural-SLAM
A=apptainer/ans.sh
$A python -c "
import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))
x=torch.randn(64,64,device='cuda'); print('matmul ok', float((x@x).sum())==float((x@x).sum()))
import habitat_sim, habitat, skfmm, skimage; print('habitat imports ok')
import env.habitat.utils.noisy_actions; print('noisy_actions (register_move_fn) ok')
"
$A python apptainer/render_test.py data/test_scenes/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb
if ls data/scene_datasets/gibson/*.glb >/dev/null 2>&1; then
  $A python main.py -n1 --auto_gpu_config 0 --split val --max_episode_length 100 --num_episodes 1 \
      --exp_name smoke --dump_location $HOME/scratch/ans_runs --print_images 0
  echo "=== eval with pretrained models (1 episode) ==="
  $A python main.py -n1 --auto_gpu_config 0 --split val --eval 1 --train_global 0 --train_local 0 --train_slam 0 \
      --load_global pretrained_models/model_best.global --load_local pretrained_models/model_best.local \
      --load_slam pretrained_models/model_best.slam --num_episodes 1 --max_episode_length 200 \
      --exp_name smoke_eval --dump_location $HOME/scratch/ans_runs --print_images 1
else
  echo "No Gibson scenes in data/scene_datasets/gibson yet: skipping main.py run"
fi
