#!/bin/bash
#SBATCH --account=rrg-lajoiepy
#SBATCH --job-name=ans-eval
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/home/yichenwu/scratch/ans_runs/logs/%x-%j.out
# Evaluate the pretrained ANS models on Gibson val (paper protocol: val_mt, 14 scenes x 71 eps, 1000 steps).
# Usage: sbatch --time=HH:MM:SS apptainer/eval.sbatch <exp_name> <num_episodes> [split=val_mt] [extra main.py args...]
set -eo pipefail
EXP=${1:?exp_name}; EPS=${2:?num_episodes}; SPLIT=${3:-val_mt}; shift 3 2>/dev/null || shift $#
case $SPLIT in val_mt) N=14;; val_mt_small) N=10;; val_mt_large) N=4;; *) echo "unknown split $SPLIT"; exit 1;; esac
cd /project/6114327/yichen/Neural-SLAM
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
apptainer/ans.sh python main.py --split $SPLIT --eval 1 --auto_gpu_config 0 -n $N --num_episodes $EPS \
    --max_episode_length 1000 --train_global 0 --train_local 0 --train_slam 0 \
    --load_global pretrained_models/model_best.global \
    --load_local pretrained_models/model_best.local \
    --load_slam pretrained_models/model_best.slam \
    --print_images 0 --log_interval 100 \
    --dump_location $HOME/scratch/ans_runs --exp_name $EXP "$@"
