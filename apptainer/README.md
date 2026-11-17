# Running Neural-SLAM on Nibi (Apptainer)

Image: `neural-slam.sif` in the repo root (Ubuntu 18.04, Python 3.8, PyTorch 2.4.1+cu118,
habitat-sim @ 9575dcd headless/EGL, habitat-api @ b5f2b00). The repo is mounted into the
container, so code edits apply without a rebuild.

Always request a **full** H100 (`--gpus=h100:1`); MIG slices cannot do EGL rendering.

```bash
# interactive
salloc --account=rrg-lajoiepy --gpus=h100:1 --cpus-per-task=4 --mem=32G --time=1:00:00
cd /project/6114327/yichen/Neural-SLAM
apptainer/ans.sh python main.py -n1 --auto_gpu_config 0 --split val

# evaluate pretrained models
apptainer/ans.sh python main.py --split val --eval 1 --train_global 0 --train_local 0 --train_slam 0 \
  --load_global pretrained_models/model_best.global \
  --load_local pretrained_models/model_best.local \
  --load_slam pretrained_models/model_best.slam
```
Write all experiment output to scratch: `-d ~/scratch/ans_runs --exp_name <name>`
(frames from `--print_images 1` land in `~/scratch/ans_runs/dump/<name>/episodes/`).
Batch template: `apptainer/smoke.sbatch`.

## Files
- `neural-slam.def` – full image recipe; `build.sbatch` builds it on a compute node (~1.5 h).
- `patch-egl.def` / `patch.sbatch` – layer applied on top: turns habitat-sim's
  `/dev/nvidia<EGL index>` readability CHECK into a warning (it breaks under Slurm GPU cgroups).
  Re-run after any full rebuild.
- `ans.sh` – runs a command in the container with `--nv`; puts the image's own glvnd
  `libEGL.so.1` ahead of the host's (host one needs glibc 2.33, image has 2.27).
- `render_test.py` – one-frame habitat-sim render check.

## Code changes vs upstream (for newer PyTorch)
- `F.grid_sample` / `F.affine_grid`: `align_corners=True` (PyTorch 1.2 behaviour, matches pretrained models).
- `utils/optimization.py`: `inspect.getargspec` → `getfullargspec`.
