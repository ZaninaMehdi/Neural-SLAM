# Active Neural SLAM (ANS) – Docker setup

Reproducible environment for [devendrachaplot/Neural-SLAM](https://github.com/devendrachaplot/Neural-SLAM)
(Python 3.6, PyTorch 1.2.0 / CUDA 10.0, pinned habitat-sim + habitat-api).

> The Dockerfile and `run.sh` below are **untested drafts**. Expect to fix a step or two,
> most likely the habitat-sim compile.

## 0. Prerequisites

**Windows (WSL2)**
1. Install WSL2 with Ubuntu, and open an Ubuntu terminal. Run everything below **inside WSL**, not PowerShell.
2. Install Docker Desktop, then enable *Settings → General → Use the WSL 2 based engine* and
   *Settings → Resources → WSL integration → (your Ubuntu distro)*.
3. Limit WSL so the habitat-sim compile doesn't freeze your machine. Create
   `C:\Users\<YOU>\.wslconfig` (adjust to your RAM, keep ≥4 GB for Windows):
   ```ini
   [wsl2]
   memory=8GB
   swap=8GB
   processors=2
   ```
   Then in PowerShell: `wsl --shutdown`, and restart Docker Desktop.

**GPU (optional for a first smoke test, required for real runs)**
- An NVIDIA GPU + recent Windows/Linux NVIDIA driver + NVIDIA Container Toolkit (Linux) or Docker Desktop WSL2 GPU support (Windows).
- PyTorch 1.2.0 / CUDA 10.0 only supports older GPU architectures (as far as I know up to roughly RTX 20-series / T4).
  RTX 30/40-series and A100 will fail.
- No NVIDIA GPU (e.g. Intel only): you can still build and try a CPU smoke test, see step 6.

**Useful tools in WSL**
```bash
sudo apt-get update && sudo apt-get install -y git unzip wget curl
```

## 1. Get the code and move the Docker files next to it

`Dockerfile` and `run.sh` are committed *inside* the `Neural-SLAM` repo (git restriction), but they must live in
the **parent** folder, next to `Neural-SLAM/`, so the repo can be mounted into the container.

```bash
mkdir -p ~/ans && cd ~/ans

# replace <YOUR_REPO_URL> with the repo that contains Dockerfile + run.sh
git clone --recurse-submodules <YOUR_REPO_URL> Neural-SLAM

# if you already cloned without submodules:
# cd Neural-SLAM && git submodule update --init --recursive && cd ..

# COPY (not move) so git doesn't see the files as deleted
cp Neural-SLAM/Dockerfile Neural-SLAM/run.sh .

# fix Windows line endings (if edited on Windows) and make the script executable
sed -i 's/\r$//' run.sh
chmod +x run.sh
```

Do **not** work from `/mnt/c/...`: it is very slow for Docker. Keep everything in the WSL filesystem (`~/ans`).

Expected layout:
```
~/ans/
  Dockerfile
  run.sh
  .dockerignore            <- created in step 4
  Neural-SLAM/
    data/                  <- step 3
    pretrained_models/     <- step 2
    env/habitat/habitat_api/   (submodule, must not be empty)
```

Quick check that the bundled habitat-api submodule is present:
```bash
ls Neural-SLAM/env/habitat/habitat_api | head
```

## 2. Download the pretrained models

```bash
cd ~/ans/Neural-SLAM
mkdir -p pretrained_models

curl -L "https://drive.google.com/uc?export=download&id=1UK2hT0GWzoTaVR5lAI6i8o27tqEmYeyY" -o pretrained_models/model_best.global
curl -L "https://drive.google.com/uc?export=download&id=1A1s_HNnbpvdYBUAiw2y1JmmELRLfAJb8" -o pretrained_models/model_best.local
curl -L "https://drive.google.com/uc?export=download&id=1o5OG7DIUKZyvi5stozSqRpAEae1F2BmX" -o pretrained_models/model_best.slam

ls -lh pretrained_models
```
Each file should be hundreds of KB or more. If one is only a few KB, Google Drive returned an HTML error page.
Fallback mirror used by some forks of the repo:
```bash
for m in global local slam; do
  wget -O pretrained_models/model_best.$m http://www.cs.cmu.edu/~dchaplot/projects/active_neural_slam/model_best.$m
done
```

## 3. Download the data (Gibson) into `Neural-SLAM/data`

Required layout (same as habitat-api):
```
Neural-SLAM/data/
  scene_datasets/gibson/Adrian.glb, Adrian.navmesh, ...
  datasets/pointnav/gibson/v1/train/, val/, ...
```

```bash
cd ~/ans/Neural-SLAM
mkdir -p data/scene_datasets/gibson data/datasets/pointnav/gibson/v1
```

### 3a. Gibson scenes (`gibson_habitat_trainval.zip`, roughly 10-11 GB)
You must first sign the Gibson license agreement (the form is linked from the "Scenes datasets" section of the
habitat-lab README). You then get a download link. Either download it directly:
```bash
GIBSON_ZIP_URL="<paste the link you received after signing the agreement>"
wget -O gibson_habitat_trainval.zip "$GIBSON_ZIP_URL"
```
or copy a zip you already downloaded on Windows:
```bash
cp /mnt/c/Users/<YOU>/Downloads/gibson_habitat_trainval.zip .
```
Then extract and clean up:
```bash
unzip gibson_habitat_trainval.zip -d data/scene_datasets/gibson/
rm gibson_habitat_trainval.zip
```

### 3b. PointNav Gibson episodes (`pointnav_gibson_v1.zip`, about 385 MB, **v1**)
The link is in the "Data" section of the habitat-api / habitat-lab README. As far as I know it is:
```bash
wget -O pointnav_gibson_v1.zip https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip
unzip pointnav_gibson_v1.zip -d data/datasets/pointnav/gibson/v1/
rm pointnav_gibson_v1.zip
```
(If that URL fails, take the current one from the habitat README.)

### 3c. Verify the layout (watch for nested folders!)
```bash
ls data/scene_datasets/gibson | head          # should list X.glb and X.navmesh directly
ls data/datasets/pointnav/gibson/v1           # should list train, val, ...
```
If you see `gibson/gibson/...` or `v1/v1/...`, move the contents up one level, otherwise you get "scene not found" errors.

## 4. Keep the data out of the Docker build

The build context is `~/ans`, which contains the 11 GB of data. The build does not need `Neural-SLAM` at all:
```bash
cd ~/ans
echo "Neural-SLAM" > .dockerignore
```

## 5. Build the image (slow: compiles habitat-sim)

```bash
cd ~/ans
./run.sh build
```
- The first build can take 30-60+ minutes. The habitat-sim compile prints very little.
- Close heavy apps, keep the laptop plugged in, and don't let it sleep.
- For a readable log:
  ```bash
  docker build --progress=plain -t ans-env . 2>&1 | tee build.log
  tail -50 build.log
  ```

## 6. Check the environment

**With an NVIDIA GPU** (the script uses `--gpus all`):
```bash
./run.sh check
```
You want: `cuda available: True`, your GPU name, `cuda matmul ok`, and `habitat_sim + habitat import ok`.

**Without an NVIDIA GPU** (CPU smoke test only): remove the GPU flag from the script first.
```bash
sed -i 's/--gpus all //' run.sh
./run.sh check
```
Expected: `cuda available: False` and `habitat_sim + habitat import ok`.

## 7. Run ANS

Open a shell in the container (your `Neural-SLAM` folder is mounted at `/workspace/Neural-SLAM`):
```bash
./run.sh shell
```
Inside the container:
```bash
# sanity check from the ANS README (verifies dependencies + data)
python main.py -n1 --auto_gpu_config 0 --split val

# evaluate the pretrained models on the Gibson val split
python main.py --split val --eval 1 --train_global 0 --train_local 0 --train_slam 0 \
  --load_global pretrained_models/model_best.global \
  --load_local pretrained_models/model_best.local \
  --load_slam pretrained_models/model_best.slam
```
Reference numbers from the paper (Gibson val): about 32.7 m² coverage / 0.948 % coverage. Expect to land close, not exactly on them.
Check `arguments.py` for options to limit the number of episodes/processes.

Edit code on the host (in `~/ans/Neural-SLAM`); changes are visible immediately in the container.
Containers are deleted on exit (`--rm`): anything installed by hand inside is lost. Put permanent changes in the Dockerfile.

## Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `wget` / `Invoke-WebRequest` error in PowerShell | You are in PowerShell. Use the WSL terminal, or `curl.exe` on Windows. |
| `docker: ... nvidia-container-cli ... no adapters were found` | No NVIDIA GPU visible. Use the CPU variant in step 6, or another machine. |
| Build "freezes", WSL terminal takes forever to open | Out of RAM/disk during the habitat-sim compile. Set `.wslconfig` limits (step 0), `wsl --shutdown`, rebuild. Check `free -h` and `df -h /`. |
| `bad interpreter` when running `run.sh` | Windows line endings: `sed -i 's/\r$//' run.sh` |
| `ImportError: cannot import name 'register_move_fn'` | habitat-sim version mismatch. It must be the exact commit pinned in the Dockerfile. |
| Scene/episodes not found | Nested folders after unzipping (step 3c), or run from the wrong directory. |
| `env/habitat/habitat_api` is empty | Submodules not initialised: `git submodule update --init --recursive` in `Neural-SLAM`. |
| Container is very slow on file access | Project is under `/mnt/c/...`. Move it into the WSL filesystem (`~/ans`). |
