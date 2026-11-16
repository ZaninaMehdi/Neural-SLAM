# Environment for Active Neural SLAM (devendrachaplot/Neural-SLAM)
# Mirrors the README: Python 3.6, PyTorch 1.2.0 / CUDA 10.0, and the pinned
# habitat-sim + habitat-api commits. UNTESTED: expect to tweak a step or two.
#
# Needs on the host: an NVIDIA driver + nvidia-container-toolkit.
# CUDA itself comes from the conda cudatoolkit=10.0 package, so no CUDA base image.

FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display

# Build tools + GL/EGL libs that habitat-sim (headless) needs
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget curl ca-certificates pkg-config \
        libjpeg-dev libpng-dev libglm-dev \
        libgl1-mesa-dev libegl1-mesa-dev xorg-dev freeglut3-dev \
        libglvnd0 libgl1 libglx0 libegl1 libgles2 \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Lets the NVIDIA driver (injected at runtime) provide EGL for headless rendering
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Old Miniconda (pre-dates the newer conda terms-of-service prompts), then a py3.6 env
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    && $CONDA_DIR/bin/conda create -y -n ans python=3.6 \
    && $CONDA_DIR/bin/conda clean -afy
ENV PATH=$CONDA_DIR/envs/ans/bin:$CONDA_DIR/bin:$PATH

# PyTorch 1.2.0 with the CUDA 10.0 runtime bundled
RUN conda install -y pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch \
    && conda clean -afy

# habitat-sim at the commit pinned in the ANS README
RUN git clone https://github.com/facebookresearch/habitat-sim.git /opt/habitat-sim \
    && cd /opt/habitat-sim \
    && git checkout 9575dcd45fe6f55d2a44043833af08972a7895a9 \
    && git submodule update --init --recursive \
    && pip install -r requirements.txt \
    && python setup.py install --headless

# habitat-api at the pinned commit
RUN git clone https://github.com/facebookresearch/habitat-api.git /opt/habitat-api \
    && cd /opt/habitat-api \
    && git checkout b5f2b00a25627ecb52b43b13ea96b05998d9a121 \
    && pip install -e .

# ANS python requirements only (your own clone of the repo gets mounted at runtime)
RUN git clone --depth 1 --recurse-submodules https://github.com/devendrachaplot/Neural-SLAM /tmp/ans \
    && pip install -r /tmp/ans/requirements.txt \
    && rm -rf /tmp/ans

WORKDIR /workspace/Neural-SLAM
CMD ["bash"]
