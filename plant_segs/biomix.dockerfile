FROM nvidia/cuda:12.0.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LIBGL_ALWAYS_INDIRECT=1

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      gnupg \
      sudo \
      build-essential \
      git \
      vim-nox \
      cmake-curses-gui \
      kmod \
      file \
      python3 \
      python3-pip \
      python3-venv \
      libelf1 \
      libnuma-dev \
      libx11-xcb1 \
      libxcb1 \
      libx11-dev \
      libxcb-xinerama0 \
      libxcb-xinerama0-dev \
      libxcb-randr0 \
      libxcb-randr0-dev \
      libxcb-shape0 \
      libxcb-shape0-dev \
      libxcb-xfixes0 \
      libxcb-xfixes0-dev \
      libxcb-sync1 \
      libxcb-sync-dev \
      libxcb-xtest0 \
      libxcb-xtest0-dev \
      libxcb-xkb1 \
      libxcb-xkb-dev \
      libxkbcommon-x11-0 \
      libgl1-mesa-glx \
      libglib2.0-0 \
      libxcb-icccm4 \
      libxcb-image0 \
      libxcb-keysyms1 \
      libxcb-render-util0 \
      libdbus-1-3 \
      libsm6 \
      fontconfig-config \
      fonts-dejavu-core \
      libfontconfig1 \
      libfreetype6 \
      libpng16-16 \
      ucf && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create and activate Python venv, install CUDA-enabled PyTorch
RUN python3 -m venv /opt/sam2_env && \
    /opt/sam2_env/bin/pip install --upgrade pip && \
    /opt/sam2_env/bin/pip install \
      torch \
      torchvision \
      torchaudio \
      --index-url https://download.pytorch.org/whl/cu118 && \
    /opt/sam2_env/bin/pip install virtualenv-clone

# Clone & install the DigitalSreeni image annotator
RUN git clone https://github.com/KopiousKarp/digitalsreeni-image-annotator.git /opt/digitalsreeni-image-annotator && \
    cd /opt/digitalsreeni-image-annotator && \
    /opt/sam2_env/bin/pip install -e .

# Replace opencv-python with headless & install imaging deps
RUN /opt/sam2_env/bin/pip uninstall -y opencv-python && \
    /opt/sam2_env/bin/pip install \
      opencv-python-headless \
      hydra-core>=1.3.2 \
      iopath>=0.1.10 \
      pillow>=9.4.0

# Clone & install Segment‑Anything 2 (SAM2)
RUN git clone https://github.com/facebookresearch/segment-anything-2.git /opt/sam2 && \
    cd /opt/sam2 && \
    /opt/sam2_env/bin/pip install --no-deps -e ".[demo]" && \
    cd /opt/sam2/checkpoints && ./download_ckpts.sh

# Install remaining Python packages
RUN /opt/sam2_env/bin/pip install \
      matplotlib \
      pycocotools \
      scikit-learn \
      jupyter \
      segmentation-models-pytorch \
      pandas \
      ftfy \
      regex \
      git+https://github.com/openai/CLIP.git
