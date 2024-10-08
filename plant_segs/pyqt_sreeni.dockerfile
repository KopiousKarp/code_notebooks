FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Add user
# RUN adduser --quiet --disabled-password qtuser && usermod -a -G audio qtuser
ARG ROCM_VERSION=6.1
ARG AMDGPU_VERSION=6.1

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl gnupg && \
  curl -sL http://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /usr/share/keyrings/rocm-archive-keyring.gpg && \
  sh -c 'echo deb [arch=amd64 signed-by=/usr/share/keyrings/rocm-archive-keyring.gpg] http://repo.radeon.com/rocm/apt/$ROCM_VERSION/ jammy main > /etc/apt/sources.list.d/rocm.list' && \
  sh -c 'echo deb [arch=amd64 signed-by=/usr/share/keyrings/rocm-archive-keyring.gpg] https://repo.radeon.com/amdgpu/$AMDGPU_VERSION/ubuntu jammy main > /etc/apt/sources.list.d/amdgpu.list'
  
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  sudo \
  libelf1 \
  libnuma-dev \
  build-essential \
  git \
  vim-nox \
  cmake-curses-gui \
  kmod \
  file \
  python3 \
  python3-pip \
  rocm-cmake=0.12.0.60100-82~22.04 \
  rocm-device-libs=1.0.0.60100-82~22.04 \
  rocminfo=1.0.0.60100-82~22.04 \
  rocm-dev \
  rocm-utils && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*
# This fix: libGL error: No matching fbConfigs or visuals found
ENV LIBGL_ALWAYS_INDIRECT=1

# Install Python 3, PyQt5
RUN apt-get update && apt-get install -y \
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
    python3-pip \
    python3-venv \
    git \
    libsm6 

RUN python3 -m venv /opt/sam2_env && \
    /opt/sam2_env/bin/pip3 install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.1 && \
    /opt/sam2_env/bin/pip3 install virtualenv-clone    
        
RUN /opt/sam2_env/bin/pip3 install digitalsreeni-image-annotator

RUN /opt/sam2_env/bin/pip3 uninstall -y opencv-python
RUN /opt/sam2_env/bin/pip3 install opencv-python-headless
RUN apt-get install -y \
    fontconfig-config \
    fonts-dejavu-core \
    libfontconfig1 \
    libfreetype6 \
    libpng16-16 \
    ucf