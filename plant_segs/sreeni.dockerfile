# This dockerfile is meant to be personalized, and serves as a template and demonstration.
# Modify it directly, but it is recommended to copy this dockerfile into a new build context (directory),
# modify to taste and modify docker-compose.yml.template to build and run it.

# It is recommended to control docker containers through 'docker-compose' https://docs.docker.com/compose/
# Docker compose depends on a .yml file to control container sets
# rocm-setup.sh can generate a useful docker-compose .yml file
# `docker-compose run --rm <rocm-terminal>`

# If it is desired to run the container manually through the docker command-line, the following is an example
# 'docker run -it --rm -v [host/directory]:[container/directory]:ro <user-name>/<project-name>'.

FROM ubuntu:20.04
LABEL maintainer=dl.mlsedevops@amd.com

# Initialize the image
# Modify to pre-install dev tools and ROCm packages
ARG ROCM_VERSION=6.1
ARG AMDGPU_VERSION=6.1

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl gnupg && \
  curl -sL http://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - && \
  sh -c 'echo deb [arch=amd64] http://repo.radeon.com/rocm/apt/$ROCM_VERSION/ focal main > /etc/apt/sources.list.d/rocm.list' && \
  sh -c 'echo deb [arch=amd64] https://repo.radeon.com/amdgpu/$AMDGPU_VERSION/ubuntu focal main > /etc/apt/sources.list.d/amdgpu.list' && \
  apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
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
  rocm-dev && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*
  ENV DEBIAN_FRONTEND=noninteractive
# Install dependencies
RUN apt-get update && apt-get install -y \
  kmod \
  file \
  python3 \
  python3-pip \
  rocm-dev \
  software-properties-common \
  libxkbcommon-x11-0 \
  libxcb-xinerama0 \
  libxcb-xinput0 \
  libxcb-xfixes0 \
  libxcb-sync1 \
  libxcb-shape0 \
  libxcb-render-util0 \
  libxcb-render0 \
  libxcb-randr0 \
  libxcb-present0 \
  libxcb-glx0 \
  libxcb-dri3-0 \
  libxcb-dri2-0 \
  libxcb-composite0 \
  libxcb-cursor0 \
  libxcb-damage0 \
  libxcb-icccm4 \
  libxcb-image0 \
  libxcb-keysyms1 \
  libxcb-render0-dev \
  libxcb-shm0 \
  libxcb-util1 \
  libxcb-xkb1 \
  libx11-xcb-dev \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libxrender1 \
  libfontconfig1 \
  libfreetype6 \
  libxext6 \
  libx11-6 \
  qt5-default \
  qtwebengine5-dev \
  qtwebengine5-examples && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*
# Grant members of 'sudo' group passwordless privileges
# Comment out to require sudo
COPY sudo-nopasswd /etc/sudoers.d/sudo-nopasswd



# The following are optional enhancements for the command-line experience
# Uncomment the following to install a pre-configured vim environment based on http://vim.spf13.com/
# 1.  Sets up an enhanced command line dev environment within VIM
# 2.  Aliases GDB to enable TUI mode by default
#RUN curl -sL https://j.mp/spf13-vim3 | bash && \
#    echo "alias gdb='gdb --tui'\n" >> ~/.bashrc


RUN sudo apt update && \
    apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.10 python3.10-venv python3.10-dev 
# Install pytorch for ROCM 
# venv for SAM2 is created here
RUN python3.10 -m venv /opt/sam2_env && \
    /opt/sam2_env/bin/pip3 install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.1 && \
    /opt/sam2_env/bin/pip3 install virtualenv-clone   

# # Create python env for root_painter and install
# RUN /opt/sam2_env/bin/virtualenv-clone /opt/sam2_env /opt/root_painter_env && \
#     git clone https://github.com/KopiousKarp/root_painter.git /opt/root_painter && \
#     cd /opt/root_painter && git checkout multiclass && \
#     /opt/root_painter_env/bin/pip install \
#     --upgrade-strategy eager \
#     -r /opt/root_painter/trainer/requirements.txt && \
#     /opt/root_painter_env/bin/pip install \ 
#     --upgrade-strategy eager \
#     -r /opt/root_painter/painter/requirements.txt


# Finish the SAM2 installation
# RUN git clone https://github.com/facebookresearch/segment-anything-2.git /opt/sam2 && \
#     cd /opt/sam2 && /opt/sam2_env/bin/pip install -e ".[demo]" && \
#     cd /opt/sam2/checkpoints && ./download_ckpts.sh
RUN /opt/sam2_env/bin/pip install digitalsreeni-image-annotator

# This is meant to be used as an interactive developer container
# Create user rocm-user as member of sudo group
# Append /opt/rocm/bin to the system PATH variable
RUN useradd --create-home -G sudo,video --shell /bin/bash rocm-user
#    sed --in-place=.rocm-backup 's|^\(PATH=.*\)"$|\1:/opt/rocm/bin"|' /etc/environment

USER rocm-user
WORKDIR /home/rocm-user
ENV PATH="${PATH}:/opt/rocm/bin"
# Default to a login shell
#CMD ["opt/sam2_env/bin/python3", "-m", "digitalsreeni-image-annotator.main"]
ENV QTWEBENGINE_DISABLE_SANDBOX=1
ENV QT_QPA_PLATFORM=webgl
# Expose the port for WebGL streaming
EXPOSE 8080
CMD ["bash","-l"]