FROM rocm/pytorch:latest

# Start with installing python 3.10 
# This image is based off of ubunutu 20.04 which has python 3.8 as default
RUN apt update && \
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

# Create python env for root_painter and install
RUN /opt/sam2_env/bin/virtualenv-clone /opt/sam2_env /opt/root_painter_env && \
    git clone https://github.com/KopiousKarp/root_painter.git /opt/root_painter && \
    cd /opt/root_painter && git checkout multiclass && \
    /opt/root_painter_env/bin/pip install \
    --upgrade-strategy eager \
    -r /opt/root_painter/trainer/requirements.txt && \
    /opt/root_painter_env/bin/pip install \ 
    --upgrade-strategy eager \
    -r /opt/root_painter/painter/requirements.txt


# Finish the SAM2 installation
RUN git clone https://github.com/facebookresearch/segment-anything-2.git /opt/sam2 && \
    cd /opt/sam2 && /opt/sam2_env/bin/pip install -e ".[demo]" && \
    cd /opt/sam2/checkpoints && ./download_ckpts.sh


