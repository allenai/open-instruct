ARG CUDA
ARG DIST
ARG TARGET
FROM --platform=linux/amd64 nvidia/cuda:${CUDA}-${TARGET}-${DIST}

ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/Los_Angeles"

# Install base tools.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    jq \
    language-pack-en \
    make \
    man-db \
    manpages \
    manpages-dev \
    manpages-posix \
    manpages-posix-dev \
    sudo \
    unzip \
    vim \
    wget \
    fish \
    parallel \
    iputils-ping \
    htop \
    emacs \
    zsh \
    rsync \
    tmux

# This ensures the dynamic linker (or NVIDIA's container runtime, I'm not sure)
# puts the right NVIDIA things in the right place (that THOR requires).
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# The -l flag makes bash act as a login shell and load /etc/profile, etc.
ENTRYPOINT ["bash", "-l"]

WORKDIR /stage/

# TODO When updating flash-attn or torch in the future, make sure to update the version in the requirements.txt file. 
COPY requirements.txt .
RUN python -m pip install --upgrade pip "setuptools<70.0.0" wheel 
# TODO, unpin setuptools when this issue in flash attention is resolved
RUN python -m pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
RUN python -m pip install packaging
RUN python -m pip install flash-attn==2.5.8 --no-build-isolation
RUN python -m pip install -r requirements.txt

# NLTK download
RUN python -m nltk.downloader punkt

COPY open_instruct open_instruct
COPY eval eval
COPY configs configs
COPY scripts scripts
RUN chmod +x scripts/*

# for interactive session
RUN chmod -R 777 /stage/
