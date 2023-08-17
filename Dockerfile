FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV CUDA_HOME=/usr/local/cuda/

RUN apt-get -y update
RUN apt-get -y install git vim jq curl wget zip unzip python3 python3-pip

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get -y install git-lfs

WORKDIR /stage/

COPY requirements.txt .
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --upgrade pip setuptools wheel
RUN pip install packaging
RUN pip install flash-attn --no-build-isolation
RUN pip install -r requirements.txt

COPY open_instruct open_instruct
COPY eval eval
COPY ds_configs ds_configs
COPY scripts scripts
RUN chmod +x scripts/*

# for interactive session
RUN chmod -R 777 /stage/
