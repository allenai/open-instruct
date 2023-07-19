# see https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
# we want an image with support for the sm_90 compute capability, which is
# needed for Hopper architecture / H100 GPUs
FROM nvcr.io/nvidia/pytorch:23.06-py3

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV CUDA_HOME=/usr/local/cuda/

RUN apt-get -y update
RUN apt-get -y install git vim jq curl wget

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get -y install git-lfs

WORKDIR /stage/

RUN pip install --upgrade pip setuptools wheel
RUN pip install packaging

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY open_instruct open_instruct
COPY eval eval
COPY ds_configs ds_configs
COPY scripts scripts
RUN chmod +x scripts/*

# for interactive session
RUN chmod -R 777 /stage/
