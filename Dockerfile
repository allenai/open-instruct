FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV CUDA_HOME=/usr/local/cuda/

RUN apt-get -y update
RUN apt-get -y install git vim jq curl wget

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get -y install git-lfs

WORKDIR /stage/

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install packaging
RUN pip install -r requirements.txt

COPY open_instruct open_instruct
COPY eval eval
COPY ds_configs ds_configs
COPY scripts scripts
RUN chmod +x scripts/*

# for interactive session
RUN chmod -R 777 /stage/