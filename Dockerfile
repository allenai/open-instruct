ARG BASE_IMAGE=ghcr.io/allenai/cuda:12.8-dev-ubuntu22.04-torch2.7.0-v1.2.170

FROM ${BASE_IMAGE}

WORKDIR /stage/

# Install nginx and create conf.d directory
RUN apt-get update --no-install-recommends && apt-get install -y nginx && mkdir -p /etc/nginx/conf.d && rm -rf /var/lib/apt/lists/*

# TODO When updating flash-attn or torch in the future, make sure to update the version in the requirements.txt file. 
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
RUN pip install packaging --no-cache-dir
RUN pip install flash-attn==2.8.0.post2 flashinfer-python>=0.2.7.post1 --no-build-isolation --no-cache-dir
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader punkt_tab

COPY open_instruct open_instruct
COPY oe-eval-internal oe-eval-internal

# install the package in editable mode
COPY pyproject.toml .
RUN pip install -e .
COPY .git/ ./.git/
COPY eval eval
COPY configs configs
COPY scripts scripts
COPY mason.py mason.py
RUN chmod +x scripts/*

