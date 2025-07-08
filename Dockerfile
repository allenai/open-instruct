FROM ghcr.io/allenai/cuda:12.8-dev-ubuntu22.04-torch2.6.0-v1.2.170

WORKDIR /stage/

# install google cloud sdk
RUN apt-get update && apt-get install -y gnupg curl
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y && apt-get install google-cloud-cli -y

# Install nginx and create conf.d directory
RUN apt-get update && apt-get install -y nginx && mkdir -p /etc/nginx/conf.d

# TODO When updating flash-attn or torch in the future, make sure to update the version in the requirements.txt file. 
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN pip install --upgrade pip "setuptools<70.0.0" wheel 
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip install packaging
RUN pip install flash-attn==2.7.2.post1 --no-build-isolation
COPY requirements.txt .
RUN pip install -r requirements.txt
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

