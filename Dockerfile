FROM ghcr.io/allenai/cuda:12.1-cudnn8-dev-ubuntu20.04-v1.2.116 

WORKDIR /stage/

# TODO When updating flash-attn or torch in the future, make sure to update the version in the requirements.txt file. 
ENV HF_HUB_ENABLE_HF_TRANSFER=1
COPY requirements.txt .
RUN pip install --upgrade pip "setuptools<70.0.0" wheel 
# TODO, unpin setuptools when this issue in flash attention is resolved
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install packaging
RUN pip install flash-attn==2.6.3 --no-build-isolation
RUN pip install -r requirements.txt

# NLTK download
RUN python -m nltk.downloader punkt
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
RUN pip cache purge

# for interactive session
RUN chmod -R 777 /stage/
