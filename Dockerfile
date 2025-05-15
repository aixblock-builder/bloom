FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
# wowai/base-hf:v1.12.0
WORKDIR /app
COPY . ./
COPY requirements.txt .

ENV MODEL_DIR=/data/models
ENV RQ_QUEUE_NAME=default
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV PORT=9090
ENV AIXBLOCK_USE_REDIS=false
ENV HOST_NAME=https://app.aixblock.io
ENV HF_TOKEN=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

RUN apt-get -qq update && \
   DEBIAN_FRONTEND=noninteractive \ 
   apt-get install --no-install-recommends --assume-yes \
    git

RUN apt-get -y purge python3.8
RUN apt-get -y autoremove

RUN apt-get install --reinstall ca-certificates
# Setup
RUN apt update && apt upgrade -y
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y

# Python 3.10
RUN apt install python3.10 -y
RUN apt install python3.10-dev -y
RUN apt install python3.10-distutils -y
RUN apt install python3.10-venv -y
RUN apt install libpq-dev -y uwsgi
RUN apt install build-essential
RUN apt install -y libpq-dev python3-dev
RUN apt install -y python3-pip

RUN python3.10 -m pip install psycopg2
RUN python3.10 -m pip install python-box
RUN python3.10 -m pip install --upgrade colorama
RUN apt install -y nvidia-cuda-toolkit --fix-missing
RUN python3.10 -m pip install torch torchvision torchaudio 
RUN apt-get -qq -y install curl --fix-missing
RUN apt-get update
WORKDIR /app

ENV PYTHONUNBUFFERED=True \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache

RUN --mount=type=cache,target=/root/.cache 
RUN python3.10 -m pip install -r requirements.txt

RUN python3.10 -m pip install --upgrade Flask

RUN python3.10 -m pip install sgl-kernel --force-reinstall --no-deps
RUN python3.10 -m pip install sglang

RUN python3.10 -m pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4
RUN python3.10 -m pip install decord
RUN python3.10 -m pip install python-multipart

RUN python3.10 -m pip install vllm-flash-attn
# RUN python3.10 -m pip install flash_attn==2.5.8
RUN python3.10 -m pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

# RUN python3.10 -m pip install cmake
# RUN python3.10 -m pip install horovod[pytorch] 
#tensorflow,keras,mxnet

RUN python3.10 -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')"
# RUN huggingface-cli download meta-llama/Llama-3.2-1B-Instruct

RUN python3.10 /app/load_model.py
# COPY . ./
EXPOSE 9090 6006 12345
CMD exec gunicorn --preload --bind :${PORT} --workers 1 --threads 1 --timeout 0 _wsgi:app

