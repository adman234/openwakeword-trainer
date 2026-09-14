# openWakeWord Trainer: Kokoro TTS, openWakeWord training and a web UI in one image.
FROM python:3.11-slim-bookworm

# cu126 wheels support NVIDIA GPUs from Pascal (GTX 10xx, Quadro P) up to Ada (RTX 40xx).
# For RTX 50xx (Blackwell) build with: --build-arg TORCH_VERSION=2.8.0 --build-arg TORCH_CUDA=cu128
ARG TORCH_VERSION=2.8.0
ARG TORCH_CUDA=cu126

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/data \
    OUTPUT_DIR=/output \
    ASSET_DIR=/opt/assets \
    TFLITE_PYTHON=/opt/tfenv/bin/python \
    HF_HOME=/data/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HTTP_PORT=8080 \
    HTTPS_PORT=8443 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg espeak-ng libsndfile1 openssl tini ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# TensorFlow lives in its own venv and is only used to write .tflite files for Home Assistant
RUN python -m venv /opt/tfenv \
 && /opt/tfenv/bin/pip install "tensorflow-cpu==2.19.0"

WORKDIR /app

RUN pip install "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
 && echo "torch==${TORCH_VERSION}" > /tmp/constraints.txt

COPY requirements.txt .
RUN pip install -c /tmp/constraints.txt -r requirements.txt \
 && pip install --no-deps openwakeword==0.6.0 \
 && pip install -c /tmp/constraints.txt \
      "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

# Kokoro weights, all voices and the openWakeWord feature models are baked in, so nothing is fetched at runtime
COPY docker/fetch_assets.py docker/fetch_assets.py
RUN python docker/fetch_assets.py

COPY owwt owwt
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
 && python docker/fetch_assets.py --smoke-test \
 && python -c "import owwt.web, owwt.train, owwt.model"

EXPOSE 8080 8443
VOLUME ["/data", "/output"]
HEALTHCHECK --interval=60s --timeout=10s --start-period=90s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/job' % os.environ['HTTP_PORT'], timeout=5)" || exit 1
ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
