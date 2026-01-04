FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV HF_HOME=/runpod-volume
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-distutils \
    git \
    wget \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Make python = python3.11
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python

# Install pip for Python 3.11 and upgrade it
RUN python -m ensurepip --upgrade && python -m pip install --upgrade pip

# Install deps (use the SAME interpreter)
COPY requirements.txt /requirements.txt
RUN python -m pip install --no-cache-dir -r /requirements.txt

# Install torch cu124
RUN python -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Pin BetterTransformer-compatible stack (key part)
RUN python -m pip install --no-cache-dir "optimum<2.0" "transformers<4.49"

# Smoke test
RUN python -c "import optimum; import optimum.bettertransformer; import transformers; print('ok', optimum.__version__, transformers.__version__)"

ADD src .
COPY test_input.json /test_input.json

CMD ["python", "-u", "/handler.py"]
