FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY ./logit.py /workspace/logit.py
