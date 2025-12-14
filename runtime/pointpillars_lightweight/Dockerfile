# 1. Use CUDA 12.8 Developer Image (Supports RTX 50-series / sm_120)
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# We install 'python3-venv' to create our isolated environment.
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    ninja-build \
    git \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# This creates a folder '/opt/venv' that holds python packages.
RUN python3 -m venv /opt/venv
# Activate the Environment Automatically
# By adding the venv/bin to PATH, every command (pip, python) automatically uses it.
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch 2.9.0 with CUDA 12.8 Support
# supports sm_120 architecture.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.9.0 \
    torchvision==0.24.0 \
    torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Copy Files
COPY PointPillars/ /workspace/PointPillars/
COPY requirements.txt /workspace/

RUN pip install --no-cache-dir -r /workspace/requirements.txt

WORKDIR /workspace/PointPillars

# 12.0 is the specific architecture code for Blackwell consumer cards (RTX 50-series).
RUN export TORCH_CUDA_ARCH_LIST="12.0" && \
    export FORCE_CUDA="1" && \
    python3 setup.py build_ext --inplace

# Runtime
WORKDIR /workspace
ENV PYTHONPATH="${PYTHONPATH}:/workspace/PointPillars"

CMD ["python3", "run/inference.py"]