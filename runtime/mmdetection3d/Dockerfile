# Base image: PyTorch 2.1 + CUDA 12.1 + cuDNN 8
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Environment variables
ENV TORCH_CUDA_ARCH_LIST="native"
ENV PYTHONUNBUFFERED=1
ENV CUDA_LAUNCH_BLOCKING=1

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Build-time dependencies for MMDetection3D
RUN pip install cython numpy<1.20.0

# Install OpenMMLab mim
RUN pip install -U openmim

# Install MMDetection
RUN mim install mmdet

# Clone and install MMDetection3D v0.18.1 (editable, no build isolation)
RUN git clone -b v0.18.1 https://github.com/open-mmlab/mmdetection3d.git /mmdetection3d
WORKDIR /mmdetection3d
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir --no-build-isolation -e .

# Extra dependencies for inference scripts
RUN pip install opencv-python pyyaml tqdm scipy open3d

# Create workspace
WORKDIR /workspace

# Default command
CMD ["python3", "inference.py"]
