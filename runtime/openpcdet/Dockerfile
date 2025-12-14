FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.9 python3.9-venv python3.9-dev python3-pip \
    git cmake curl unzip libgl1-mesa-glx libglib2.0-0 && \
    ln -s /usr/bin/python3.9 /usr/bin/python && \
    python -m pip install --upgrade pip

# Set working directory
WORKDIR /workspace

# Install PyTorch (CUDA 11.8)
RUN pip install torch==2.0.1+cu118 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Install spconv (from precompiled wheel)
RUN pip install spconv-cu118

# Copy OpenPCDet dependencies and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy OpenPCDet source code
COPY OpenPCDet/ /workspace/OpenPCDet/
WORKDIR /workspace/OpenPCDet

# spconv and openpcdet needs this set explcitly
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Should fix the list out of range error
ENV TORCH_CUDA_ARCH_LIST="9.0"


RUN python setup.py develop

WORKDIR /workspace
CMD ["python3", "inference.py"]
