FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
# BASE IMAGE: Must be CUDA 12.1+ for RTX 50 Series support

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git cmake curl unzip libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# 3. PYTORCH FOR CUDA 12.1
# We install PyTorch 2.1.0, which supports both Python 3.9 and RTX 50 Series GPU.
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install spconv (from precompiled wheel)
RUN pip install spconv-cu120

# Extra settings -> fixme remove later if it doesn't work
# 'RSCK' allows old checkpoints to load in the new spconv
# '9.0+PTX' points OpenPCDet (and other libraries) to the right cuda, otherwise setup.py would fail
ENV SPCONV_SAVED_WEIGHT_LAYOUT="RSCK"
ENV TORCH_CUDA_ARCH_LIST="9.0+PTX"

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

RUN python3 setup.py develop

WORKDIR /workspace
CMD ["python3", "inference.py"]
