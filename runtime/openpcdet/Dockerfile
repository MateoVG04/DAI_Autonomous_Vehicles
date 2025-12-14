# BASE IMAGE: Clean CUDA 12.4.1 (Supports RTX 50 drivers)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. Install System Dependencies
# We include 'ninja-build' to speed up the compilation of spconv
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python-is-python3 \
    git cmake curl unzip libgl1-mesa-glx libglib2.0-0 libboost-all-dev ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 2. INSTALL PYTORCH NIGHTLY (Preview)
# We use the nightly channel to get support for CUDA 12.4/RTX 50 without the NGC bloat.
RUN pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124
# Install Torchvision ignoring the strict version check
RUN pip3 install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu124 --no-deps

# --------------------------------------------------------------------------
# 3. COMPILE SPCONV FROM SOURCE
# We must compile this because pre-built wheels don't exist for PyTorch Nightly yet.
# --------------------------------------------------------------------------
RUN pip3 install pccm

# Build CUMM
RUN git clone https://github.com/Finddefinition/cumm.git && \
    cd cumm && \
    export TORCH_CUDA_ARCH_LIST="9.0+PTX" && \
    python3 setup.py bdist_wheel && \
    pip3 install dist/*.whl && \
    cd .. && rm -rf cumm

# Build SPCONV
RUN git clone https://github.com/traveller59/spconv.git --recursive && \
    cd spconv && \
    export TORCH_CUDA_ARCH_LIST="9.0+PTX" && \
    python3 setup.py bdist_wheel && \
    pip3 install dist/*.whl && \
    cd .. && rm -rf spconv
# --------------------------------------------------------------------------

# Environment Configs
ENV SPCONV_SAVED_WEIGHT_LAYOUT="RSCK"
ENV TORCH_CUDA_ARCH_LIST="9.0+PTX"

# 4. INSTALL REQUIREMENTS
COPY requirements.txt .
# We remove torch/torchvision/numpy pins to prevent conflicts with Nightly versions
RUN sed -i '/torch/d' requirements.txt && \
    sed -i '/torchvision/d' requirements.txt && \
    sed -i '/numpy/d' requirements.txt && \
    pip3 install --no-cache-dir -r requirements.txt

# 5. COPY & BUILD OPENPCDET
COPY OpenPCDet/ /workspace/OpenPCDet/
WORKDIR /workspace/OpenPCDet

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ==============================================================================
# Force OpenPCDet to use Spconv 2.x
# We overwrite the version detection script to hardcode the v2 import.
# This fixes "AttributeError: module 'spconv' has no attribute 'SparseModule'"
# This was LLM generated fixme remove if it doesn't work
# ==============================================================================
RUN echo "import spconv.pytorch as spconv" > pcdet/utils/spconv_utils.py

RUN python3 setup.py develop

WORKDIR /workspace
CMD ["python3", "run/inference.py"]