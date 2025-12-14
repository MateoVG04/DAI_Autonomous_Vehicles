# 1. Use Official PyTorch DEVEL image
# - CUDA 12.1 (Supports RTX 5070 Ti)
# - Devel version includes 'nvcc' (Required to compile PointPillars setup.py)
# - More compatible with VMs (less AVX crashes than NVIDIA NGC images)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# 2. Install System Dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 3. Copy Files
COPY PointPillars/ /workspace/PointPillars/
COPY requirements.txt /workspace/

# 4. Install Dependencies
WORKDIR /workspace/PointPillars

# We install requirements using the system PyTorch (No torch in requirements.txt!)
RUN pip install --ignore-installed -r /workspace/requirements.txt

# 5. Compile Custom CUDA Kernels
# This step requires the 'devel' image (nvcc must be present)
RUN export TORCH_CUDA_ARCH_LIST="8.6" && python3 setup.py build_ext --inplace

# 6. Prepare Runtime
WORKDIR /workspace
# Copy inference.py to /workspace/inference.py
COPY inference.py .

# 7. Execution Setup
# Add PointPillars to python path so the script can do "import pointpillars"
ENV PYTHONPATH="${PYTHONPATH}:/workspace/PointPillars"

CMD ["python3", "inference.py"]