FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set a working directory
WORKDIR /workspace

# Copy current folder into /workspace inside the container
COPY . .

# Other libs which cannot be installed via pip
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Pip installing requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Run script
CMD ["python3", "object_tracking.py"]

