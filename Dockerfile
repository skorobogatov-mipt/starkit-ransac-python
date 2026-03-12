FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies including GUI support
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgl1 \
    libgomp1 \
    libglu1-mesa \
    libxrender1 \
    libxcursor1 \
    libxinerama1 \
    libxrandr2 \
    libxi6 \
    python3-pyqt5 \
    && rm -rf /var/lib/apt/lists/*

# Install Open3D
RUN pip install --upgrade pip setuptools "numpy<2,>=1.23.5" pyqt5
RUN pip3 install --no-cache-dir open3d pytest

WORKDIR /root/starkit-ransac

CMD ["bash"]
