FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    ffmpeg \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python libraries
RUN pip install --no-cache-dir \
    jupyterlab \
    numpy \
    scipy \
    pandas \
    scikit-learn \
    matplotlib \
    opencv-python \
    albumentations \
    timm \
    tqdm \
    huggingface_hub \
    datasets

WORKDIR /workspace

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
jupyter lab -ip=0.0.0.0 --no-browser --allow-root