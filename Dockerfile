# Base image with PyTorch 2.1.0 and CUDA 11.8 (verified existing tag)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    libsndfile1 \
    ffmpeg \
    libavcodec-extra \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements.txt first for better caching
COPY ./requirements.txt .

# Install specific PyTorch versions with CUDA support
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.6.0+cu118 \
    torchaudio==2.6.0+cu118 \
    torchvision==0.21.0+cu118

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir bitsandbytes==0.45.3 demucs soundfile wandb

# Set cache directory for Hugging Face
ENV HF_HOME=/workspace/.cache/huggingface

# Copy the project code
COPY ./ /workspace/

# Create necessary directories
RUN mkdir -p /workspace/Data /workspace/ProcessedFiles /workspace/checkpoints

# Update entrypoint script to handle wandb authentication
RUN echo '#!/bin/bash\n\
if [ ! -z "$HF_TOKEN" ]; then\n\
  echo "Setting up Hugging Face credentials..."\n\
  huggingface-cli login --token $HF_TOKEN\n\
  echo "Credentials configured."\n\
fi\n\
\n\
if [ ! -z "$WANDB_API_KEY" ]; then\n\
  echo "Setting up Weights & Biases credentials..."\n\
  wandb login $WANDB_API_KEY\n\
  echo "W&B credentials configured."\n\
fi\n\
\n\
# Execute the command passed to the container\n\
if [ "$#" -eq 0 ]; then\n\
  # Default command if none provided\n\
  python /workspace/main.py\n\
else\n\
  # Execute the command provided\n\
  exec "$@"\n\
fi' > /entrypoint.sh

RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]