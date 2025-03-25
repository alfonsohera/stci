# Start with a minimal base image that includes conda
FROM continuumio/miniconda3:latest

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

# Copy environment.yml first for better caching
COPY ./environment.yml .

# Create conda environment from yml file
RUN conda env create -f environment.yml && \
    conda clean -afy

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "stci", "/bin/bash", "-c"]

# Set cache directory for Hugging Face
ENV HF_HOME=/workspace/.cache/huggingface

# Copy the project code
COPY ./ /workspace/

# Create necessary directories
RUN mkdir -p /workspace/Data /workspace/ProcessedFiles /workspace/checkpoints

# Copy the entrypoint script to the root directory
COPY ./entrypoint.sh /entrypoint.sh

# Ensure the script is executable
RUN chmod +x /entrypoint.sh

# Modify the entrypoint script to also activate conda
RUN sed -i '1a source /opt/conda/etc/profile.d/conda.sh\nconda activate stci' /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]