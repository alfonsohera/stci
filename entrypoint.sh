#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate stci

# Clone or pull the repository based on GITHUB_TOKEN
if [ ! -z "$GITHUB_TOKEN" ]; then
  echo "Updating repository from GitHub..."
  repo_owner="alfonsohera"
  repo_name="stci"
  
  if [ -d "/workspace/.git" ]; then
    # If repo already exists, pull the latest changes
    git -C /workspace pull "https://${GITHUB_TOKEN}@github.com/${repo_owner}/${repo_name}.git"
  else
    # If not, clone the repository (after backing up any existing files)
    mkdir -p /workspace_backup
    mv /workspace/* /workspace_backup/ 2>/dev/null || true
    git clone "https://${GITHUB_TOKEN}@github.com/${repo_owner}/${repo_name}.git" /workspace_temp
    mv /workspace_temp/* /workspace/
    mv /workspace_temp/.* /workspace/ 2>/dev/null || true
    rm -rf /workspace_temp
  fi
  echo "Repository updated successfully."
fi

# Rest of your entrypoint script
if [ ! -z "$HF_TOKEN" ]; then
  echo "Setting up Hugging Face credentials..."
  huggingface-cli login --token $HF_TOKEN
  echo "Credentials configured."
fi

# Execute the command passed to the container
if [ "$#" -eq 0 ]; then
  echo "No command provided. Keeping container alive..."
  tail -f /dev/null
else
  exec "$@"
fi