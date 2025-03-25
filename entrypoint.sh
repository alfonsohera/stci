#!/bin/bash
if [ ! -z "$HF_TOKEN" ]; then
  echo "Setting up Hugging Face credentials..."
  huggingface-cli login --token $HF_TOKEN
  echo "Credentials configured."
fi

if [ ! -z "$WANDB_API_KEY" ]; then
  echo "Setting up Weights & Biases credentials..."
  wandb login $WANDB_API_KEY
  echo "W&B credentials configured."
fi

# Execute the command passed to the container
if [ "$#" -eq 0 ]; then
  # Default command if none provided
  python /workspace/main.py
else
  # Execute the command provided
  exec "$@"
fi