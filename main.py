import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import warnings
from zipfile import ZipFile
import torch
import pandas as pd
import numpy
from transformers import logging
# <local imports>
import myConfig
import myData
import myModel
import myFunctions
# </local imports>
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F 


# Add this function to your main.py at the top level
def log_memory_usage(label):
    import psutil
    import gc
    
    # Force garbage collection before measurement
    gc.collect()
    
    # Get process memory info
    process = psutil.Process()
    mem_info = process.memory_info()
    
    # Calculate CPU memory usage
    rss_mb = mem_info.rss / (1024 * 1024)
    vms_mb = mem_info.vms / (1024 * 1024)
    
    # System memory info
    sys_mem = psutil.virtual_memory()
    sys_percent = sys_mem.percent
    
    print(f"[{label}] CPU Memory: RSS={rss_mb:.1f}MB, VMS={vms_mb:.1f}MB, System={sys_percent}%")


# Suppress specific warnings
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized from the model checkpoint.*")
logging.set_verbosity_error()  # Set transformers logging to show only errors


class FocalLoss(nn.Module):
        def __init__(self, gamma=2, weight=None):
            super(FocalLoss, self).__init__()
            self.gamma = gamma
            self.weight = weight
            
        def forward(self, input, target):
            ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.weight)
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            return focal_loss.mean()


def collate_fn(batch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, processor, _ = myModel.getModelDefinitions()
    input_values = processor(
        [item['audio']['array'] for item in batch],  # Access the 'array' key
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    ).input_values

    prosodic_features = torch.stack([
        torch.tensor(item["prosodic_features"]) for item in batch
    ])

    labels = torch.tensor([item["label"] for item in batch])

    return {
        "input_values": input_values.to(device),
        "prosodic_features": prosodic_features.to(device),
        "labels": labels.to(device)
    }


def testModel(model, dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loader = DataLoader(dataset["test"], batch_size=8, collate_fn=collate_fn)

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            logits = model(
                input_values=batch["input_values"],
                prosodic_features=batch["prosodic_features"]
            ).logits

            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    # Calculate metrics 
    test_accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=["Healthy", "MCI", "AD"]
    )
    # Print results 
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Classification Report:")
    print(report)


def main_fn():
    # Ensure paths are configured correctly
    path_config = myConfig.configure_paths()
    for key, path in path_config.items():
        setattr(myConfig, key, path)
    
    model_name, processor, base_model = myModel.getModelDefinitions()
    # Data extraction and feature engineering
    myData.DownloadAndExtract()    
    
    # Check if dataframe.csv exists in the Data directory
    data_file_path = os.path.join(myConfig.DATA_DIR, "dataframe.csv")   
    if os.path.exists(data_file_path):
        # Load existing dataframe
        data_df = pd.read_csv(data_file_path)
        print(f"Loaded existing dataframe from {data_file_path}")
        
        # Check if paths are absolute and convert if needed
        if '/' in data_df['file_path'].iloc[0] and not data_df['file_path'].iloc[0].startswith(('Healthy', 'MCI', 'AD')):
            print("Converting absolute paths to relative paths...")
            data_df = myFunctions.convert_absolute_to_relative_paths(data_df)
            # Save the updated dataframe
            data_df.to_csv(data_file_path, index=False)
    else:
        # Create dataframe and save it
        data_df = myFunctions.createDataframe()        
        data_df = myFunctions.featureEngineering(data_df)
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(data_file_path), exist_ok=True)        
        # Save dataframe
        data_df.to_csv(data_file_path, index=False)
        print(f"Created and saved dataframe to {data_file_path}")
    _, weights_tensor = myFunctions.setWeightedCELoss()
    # Feature engineering    
    
    if not os.path.exists(myConfig.OUTPUT_PATH) or (os.path.exists(myConfig.OUTPUT_PATH) and len(os.listdir(myConfig.OUTPUT_PATH)) == 0):
        # Data splits
        train_df, val_df, test_df = myData.datasetSplit(data_df)
        # Apply standard scaling to the splits
        train_df, val_df, test_df = myData.ScaleDatasets(train_df, val_df, test_df)
        # Create HF's dataset
        myData.createHFDatasets(train_df, val_df, test_df)    
    # Load HF's dataset
    dataset = myData.loadHFDataset()
    # Load model
    model, optimizer = myModel.loadModel(model_name)
    # Create trainer
    trainer = myModel.createTrainer(model, optimizer, dataset, weights_tensor)
    trainer.train()
    
    # Save model and processor
    output_dir = os.path.join(myConfig.training_args.output_dir, "final-model")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    
    # Save model in safetensors format
    from safetensors.torch import save_file
    save_file(model.state_dict(), os.path.join(output_dir, "model.safetensors"))
    
    # Save processor
    processor.save_pretrained(output_dir)
    
    # Save config
    if myConfig.training_from_scratch:
        model.config.save_pretrained(myConfig.checkpoint_dir)
    
    # Log final model to wandb if enabled
    if not myConfig.running_offline and "wandb" in myConfig.training_args.report_to:
        import wandb
        if wandb.run and myConfig.wandb_log_model:
            artifact = wandb.Artifact(
                f"final-model-{wandb.run.id}", 
                type="model",
                description="Final trained model"
            )
            
            # Add files to artifact
            artifact.add_dir(output_dir)
            wandb.log_artifact(artifact)
            
            # Mark run as complete
            wandb.run.finish()
    
    print("Training complete! Model saved to", output_dir)


def test():
    model_name, _, _ = myModel.getModelDefinitions()
    model, _ = myModel.loadModel(model_name)
    dataset = myData.loadHFDataset()
    testModel(model, dataset)


if __name__ == "__main__":
    import argparse
    import numpy as np  
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Cognitive Impairment Detection Model")
    parser.add_argument("mode", choices=["train", "finetune", "test", "optimize", "test_thresholds"], 
                        help="Mode of operation: train (from scratch), finetune (existing model), "
                             "test (evaluate model), optimize (threshold optimization), or "
                             "test_thresholds (evaluate with optimized thresholds)")
    parser.add_argument("--pipeline", choices=["wav2vec2", "cnn_rnn"], default="wav2vec2",
                        help="Specify the pipeline to use: wav2vec2 (transformer-based) or cnn_rnn")
    parser.add_argument("--online", action="store_true", 
                        help="Run with online services (WandB logging)")
    parser.add_argument("--no_manual", action="store_true",
                        help="Disable manual features for cnn_rnn pipeline")
    
    args = parser.parse_args()
    
    # Configure offline/online mode
    myConfig.running_offline = not args.online
    
    # Set up the selected pipeline
    if args.pipeline == "wav2vec2":
        # Set training mode and call appropriate function for wav2vec2 pipeline
        if args.mode == "train":
            myConfig.training_from_scratch = True
            print("Starting training from scratch (Wav2Vec2 pipeline)...")
            main_fn()
        elif args.mode == "finetune":
            myConfig.training_from_scratch = False
            print("Starting fine-tuning of existing model (Wav2Vec2 pipeline)...")
            main_fn()
        elif args.mode == "test":
            myConfig.training_from_scratch = False
            print("Running model evaluation (Wav2Vec2 pipeline)...")
            test()
        elif args.mode == "optimize":
            myConfig.training_from_scratch = False
            print("Running threshold optimization (Wav2Vec2 pipeline)...")
            optimize()
        elif args.mode == "test_thresholds":
            myConfig.training_from_scratch = False
            print("Testing model with optimized thresholds (Wav2Vec2 pipeline)...")
            test_with_thresholds()
    elif args.pipeline == "cnn_rnn":
        # Import CNN+RNN functions only when needed
        from cnn_rnn_train import main_cnn_rnn, test_cnn_rnn
        # Import CNN+RNN threshold optimization if it exists
        try:
            from cnn_rnn_train import optimize_cnn_rnn, test_cnn_rnn_with_thresholds
            has_threshold_functions = True
        except ImportError:
            has_threshold_functions = False
            
        use_manual = not args.no_manual
        feature_text = "without" if args.no_manual else "with"
        
        if args.mode == "train":
            myConfig.training_from_scratch = True
            print(f"Starting training from scratch (CNN+RNN pipeline {feature_text} manual features)...")
            main_cnn_rnn(use_prosodic_features=use_manual)
        elif args.mode == "finetune":
            myConfig.training_from_scratch = False
            print(f"Starting fine-tuning (CNN+RNN pipeline {feature_text} manual features)...")
            main_cnn_rnn(use_prosodic_features=use_manual)
        elif args.mode == "test":
            myConfig.training_from_scratch = False
            print(f"Running model evaluation (CNN+RNN pipeline {feature_text} manual features)...")
            test_cnn_rnn(use_prosodic_features=use_manual)
        elif args.mode == "optimize":
            myConfig.training_from_scratch = False
            if has_threshold_functions:
                print(f"Running threshold optimization (CNN+RNN pipeline {feature_text} manual features)...")
                optimize_cnn_rnn(use_prosodic_features=use_manual)
            else:
                print("Threshold optimization not implemented for CNN+RNN pipeline.")
                print("Please use the wav2vec2 pipeline for threshold optimization.")
        elif args.mode == "test_thresholds":
            myConfig.training_from_scratch = False
            if has_threshold_functions:
                print(f"Testing with optimized thresholds (CNN+RNN pipeline {feature_text} manual features)...")
                test_cnn_rnn_with_thresholds(use_prosodic_features=use_manual)
            else:
                print("Testing with thresholds not implemented for CNN+RNN pipeline.")
                print("Please use the wav2vec2 pipeline for threshold testing.")
