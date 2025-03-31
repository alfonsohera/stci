import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import warnings
from zipfile import ZipFile
import torch
import pandas as pd
from transformers import logging
# <local imports>
import myConfig
import myData
import myModel
import myFunctions
import myModel
# </local imports>
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader


# Suppress specific warnings
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized from the model checkpoint.*")
logging.set_verbosity_error()  # Set transformers logging to show only errors

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


def testModelWithThresholds(model, dataset, thresholds=None, threshold_type="youden"):
    """
    Test model on dataset using optimized thresholds.
    
    Args:
        model: The trained model
        dataset: Dataset dictionary containing 'test' split
        thresholds: Dictionary mapping class names to threshold values
                   If None, will use standard argmax prediction
        threshold_type: Type of threshold to use in reporting ("youden" or "f1")
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loader = DataLoader(dataset["test"], batch_size=8, collate_fn=collate_fn)
    class_names = ["Healthy", "MCI", "AD"]
    
    model.to(device)
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            logits = model(
                input_values=batch["input_values"],
                prosodic_features=batch["prosodic_features"]
            ).logits
            
            # Get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
    
    # Convert to numpy arrays
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # Standard argmax prediction (baseline)
    standard_preds = np.argmax(all_probs, axis=-1)
    
    # Calculate baseline metrics
    baseline_accuracy = accuracy_score(all_labels, standard_preds)
    baseline_report = classification_report(
        all_labels, standard_preds, target_names=class_names, output_dict=True
    )
    
    # Print baseline results
    print("=" * 50)
    print("BASELINE (ARGMAX) RESULTS:")
    print(f"Test Accuracy: {baseline_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(all_labels, standard_preds, target_names=class_names))
    print("=" * 50)
    
    # If thresholds are provided, use them for prediction
    if thresholds:
        print(f"\n{threshold_type.upper()} THRESHOLD RESULTS:")
        print(f"Using thresholds: {thresholds}")
        
        # Make predictions with thresholds
        threshold_preds = np.zeros_like(all_labels)
        
        # For each sample, predict class with highest probability/threshold ratio
        for i in range(len(all_labels)):
            best_score = -float('inf')
            best_class = -1
            
            for j, class_name in enumerate(class_names):
                threshold = thresholds[class_name]
                # Calculate how much the probability exceeds the threshold
                score = all_probs[i, j] - threshold
                
                if score > best_score:
                    best_score = score
                    best_class = j
            
            threshold_preds[i] = best_class
        
        # Calculate metrics with optimized thresholds
        threshold_accuracy = accuracy_score(all_labels, threshold_preds)
        threshold_report = classification_report(
            all_labels, threshold_preds, target_names=class_names, output_dict=True
        )
        
        # Print thresholded results
        print(f"Test Accuracy: {threshold_accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(all_labels, threshold_preds, target_names=class_names))
        print("=" * 50)
        
        # Print comparison
        print("\nCOMPARISON (Threshold vs Baseline):")
        print(f"Overall Accuracy: {threshold_accuracy:.4f} vs {baseline_accuracy:.4f} " +
              f"({'better' if threshold_accuracy > baseline_accuracy else 'worse'})")
        
        # Per-class comparison
        for i, class_name in enumerate(class_names):
            baseline_f1 = baseline_report[class_name]['f1-score']
            threshold_f1 = threshold_report[class_name]['f1-score']
            print(f"{class_name} F1: {threshold_f1:.4f} vs {baseline_f1:.4f} " +
                  f"({'better' if threshold_f1 > baseline_f1 else 'worse'})")
            
            baseline_recall = baseline_report[class_name]['recall']
            threshold_recall = threshold_report[class_name]['recall']
            print(f"{class_name} Recall: {threshold_recall:.4f} vs {baseline_recall:.4f} " +
                  f"({'better' if threshold_recall > baseline_recall else 'worse'})")
        
        # Plot confusion matrices
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set up plot 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Baseline confusion matrix
        cm_baseline = confusion_matrix(all_labels, standard_preds)
        sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', 
                  xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_xlabel('Predicted labels')
        ax1.set_ylabel('True labels')
        ax1.set_title('Baseline Confusion Matrix (argmax)')
        
        # Threshold confusion matrix
        cm_threshold = confusion_matrix(all_labels, threshold_preds)
        sns.heatmap(cm_threshold, annot=True, fmt='d', cmap='Blues', 
                  xticklabels=class_names, yticklabels=class_names, ax=ax2)
        ax2.set_xlabel('Predicted labels') 
        ax2.set_ylabel('True labels')
        ax2.set_title(f'Threshold Confusion Matrix ({threshold_type})')
        
        # Save the plot
        output_dir = os.path.join(myConfig.OUTPUT_PATH, "threshold_comparison")
        os.makedirs(output_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_comparison_{threshold_type}.png"))
        plt.close()
        
        print(f"\nConfusion matrix comparison saved to {output_dir}")
        
        return {
            "baseline": {
                "accuracy": baseline_accuracy,
                "report": baseline_report,
                "predictions": standard_preds
            },
            "threshold": {
                "accuracy": threshold_accuracy,
                "report": threshold_report,
                "predictions": threshold_preds
            }
        }
    
    # If no thresholds provided, just return baseline results
    return {
        "baseline": {
            "accuracy": baseline_accuracy,
            "report": baseline_report,
            "predictions": standard_preds
        }
    }


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
    trainer = myModel.createTrainer(model, optimizer, dataset)
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


def test_with_thresholds():
    """Test model using the optimized thresholds"""
    # Configure paths
    path_config = myConfig.configure_paths()
    for key, path in path_config.items():
        setattr(myConfig, key, path)
    
    # Load model
    model_name, _, _ = myModel.getModelDefinitions()
    model, _ = myModel.loadModel(model_name)
    
    # Load dataset
    dataset = myData.loadHFDataset()
    
    # Try to load threshold values from the optimization results
    threshold_results_path = os.path.join(myConfig.OUTPUT_PATH, "threshold_optimization", 
                                          "threshold_optimization_results.json")
    
    if os.path.exists(threshold_results_path):
        print(f"Loading threshold values from {threshold_results_path}")
        import json
        with open(threshold_results_path, 'r') as f:
            results = json.load(f)
        
        # Extract Youden thresholds
        youden_thresholds = {
            class_name: metrics["best_youden"]["threshold"]
            for class_name, metrics in results.items()
        }
        
        # Extract F1 thresholds
        f1_thresholds = {
            class_name: metrics["best_f1"]["threshold"]
            for class_name, metrics in results.items()
        }
        
        print("Testing with Youden's index optimized thresholds...")
        testModelWithThresholds(model, dataset, youden_thresholds, "youden")
        
        print("\nTesting with F1-score optimized thresholds...")
        testModelWithThresholds(model, dataset, f1_thresholds, "f1")
    else:
        print(f"No threshold optimization results found at {threshold_results_path}")
        print("Testing with standard argmax prediction...")
        testModelWithThresholds(model, dataset)


def optimize():
    """Function to run threshold optimization"""
    from myThresholdOptimization import optimize_thresholds_for_model
    
    # Configure paths
    path_config = myConfig.configure_paths()
    for key, path in path_config.items():
        setattr(myConfig, key, path)
    
    # Get model and load it
    model_name, _, _ = myModel.getModelDefinitions()
    model, _ = myModel.loadModel(model_name)
    
    # Load dataset
    dataset = myData.loadHFDataset()
    
    # Create validation dataloader
    val_loader = DataLoader(dataset["validation"], batch_size=8, collate_fn=collate_fn)
    
    # Set output directory
    output_dir = os.path.join(myConfig.OUTPUT_PATH, "threshold_optimization")
    
    # Run threshold optimization
    class_names = ["Healthy", "MCI", "AD"]
    optimize_thresholds_for_model(
        model=model,
        dataloader=val_loader,
        class_names=class_names,
        output_dir=output_dir,
        use_manual_features=False,
        is_cnn_rnn=False,
        log_to_wandb=not myConfig.running_offline
    )


if __name__ == "__main__":
    import argparse
    import numpy as np  
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Cognitive Impairment Detection Model")
    parser.add_argument("mode", choices=["train", "finetune", "test", "optimize", "test_thresholds"], 
                        help="Mode of operation: train (from scratch), finetune (existing model), "
                             "test (evaluate model), optimize (threshold optimization), or "
                             "test_thresholds (evaluate with optimized thresholds)")
    parser.add_argument("--online", action="store_true", help="Run with online services (WandB logging)")
    
    args = parser.parse_args()
    
    # Configure offline/online mode
    myConfig.running_offline = not args.online
    
    # Set training mode and call appropriate function
    if args.mode == "train":
        myConfig.training_from_scratch = True
        print("Starting training from scratch...")
        main_fn()
    elif args.mode == "finetune":
        myConfig.training_from_scratch = False
        print("Starting fine-tuning of existing model...")
        main_fn()
    elif args.mode == "test":
        myConfig.training_from_scratch = False
        print("Running model evaluation...")
        test()
    elif args.mode == "optimize":
        myConfig.training_from_scratch = False
        print("Running threshold optimization...")
        optimize()
    elif args.mode == "test_thresholds":
        myConfig.training_from_scratch = False
        print("Testing model with optimized thresholds...")
        test_with_thresholds()
