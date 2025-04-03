import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import warnings
import torch
import pandas as pd
import argparse
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import logging
# <local imports>
import myConfig
import myData
import myModel
import myFunctions
# </local imports>
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader


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

    with torch.inference_mode():
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
    
    with torch.inference_mode():
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
    # Create argument parser
    parser = argparse.ArgumentParser(description="Cognitive Impairment Detection Model")
    parser.add_argument("mode", choices=["train", "finetune", "test", "optimize", "test_thresholds", "cv", "hpo"], 
                        help="Mode of operation: train (from scratch), finetune (existing model), "
                             "test (evaluate model), optimize (threshold optimization), "
                             "test_thresholds (evaluate with optimized thresholds), "
                             "cv (cross-validation), or hpo (hyperparameter optimization)")
    parser.add_argument("--pipeline", choices=["wav2vec2", "cnn_rnn"], default="wav2vec2",
                        help="Specify the pipeline to use: wav2vec2 (transformer-based) or cnn_rnn")
    parser.add_argument("--online", action="store_true", 
                        help="Run with online services (WandB logging)")
    parser.add_argument("--no_manual", action="store_true",
                        help="Disable manual features for cnn_rnn pipeline")
    # Add new arguments for cross-validation and hyperparameter optimization
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of folds for cross-validation (default: 5)")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of trials for hyperparameter optimization (default: 50)")
    
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
            from cnn_rnn_train import run_cross_validation, run_bayesian_optimization
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
        elif args.mode == "cv":
            if has_threshold_functions:
                print(f"Running {args.folds}-fold cross-validation (CNN+RNN pipeline {feature_text} manual features)...")
                run_cross_validation(use_prosodic_features=use_manual, n_folds=args.folds)
            else:
                print("Cross-validation not implemented. Please check your installation.")
        elif args.mode == "hpo":
            if has_threshold_functions:
                print(f"Running hyperparameter optimization with {args.trials} trials (CNN+RNN pipeline {feature_text} manual features)...")
                run_bayesian_optimization(use_prosodic_features=use_manual, n_trials=args.trials)
            else:
                print("Hyperparameter optimization not implemented. Please check your installation.")
