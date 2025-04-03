import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, 
    confusion_matrix, accuracy_score, f1_score,
    precision_score, recall_score
)
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd
import seaborn as sns
from typing import Dict, Tuple, List, Any, Optional, Union
import wandb
import argparse
import myConfig
import myData
import myModel
from torch.utils.data import DataLoader
from safetensors.torch import load_file


def get_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    use_prosodic_features: bool = False,
    is_cnn_rnn: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
Get model predictions and true labels from a dataloader.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader containing validation/test data
        use_prosodic_features: Whether the model uses prosodic features
        is_cnn_rnn: Whether the model is CNN+RNN type
        
    Returns:
        Tuple of (all_probs, all_labels) as numpy arrays
"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Getting predictions"):
            # Move all tensor values to the correct device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Handle different model architectures
            if is_cnn_rnn:
                if use_prosodic_features and "prosodic_features" in batch:
                    logits = model(
                        batch["audio"],
                        audio_lengths=batch["audio_lengths"],
                        prosodic_features=batch["prosodic_features"]
                    )
                else:
                    logits = model(
                        batch["audio"],
                        audio_lengths=batch["audio_lengths"]
                    )
            else:
                # Wav2Vec2 type models
                logits = model(
                    input_values=batch["input_values"],
                    prosodic_features=batch["prosodic_features"]
                ).logits
            
            # Get labels from batch
            labels = batch["labels"]
            
            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    return np.vstack(all_probs), np.concatenate(all_labels)


def calculate_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray, 
    threshold: float,
    positive_class: int
) -> Dict[str, float]:
    """
    Calculate classification metrics at a specific threshold.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities for positive class
        threshold: Classification threshold
        positive_class: Index of the positive class
        
    Returns:
        Dictionary of metrics
    """
    # Convert multi-class probabilities to binary classification problem
    binary_y_true = (y_true == positive_class).astype(int)
    
    # Make binary prediction using threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(binary_y_true, y_pred).ravel()
    
    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    youdens_index = sensitivity + specificity - 1
    
    return {
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "accuracy": accuracy,
        "f1": f1,
        "youdens_index": youdens_index,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def optimize_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    threshold_range: np.ndarray = None,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Optimize thresholds for each class based on various metrics.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities (n_samples, n_classes)
        class_names: List of class names
        threshold_range: Array of thresholds to check (default: 0.01 to 0.99 in 0.01 steps)
        output_dir: Directory to save plots
        
    Returns:
        Dictionary with optimal thresholds and metrics for each class
    """
    if threshold_range is None:
        threshold_range = np.arange(0.01, 1.00, 0.01)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # For each class, treat it as the positive class in a one-vs-rest scenario
    for class_idx, class_name in enumerate(class_names):
        class_results = []
        
        # Extract probabilities for this class
        class_probs = y_prob[:, class_idx]
        
        # Calculate ROC curve and AUC
        fpr, tpr, roc_thresholds = roc_curve(y_true == class_idx, class_probs)
        roc_auc = auc(fpr, tpr)
        
        # Calculate Precision-Recall curve
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true == class_idx, class_probs)
        
        # For each threshold, calculate metrics
        for threshold in threshold_range:
            metrics = calculate_metrics_at_threshold(
                y_true, class_probs, threshold, positive_class=class_idx
            )
            class_results.append(metrics)
        
        # Convert results to DataFrame for easier analysis
        df_results = pd.DataFrame(class_results)
        
        # Find optimal thresholds
        best_youden_idx = df_results["youdens_index"].idxmax()
        best_f1_idx = df_results["f1"].idxmax()
        
        best_youden_threshold = df_results.loc[best_youden_idx, "threshold"]
        best_f1_threshold = df_results.loc[best_f1_idx, "threshold"]
        
        # Generate plots if output_dir is provided
        if output_dir:
            # Plot metrics vs threshold
            plt.figure(figsize=(12, 8))
            plt.plot(df_results["threshold"], df_results["sensitivity"], label="Sensitivity")
            plt.plot(df_results["threshold"], df_results["specificity"], label="Specificity")
            plt.plot(df_results["threshold"], df_results["precision"], label="Precision")
            plt.plot(df_results["threshold"], df_results["f1"], label="F1 Score")
            plt.plot(df_results["threshold"], df_results["youdens_index"], label="Youden's Index")
            plt.axvline(x=best_youden_threshold, color='r', linestyle='--', 
                       label=f'Best Youden ({best_youden_threshold:.2f})')
            plt.axvline(x=best_f1_threshold, color='g', linestyle='--', 
                       label=f'Best F1 ({best_f1_threshold:.2f})')
            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.title(f"Metrics vs Threshold for Class: {class_name}")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{class_name}_metrics_vs_threshold.png"))
            plt.close()
            
            # Plot ROC curve
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for Class: {class_name}')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{class_name}_roc_curve.png"))
            plt.close()
            
            # Plot Precision-Recall curve
            plt.figure(figsize=(10, 8))
            plt.plot(recall_curve, precision_curve, lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve for Class: {class_name}')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{class_name}_precision_recall_curve.png"))
            plt.close()
        
        # Save best metrics for this class
        best_youden_metrics = df_results.iloc[best_youden_idx].to_dict()
        best_f1_metrics = df_results.iloc[best_f1_idx].to_dict()
        
        results[class_name] = {
            "best_youden": {
                "threshold": best_youden_threshold,
                **{k: v for k, v in best_youden_metrics.items() if k != "threshold"}
            },
            "best_f1": {
                "threshold": best_f1_threshold,
                **{k: v for k, v in best_f1_metrics.items() if k != "threshold"}
            },
            "roc_auc": roc_auc,
            "all_thresholds": df_results.to_dict('records'),
        }
    
    return results


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Dict[str, float],
    class_names: List[str],
    output_dir: str
) -> None:
    """
    Plot confusion matrices for different thresholds.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        thresholds: Dictionary mapping class names to thresholds
        class_names: List of class names
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create predictions using the optimal thresholds
    y_pred = np.zeros(y_true.shape)
    
    # For each sample, predict the class with the highest probability/threshold ratio
    for i in range(len(y_true)):
        class_scores = []
        for j, class_name in enumerate(class_names):
            # Calculate how much the probability exceeds the threshold
            threshold = thresholds[class_name]
            score = y_prob[i, j] / threshold if threshold > 0 else float('inf')
            class_scores.append((j, score))
        
        # Choose class with highest score
        best_class, _ = max(class_scores, key=lambda x: x[1])
        y_pred[i] = best_class
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix with Optimized Thresholds')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_optimized.png"))
    plt.close()
    
    # Calculate and return metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    # Calculate per-class metrics
    metrics = {}
    for i, class_name in enumerate(class_names):
        binary_y_true = (y_true == i).astype(int)
        binary_y_pred = (y_pred == i).astype(int)
        
        precision = precision_score(binary_y_true, binary_y_pred)
        recall = recall_score(binary_y_true, binary_y_pred)
        f1 = f1_score(binary_y_true, binary_y_pred)
        
        metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    # Save metrics to a text file
    with open(os.path.join(output_dir, "optimized_metrics.txt"), "w") as f:
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Macro: {f1_macro:.4f}\n")
        f.write(f"F1 Weighted: {f1_weighted:.4f}\n\n")
        
        for class_name, class_metrics in metrics.items():
            f.write(f"{class_name} Metrics:\n")
            f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
            f.write(f"  Recall: {class_metrics['recall']:.4f}\n")
            f.write(f"  F1 Score: {class_metrics['f1']:.4f}\n\n")


def optimize_thresholds_for_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    class_names: List[str],
    output_dir: str,
    use_prosodic_features: bool = False,
    is_cnn_rnn: bool = False,
    log_to_wandb: bool = False
) -> Dict[str, float]:
    """
    Main function to optimize thresholds for a given model.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for validation or test data
        class_names: List of class names
        output_dir: Directory to save results
        use_prosodic_features: Whether the model uses prosodic features
        is_cnn_rnn: Whether the model is CNN+RNN type
        log_to_wandb: Whether to log results to wandb
        
    Returns:
        Dictionary of optimal thresholds for each class
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions
    print("Getting model predictions...")
    y_prob, y_true = get_predictions(
        model, dataloader, 
        use_prosodic_features=use_prosodic_features,
        is_cnn_rnn=is_cnn_rnn
    )
    
    # Optimize thresholds
    print("Optimizing thresholds...")
    results = optimize_thresholds(
        y_true, y_prob, 
        class_names=class_names,
        output_dir=os.path.join(output_dir, "threshold_plots")
    )
    
    # Collect best thresholds according to Youden's index
    best_youden_thresholds = {
        class_name: metrics['best_youden']['threshold']
        for class_name, metrics in results.items()
    }
    
    # Collect best thresholds according to F1
    best_f1_thresholds = {
        class_name: metrics['best_f1']['threshold']
        for class_name, metrics in results.items()
    }
    
    # Plot confusion matrices
    print("Plotting confusion matrices...")
    plot_confusion_matrices(
        y_true, y_prob, 
        thresholds=best_youden_thresholds,
        class_names=class_names,
        output_dir=os.path.join(output_dir, "youden_optimization")
    )
    
    plot_confusion_matrices(
        y_true, y_prob, 
        thresholds=best_f1_thresholds,
        class_names=class_names,
        output_dir=os.path.join(output_dir, "f1_optimization")
    )
    
    # Log to wandb if enabled
    if log_to_wandb and wandb.run:
        # Prepare wandb logging data
        wandb_data = {}
        
        for class_name, metrics in results.items():
            # Add best thresholds
            wandb_data[f"best_youden_threshold_{class_name}"] = metrics['best_youden']['threshold']
            wandb_data[f"best_f1_threshold_{class_name}"] = metrics['best_f1']['threshold']
            
            # Add metrics at best Youden threshold
            for metric_name, value in metrics['best_youden'].items():
                if metric_name != "threshold":
                    wandb_data[f"youden_{metric_name}_{class_name}"] = value
            
            # Add metrics at best F1 threshold
            for metric_name, value in metrics['best_f1'].items():
                if metric_name != "threshold":
                    wandb_data[f"f1_opt_{metric_name}_{class_name}"] = value
            
            # Add ROC AUC
            wandb_data[f"roc_auc_{class_name}"] = metrics['roc_auc']
        
        # Log to wandb
        wandb.log(wandb_data)
        
        # Log plots to wandb
        for class_name in class_names:
            wandb.log({
                f"{class_name}_metrics_vs_threshold": wandb.Image(
                    os.path.join(output_dir, "threshold_plots", f"{class_name}_metrics_vs_threshold.png")
                ),
                f"{class_name}_roc_curve": wandb.Image(
                    os.path.join(output_dir, "threshold_plots", f"{class_name}_roc_curve.png")
                ),
                f"{class_name}_precision_recall_curve": wandb.Image(
                    os.path.join(output_dir, "threshold_plots", f"{class_name}_precision_recall_curve.png")
                )
            })
        
        # Log confusion matrices
        wandb.log({
            "confusion_matrix_youden": wandb.Image(
                os.path.join(output_dir, "youden_optimization", "confusion_matrix_optimized.png")
            ),
            "confusion_matrix_f1": wandb.Image(
                os.path.join(output_dir, "f1_optimization", "confusion_matrix_optimized.png")
            )
        })
    
    # Save results to JSON
    import json
    with open(os.path.join(output_dir, "threshold_optimization_results.json"), "w") as f:
        # Convert numpy values to Python native types for JSON serialization
        serializable_results = {}
        for class_name, metrics in results.items():
            serializable_results[class_name] = {
                "best_youden": {
                    k: float(v) if isinstance(v, np.number) else v
                    for k, v in metrics["best_youden"].items()
                },
                "best_f1": {
                    k: float(v) if isinstance(v, np.number) else v
                    for k, v in metrics["best_f1"].items()
                },
                "roc_auc": float(metrics["roc_auc"]),
                # Skip the detailed threshold data to keep the JSON size reasonable
            }
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"Threshold optimization results saved to {output_dir}")
    print("\nOptimal thresholds (Youden's index):")
    for class_name, threshold in best_youden_thresholds.items():
        print(f"  {class_name}: {threshold:.4f}")
    
    print("\nOptimal thresholds (F1 score):")
    for class_name, threshold in best_f1_thresholds.items():
        print(f"  {class_name}: {threshold:.4f}")
    
    return best_youden_thresholds, best_f1_thresholds


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Optimize classification thresholds")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained model (.pth or .safetensors)")
    parser.add_argument("--model_type", type=str, choices=["wav2vec2", "cnn_rnn"], default="wav2vec2",
                        help="Type of model architecture")
    parser.add_argument("--use_manual", action="store_true", 
                        help="Use manual features (for CNN+RNN model)")
    parser.add_argument("--dataset_split", type=str, choices=["validation", "test"], default="validation",
                        help="Dataset split to optimize thresholds on")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for inference")
    parser.add_argument("--output_dir", type=str, default="threshold_optimization",
                        help="Directory to save results")
    parser.add_argument("--log_wandb", action="store_true", 
                        help="Log results to Weights & Biases")
    
    args = parser.parse_args()
    
    # Ensure config paths are set up
    path_config = myConfig.configure_paths()
    for key, path in path_config.items():
        setattr(myConfig, key, path)
    
    # Define class names
    class_names = ["Healthy", "MCI", "AD"]
    
    # Load dataset
    print("Loading dataset...")
    dataset = myData.loadHFDataset()
    
    # Safe loading function that handles both .pth and .safetensors formats
    def safe_load_model_weights(model, path):
        print(f"Loading model weights from {path}...")
        if path.endswith('.safetensors'):
            state_dict = load_file(path)
            model.load_state_dict(state_dict)
        else:
            # For .pth files, we need to handle the PyTorch 2.6+ change in weights_only default
            try:
                state_dict = torch.load(path)
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Failed to load with default settings, trying with weights_only=False: {e}")
                state_dict = torch.load(path, weights_only=False)
                model.load_state_dict(state_dict)
        return model
    
    # Prepare collate function and dataset based on model type
    if args.model_type == "wav2vec2":
        from main import collate_fn
        split_dataset = dataset[args.dataset_split]
        dataloader = DataLoader(
            split_dataset, batch_size=args.batch_size, collate_fn=collate_fn
        )
        
        # Load model
        print(f"Loading Wav2Vec2 model from {args.model_path}...")
        model_name, processor, base_model = myModel.getModelDefinitions()
        model, _ = myModel.loadModel(model_name)
        
        # Load model weights if specified
        if args.model_path:
            model = safe_load_model_weights(model, args.model_path)
        
        # Optimize thresholds
        optimize_thresholds_for_model(
            model=model,
            dataloader=dataloader,
            class_names=class_names,
            output_dir=args.output_dir,
            use_prosodic_features=False,
            is_cnn_rnn=False,
            log_to_wandb=args.log_wandb
        )
    
    elif args.model_type == "cnn_rnn":
        # Import here to avoid issues if the module doesn't exist
        try:
            from main import collate_fn_cnn_rnn
            from cnn_rnn_model import DualPathAudioClassifier
        except ImportError:
            print("Error: Could not import CNN+RNN model. Make sure cnn_rnn_model.py exists and contains the DualPathAudioClassifier class.")
            return
        
        # Prepare dataset for CNN+RNN model
        print("Preparing dataset for CNN+RNN model...")
        dataset = dataset.map(myData.prepare_for_cnn_rnn)
        
        # Create data loader
        split_dataset = dataset[args.dataset_split]
        dataloader = DataLoader(
            split_dataset, batch_size=args.batch_size, collate_fn=collate_fn_cnn_rnn
        )
        
        # Create model
        print("Creating CNN+RNN model...")
        model = DualPathAudioClassifier(
            num_classes=3,
            sample_rate=16000,
            use_prosodic_features=args.use_manual,
            manual_features_dim=len(myData.extracted_features)
        )
        
        # Load model weights if specified
        if args.model_path:
            model = safe_load_model_weights(model, args.model_path)
        
        # Optimize thresholds
        optimize_thresholds_for_model(
            model=model,
            dataloader=dataloader,
            class_names=class_names,
            output_dir=args.output_dir,
            use_prosodic_features=args.use_manual,
            is_cnn_rnn=True,
            log_to_wandb=args.log_wandb
        )
if __name__ == "__main__":
    import myModel  # Import here to avoid circular imports
    main()


