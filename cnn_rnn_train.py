import os
import torch
import numpy as np
import wandb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm
import gc

# Local imports
import myConfig
import myData
from myThresholdOptimization import optimize_thresholds_for_model
from main import  log_memory_usage
from cnn_rnn_data import prepare_cnn_rnn_dataset, get_cnn_rnn_dataloaders
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, input, target):
        # Compute cross entropy with class weights if provided
        ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.weight)
        # Get prediction probabilities
        pt = torch.exp(-ce_loss)
        # Apply focal weighting
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train_epoch(model, train_loader, optimizer, criterion, device, scheduler, use_prosodic_features=True):
    """Train the model for one epoch."""
    import gc
    model.train()
    train_loss = 0.0
    
    for i, batch in enumerate(tqdm(train_loader, desc="Training")):
        # Move batch to GPU if available
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Zero gradients more efficiently
        optimizer.zero_grad()
        
        # Forward pass
        if use_prosodic_features and "prosodic_features" in batch:
            logits = model(batch["audio"], batch["prosodic_features"], batch.get("augmentation_id"))
        else:
            logits = model(batch["audio"], augmentation_id=batch.get("augmentation_id"))
            
        # Calculate loss                
        loss = criterion(logits, batch["labels"].to(device))
        
        # Backpropagation
        loss.backward()                        
        # Update weights
        optimizer.step()        
        # Update LR
        scheduler.step()
        # Track loss 
        train_loss += loss.item()        
        # cleanup of tensors and loss
        logits_detached = logits.detach()        
        del logits, loss, batch
        del logits_detached
        
        # Periodic garbage collection 
        if i % 10 == 0:  # Less frequent to reduce overhead
            gc.collect()
            
    # Explicit cleanup after training phase
    torch.cuda.empty_cache()
    gc.collect()
    
    # Calculate average loss
    avg_train_loss = train_loss / len(train_loader)
    
    return avg_train_loss


def evaluate(model, val_loader, criterion, device, use_prosodic_features=True):
    """Evaluate the model on validation data."""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.inference_mode():
        for batch in tqdm(val_loader, desc="Validation"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            if use_prosodic_features and "prosodic_features" in batch:
                logits = model(batch["audio"], batch["prosodic_features"])
            else:
                logits = model(batch["audio"])
                
            # Calculate loss
            loss = criterion(logits, batch["labels"])
            # Track loss
            val_loss += loss.item()
            # Get predictions
            preds = torch.argmax(logits, dim=-1)
            # Track predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    
    return val_loss, all_labels, all_preds


def train_cnn_rnn_model(model, dataloaders, num_epochs=10, use_prosodic_features=True):
    """Train the CNN+RNN model."""        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Initialize wandb
    if not myConfig.running_offline and not wandb.run:
        wandb.init(
            project=myConfig.wandb_project,
            entity=myConfig.wandb_entity,
            name=f"cnn_rnn{'_manual' if use_prosodic_features else '_no_manual'}",
            config={
                "model_type": "CNN+RNN",
                "use_prosodic_features": use_prosodic_features,
                "learning_rate": 1e-4,
                "epochs": num_epochs,
                "batch_size": 64,
                "weight_decay": 5e-4,
                "prosodic_features_dim": len(myData.extracted_features) if use_prosodic_features else 0
            }
        )
        
        # Watch model parameters and gradients
        if myConfig.wandb_watch_model:
            wandb.watch(model, log="all", log_freq=100)

    # Set up the loss function with default weighting
    criterion = FocalLoss(gamma=0, weight=None)
    
    # Set up the optimizer with proper hyperparameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,            # Starting LR (will be scaled by OneCycleLR)
        weight_decay=0.01,  # L2 regularization
        betas=(0.9, 0.999)  # Default Adam betas
    )
    
    # Calculate total steps for 1cycle scheduler
    total_steps = len(dataloaders["train"]) * num_epochs
    
    # 1cycle LR scheduler with optimized parameters
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,           
        total_steps=total_steps,
        pct_start=0.3,          # Warm up for 30% of training
        div_factor=25,          # Initial LR = max_lr/25
        final_div_factor=1000,  # Final LR = max_lr/1000
        anneal_strategy='cos',  # Cosine annealing
        three_phase=False       # Use standard two-phase schedule
    )
    
    # Create output directory for CNN+RNN model
    cnn_rnn_output_dir = os.path.join(myConfig.training_args.output_dir, "cnn_rnn")
    os.makedirs(cnn_rnn_output_dir, exist_ok=True)
    
    # Tracking variables
    best_f1_macro = 0.0  
        
    # Training loop
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(f"Epoch {epoch+1} start")
        
        # Training phase
        avg_train_loss = train_epoch(
            model, 
            dataloaders["train"], 
            optimizer, 
            criterion, 
            device, 
            scheduler, 
            use_prosodic_features
        )
        
        # Validation phase
        val_loss, all_labels, all_preds = evaluate(
            model, 
            dataloaders["validation"], 
            criterion, 
            device, 
            use_prosodic_features
        )
                       
        # Calculate metrics
        avg_val_loss = val_loss / len(dataloaders["validation"])
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1_macro = f1_score(all_labels, all_preds, average='macro')
        val_f1_per_class = f1_score(all_labels, all_preds, average=None)
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Calculate class-specific metrics
        class_names = ["healthy", "mci", "ad"]
        
        # Calculate per-class metrics
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "val_f1_macro": val_f1_macro,
            "learning_rate": scheduler.get_last_lr()[0]
        }
        
        # Compute class-specific metrics
        tp = {}
        fp = {}
        tn = {}
        fn = {}
        
        # Calculate TP, FP, TN, FN for each class
        for i, class_name in enumerate(class_names):
            tp[i] = sum((np.array(all_preds) == i) & (np.array(all_labels) == i))
            fp[i] = sum((np.array(all_preds) == i) & (np.array(all_labels) != i))
            tn[i] = sum((np.array(all_preds) != i) & (np.array(all_labels) != i))
            fn[i] = sum((np.array(all_preds) != i) & (np.array(all_labels) == i))
            
            # Precision = TP / (TP + FP)
            precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0
            # Recall/Sensitivity = TP / (TP + FN)
            recall = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0
            # Specificity = TN / (TN + FP)
            specificity = tn[i] / (tn[i] + fp[i]) if (tn[i] + fp[i]) > 0 else 0
            # NPV = TN / (TN + FN)
            npv = tn[i] / (tn[i] + fn[i]) if (tn[i] + fn[i]) > 0 else 0
            # F1 = 2 * (precision * recall) / (precision + recall)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Add to the log dictionary
            log_dict[f"val_precision_{class_name}"] = precision
            log_dict[f"val_recall_{class_name}"] = recall
            log_dict[f"val_specificity_{class_name}"] = specificity
            log_dict[f"val_npv_{class_name}"] = npv
            log_dict[f"val_f1_{class_name}"] = f1
        
        # Log metrics to wandb if applicable
        if not myConfig.running_offline and wandb.run:
            wandb.log(log_dict)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")
        print(f"  Val F1-Macro: {val_f1_macro:.4f}")
        print(f"  Val F1 per class: {val_f1_per_class}")
        print(cm)
        
        # Save best model based on F1-macro
        if val_f1_macro > best_f1_macro:
            best_f1_macro = val_f1_macro
            
            # Save model to CNN+RNN specific directory
            model_path = os.path.join(cnn_rnn_output_dir, "cnn_rnn_best.pt")
            torch.save(model.state_dict(), model_path)
            print(f"  Saved new best model with F1-macro: {best_f1_macro:.4f} to {model_path}!")
            
            # Also save in safetensors format if available
            try:
                from safetensors.torch import save_file
                safetensors_path = os.path.join(cnn_rnn_output_dir, "cnn_rnn_best.safetensors")
                save_file(model.state_dict(), safetensors_path)
                print(f"  Also saved model in safetensors format to {safetensors_path}")
            except ImportError:
                print("  safetensors not available, skipping safetensors format")
    
    # End of training, log best model if enabled    
    if not myConfig.running_offline and wandb.run:
        wandb.run.summary["best_f1_macro"] = best_f1_macro
        
        # Log final model if configured
        if myConfig.wandb_log_model:
            model_path = os.path.join(cnn_rnn_output_dir, "cnn_rnn_best.pt")
            safetensors_path = os.path.join(cnn_rnn_output_dir, "cnn_rnn_best.safetensors")
            
            if os.path.exists(model_path):
                artifact = wandb.Artifact(
                    f"cnn-rnn-best-{wandb.run.id}", 
                    type="model",
                    description=f"Best CNN+RNN model with F1-macro={best_f1_macro:.4f}"
                )
                artifact.add_file(model_path, name="model.pt")
                
                if os.path.exists(safetensors_path):
                    artifact.add_file(safetensors_path, name="model.safetensors")
                    
                wandb.log_artifact(artifact)


def test_cnn_rnn_model(model, test_loader, use_prosodic_features=True):
    """Test the CNN+RNN model on the test set."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            if use_prosodic_features and "prosodic_features" in batch:
                logits = model(batch["audio"], batch["prosodic_features"])
            else:
                logits = model(batch["audio"])
                
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    
    # Calculate metrics
    test_accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=["Healthy", "MCI", "AD"]
    )
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Classification Report:")
    print(report)


def main_cnn_rnn(use_prosodic_features=True):
    """Main function for the CNN+RNN pipeline."""
    from myCnnRnnModel import BalancedAugmentedDataset, DualPathAudioClassifier
    
    print(f"Running CNN+RNN model {'with' if use_prosodic_features else 'without'} manual features")
    
    # Load and prepare dataset using the dedicated cnn_rnn_data module
    dataset = prepare_cnn_rnn_dataset()
    
    # Create balanced training dataset with augmentations
    print("Creating balanced training dataset with augmentations...")
    balanced_train_dataset = BalancedAugmentedDataset(
        original_dataset=dataset["train"],
        total_target_samples=1000,
        num_classes=3
    )
    
    # Display class distribution
    balanced_train_dataset.print_distribution_stats()
    
    # Update dataset with balanced training set
    balanced_dataset = {
        "train": balanced_train_dataset,
        "validation": dataset["validation"],
        "test": dataset["test"]
    }
    
    # Get dataloaders optimized for CNN+RNN training
    dataloaders = get_cnn_rnn_dataloaders(
        balanced_dataset, 
        batch_size=64,
        use_prosodic_features=use_prosodic_features
    )
    
    # Create model
    model = DualPathAudioClassifier(
        num_classes=3,
        sample_rate=16000,
        use_prosodic_features=use_prosodic_features,
        prosodic_features_dim=len(myData.extracted_features) if use_prosodic_features else 0
    )
    print("Model created!")
    
    # Train model
    print("Training model...")
    train_cnn_rnn_model(
        model, 
        dataloaders, 
        num_epochs=10, 
        use_prosodic_features=use_prosodic_features
    )
    print("Training complete!")


def test_cnn_rnn(use_prosodic_features=True):
    """Test function for the CNN+RNN pipeline."""
    from myCnnRnnModel import DualPathAudioClassifier
    
    # Prepare data
    dataset = prepare_cnn_rnn_dataset()
    
    # Get dataloaders
    dataloaders = get_cnn_rnn_dataloaders(
        dataset, 
        batch_size=8,
        use_prosodic_features=use_prosodic_features
    )
    
    # Create model
    model = DualPathAudioClassifier(
        num_classes=3,
        sample_rate=16000,
        use_prosodic_features=use_prosodic_features,
        prosodic_features_dim=len(myData.extracted_features)
    )
    
    # Load the best model weights if available
    model_path = os.path.join(myConfig.training_args.output_dir, "cnn_rnn", "cnn_rnn_best.pt")
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("No pre-trained model found. Using randomly initialized weights.")
    
    # Run evaluation
    test_cnn_rnn_model(model, dataloaders["test"], use_prosodic_features)


def optimize_cnn_rnn(use_prosodic_features=True):
    """Function to run threshold optimization for CNN+RNN model"""    
    
    # Configure paths
    path_config = myConfig.configure_paths()
    for key, path in path_config.items():
        setattr(myConfig, key, path)
    
    # Prepare the dataset
    print("Loading dataset for threshold optimization...")
    dataset = prepare_cnn_rnn_dataset()
    
    # Create model
    from myCnnRnnModel import DualPathAudioClassifier
    model = DualPathAudioClassifier(
        num_classes=3,
        sample_rate=16000,
        use_prosodic_features=use_prosodic_features,
        prosodic_features_dim=len(myData.extracted_features) if use_prosodic_features else 0
    )
    
    # Load the best model weights if available
    model_path = os.path.join(myConfig.training_args.output_dir, "cnn_rnn", "cnn_rnn_best.pt")
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        raise FileNotFoundError(f"No pre-trained model found at {model_path}. Please train the model first.")
    
    # Create validation dataloader
    dataloader = get_cnn_rnn_dataloaders(
        dataset, 
        batch_size=8,
        use_prosodic_features=use_prosodic_features
    )["validation"]
    
    # Set output directory
    output_dir = os.path.join(myConfig.OUTPUT_PATH, "threshold_optimization", "cnn_rnn")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run threshold optimization
    class_names = ["Healthy", "MCI", "AD"]
    
    # Custom prediction function for CNN+RNN model
    def predict_fn(model, batch, device):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        if use_prosodic_features and "prosodic_features" in batch:
            logits = model(batch["audio"], batch["prosodic_features"])
        else:
            logits = model(batch["audio"])
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        # Get true labels
        labels = batch["labels"]
        
        return probs, labels
    
    # Run optimization
    print(f"Running threshold optimization for CNN+RNN model {'with' if use_prosodic_features else 'without'} manual features...")
    optimize_thresholds_for_model(
        model=model,
        dataloader=dataloader,
        class_names=class_names,
        output_dir=output_dir,
        use_manual_features=use_prosodic_features,
        is_cnn_rnn=True,  # Flag to indicate CNN+RNN model
        log_to_wandb=not myConfig.running_offline,
        prediction_fn=predict_fn  # Pass custom prediction function
    )
    
    print(f"Threshold optimization completed. Results saved to {output_dir}")


def test_cnn_rnn_with_thresholds(use_prosodic_features=True):
    """Test CNN+RNN model using the optimized thresholds"""
    import os
    import json
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    
    # Configure paths
    path_config = myConfig.configure_paths()
    for key, path in path_config.items():
        setattr(myConfig, key, path)
    
    # Prepare dataset
    print(f"Loading dataset for testing with thresholds ({'with' if use_prosodic_features else 'without'} manual features)...")
    dataset = prepare_cnn_rnn_dataset()
    
    # Create model
    from myCnnRnnModel import DualPathAudioClassifier
    model = DualPathAudioClassifier(
        num_classes=3,
        sample_rate=16000,
        use_prosodic_features=use_prosodic_features,
        prosodic_features_dim=len(myData.extracted_features) if use_prosodic_features else 0
    )
    
    # Load the best model weights
    model_path = os.path.join(myConfig.training_args.output_dir, "cnn_rnn", "cnn_rnn_best.pt")
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        raise FileNotFoundError(f"No pre-trained model found at {model_path}. Please train the model first.")
    
    # Create test dataloader
    dataloader = get_cnn_rnn_dataloaders(
        dataset, 
        batch_size=8,
        use_prosodic_features=use_prosodic_features
    )["test"]
    
    # Try to load threshold values from the optimization results
    threshold_results_path = os.path.join(
        myConfig.OUTPUT_PATH, 
        "threshold_optimization", 
        "cnn_rnn", 
        "threshold_optimization_results.json"
    )
    
    if os.path.exists(threshold_results_path):
        print(f"Loading thresholds from {threshold_results_path}")
        with open(threshold_results_path, "r") as f:
            threshold_results = json.load(f)
            
        # We'll test with both Youden and F1 thresholds
        for threshold_type in ["youden", "f1"]:
            print(f"\nTesting with {threshold_type.upper()} thresholds...")
            
            # Get thresholds for current type
            thresholds = threshold_results[f"{threshold_type}_thresholds"]
            
            # Run evaluation with thresholds
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            
            all_probs = []
            all_labels = []
            
            # Collect all predictions and labels
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Evaluating"):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    
                    if use_prosodic_features and "prosodic_features" in batch:
                        logits = model(batch["audio"], batch["prosodic_features"])
                    else:
                        logits = model(batch["audio"])
                    
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
            class_names = ["Healthy", "MCI", "AD"]
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
            print(f"\n{threshold_type.upper()} THRESHOLD RESULTS:")
            print(f"Using thresholds: {thresholds}")
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
            output_dir = os.path.join(myConfig.OUTPUT_PATH, "threshold_comparison", "cnn_rnn")
            os.makedirs(output_dir, exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"confusion_matrix_comparison_{threshold_type}.png"))
            plt.close()
            
            print(f"\nConfusion matrix comparison saved to {output_dir}")
    else:
        print(f"Threshold optimization results not found at {threshold_results_path}")
        print("Please run optimize_cnn_rnn() first to generate threshold values.")