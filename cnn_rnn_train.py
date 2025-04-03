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
import optuna
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback


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
            logits = model(
                batch["audio"], 
                audio_lengths=batch["audio_lengths"],  
                prosodic_features=batch["prosodic_features"], 
                augmentation_id=batch.get("augmentation_id")
            )
        else:
            logits = model(
                batch["audio"], 
                audio_lengths=batch["audio_lengths"],  
                augmentation_id=batch.get("augmentation_id")
            )
            
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
    if not wandb.run:
        wandb.init(
            project=myConfig.wandb_project,
            entity=myConfig.wandb_entity,
            name=f"cnn_rnn{'_manual' if use_prosodic_features else '_no_manual'}",
            config={
                "model_type": "CNN+RNN",
                "use_prosodic_features": use_prosodic_features,
                "learning_rate": 1e-4,
                "epochs": num_epochs,
                "batch_size": 32,
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
        if wandb.run:
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
    if wandb.run:
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
    
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
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
    from cnn_rnn_model import BalancedAugmentedDataset, DualPathAudioClassifier
    
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
        batch_size=32,
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
    from cnn_rnn_model import DualPathAudioClassifier
    
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
    from cnn_rnn_model import DualPathAudioClassifier
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
        use_prosodic_features=use_prosodic_features,
        is_cnn_rnn=True, 
        log_to_wandb=not myConfig.running_offline,        
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
    from cnn_rnn_model import DualPathAudioClassifier
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
            
        # test with both Youden and F1 thresholds
        for threshold_type in ["youden", "f1"]:
            print(f"\nTesting with {threshold_type.upper()} thresholds...")
            
            # Extract thresholds from the JSON results
            thresholds = {}
            for class_name in ["Healthy", "MCI", "AD"]:
                if threshold_type == "youden":
                    thresholds[class_name] = threshold_results[class_name]["best_youden"]["threshold"]
                else:  # f1
                    thresholds[class_name] = threshold_results[class_name]["best_f1"]["threshold"]
            
            # Run evaluation with thresholds
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            
            all_probs = []
            all_labels = []
            
            # Collect all predictions and labels
            with torch.inference_mode():
                for batch in tqdm(dataloader, desc="Evaluating"):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    
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


def run_cross_validation(use_prosodic_features=True, n_folds=5):
    """Run k-fold cross-validation for the CNN+RNN model."""
    from sklearn.model_selection import KFold
    from cnn_rnn_model import DualPathAudioClassifier, BalancedAugmentedDataset
    import numpy as np
    import json
    
    print(f"Running {n_folds}-fold cross-validation...")
    
    # Load and prepare dataset
    dataset = prepare_cnn_rnn_dataset()
    
    # Combine train and validation for cross-validation
    combined_data = []
    for split in ["train", "validation"]:
        for i in range(len(dataset[split])):
            item = dataset[split][i]
            combined_data.append(item)
    
    # Prepare indices for k-fold splits
    indices = np.arange(len(combined_data))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    cv_results = []
    fold_metrics = []
    
    # Run training for each fold
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n--- Processing Fold {fold_idx+1}/{n_folds} ---")
        
        # Create fold-specific datasets
        fold_train = [combined_data[i] for i in train_idx]
        fold_val = [combined_data[i] for i in val_idx]
        fold_test = dataset["test"]  # Use original test set
        
        # Create balanced training dataset
        print("Creating balanced dataset for this fold...")
        fold_train_balanced = BalancedAugmentedDataset(
            original_dataset=fold_train,
            total_target_samples=1000,
            num_classes=3
        )
        fold_train_balanced.print_distribution_stats()
        
        # Create temporary dataset with the fold splits
        fold_dataset = {
            "train": fold_train_balanced,
            "validation": fold_val,
            "test": fold_test
        }
        
        # Get dataloaders for this fold
        fold_dataloaders = get_cnn_rnn_dataloaders(
            fold_dataset,
            batch_size=32,
            use_prosodic_features=use_prosodic_features
        )
        
        # Create new model for this fold
        fold_model = DualPathAudioClassifier(
            num_classes=3,
            sample_rate=16000,
            use_prosodic_features=use_prosodic_features,
            prosodic_features_dim=len(myData.extracted_features) if use_prosodic_features else 0
        )
        
        # Train the model on this fold
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fold_model.to(device)
        
        # Setup training for this fold with progress tracking
        criterion = FocalLoss(gamma=0, weight=None)
        optimizer = torch.optim.AdamW(
            fold_model.parameters(),
            lr=2e-5,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Calculate total steps for 1cycle scheduler
        total_steps = len(fold_dataloaders["train"]) * 5  # Use 5 epochs for CV
        
        # 1cycle LR scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=5e-4,
            total_steps=total_steps,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1000,
            anneal_strategy='cos',
            three_phase=False
        )
        
        # Train for fewer epochs in cross-validation
        best_val_f1 = 0.0
        fold_best_metrics = {}
        
        print(f"Training fold {fold_idx+1}...")
        for epoch in range(5):  # 5 epochs per fold
            # Train
            train_loss = train_epoch(
                fold_model, fold_dataloaders["train"], optimizer, 
                criterion, device, scheduler, use_prosodic_features
            )
            
            # Validate
            val_loss, val_labels, val_preds = evaluate(
                fold_model, fold_dataloaders["validation"], 
                criterion, device, use_prosodic_features
            )
            
            # Calculate metrics
            val_accuracy = accuracy_score(val_labels, val_preds)
            val_f1_macro = f1_score(val_labels, val_preds, average='macro')
            
            print(f"Fold {fold_idx+1}, Epoch {epoch+1}: Val Acc={val_accuracy:.4f}, F1={val_f1_macro:.4f}")
            
            # Track best model
            if val_f1_macro > best_val_f1:
                best_val_f1 = val_f1_macro
                fold_best_metrics = {
                    "fold": fold_idx+1,
                    "val_accuracy": val_accuracy,
                    "val_f1_macro": val_f1_macro,
                    "val_loss": val_loss / len(fold_dataloaders["validation"]),
                    "epoch": epoch + 1
                }
        
        # Test the best model on this fold
        test_loss, test_labels, test_preds = evaluate(
            fold_model, fold_dataloaders["test"],
            criterion, device, use_prosodic_features
        )
        
        # Calculate test metrics
        test_accuracy = accuracy_score(test_labels, test_preds)
        test_f1_macro = f1_score(test_labels, test_preds, average='macro')
        
        # Add test metrics to fold results
        fold_best_metrics.update({
            "test_accuracy": test_accuracy,
            "test_f1_macro": test_f1_macro,
            "test_loss": test_loss / len(fold_dataloaders["test"])
        })
        
        fold_metrics.append(fold_best_metrics)
        print(f"Fold {fold_idx+1} test results: Acc={test_accuracy:.4f}, F1={test_f1_macro:.4f}")
        
        # Clean up to free memory
        del fold_model, optimizer, scheduler, criterion
        del fold_dataloaders, fold_dataset, fold_train, fold_val
        gc.collect()
        torch.cuda.empty_cache()
    
    # Calculate average metrics across folds
    avg_metrics = {
        "val_accuracy": np.mean([fold["val_accuracy"] for fold in fold_metrics]),
        "val_f1_macro": np.mean([fold["val_f1_macro"] for fold in fold_metrics]),
        "test_accuracy": np.mean([fold["test_accuracy"] for fold in fold_metrics]),
        "test_f1_macro": np.mean([fold["test_f1_macro"] for fold in fold_metrics]),
    }
    
    # Store full results
    cv_results = {
        "fold_metrics": fold_metrics,
        "avg_metrics": avg_metrics
    }
    
    # Print cross-validation summary
    print("\n=== Cross-Validation Results ===")
    print(f"Average validation accuracy: {avg_metrics['val_accuracy']:.4f}")
    print(f"Average validation F1-macro: {avg_metrics['val_f1_macro']:.4f}")
    print(f"Average test accuracy: {avg_metrics['test_accuracy']:.4f}")
    print(f"Average test F1-macro: {avg_metrics['test_f1_macro']:.4f}")
    
    # Save results
    output_dir = os.path.join(myConfig.OUTPUT_PATH, "cross_validation")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, f"cv_{n_folds}fold_results.json"), "w") as f:
        json.dump(cv_results, f, indent=2)
    
    print(f"Cross-validation results saved to {output_dir}")
    return cv_results


def run_bayesian_optimization(use_prosodic_features=True, n_trials=50):
    """Run Bayesian hyperparameter optimization for the CNN+RNN model."""
    from cnn_rnn_model import DualPathAudioClassifier
    import json
    
    # Initialize wandb if not already running
    if not wandb.run:
        wandb.init(
            project="CNN-RNN-HPO",
            entity=myConfig.wandb_entity,
            name="hpo_cnn_rnn",
            config={
                "model_type": "CNN+RNN HPO",
                "use_prosodic_features": use_prosodic_features,
                "n_trials": n_trials,
                "optimization_type": "bayesian"
            },
            tags=["hpo", "bayesian-optimization", "cnn-rnn"]
        )
    
    # Load and prepare dataset
    print("Loading dataset for hyperparameter optimization...")
    dataset = prepare_cnn_rnn_dataset()
    
    # Create dataloaders for optimization
    # Use the entire train set for training and validation set for hyperparameter evaluation
    train_loader = get_cnn_rnn_dataloaders(
        {"train": dataset["train"]}, batch_size=64, use_prosodic_features=use_prosodic_features
    )["train"]
    
    val_loader = get_cnn_rnn_dataloaders(
        {"validation": dataset["validation"]}, batch_size=64, use_prosodic_features=use_prosodic_features
    )["validation"]
    
    # Test set will only be used for final evaluation of the best model
    test_loader = get_cnn_rnn_dataloaders(
        {"test": dataset["test"]}, batch_size=64, use_prosodic_features=use_prosodic_features
    )["test"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def objective(trial):
        # Current parameters
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        max_lr = trial.suggest_float("max_lr", lr, 1e-2, log=True)
        pct_start = trial.suggest_float("pct_start", 0.1, 0.5)
        gamma = trial.suggest_float("focal_loss_gamma", 0.0, 3.0)        
        n_mels = trial.suggest_int("n_mels", 64, 256, log=True)
        time_mask_param = trial.suggest_int("time_mask_param", 10, 100)
        freq_mask_param = trial.suggest_int("freq_mask_param", 10, 100)
        
        trial_history = []
        
        """ dropout = trial.suggest_float("dropout", 0.0, 0.5)
        rnn_hidden_size = trial.suggest_int("rnn_hidden_size", 64, 256)
        rnn_num_layers = trial.suggest_int("rnn_num_layers", 1, 3)
        rnn_type = trial.suggest_categorical("rnn_type", ["lstm", "gru"])
        
        # Class weighting for focal loss
        use_class_weights = trial.suggest_categorical("use_class_weights", [True, False])
        
        # For CNN path complexity
        cnn_channels = trial.suggest_int("cnn_channels", 16, 64)
        cnn_layers = trial.suggest_int("cnn_layers", 2, 4) """
        
        # Create model with trial hyperparameters
        model = DualPathAudioClassifier(
            num_classes=3,
            sample_rate=16000,
            use_prosodic_features=use_prosodic_features,
            prosodic_features_dim=len(myData.extracted_features) if use_prosodic_features else 0,
            n_mels=n_mels,
            apply_specaugment=True
        )
        
        # Update SpecAugment parameters if the model has that module
        if hasattr(model, 'spec_augment'):
            model.spec_augment.time_mask_param = time_mask_param
            model.spec_augment.freq_mask_param = freq_mask_param
        
        model.to(device)
        
        # Create optimizer with trial hyperparameters
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Create focal loss with trial gamma
        criterion = FocalLoss(gamma=gamma)
        
        # Calculate total steps for shorter training (3 epochs)
        total_steps = len(train_loader) * 3
        
        # Create scheduler with trial hyperparameters
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=25,
            final_div_factor=1000,
            anneal_strategy='cos',
            three_phase=False
        )
        
        # Short training loop (5 epochs)
        max_training_epochs = 5
        
        best_trial_f1 = 0.0
        
        trial_name = f"trial_{trial.number}"
        
        # Log trial parameters to wandb
        if wandb.run:
            wandb.run.summary[f"{trial_name}_params"] = {
                param: value for param, value in trial.params.items()
            }
        
        for epoch in range(max_training_epochs):
            # Train
            train_loss = train_epoch(
                model, train_loader, optimizer, 
                criterion, device, scheduler, use_prosodic_features
            )
            # Evaluate 
            val_loss, val_labels, val_preds = evaluate(
                model, val_loader, criterion, device, use_prosodic_features
            )
            val_f1_macro = f1_score(val_labels, val_preds, average='macro')            
            # Append to history 
            trial_history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_f1": val_f1_macro
            })            
            # Log to wandb after each epoch evaluation
            if wandb.run:
                wandb.log({
                    f"{trial_name}/epoch": epoch,
                    f"{trial_name}/train_loss": train_loss,
                    f"{trial_name}/val_f1": val_f1_macro
                })
            # Check for pruning
            trial.report(val_f1_macro, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if val_f1_macro > best_trial_f1:
            best_trial_f1 = val_f1_macro
            # Remove any previous model for this trial if it exists
            trial_model_path = os.path.join(
                myConfig.OUTPUT_PATH, 
                "hyperparameter_optimization", 
                f"trial_{trial.number}_model.pt"
            )
            if os.path.exists(trial_model_path):
                os.remove(trial_model_path)
            torch.save(model.state_dict(), trial_model_path)
        
        # Clean up to free memory
        del model, optimizer, scheduler, criterion
        torch.cuda.empty_cache()
        gc.collect()
        
        # Store history
        with open(os.path.join(
            myConfig.OUTPUT_PATH, 
            "hyperparameter_optimization",
            f"trial_{trial.number}_history.json"
        ), "w") as f:
            json.dump(trial_history, f)
        
        # Log the final result to wandb
        if wandb.run:
            wandb.run.summary[f"{trial_name}_best_f1"] = best_trial_f1
        
        return best_trial_f1
    
    # Create study for maximizing F1-macro
    print(f"Running Bayesian optimization with {n_trials} trials...")
    study_name = f"cnn_rnn_{'with' if use_prosodic_features else 'without'}_manual_features"
        
    # Pruner that balances exploration and exploitation
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, 
        reduction_factor=4, 
        min_early_stopping_rate=0
    )

    # Sampler for more efficient search
    sampler = optuna.samplers.TPESampler(seed=42)

    wandb_callback = WeightsAndBiasesCallback(
        metric_name="f1_macro",
        wandb_kwargs={
            "project": "CNN-RNN-HPO",
            "entity": myConfig.wandb_entity,
            # Use existing wandb run
            "group": wandb.run.name if wandb.run else None
        }
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=pruner,
        sampler=sampler
    )
    
    study.optimize(objective, n_trials=n_trials, callbacks=[wandb_callback])
    
    # Get and print best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    print("\n=== Bayesian Optimization Results ===")
    print(f"Best F1-macro on validation set: {best_value:.4f}")
    print("Best hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Save the results
    output_dir = os.path.join(myConfig.OUTPUT_PATH, "hyperparameter_optimization")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save best hyperparameters
    result = {
        "best_params": best_params,
        "best_value": best_value,
        "feature_type": "with_manual" if use_prosodic_features else "without_manual"
    }
    
    with open(os.path.join(output_dir, f"best_hyperparams_{'with' if use_prosodic_features else 'no'}_manual.json"), "w") as f:
        json.dump(result, f, indent=2)
    
    # Print importance of hyperparameters if optuna has enough trials
    if n_trials >= 20:
        param_importances = optuna.importance.get_param_importances(study)
        print("\nHyperparameter Importance:")
        for param, importance in param_importances.items():
            print(f"  {param}: {importance:.4f}")
    
    # Create visualization if not in offline mode
    if not myConfig.running_offline:
        try:
            # Save optimization plots
            import matplotlib.pyplot as plt
            from optuna.visualization import plot_optimization_history, plot_param_importances
            
            # History plot
            fig1 = plot_optimization_history(study)
            fig1.write_image(os.path.join(output_dir, f"optimization_history_{'with' if use_prosodic_features else 'no'}_manual.png"))
            
            # Parameter importance plot
            fig2 = plot_param_importances(study)
            fig2.write_image(os.path.join(output_dir, f"param_importances_{'with' if use_prosodic_features else 'no'}_manual.png"))
            
            print(f"Optimization visualizations saved to {output_dir}")
        except Exception as e:
            print(f"Could not create visualizations: {str(e)}")
    
    print(f"Optimization results saved to {output_dir}")
    
    if wandb.run:
        # Log best parameters
        wandb.run.summary["best_f1_macro"] = best_value
        wandb.run.summary["best_params"] = best_params
        
        # Log best model artifact
        best_trial_number = study.best_trial.number
        best_model_path = os.path.join(
            myConfig.OUTPUT_PATH, 
            "hyperparameter_optimization", 
            f"trial_{best_trial_number}_model.pt"
        )
        
        if os.path.exists(best_model_path):
            artifact = wandb.Artifact(
                f"hpo-best-model-{wandb.run.id}", 
                type="model",
                description=f"Best model from HPO with F1={best_value:.4f}"
            )
            artifact.add_file(best_model_path, name="best_model.pt")
            wandb.log_artifact(artifact)
    
    # Train final model with best hyperparameters (optional)
    if input("Train final model with best hyperparameters? (y/n): ").lower() == "y":
        train_with_best_hyperparameters(dataset, best_params, use_prosodic_features)
    
    if wandb.run and wandb.run.name.startswith("hpo_cnn_rnn"):
        wandb.finish()
    
    return result


def train_with_best_hyperparameters(dataset, best_params, use_prosodic_features=True):
    """Train a final model using the best hyperparameters from Bayesian optimization."""
    from cnn_rnn_model import DualPathAudioClassifier, BalancedAugmentedDataset
    
    print("\n=== Training with Best Hyperparameters ===")
    
    # Create balanced training dataset
    print("Creating balanced training dataset...")
    balanced_train_dataset = BalancedAugmentedDataset(
        original_dataset=dataset["train"],
        total_target_samples=1000,
        num_classes=3
    )
    balanced_train_dataset.print_distribution_stats()
    
    # Create dataset with balanced training
    balanced_dataset = {
        "train": balanced_train_dataset,
        "validation": dataset["validation"],
        "test": dataset["test"]
    }
    
    # Extract hyperparameters
    lr = best_params.get("learning_rate", 2e-5)
    weight_decay = best_params.get("weight_decay", 0.01)
    max_lr = best_params.get("max_lr", 5e-4)
    pct_start = best_params.get("pct_start", 0.3)
    gamma = best_params.get("focal_loss_gamma", 0.0)
    n_mels = best_params.get("n_mels", 128)
    time_mask_param = best_params.get("time_mask_param", 50)
    freq_mask_param = best_params.get("freq_mask_param", 50)
    
    # Create dataloaders
    dataloaders = get_cnn_rnn_dataloaders(
        balanced_dataset, 
        batch_size=64,
        use_prosodic_features=use_prosodic_features
    )
    
    # Create model with optimized hyperparameters
    model = DualPathAudioClassifier(
        num_classes=3,
        sample_rate=16000,
        use_prosodic_features=use_prosodic_features,
        prosodic_features_dim=len(myData.extracted_features) if use_prosodic_features else 0,
        n_mels=n_mels,
        apply_specaugment=True
    )
    
    # Update SpecAugment parameters if the model has that module
    if hasattr(model, 'spec_augment'):
        model.spec_augment.time_mask_param = time_mask_param
        model.spec_augment.freq_mask_param = freq_mask_param
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Create optimizer with optimized hyperparameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Create focal loss with optimized gamma
    criterion = FocalLoss(gamma=gamma)
    
    # Calculate total steps for full training (10 epochs)
    total_steps = len(dataloaders["train"]) * 10
    
    # Create scheduler with optimized hyperparameters
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        div_factor=25,
        final_div_factor=1000,
        anneal_strategy='cos',
        three_phase=False
    )
    
    # Create output directory for optimized CNN+RNN model
    optimized_output_dir = os.path.join(myConfig.OUTPUT_PATH, "cnn_rnn_optimized")
    os.makedirs(optimized_output_dir, exist_ok=True)
    
    # Tracking variables
    best_f1_macro = 0.0
    
    # Training loop
    for epoch in range(10):
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
        
        # Print metrics
        print(f"Epoch {epoch+1}/10:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")
        print(f"  Val F1-Macro: {val_f1_macro:.4f}")
        print(f"  Val F1 per class: {val_f1_per_class}")
        
        # Save best model based on F1-macro
        if val_f1_macro > best_f1_macro:
            best_f1_macro = val_f1_macro
            
            # Save model to optimized directory
            model_path = os.path.join(optimized_output_dir, "cnn_rnn_optimized.pt")
            torch.save(model.state_dict(), model_path)
            print(f"  Saved new best model with F1-macro: {best_f1_macro:.4f} to {model_path}!")
            
            # Save in safetensors format if available
            try:
                from safetensors.torch import save_file
                safetensors_path = os.path.join(optimized_output_dir, "cnn_rnn_optimized.safetensors")
                save_file(model.state_dict(), safetensors_path)
                print(f"  Also saved model in safetensors format to {safetensors_path}")
            except ImportError:
                print("  safetensors not available, skipping safetensors format")
    
    # Test the best model
    # Load the best model
    model.load_state_dict(torch.load(os.path.join(optimized_output_dir, "cnn_rnn_optimized.pt")))
    
    # Test on test set
    print("\nTesting optimized model on test set...")
    test_loss, test_labels, test_preds = evaluate(
        model, dataloaders["test"], criterion, device, use_prosodic_features
    )
    
    # Calculate test metrics
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_f1_macro = f1_score(test_labels, test_preds, average='macro')
    test_report = classification_report(test_labels, test_preds, target_names=["Healthy", "MCI", "AD"])
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1-Macro: {test_f1_macro:.4f}")
    print("Classification Report:")
    print(test_report)
    
    # Save hyperparameters and test results
    results = {
        "hyperparameters": best_params,
        "test_accuracy": test_accuracy,
        "test_f1_macro": test_f1_macro,
        "val_f1_macro": best_f1_macro
    }
    
    with open(os.path.join(optimized_output_dir, "results.json"), "w") as f:
        import json
        json.dump(results, f, indent=2)
    
    print(f"Optimized model and results saved to {optimized_output_dir}")