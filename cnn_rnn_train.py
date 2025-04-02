import os
import torch
import numpy as np
import pandas as pd
import wandb
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm
import gc

# Local imports
import myConfig
import myData
from main import FocalLoss, log_memory_usage
from cnn_rnn_data import prepare_cnn_rnn_dataset, get_cnn_rnn_dataloaders


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
    from sklearn.utils.class_weight import compute_class_weight
    
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