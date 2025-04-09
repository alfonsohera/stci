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
from main import log_memory_usage
from cnn_rnn_data import prepare_cnn_rnn_dataset, get_cnn_rnn_dataloaders
from torch import nn
from torch.nn import functional as F
import optuna


class WandbCallback:
    
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.trial_count = 0
    
    def __call__(self, study, trial):
        """Called after each trial."""
        self.trial_count += 1
        if wandb.run:
            # Existing code
            wandb.log({
                "best_value": study.best_value,
                f"trial_{trial.number}_value": trial.value,
                "trial_number": trial.number,
                "completed_trials": self.trial_count
            })
                        
            if 'val_loss' in trial.user_attrs:
                wandb.log({f"trial_{trial.number}_val_loss": trial.user_attrs['val_loss']})                            
            for key, value in trial.params.items():
                wandb.log({f"trial_{trial.number}_{key}": value})


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


def train_epoch(model, train_loader, optimizer, criterion, device, scheduler=None):
    """Train the model for one epoch."""
    import gc
    model.train()
    train_loss = 0.0
    total_samples = 0  # Track total number of samples or recordings
    
    # Dictionary to track chunks by audio_id
    audio_chunks = {}
    audio_labels = {}
    
    for i, batch in enumerate(tqdm(train_loader, desc="Training")):
        # Move batch to GPU if available
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Extract audio_ids if available
        audio_ids = batch.get("audio_id", None)
        
        # If no audio_ids, process normally (no chunking)
        if audio_ids is None:
            # Zero gradients more efficiently
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(
                batch["audio"], 
                audio_lengths=batch["audio_lengths"],
                augmentation_id=batch.get("augmentation_id", None),
                prosodic_features=batch.get("prosodic_features", None),
                chunk_context=batch.get("chunk_context", None)
            )
                
            # Calculate loss                
            loss = criterion(logits, batch["labels"].to(device))
            
            # Backpropagation
            loss.backward() 
            # Gradient clipping
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)                       
            # Update weights
            optimizer.step()        
            # Update LR
            #scheduler.step()
            
            # Track loss (count each sample)
            batch_size = batch["audio"].size(0)
            train_loss += loss.item() * batch_size
            total_samples += batch_size
        else:
            # Process batches with audio_ids for chunking
            # Group logits by audio_id for later aggregation
            
            # Forward pass to get logits for each chunk
            logits = model(
                batch["audio"], 
                audio_lengths=batch["audio_lengths"],
                augmentation_id=batch.get("augmentation_id", None),
                prosodic_features=batch.get("prosodic_features", None),
                chunk_context=batch.get("chunk_context", None)
            )
            
            # Store chunks by audio_id
            for j, audio_id in enumerate(audio_ids):
                if audio_id not in audio_chunks:
                    audio_chunks[audio_id] = []
                    # Store the label for this audio (all chunks should have the same label)
                    audio_labels[audio_id] = batch["labels"][j]
                
                # Store the logits for this chunk
                audio_chunks[audio_id].append(logits[j])
            
            # Process complete audios (all chunks received)
            # Here we could add logic to determine when we have all chunks for an audio
            # For simplicity, let's process any audio that has accumulated chunks each batch
            complete_audio_ids = list(audio_chunks.keys())
            
            if complete_audio_ids:
                # Zero gradients before processing complete audios
                optimizer.zero_grad()
                
                # Accumulate loss for all complete audios
                batch_loss = 0.0
                
                for audio_id in complete_audio_ids:
                    # Aggregate predictions from all chunks
                    chunk_outputs = audio_chunks[audio_id]
                    aggregated_output = model.aggregate_chunk_predictions(chunk_outputs)
                    
                    # Get label for this audio
                    label = audio_labels[audio_id]
                    
                    # Calculate loss using the aggregated output
                    # Need to unsqueeze to match expected dimensions [batch_size, num_classes]
                    loss = criterion(aggregated_output.unsqueeze(0), label.unsqueeze(0))
                    batch_loss += loss
                
                # Average loss across all processed audios
                batch_loss = batch_loss / len(complete_audio_ids)
                
                # Backpropagation
                batch_loss.backward()
                
                # Update weights
                optimizer.step()
                
                # Update LR
                #scheduler.step()
                
                # Track loss by number of recordings (not multiplying by length)
                train_loss += batch_loss.item() * len(complete_audio_ids)
                total_samples += len(complete_audio_ids)
                
                # Clear processed audio chunks and labels
                for audio_id in complete_audio_ids:
                    del audio_chunks[audio_id]
                    del audio_labels[audio_id]
        
        # cleanup of tensors and loss
        logits_detached = logits.detach()        
        del logits, loss, batch
        del logits_detached
        
        # Periodic garbage collection 
        if i % 50 == 0:  # Less frequent to reduce overhead
            gc.collect()
            
    # Explicit cleanup after training phase
    torch.cuda.empty_cache()
    gc.collect()
    
    # Calculate average loss
    avg_train_loss = train_loss / total_samples if total_samples > 0 else 0
    
    return avg_train_loss


def evaluate(model, val_loader, criterion, device):
    """Evaluate the model on validation data."""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    # Dictionary to track chunks by audio_id
    audio_chunks = {}
    audio_labels = {}
    
    with torch.inference_mode():
        for batch in tqdm(val_loader, desc="Validation"):
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Extract audio_ids if available
            audio_ids = batch.get("audio_id", None)
            
            # If no audio_ids, process normally (no chunking)
            if audio_ids is None:
                # Forward pass
                logits = model(
                    batch["audio"], 
                    audio_lengths=batch["audio_lengths"],
                    augmentation_id=batch.get("augmentation_id", None),
                    prosodic_features=batch.get("prosodic_features", None),
                    chunk_context=batch.get("chunk_context", None)
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
            else:
                # Process batches with audio_ids for chunking
                # Forward pass to get logits for each chunk
                logits = model(
                    batch["audio"], 
                    audio_lengths=batch["audio_lengths"],
                    augmentation_id=batch.get("augmentation_id", None),
                    prosodic_features=batch.get("prosodic_features", None),
                    chunk_context=batch.get("chunk_context", None)
                )
                
                # Store chunks by audio_id
                for j, audio_id in enumerate(audio_ids):
                    if audio_id not in audio_chunks:
                        audio_chunks[audio_id] = []
                        # Store the label for this audio
                        audio_labels[audio_id] = batch["labels"][j]
                    
                    # Store the logits for this chunk
                    audio_chunks[audio_id].append(logits[j])
    
    # Process all remaining audios after going through the entire dataset
    if audio_chunks:        
        for audio_id, chunk_outputs in audio_chunks.items():
            # Aggregate predictions from all chunks
            aggregated_output = model.aggregate_chunk_predictions(chunk_outputs)
            
            # Get label for this audio
            label = audio_labels[audio_id]
            
            # Calculate loss using the aggregated output
            loss = criterion(aggregated_output.unsqueeze(0), label.unsqueeze(0))
            val_loss += loss.item() 
            
            # Get prediction from aggregated output
            pred = torch.argmax(aggregated_output)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(label.cpu().numpy())
    
    return val_loss, all_labels, all_preds


def train_cnn_rnn_model(model, dataloaders, num_epochs=10):
    """Train the CNN+RNN model."""        
    from sklearn.utils.class_weight import compute_class_weight
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # From HPO:        
    hpo_max_lr = 0.0001749508037956646
    hpo_focal_loss_gamma = 1.412267303911459
    hpo_weight_scaling_factor = 0.9725833997090791
    hpo_weight_decay = 1.072231283804023e-06
    hpo_dropout_factor = 0.9097065013330266

    # Initialize wandb
    if not wandb.run:
        wandb.init(
            project=myConfig.wandb_project,
            entity=myConfig.wandb_entity,
            name="cnn_rnn",
            config={
                "model_type": "CNN+RNN",
                "learning_rate": hpo_max_lr,
                "epochs": num_epochs,
                "batch_size": 96,
                "weight_decay": hpo_weight_decay
            }
        )
        
        # Watch model parameters and gradients
        if myConfig.wandb_watch_model:
            wandb.watch(model, log="all", log_freq=100)

    # Calculate class weights with scaling factor
    classes = np.array([0, 1, 2])  
    class_counts = myConfig.num_samples_per_class
    y = np.array([])
    # Create array with labels based on known counts
    for class_id, count in class_counts.items():
        y = np.append(y, [class_id] * count)
    # Compute balanced weights
    raw_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)  
    # Apply scaling factor to make weights less extreme
    scaled_weights = np.power(raw_weights, hpo_weight_scaling_factor)
    # Normalize to maintain sum proportionality
    scaled_weights = scaled_weights * (len(classes) / np.sum(scaled_weights))
    # Convert to tensor
    weight_tensor = torch.tensor(scaled_weights, device=device, dtype=torch.float32)
    # Set up the loss function with class weighting
    criterion = FocalLoss(gamma=hpo_focal_loss_gamma, weight=weight_tensor)
    
    # Adjust all dropout rates in the model
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            # Get current dropout probability
            current_p = module.p
            # Apply dropout factor, but cap at 0.7 to avoid excessive dropout
            new_p = min(current_p * hpo_dropout_factor, 0.7)
            # Set new dropout probability
            module.p = new_p
    model.to(device)


    # Set up the optimizer with proper hyperparameters
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hpo_max_lr,
        weight_decay=hpo_weight_decay,  # L2 regularization
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
            device
        )
        
        # Validation phase
        val_loss, all_labels, all_preds = evaluate(
            model, 
            dataloaders["validation"], 
            criterion, 
            device
        )
                       
        # Calculate metrics
        total_val_recordings = len(all_labels)  # This now represents total processed recordings
        avg_val_loss = val_loss / total_val_recordings
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
            "learning_rate": hpo_max_lr
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


def test_cnn_rnn_model(model, test_loader):
    """Test the CNN+RNN model on the test set."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # Dictionary to track chunks by audio_id
    audio_chunks = {}
    audio_labels = {}
    
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Extract audio_ids if available
            audio_ids = batch.get("audio_id", None)
            
            # If no audio_ids, process normally (no chunking)
            if audio_ids is None:
                logits = model(
                    batch["audio"], 
                    audio_lengths=batch["audio_lengths"],
                    augmentation_id=batch.get("augmentation_id", None),
                    prosodic_features=batch.get("prosodic_features", None),
                    chunk_context=batch.get("chunk_context", None)
                )
                
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
            else:
                # Process batches with audio_ids for chunking
                logits = model(
                    batch["audio"], 
                    audio_lengths=batch["audio_lengths"],
                    augmentation_id=batch.get("augmentation_id", None),
                    prosodic_features=batch.get("prosodic_features", None),
                    chunk_context=batch.get("chunk_context", None)
                )
                
                # Store chunks by audio_id
                for j, audio_id in enumerate(audio_ids):
                    if audio_id not in audio_chunks:
                        audio_chunks[audio_id] = []
                        # Store the label for this audio
                        audio_labels[audio_id] = batch["labels"][j]
                    
                    # Store the logits for this chunk
                    audio_chunks[audio_id].append(logits[j])
    
    # Process all remaining audios after going through the entire dataset
    if audio_chunks:
        for audio_id, chunk_outputs in audio_chunks.items():
            # Aggregate predictions from all chunks
            aggregated_output = model.aggregate_chunk_predictions(chunk_outputs)
            
            # Get label for this audio
            label = audio_labels[audio_id]
            
            # Get prediction from aggregated output
            pred = torch.argmax(aggregated_output)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(label.cpu().numpy())
    
    # Calculate metrics
    test_accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=["Healthy", "MCI", "AD"]
    )
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Classification Report:")
    print(report)


def main_cnn_rnn(use_prosodic_features=False):
    """Main function for the CNN+RNN pipeline."""
    from cnn_rnn_model import AugmentedDataset, DualPathAudioClassifier
    hpo_n_mels = 128
    print("Running CNN+RNN model")
    
    # Load and prepare dataset using the dedicated cnn_rnn_data module
    dataset = prepare_cnn_rnn_dataset()
    
    # Create balanced training dataset with augmentations
    print("Creating balanced training dataset with augmentations...")
    augmented_train_dataset = AugmentedDataset(
        original_dataset=dataset["train"],        
        num_classes=3
    )
    
    # Display class distribution
    augmented_train_dataset.print_distribution_stats()
    
    # Update dataset with balanced training set
    balanced_dataset = {
        "train": augmented_train_dataset,
        "validation": dataset["validation"],
        "test": dataset["test"]
    }
    
    # Get dataloaders optimized for CNN+RNN training
    dataloaders = get_cnn_rnn_dataloaders(
        balanced_dataset, 
        batch_size=96
    )
    
    # Create model
    model = DualPathAudioClassifier(
        num_classes=3,
        sample_rate=16000,
        n_mels=hpo_n_mels
    )
    print("Model created!")
    
    # Train model
    print("Training model...")
    train_cnn_rnn_model(
        model, 
        dataloaders, 
        num_epochs=15
    )
    print("Training complete!")


def test_cnn_rnn():
    """Test function for the CNN+RNN pipeline."""
    from cnn_rnn_model import DualPathAudioClassifier
    hpo_n_mels = 128
    # Prepare data
    dataset = prepare_cnn_rnn_dataset()
    
    # Get dataloaders
    dataloaders = get_cnn_rnn_dataloaders(
        dataset, 
        batch_size=96
    )
    
    # Create model
    model = DualPathAudioClassifier(
        num_classes=3,
        sample_rate=16000,
        n_mels=hpo_n_mels
    )
    
    # Load the best model weights if available
    model_path = os.path.join(myConfig.training_args.output_dir, "cnn_rnn", "cnn_rnn_best.pt")
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("No pre-trained model found. Using randomly initialized weights.")
    
    # Run evaluation
    test_cnn_rnn_model(model, dataloaders["test"])


def optimize_cnn_rnn():
    """Function to run threshold optimization for CNN+RNN model"""    
    hpo_n_mels = 128
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
        n_mels=hpo_n_mels
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
        batch_size=96
    )["validation"]
    
    # Set output directory
    output_dir = os.path.join(myConfig.OUTPUT_PATH, "threshold_optimization", "cnn_rnn")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run threshold optimization
    class_names = ["Healthy", "MCI", "AD"]
    
    # Run optimization
    print("Running threshold optimization for CNN+RNN model...")
    optimize_thresholds_for_model(
        model=model,
        dataloader=dataloader,
        class_names=class_names,
        output_dir=output_dir,
        is_cnn_rnn=True, 
        log_to_wandb=not myConfig.running_offline        
    )
    
    print(f"Threshold optimization completed. Results saved to {output_dir}")


def test_cnn_rnn_with_thresholds():
    """Test CNN+RNN model using the optimized thresholds"""
    import os
    import json
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    
    hpo_n_mels = 128
    # Configure paths
    path_config = myConfig.configure_paths()
    for key, path in path_config.items():
        setattr(myConfig, key, path)
    
    # Prepare dataset
    print("Loading dataset for testing with thresholds...")
    dataset = prepare_cnn_rnn_dataset()
    
    # Create model
    from cnn_rnn_model import DualPathAudioClassifier
    model = DualPathAudioClassifier(
        num_classes=3,
        sample_rate=16000,
        n_mels=hpo_n_mels
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
        batch_size=96
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
            
            # Dictionary to track chunks by audio_id
            audio_chunks = {}
            audio_labels = {}

            with torch.inference_mode():
                for batch in tqdm(dataloader, desc="Evaluating"):
                    batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    
                    # Extract audio_ids if available
                    audio_ids = batch.get("audio_id", None)
                    
                    # If no audio_ids, process normally (no chunking)
                    if audio_ids is None:
                        logits = model(
                            batch["audio"], 
                            audio_lengths=batch["audio_lengths"],
                            augmentation_id=batch.get("augmentation_id", None),
                            prosodic_features=batch.get("prosodic_features", None),
                            chunk_context=batch.get("chunk_context", None)
                        )
                        
                        # Get probabilities
                        probs = torch.softmax(logits, dim=-1)
                        
                        all_probs.append(probs.cpu().numpy())
                        all_labels.append(batch["labels"].cpu().numpy())
                    else:
                        # Process batches with audio_ids for chunking
                        logits = model(
                            batch["audio"], 
                            audio_lengths=batch["audio_lengths"],
                            augmentation_id=batch.get("augmentation_id", None),
                            prosodic_features=batch.get("prosodic_features", None),
                            chunk_context=batch.get("chunk_context", None)
                        )
                        
                        # Store chunks by audio_id
                        for j, audio_id in enumerate(audio_ids):
                            if audio_id not in audio_chunks:
                                audio_chunks[audio_id] = []
                                # Store the label for this audio
                                audio_labels[audio_id] = batch["labels"][j]
                            
                            # Store the logits for this chunk
                            audio_chunks[audio_id].append(logits[j])

            # Process all remaining audios after going through the entire dataset
            if audio_chunks:
                for audio_id, chunk_outputs in audio_chunks.items():
                    # Aggregate predictions from all chunks
                    aggregated_output = model.aggregate_chunk_predictions(chunk_outputs)
                    
                    # Get label for this audio
                    label = audio_labels[audio_id]
                    
                    # Get probabilities from aggregated output
                    probs = torch.softmax(aggregated_output, dim=-1)
                    
                    all_probs.append(probs.cpu().numpy().reshape(1, -1))
                    all_labels.append(label.cpu().numpy().reshape(1))
            
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


def run_cross_validation(n_folds=5):
    from sklearn.utils.class_weight import compute_class_weight
    """Run k-fold cross-validation for the CNN+RNN model."""
    from sklearn.model_selection import KFold
    from cnn_rnn_model import DualPathAudioClassifier, AugmentedDataset
    import numpy as np
    import json
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # From HPO:    
    hpo_max_lr = 0.0001749508037956646
    hpo_focal_loss_gamma = 1.412267303911459
    hpo_weight_scaling_factor = 0.9725833997090791
    hpo_weight_decay = 1.072231283804023e-06
    hpo_dropout_factor = 0.9097065013330266
    hpo_n_mels = 128

    cv_epochs = 15
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
        fold_train_balanced = AugmentedDataset(
            original_dataset=fold_train,            
            num_classes=3
        )        
        
        # Create temporary dataset with the fold splits
        fold_dataset = {
            "train": fold_train_balanced,
            "validation": fold_val,
            "test": fold_test
        }
        
        # Get dataloaders for this fold
        fold_dataloaders = get_cnn_rnn_dataloaders(
            fold_dataset,
            batch_size=96
        )
        
        # Create new model for this fold
        fold_model = DualPathAudioClassifier(
            num_classes=3,
            sample_rate=16000,
            n_mels=hpo_n_mels
        )
        
        # Calculate class weights with scaling factor
        classes = np.array([0, 1, 2])  
        class_counts = myConfig.num_samples_per_class
        y = np.array([])
        # Create array with labels based on known counts
        for class_id, count in class_counts.items():
            y = np.append(y, [class_id] * count)
        # Compute balanced weights
        raw_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)  
        # Apply scaling factor to make weights less extreme
        scaled_weights = np.power(raw_weights, hpo_weight_scaling_factor)
        # Normalize to maintain sum proportionality
        scaled_weights = scaled_weights * (len(classes) / np.sum(scaled_weights))
        # Convert to tensor
        weight_tensor = torch.tensor(scaled_weights, device=device, dtype=torch.float32)
        # Set up the loss function with class weighting
        criterion = FocalLoss(gamma=hpo_focal_loss_gamma, weight=weight_tensor)
        
        # Adjust all dropout rates in the model
        for module in fold_model.modules():
            if isinstance(module, nn.Dropout):
                # Get current dropout probability
                current_p = module.p
                # Apply dropout factor, but cap at 0.7 to avoid excessive dropout
                new_p = min(current_p * hpo_dropout_factor, 0.7)
                # Set new dropout probability
                module.p = new_p
 
        # Train the model on this fold        
        fold_model.to(device)
                        
        optimizer = torch.optim.Adam(
            fold_model.parameters(),
            lr=hpo_max_lr,
            weight_decay=hpo_weight_decay,            
        )
        
        # Train for fewer epochs in cross-validation
        best_val_f1 = 0.0
        fold_best_metrics = {}
        
        print(f"Training fold {fold_idx+1}...")
        for epoch in range(cv_epochs):  # 5 epochs per fold
            # Train
            train_loss = train_epoch(
                fold_model, fold_dataloaders["train"], optimizer, 
                criterion, device
            )
            
            # Validate
            val_loss, val_labels, val_preds = evaluate(
                fold_model, fold_dataloaders["validation"], 
                criterion, device
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
            criterion, device
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
        del fold_model, optimizer, criterion
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


def run_bayesian_optimization(n_trials=50, resume_study=False, n_folds=5):
    """Run Bayesian hyperparameter optimization for the CNN+RNN model with k-fold cross-validation."""
    from cnn_rnn_model import DualPathAudioClassifier, AugmentedDataset
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.model_selection import KFold
    import json
    import joblib
    
    # Initialize wandb if not already running
    if not wandb.run:
        wandb.init(
            project="CNN-RNN-HPO",
            entity=myConfig.wandb_entity,
            name=f"hpo_cnn_rnn_{n_folds}fold",
            config={
                "model_type": "CNN+RNN HPO with CV",
                "n_trials": n_trials,
                "optimization_type": "bayesian",
                "n_folds": n_folds
            },
            tags=["hpo", "bayesian-optimization", "cnn-rnn", "cross-validation"]
        )
    
    # Create output directory for hyperparameter optimization    
    output_dir = os.path.join(myConfig.OUTPUT_PATH, "hyperparameter_optimization")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare dataset
    print("Loading dataset for hyperparameter optimization with cross-validation...")
    dataset = prepare_cnn_rnn_dataset()
    
    # Combine train and validation for cross-validation
    combined_data = []
    for split in ["train", "validation"]:
        for i in range(len(dataset[split])):
            combined_data.append(dataset[split][i])
    
    # Prepare indices for k-fold splits
    indices = np.arange(len(combined_data))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def objective(trial):
        try:            
            torch.cuda.empty_cache()
            gc.collect()
                        
            fixed_lr = trial.suggest_float("fixed_lr", 0.00012, 0.00025, log=True)
            focal_loss_gamma = trial.suggest_float("focal_loss_gamma", 0, 2)
            weight_scaling_factor = trial.suggest_float("weight_scaling_factor", 0.1, 1.0)            
            time_mask_param = trial.suggest_int("time_mask_param", 35, 50)
            freq_mask_param = trial.suggest_int("freq_mask_param", 35, 50)            
            dropout_factor = trial.suggest_float("dropout_factor", 0.85, 1.0)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
            
            trial_name = f"trial_{trial.number}"
            
            # Log trial parameters to wandb
            if wandb.run:
                wandb.log({
                    f"{trial_name}_params": {
                        param: value for param, value in trial.params.items()
                    }
                })
            
            # Run k-fold cross-validation for this trial
            fold_f1_scores = []
            fold_val_losses = []
            n_mels = 128
            
            # Number of epochs for HPO cross-validation (reduced)
            n_epochs = 6  # Reduced from 10 to make HPO faster with k-fold
            
            print(f"\n--- Trial {trial.number}: Running {n_folds}-fold cross-validation ---")
            
            # Store fold histories
            fold_histories = []
            
            # For each fold
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
                print(f"  Processing fold {fold_idx+1}/{n_folds} for trial {trial.number}")
                
                # Create fold-specific datasets
                fold_train = [combined_data[i] for i in train_idx]
                fold_val = [combined_data[i] for i in val_idx]
                
                # Create balanced training dataset for this fold
                fold_train_balanced = AugmentedDataset(
                    original_dataset=fold_train,            
                    num_classes=3
                )
                
                # Create temporary dataset with the fold splits
                fold_dataset = {
                    "train": fold_train_balanced,
                    "validation": fold_val
                }
                
                # Get dataloaders for this fold
                fold_train_loader = get_cnn_rnn_dataloaders(
                    {"train": fold_dataset["train"]},
                    batch_size=96
                )["train"]
                
                fold_val_loader = get_cnn_rnn_dataloaders(
                    {"validation": fold_dataset["validation"]},
                    batch_size=96
                )["validation"]
                
                # Create new model for this fold with trial hyperparameters
                fold_model = DualPathAudioClassifier(
                    num_classes=3,
                    sample_rate=16000,
                    n_mels=n_mels,
                    apply_specaugment=True
                )
                fold_model.to(device)
                
                # Update SpecAugment parameters if the model has that module
                if hasattr(fold_model, 'spec_augment'):
                    fold_model.spec_augment.time_mask_param = time_mask_param
                    fold_model.spec_augment.freq_mask_param = freq_mask_param
                
                # Calculate class weights with scaling factor
                classes = np.array([0, 1, 2])  
                class_counts = myConfig.num_samples_per_class
                y = np.array([])
                # Create array with labels based on known counts
                for class_id, count in class_counts.items():
                    y = np.append(y, [class_id] * count)
                # Compute balanced weights
                raw_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)  
                # Apply scaling factor to make weights less extreme
                scaled_weights = np.power(raw_weights, weight_scaling_factor)
                # Normalize to maintain sum proportionality
                scaled_weights = scaled_weights * (len(classes) / np.sum(scaled_weights))
                # Convert to tensor
                weight_tensor = torch.tensor(scaled_weights, device=device, dtype=torch.float32)
                # Set up the loss function with class weighting
                criterion = FocalLoss(gamma=focal_loss_gamma, weight=weight_tensor)
                
                # Adjust all dropout rates in the model
                for module in fold_model.modules():
                    if isinstance(module, nn.Dropout):
                        # Get current dropout probability
                        current_p = module.p
                        # Apply dropout factor, but cap at 0.7 to avoid excessive dropout
                        new_p = min(current_p * dropout_factor, 0.7)
                        # Set new dropout probability
                        module.p = new_p
                
                # Create optimizer with trial hyperparameters
                optimizer = torch.optim.Adam(
                    fold_model.parameters(),
                    lr=fixed_lr,
                    weight_decay=weight_decay,
                )
                
                # Train for specified epochs in each fold
                fold_history = []
                best_fold_f1 = 0.0
                
                for epoch in range(n_epochs):
                    try:
                        # Train
                        train_loss = train_epoch(
                            fold_model, fold_train_loader, optimizer, 
                            criterion, device
                        )
                        
                        # Evaluate 
                        val_loss, val_labels, val_preds = evaluate(
                            fold_model, fold_val_loader, criterion, device
                        )
                        val_f1_macro = f1_score(val_labels, val_preds, average='macro')
                        
                        # Calculate average validation loss
                        total_val_recordings = len(val_labels)
                        avg_val_loss = val_loss / total_val_recordings
                        
                        # Append to fold history
                        fold_history.append({
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "val_loss": avg_val_loss,
                            "val_f1": val_f1_macro
                        })
                        
                        # Log to wandb
                        if wandb.run:
                            wandb.log({
                                f"{trial_name}/fold_{fold_idx}/epoch": epoch,
                                f"{trial_name}/fold_{fold_idx}/train_loss": train_loss,
                                f"{trial_name}/fold_{fold_idx}/val_loss": avg_val_loss,
                                f"{trial_name}/fold_{fold_idx}/val_f1": val_f1_macro
                            })
                        
                        # Track best F1 for this fold
                        if val_f1_macro > best_fold_f1:
                            best_fold_f1 = val_f1_macro
                        
                        # Report intermediate value for pruning
                        if epoch == n_epochs // 2:  # Report halfway through training
                            trial.report(val_f1_macro, epoch + fold_idx * n_epochs)
                            if trial.should_prune():
                                raise optuna.exceptions.TrialPruned()
                        
                    except Exception as epoch_error:
                        print(f"Error in epoch {epoch} of fold {fold_idx} in trial {trial.number}: {str(epoch_error)}")
                        if wandb.run:
                            wandb.log({f"{trial_name}/fold_{fold_idx}/error": str(epoch_error)})
                        raise
                
                # Store best F1 and min val loss for this fold
                fold_f1_scores.append(best_fold_f1)
                min_fold_loss = min(entry["val_loss"] for entry in fold_history)
                fold_val_losses.append(min_fold_loss)
                fold_histories.append(fold_history)
                
                # Clean up to free memory
                del fold_model, optimizer, criterion
                del fold_train_loader, fold_val_loader
                del fold_dataset, fold_train, fold_val
                gc.collect()
                torch.cuda.empty_cache()
            
            # Calculate average performance across all folds
            avg_f1_macro = np.mean(fold_f1_scores)
            avg_val_loss = np.mean(fold_val_losses)
            
            print(f"Trial {trial.number} completed: Avg F1 across {n_folds} folds: {avg_f1_macro:.4f}")
            print(f"Individual fold F1 scores: {fold_f1_scores}")
            
            # Store all fold histories and metrics
            history_dir = os.path.join(
                myConfig.OUTPUT_PATH, 
                "hyperparameter_optimization"
            )
            os.makedirs(history_dir, exist_ok=True)
            
            trial_results = {
                "fold_f1_scores": fold_f1_scores,
                "avg_f1_macro": avg_f1_macro,
                "fold_val_losses": fold_val_losses,
                "avg_val_loss": avg_val_loss,
                "fold_histories": fold_histories
            }
            
            with open(os.path.join(
                history_dir,
                f"trial_{trial.number}_cv_history.json"
            ), "w") as f:
                json.dump(trial_results, f)
            
            # Log the final results to wandb
            if wandb.run:
                wandb.log({
                    f"{trial_name}/avg_f1_across_folds": avg_f1_macro,
                    f"{trial_name}/avg_val_loss_across_folds": avg_val_loss,
                    **{f"{trial_name}/fold_{i}_f1": score for i, score in enumerate(fold_f1_scores)},
                    **{f"{trial_name}/fold_{i}_val_loss": loss for i, loss in enumerate(fold_val_losses)}
                })
                wandb.run.summary[f"{trial_name}_avg_f1"] = avg_f1_macro
            
            # Store the average validation loss as trial user attribute
            trial.set_user_attr("val_loss", avg_val_loss)
            
            return avg_f1_macro  # Return average F1 across all folds as the objective value
            
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {str(e)}")            
            if wandb.run:
                wandb.log({
                    f"trial_{trial.number}_error": str(e),
                    f"trial_{trial.number}_error_type": type(e).__name__,
                    f"trial_{trial.number}_hyperparams": trial.params
                })
            # Return a very low score to indicate failure
            return -1.0
    
    # Define study storage path
    study_storage_path = os.path.join(
        myConfig.OUTPUT_PATH, 
        "hyperparameter_optimization",
        f"hpo_study_cnn_rnn_{n_folds}fold.pkl"
    )
    
    # Create study for maximizing F1-macro
    print(f"Running Bayesian optimization with {n_trials} trials using {n_folds}-fold cross-validation...")
    study_name = f"cnn_rnn_{n_folds}fold"
    
    # Initialize or resume study
    if resume_study and os.path.exists(study_storage_path):
        print(f"Resuming study from {study_storage_path}")
        study = joblib.load(study_storage_path)
    else:
        # Pruner that balances exploration and exploitation
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,  # Don't prune the first 5 trials
            n_warmup_steps=3,    # Don't prune before 3 epochs
            interval_steps=1
        )

        # Sampler for more efficient search
        sampler = optuna.samplers.TPESampler(seed=42)

        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            pruner=pruner,
            sampler=sampler
        )
    
    wandb_callback = WandbCallback(metric_name="f1_macro")

    study.optimize(objective, n_trials=n_trials, callbacks=[wandb_callback])
    
    # Get and print best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\n=== Bayesian Optimization Results with {n_folds}-fold CV ===")
    print(f"Best average F1-macro across folds: {best_value:.4f}")
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
        "n_folds": n_folds
    }
    
    with open(os.path.join(output_dir, f"best_hyperparams_cnn_rnn_{n_folds}fold.json"), "w") as f:
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
            fig1.write_image(os.path.join(output_dir, f"optimization_history_cnn_rnn_{n_folds}fold.png"))
            
            # Parameter importance plot
            fig2 = plot_param_importances(study)
            fig2.write_image(os.path.join(output_dir, f"param_importances_cnn_rnn_{n_folds}fold.png"))
            
            print(f"Optimization visualizations saved to {output_dir}")
        except Exception as e:
            print(f"Could not create visualizations: {str(e)}")
    
    print(f"Optimization results saved to {output_dir}")
    
    if wandb.run:
        # Log best parameters
        wandb.run.summary["best_f1_macro"] = best_value
        wandb.run.summary["best_params"] = best_params
        wandb.run.summary["n_folds"] = n_folds
    
    # Save study state for possible resumption
    os.makedirs(os.path.dirname(study_storage_path), exist_ok=True)
    joblib.dump(study, study_storage_path)
    print(f"Study state saved to {study_storage_path}")
    
    if wandb.run and wandb.run.name.startswith("hpo_cnn_rnn"):
        wandb.finish()
    
    # Optionally train final model
    if input(f"Train final model with best hyperparameters from {n_folds}-fold CV? (y/n): ").lower() == "y":
        train_with_best_hyperparameters(dataset, best_params)
    
    return result
