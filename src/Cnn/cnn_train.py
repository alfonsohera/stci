import os
import torch
import numpy as np
import wandb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_curve, auc
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
import seaborn as sns

# Import from parent package
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Local imports - individual imports from Config
from src.Common.Config import (
    cnn_hyperparams, wandb_project, wandb_entity, checkpoint_dir, 
    training_args, wandb_watch_model, wandb_log_model, num_samples_per_class,
    OUTPUT_PATH, ROOT_DIR, DATA_DIR, MODEL_DIR, configure_paths, 
    training_from_scratch  
)
from src.Common import ThresholdOptimization
from main import log_memory_usage
from .cnn_data import prepare_cnn_dataset, get_cnn_dataloaders
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
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        
    def forward(self, input, target):
        # Compute cross entropy with class weights if provided
        ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.weight, label_smoothing=self.label_smoothing)
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
            scheduler.step()
            
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
                scheduler.step()
                
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


def evaluate(model, val_loader, criterion, device, use_cam=False, cam_output_dir=None, max_cam_samples=10, epoch=None):
    """
    Evaluate the model on validation data.
    
    Args:
        model: Model to evaluate
        val_loader: DataLoader for evaluation
        criterion: Loss function
        device: Device to run evaluation on
        use_cam: Whether to generate CAM visualizations
        cam_output_dir: Directory to save CAM visualizations
        max_cam_samples: Maximum number of samples to visualize per class and prediction outcome
    """
    from cam_utils import visualize_cam
    import random
    import os
    
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    # Dictionary to track chunks by audio_id
    audio_chunks = {}
    audio_labels = {}
    audio_tensors = {} # Store audio tensors for CAM visualization
    
    # Dynamically determine number of classes for CAM visualization
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'out_features'):
        num_classes = model.classifier.out_features
    else:
        # Default to 3 classes if we can't determine from model
        num_classes = 3
    
    # Counters for CAM visualization - dynamically created based on number of classes
    cam_counters = {
        'correct': {i: 0 for i in range(num_classes)},    # Counts by class
        'incorrect': {i: 0 for i in range(num_classes)}   # Counts by class
    }
    
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
                
                # Process CAM visualization for selected samples if enabled
                if use_cam and cam_output_dir:
                    process_batch_for_cam(model, batch, preds, cam_output_dir, cam_counters, max_cam_samples)
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
                        # Store the label and audio for this audio
                        audio_labels[audio_id] = batch["labels"][j]
                        
                        # Store audio tensor for later CAM visualization
                        if use_cam:
                            # Initialize list for this audio_id
                            audio_tensors[audio_id] = []
                    
                    # Store the logits for this chunk
                    audio_chunks[audio_id].append(logits[j])
                    
                    # Also store the audio chunk itself for visualization
                    if use_cam:
                        audio_tensors[audio_id].append(batch["audio"][j:j+1].detach().clone())
    
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
            
            # Process CAM for chunked audio
            if use_cam and cam_output_dir and audio_id in audio_tensors:
                true_class = label.item()
                pred_class = pred.item()
                
                # Check if prediction is correct
                is_correct = pred_class == true_class
                status = 'correct' if is_correct else 'incorrect'
                
                # If we haven't reached max samples for this class/outcome
                if cam_counters[status][true_class] < max_cam_samples:
                    # Now we're passing all collected audio chunks
                    print(f"Processing CAM for audio_id: {audio_id} with {len(chunk_outputs)} chunks")
                    print(f"Number of audio chunks collected: {len(audio_tensors[audio_id])}")
                    
                    # Get first chunk as reference, but pass all chunks
                    first_chunk = audio_tensors[audio_id][0]
                    
                    # Generate CAM visualization - pass both pred_class and true_class
                    visualize_cam(
                        audio=first_chunk,
                        model=model,
                        target_class=pred_class,  # Pass the predicted class for CAM generation
                        true_class=true_class,    # Pass the true class separately
                        save_path=cam_output_dir,
                        audio_id=audio_id,  # Pass the original audio_id directly without formatting
                        correct=is_correct,
                        audio_paths_dir=os.path.join(cam_output_dir, "audio_paths"),
                        audio_chunks=audio_tensors[audio_id],
                        chunk_outputs=chunk_outputs,
                        show_time_domain=True  # Enable time-domain visualization
                    )
                    
                    # Update counter
                    cam_counters[status][true_class] += 1
    
    return val_loss, all_labels, all_preds

def process_batch_for_cam(model, batch, preds, cam_output_dir, cam_counters, max_cam_samples):
    """Process batch for CAM visualization"""
    import os
    from cam_utils import visualize_cam
    
    # Process some samples for CAM visualization
    for i in range(len(preds)):
        true_class = batch["labels"][i].item()
        pred_class = preds[i].item()
        
        # Check if prediction is correct
        is_correct = pred_class == true_class
        status = 'correct' if is_correct else 'incorrect'
        
        # If we haven't reached the maximum samples for this class and outcome
        if cam_counters[status][true_class] < max_cam_samples:
            # For incorrect predictions, use pred_class to visualize what the model actually saw
            target_for_cam = pred_class if not is_correct else true_class
            # Get audio for this sample
            audio = batch["audio"][i:i+1]  # Keep batch dimension
            
            # Generate CAM
            audio_id = batch["audio_id"][i] if "audio_id" in batch and batch["audio_id"] is not None else f"eval_sample_{i}"
                
            # Get file path if available - improved file path extraction
            file_path = None
            
            # Try all possible locations where file path might be stored
            if "file_path" in batch:
                # Check if it's a list or a single value
                if isinstance(batch["file_path"], list):
                    if i < len(batch["file_path"]):
                        file_path = batch["file_path"][i]
                else:
                    file_path = batch["file_path"]
            
            # Try plural version 'file_paths'
            elif "file_paths" in batch:
                if isinstance(batch["file_paths"], list):
                    if i < len(batch["file_paths"]):
                        file_path = batch["file_paths"][i]
                else:
                    file_path = batch["file_paths"]
                    
            print(f"Found file path for sample {i}: {file_path}")
            
            visualize_cam(
                audio=audio,
                model=model,
                target_class=target_for_cam,  # Use prediction for incorrect samples
                save_path=cam_output_dir,
                audio_id=audio_id,  # Pass the original audio_id directly
                correct=is_correct,
                audio_paths_dir=os.path.join(cam_output_dir, "audio_paths"),
                file_path=file_path,  # Pass the original file path
                show_time_domain=True  # Enable time-domain visualization
            )
            
            # Update counter
            cam_counters[status][true_class] += 1


def train_cnn_model(model, dataloaders, num_epochs=10):
    """Train the CNN model."""        
    from sklearn.utils.class_weight import compute_class_weight
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # From centralized hyperparameters in Config
    hpo_max_lr = cnn_hyperparams["max_lr"]
    hpo_focal_loss_gamma = cnn_hyperparams["focal_loss_gamma"]
    hpo_weight_scaling_factor = cnn_hyperparams["weight_scaling_factor"]
    hpo_weight_decay = cnn_hyperparams["weight_decay"]
    hpo_pct_start = cnn_hyperparams["pct_start"]
    hpo_div_factor = cnn_hyperparams["div_factor"]
    hpo_final_div_factor = cnn_hyperparams["final_div_factor"]
    hpo_weight_decay_cnn = cnn_hyperparams["weight_decay_cnn"]
    hpo_learning_rate_cnn = cnn_hyperparams["learning_rate_cnn"]
    # Initialize wandb
    if not wandb.run:
        # Determine if we're doing binary or multi-class classification
        num_classes = model.classifier.out_features if hasattr(model, 'classifier') else 3
        model_type = "CNN-Binary" if num_classes == 2 else "CNN"
        
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name="cnn",
            config={
                "model_type": model_type,
                "num_classes": num_classes,
                "learning_rate": hpo_max_lr,
                "epochs": num_epochs,
                "batch_size": 96,
                "weight_decay": hpo_weight_decay
            }
        )
        
        # Watch model parameters and gradients
        if wandb_watch_model:
            wandb.watch(model, log="all", log_freq=100)

    # Determine number of classes from model
    num_classes = model.classifier.out_features if hasattr(model, 'classifier') else 3
    
    # Calculate class weights with scaling factor
    if num_classes == 2:
        # Binary classification (Healthy vs Non-Healthy)
        classes = np.array([0, 1])
        # Adapt class counts for binary classification
        class_counts = {
            0: num_samples_per_class.get(0, 0),  # Healthy
            1: num_samples_per_class.get(1, 0) + num_samples_per_class.get(2, 0)  # MCI + AD
        }
    else:
        # Original 3-class classification
        classes = np.array([0, 1, 2])  
        class_counts = num_samples_per_class
    
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
    criterion = FocalLoss(gamma=0, weight=None, label_smoothing=0.12)
        
    model.to(device)

    cnn_params = []
    other_params = []

    # Group parameters for different learning rates
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "cnn_extractor" in name:
                cnn_params.append(param)
            else:
                other_params.append(param)

    # Set up the optimizer with parameter groups
    initial_lr = hpo_max_lr / hpo_div_factor
    cnn_lr = hpo_learning_rate_cnn / hpo_div_factor 

    optimizer = torch.optim.AdamW([
        {'params': cnn_params, 'lr': cnn_lr, 'weight_decay': hpo_weight_decay_cnn},  
        {'params': other_params, 'lr': initial_lr, 'weight_decay': hpo_weight_decay}
    ])

    total_steps = len(dataloaders["train"]) * num_epochs
    
    # Create scheduler with optimized hyperparameters
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[cnn_lr, hpo_max_lr], # 10x lower for CNN14 layers
        total_steps=total_steps,
        pct_start=hpo_pct_start,
        div_factor=hpo_div_factor,
        final_div_factor=hpo_final_div_factor,
        anneal_strategy='cos',
        three_phase=False
    )    
    
    # Create output directory for CNN model
    cnn_output_dir = os.path.join(training_args.output_dir, "cnn")
    if num_classes == 2:
        # For binary classification, use a different output directory
        cnn_output_dir = os.path.join(training_args.output_dir, "cnn_binary")
    
    os.makedirs(cnn_output_dir, exist_ok=True)
    
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
            scheduler
        )
        
        # Validation phase
        val_loss, all_labels, all_preds = evaluate(
            model, 
            dataloaders["validation"], 
            criterion, 
            device,
            use_cam=False,  # Always disable CAM during training
            cam_output_dir=None,  # Don't even pass output directory
            max_cam_samples=0,  # No samples
            epoch=epoch
        )
                       
        # Calculate metrics
        total_val_recordings = len(all_labels)  # This now represents total processed recordings
        avg_val_loss = val_loss / total_val_recordings
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1_macro = f1_score(all_labels, all_preds, average='macro')
        val_f1_per_class = f1_score(all_labels, all_preds, average=None)
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Set class names based on number of classes
        if num_classes == 2:
            class_names = ["healthy", "non-healthy"]
        else:
            class_names = ["healthy", "mci", "ad"]
        
        # Calculate per-class metrics
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "val_f1_macro": val_f1_macro,
            "learning_rate": scheduler.get_last_lr()[1]
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
            
            # Save model to CNN specific directory
            model_path = os.path.join(cnn_output_dir, "cnn_best.pt")
            torch.save(model.state_dict(), model_path)
            print(f"  Saved new best model with F1-macro: {best_f1_macro:.4f} to {model_path}!")
    
    # End of training, log best model if enabled    
    if wandb.run:
        wandb.run.summary["best_f1_macro"] = best_f1_macro
        
        # Log final model if configured
        if wandb_log_model:
            model_path = os.path.join(cnn_output_dir, "cnn_best.pt")
            safetensors_path = os.path.join(cnn_output_dir, "cnn_best.safetensors")
            
            if os.path.exists(model_path):
                artifact = wandb.Artifact(
                    f"cnn-best-{wandb.run.id}", 
                    type="model",
                    description=f"Best CNN model with F1-macro={best_f1_macro:.4f}"
                )
                #artifact.add_file(model_path, name="model.pt")
                
                if os.path.exists(safetensors_path):
                    artifact.add_file(safetensors_path, name="model.safetensors")
                    
                wandb.log_artifact(artifact)


def test_cnn_model(model, test_loader, use_cam=False, cam_output_dir=None, max_cam_samples=20):
    """
    Test the CNN model on the test set with optional CAM visualization.
    
    Args:
        model: The model to test
        test_loader: DataLoader for test data
        use_cam: Whether to generate CAM visualizations
        cam_output_dir: Directory to save CAM visualizations
        max_cam_samples: Maximum number of samples to visualize per class and prediction outcome
    """
    from cam_utils import visualize_cam
    import os
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []  # Store probabilities for ROC curves
    
    # Dictionary to track chunks by audio_id
    audio_chunks = {}
    audio_labels = {}
    audio_tensors = {}  # Store audio tensors for CAM visualization
    
    # Dynamically determine number of classes for CAM visualization
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'out_features'):
        num_classes = model.classifier.out_features
    else:
        # Default to 3 classes if we can't determine from model
        num_classes = 3
    
    # Counters for CAM visualization - dynamically created based on number of classes
    cam_counters = {
        'correct': {i: 0 for i in range(num_classes)},    # Counts by class
        'incorrect': {i: 0 for i in range(num_classes)}   # Counts by class
    }
    
    # Create output directories if needed
    if use_cam and cam_output_dir:
        os.makedirs(cam_output_dir, exist_ok=True)
        os.makedirs(os.path.join(cam_output_dir, 'audio_paths'), exist_ok=True)
    
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
                
                # Get class probabilities for ROC curve
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())
                
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
                
                # Process CAM visualization for selected samples if enabled
                if use_cam and cam_output_dir:
                    process_batch_for_cam(model, batch, preds, cam_output_dir, cam_counters, max_cam_samples)
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
                        
                        # Store audio tensor for later CAM visualization
                        if use_cam:
                            # Initialize list for this audio_id
                            audio_tensors[audio_id] = []
                    
                    # Store the logits for this chunk
                    audio_chunks[audio_id].append(logits[j])
                    
                    # Also store the audio chunk itself for visualization
                    if use_cam:
                        audio_tensors[audio_id].append(batch["audio"][j:j+1].detach().clone())
    
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
            
            # Get probabilities for ROC curve
            probs = torch.softmax(aggregated_output, dim=-1)
            all_probs.append(probs.cpu().numpy().reshape(1, -1))
            
            # Process CAM for chunked audio
            if use_cam and cam_output_dir and audio_id in audio_tensors:
                true_class = label.item()
                pred_class = pred.item()
                
                # Check if prediction is correct
                is_correct = pred_class == true_class
                status = 'correct' if is_correct else 'incorrect'
                
                # If we haven't reached max samples for this class/outcome
                if cam_counters[status][true_class] < max_cam_samples:
                    # Now we're passing all collected audio chunks
                    print(f"Processing CAM for audio_id: {audio_id} with {len(chunk_outputs)} chunks")
                    print(f"Number of audio chunks collected: {len(audio_tensors[audio_id])}")
                    
                    # Get first chunk as reference, but pass all chunks
                    first_chunk = audio_tensors[audio_id][0]
                    
                    # Generate CAM visualization - pass both pred_class and true_class
                    visualize_cam(
                        audio=first_chunk,
                        model=model,
                        target_class=pred_class,  # Pass the predicted class for CAM generation
                        true_class=true_class,    # Pass the true class separately
                        save_path=cam_output_dir,
                        audio_id=audio_id,  # Pass the original audio_id directly without formatting
                        correct=is_correct,
                        audio_paths_dir=os.path.join(cam_output_dir, "audio_paths"),
                        audio_chunks=audio_tensors[audio_id],
                        chunk_outputs=chunk_outputs,
                        show_time_domain=True  # Enable time-domain visualization
                    )
                    
                    # Update counter
                    cam_counters[status][true_class] += 1
    
    # Calculate metrics
    test_accuracy = accuracy_score(all_labels, all_preds)
    if num_classes == 2:
        class_names = ["Healthy", "Not Healthy"]
    else:
        class_names = ["Healthy", "MCI", "AD"]                
    report = classification_report(
            all_labels, all_preds, target_names=class_names)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    # Print CAM generation summary if enabled
    if use_cam:
        print("\nCAM Visualization Summary:")
        for status in ['correct', 'incorrect']:
            print(f"  {status.capitalize()} predictions:")
            for class_id, count in cam_counters[status].items():
                class_name = class_names[class_id]
                print(f"    Class {class_name}: {count} samples")
        print(f"\nVisualizations saved to {cam_output_dir}")
    
    # Generate ROC curves if we have probability data
    if all_probs:
        # Convert all_probs to numpy array
        all_probs = np.vstack(all_probs)
        
        # Create output directory for visualizations
        viz_output_dir = cam_output_dir if cam_output_dir else os.path.join(OUTPUT_PATH, "test_visualizations")
        os.makedirs(viz_output_dir, exist_ok=True)
        
        # Generate confusion matrix visualization
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(all_labels, all_preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                  xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix (Test Set)")
        plt.savefig(os.path.join(viz_output_dir, "test_confusion_matrix.png"))
        plt.close()
        
        # Plot ROC curves for each class
        plt.figure(figsize=(10, 8))
        
        # Colors for different classes
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # For each class, compute ROC curve and AUC
        for i, class_name in enumerate(class_names):
            # Binarize the labels for current class (one-vs-rest)
            bin_labels = (np.array(all_labels) == i).astype(int)
            
            # Get probability scores for current class
            class_probs = all_probs[:, i]
            
            # Compute ROC curve and AUC
            fpr, tpr, _ = roc_curve(bin_labels, class_probs)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                     label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Save the plot
        roc_curve_path = os.path.join(viz_output_dir, "roc_curves.png")
        plt.savefig(roc_curve_path)
        plt.close()
        
        print(f"\nROC curves saved to: {roc_curve_path}")
        
        # If multiclass (more than 2 classes), also create micro-average and macro-average ROC curves
        if len(class_names) > 2:
            plt.figure(figsize=(10, 8))
            
            # Compute micro-average ROC curve and AUC
            # Flatten predictions and labels
            all_labels_bin = []
            all_probs_bin = []
            
            for i in range(len(class_names)):
                bin_labels = (np.array(all_labels) == i).astype(int)
                all_labels_bin.extend(bin_labels)
                all_probs_bin.extend(all_probs[:, i])
            
            # Compute micro-average ROC curve and AUC
            fpr_micro, tpr_micro, _ = roc_curve(all_labels_bin, all_probs_bin)
            roc_auc_micro = auc(fpr_micro, tpr_micro)
            plt.plot(fpr_micro, tpr_micro, 'b-', lw=2,
                    label=f'Micro-average (AUC = {roc_auc_micro:.2f})')
            
            # Compute macro-average ROC curve and AUC
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr for i, (fpr, _, _) in 
                                              enumerate([roc_curve((np.array(all_labels) == i).astype(int), 
                                                                 all_probs[:, i]) for i in range(len(class_names))])]))
            
            # Then interpolate all ROC curves at these points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(len(class_names)):
                bin_labels = (np.array(all_labels) == i).astype(int)
                fpr, tpr, _ = roc_curve(bin_labels, all_probs[:, i])
                mean_tpr += np.interp(all_fpr, fpr, tpr)
            
            # Average and compute AUC
            mean_tpr /= len(class_names)
            roc_auc_macro = auc(all_fpr, mean_tpr)
            plt.plot(all_fpr, mean_tpr, 'r-', lw=2,
                    label=f'Macro-average (AUC = {roc_auc_macro:.2f})')
            
            # Plot diagonal line
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Micro and Macro Average ROC Curves')
            plt.legend(loc="lower right")
            plt.grid(True)
            
            # Save the plot
            avg_roc_curve_path = os.path.join(viz_output_dir, "average_roc_curves.png")
            plt.savefig(avg_roc_curve_path)
            plt.close()
            
            print(f"Average ROC curves saved to: {avg_roc_curve_path}")
    
    return test_accuracy, all_preds, all_labels


def main_cnn(use_prosodic_features=False, binary_classification=False):
    """Main function for the CNN pipeline.
    
    Args:
        use_prosodic_features: Whether to use prosodic features
        binary_classification: Whether to use binary classification (Healthy vs Non-Healthy)
    """
    from .cnn_model import AugmentedDataset, DualPathAudioClassifier, CNN14Classifier, PretrainedDualPathAudioClassifier
    hpo_n_mels = 128
    
    if binary_classification:
        print("Running CNN model with binary classification (Healthy vs. Non-Healthy)")
    else:
        print("Running CNN model with 3-class classification (Healthy vs. MCI vs. AD)")
    
    # Load and prepare dataset using the dedicated cnn_data module
    # Apply binary classification conversion if requested
    dataset = prepare_cnn_dataset(binary_classification=binary_classification)
    
    # Get dataloaders optimized for CNN training
    dataloaders = get_cnn_dataloaders(
        dataset,         
        batch_size=96
    )
    

    # Create model with the appropriate number of classes
    model = PretrainedDualPathAudioClassifier(
        num_classes=2 if binary_classification else 3,  # Binary or 3-class based on parameter
        sample_rate=16000,
        pretrained_cnn14_path=checkpoint_dir+'/Cnn14_mAP=0.431.pth',                            
    )

    
    if not training_from_scratch:       
        model_dir = "cnn_binary" if binary_classification else "cnn"
        model_path = os.path.join(training_args.output_dir, model_dir, "cnn_best.pt")
        
        print(f"Fine-tuning: Loading pre-trained model from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path))
            print("Successfully loaded model for fine-tuning")
            print("Selectively unfreezing CNN14 layers for fine-tuning...")
            for name, param in model.cnn_extractor.named_parameters():
                if "conv_block5" in name or "conv_block6" in name or "fc1" in name:
                    param.requires_grad = True
                    print(f"Unfreezing {name}")
                else:
                    param.requires_grad = False             
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("This could be because the saved model has a different architecture.")
            print("Proceeding with newly initialized model instead.")
    else:
        # Apply selective unfreezing directly
        print("Training a new model from scratch as per configuration.")
        print("Selectively unfreezing CNN14 layers...")
        for name, param in model.cnn_extractor.named_parameters():
            if "conv_block5" in name or "conv_block6" in name or "fc1" in name:
                param.requires_grad = True
                print(f"Unfreezing {name}")
            else:
                param.requires_grad = False


    print(f"Model created with {model.classifier.out_features} output classes!")
    
    # Train model
    print("Training model...")
    train_cnn_model(
        model, 
        dataloaders, 
        num_epochs=10
    )
    print("Training complete!")


def test_cnn(binary_classification=False, use_cam=False, max_cam_samples=20):
    """Test function for the CNN pipeline.
    
    Args:
        binary_classification: Whether to use binary classification (Healthy vs Non-Healthy)
    """
    from .cnn_model import PretrainedDualPathAudioClassifier
    hpo_n_mels = 128
    
    # Prepare data with appropriate classification mode
    if binary_classification:
        print("Testing CNN model with binary classification (Healthy vs. Non-Healthy)")
    else:
        print("Testing CNN model with 3-class classification (Healthy vs. MCI vs. AD)")
        
    dataset = prepare_cnn_dataset(binary_classification=binary_classification)
    
    # Get dataloaders
    dataloaders = get_cnn_dataloaders(
        dataset, 
        batch_size=96
    )
    
    # Create model with appropriate number of classes
    model = PretrainedDualPathAudioClassifier(
        num_classes=2 if binary_classification else 3,
        sample_rate=16000,
        pretrained_cnn14_path=checkpoint_dir+'/Cnn14_mAP=0.431.pth',            
    )
    
    # Load the best model weights from the appropriate directory
    model_dir = "cnn_binary" if binary_classification else "cnn"
    model_path = os.path.join(training_args.output_dir, model_dir, "cnn_best.pt")
    
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path))
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("This could be because the saved model has a different number of classes.")
            print("Training a new model might be required.")
            return
    else:
        print(f"No pre-trained model found at {model_path}. Using randomly initialized weights.")
    
    # Set up CAM output directory with timestamp for uniqueness
    cam_output_dir = None
    if use_cam:        
        class_type = "Binary" if binary_classification else "ThreeClass"
        cam_output_dir = os.path.join(OUTPUT_PATH, f"CAM_Test_{class_type}")
        os.makedirs(cam_output_dir, exist_ok=True)
        print(f"CAM visualizations will be saved to: {cam_output_dir}")
    
    # Run evaluation on test set with optional CAM visualizations
    print("Testing model on test set...")
    test_accuracy, test_preds, test_labels = test_cnn_model(
        model,
        dataloaders["test"],
        use_cam=use_cam,
        cam_output_dir=cam_output_dir,
        max_cam_samples=max_cam_samples
    )
    
    # Display more detailed metrics if available
    if test_labels is not None and test_preds is not None:
        target_names = ["Healthy", "Non-Healthy"] if binary_classification else ["Healthy", "MCI", "AD"]
        test_f1_macro = f1_score(test_labels, test_preds, average='macro')
        test_f1_weighted = f1_score(test_labels, test_preds, average='weighted')
        test_f1_per_class = f1_score(test_labels, test_preds, average=None)
        test_precision_macro = precision_score(test_labels, test_preds, average='macro')
        test_precision_weighted = precision_score(test_labels, test_preds, average='weighted')
        test_precision_per_class = precision_score(test_labels, test_preds, average=None)
        test_recall_macro = recall_score(test_labels, test_preds, average='macro')
        test_recall_weighted = recall_score(test_labels, test_preds, average='weighted')
        test_recall_per_class = recall_score(test_labels, test_preds, average=None)
        test_report = classification_report(test_labels, test_preds, target_names=target_names)
        
        # Calculate confusion matrix
        cm = confusion_matrix(test_labels, test_preds)
        
        # Calculate specificity and NPV for each class
        n_classes = len(target_names)
        specificity_per_class = np.zeros(n_classes)
        npv_per_class = np.zeros(n_classes)
        
        for i in range(n_classes):
            # For each class, calculate TN, FP, FN
            true_negative = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
            false_positive = np.sum(cm[:, i]) - cm[i, i]
            false_negative = np.sum(cm[i, :]) - cm[i, i]
            
            # Calculate specificity (true negative rate): TN / (TN + FP)
            specificity_per_class[i] = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
            
            # Calculate negative predictive value: TN / (TN + FN)
            npv_per_class[i] = true_negative / (true_negative + false_negative) if (true_negative + false_negative) > 0 else 0
        
        # Print detailed metrics
        print("\n" + "="*50)
        print("DETAILED TEST RESULTS")
        print("="*50)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        print("\nMACRO METRICS:")
        print(f"F1-Score (Macro): {test_f1_macro:.4f}")
        print(f"Precision (Macro): {test_precision_macro:.4f}")
        print(f"Recall (Macro): {test_recall_macro:.4f}")
        
        print("\nWEIGHTED METRICS:")
        print(f"F1-Score (Weighted): {test_f1_weighted:.4f}")
        print(f"Precision (Weighted): {test_precision_weighted:.4f}")
        print(f"Recall (Weighted): {test_recall_weighted:.4f}")
        
        print("\nPER-CLASS METRICS:")
        for i, class_name in enumerate(target_names):
            print(f"\n{class_name} Class:")
            print(f"  Precision: {test_precision_per_class[i]:.4f}")
            print(f"  Recall/Sensitivity: {test_recall_per_class[i]:.4f}")
            print(f"  Specificity: {specificity_per_class[i]:.4f}")
            print(f"  NPV: {npv_per_class[i]:.4f}")
            print(f"  F1-Score: {test_f1_per_class[i]:.4f}")
            
            # Calculate and print additional class metrics from confusion matrix
            tp = cm[i, i]  # True positives for this class
            fp = np.sum(cm[:, i]) - tp  # False positives
            fn = np.sum(cm[i, :]) - tp  # False negatives
            tn = np.sum(cm) - tp - fp - fn  # True negatives
            
            accuracy_class = (tp + tn) / np.sum(cm) if np.sum(cm) > 0 else 0
            balanced_accuracy = (test_recall_per_class[i] + specificity_per_class[i]) / 2
            y_true_binary = np.array(test_labels) == i
            y_pred_binary = np.array(test_preds) == i
            mcc = matthews_corrcoef(y_true_binary, y_pred_binary)

            print(f"  Balanced Accuracy: {balanced_accuracy:.4f}")
            print(f"  Matthews Correlation Coefficient: {mcc:.4f}")
            
        # Calculate and print aggregate balanced accuracy
        balanced_accuracy_score_all = balanced_accuracy_score(test_labels, test_preds)
        print(f"\nBalanced Accuracy (All Classes): {balanced_accuracy_score_all:.4f}")
        
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(test_report)
        print("="*50)
    
    return test_accuracy, test_preds, test_labels


def optimize_cnn(binary_classification=False):
    """Function to run threshold optimization for CNN model
    
    Args:
        binary_classification: Whether to use binary classification (Healthy vs Non-Healthy)
    """    
    hpo_n_mels = 128
    # Configure paths
    path_config = configure_paths()    
    from src.Common import Config
    for key, path in path_config.items():
        setattr(Config, key, path)        
    
    if binary_classification:
        print("Running threshold optimization for CNN model with binary classification (Healthy vs. Non-Healthy)")
    else:
        print("Running threshold optimization for CNN model with 3-class classification (Healthy vs. MCI vs. AD)")
        
    # Prepare the dataset with appropriate classification mode
    dataset = prepare_cnn_dataset(binary_classification=binary_classification)
    
    # Create model with appropriate number of classes
    from .cnn_model import PretrainedDualPathAudioClassifier
    

    model = PretrainedDualPathAudioClassifier(
        num_classes=2 if binary_classification else 3,
        sample_rate=16000,
        pretrained_cnn14_path=checkpoint_dir+'/Cnn14_mAP=0.431.pth',            
    )
    
    # Load the best model weights from the appropriate directory
    model_dir = "cnn_binary" if binary_classification else "cnn"
    model_path = os.path.join(training_args.output_dir, model_dir, "cnn_best.pt")
    
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path))
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("This could be because the saved model has a different number of classes.")
            raise FileNotFoundError(f"Failed to load model from {model_path}. Please train the model first.")
    else:
        raise FileNotFoundError(f"No pre-trained model found at {model_path}. Please train the model first.")
    
    # Create validation dataloader
    dataloader = get_cnn_dataloaders(
        dataset, 
        batch_size=96
    )["validation"]
    
    # Set output directory with appropriate name for binary/multi-class
    output_dir_name = "cnn_binary" if binary_classification else "cnn"
    output_dir = os.path.join(OUTPUT_PATH, "threshold_optimization", output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set class names based on binary/multi-class
    if binary_classification:
        class_names = ["Healthy", "Non-Healthy"]
    else:
        class_names = ["Healthy", "MCI", "AD"]
    
    # Run optimization
    print(f"Running threshold optimization for {len(class_names)} classes: {class_names}...")
    ThresholdOptimization.optimize_thresholds_for_model(
        model=model,
        dataloader=dataloader,
        class_names=class_names,
        output_dir=output_dir,
        is_cnn=True, 
        log_to_wandb=True
    )
    
    print(f"Threshold optimization completed. Results saved to {output_dir}")


def test_cnn_with_thresholds(binary_classification=False):
    """Test CNN model using the optimized thresholds
    
    Args:
        binary_classification: Whether to use binary classification (Healthy vs Non-Healthy)
    """
    import os
    import json
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    
    # Configure paths
    path_config = configure_paths()    
    from src.Common import Config
    for key, path in path_config.items():
        setattr(Config, key, path)        
    
    # Prepare dataset with appropriate classification mode
    if binary_classification:
        print("Testing CNN model with binary classification (Healthy vs. Non-Healthy) using optimized thresholds")
    else:
        print("Testing CNN model with 3-class classification (Healthy vs. MCI vs. AD) using optimized thresholds")
        
    dataset = prepare_cnn_dataset(binary_classification=binary_classification)
        

    # Create model with appropriate number of classes
    from .cnn_model import PretrainedDualPathAudioClassifier
    model = PretrainedDualPathAudioClassifier(
        num_classes=2 if binary_classification else 3,
        sample_rate=16000,
        pretrained_cnn14_path=checkpoint_dir+'/Cnn14_mAP=0.431.pth',                        
    )
    
    # Load the best model weights from the appropriate directory
    model_dir = "cnn_binary" if binary_classification else "cnn"
    model_path = os.path.join(training_args.output_dir, model_dir, "cnn_best.pt")
    
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path))
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("This could be because the saved model has a different number of classes.")
            raise FileNotFoundError(f"Failed to load model from {model_path}.")
    else:
        raise FileNotFoundError(f"No pre-trained model found at {model_path}. Please train the model first.")
    
    # Create test dataloader
    dataloader = get_cnn_dataloaders(
        dataset, 
        batch_size=96
    )["test"]
    
    # Try to load threshold values from the optimization results
    threshold_results_path = os.path.join(
        OUTPUT_PATH, 
        "threshold_optimization", 
        model_dir, 
        "threshold_optimization_results.json"
    )
    
    # Set class names based on binary/multi-class
    if binary_classification:
        class_names = ["Healthy", "Non-Healthy"]
    else:
        class_names = ["Healthy", "MCI", "AD"]
        
    print(f"Using {len(class_names)} classes: {class_names}")
    
    if os.path.exists(threshold_results_path):
        print(f"Loading thresholds from {threshold_results_path}")
        with open(threshold_results_path, "r") as f:
            threshold_results = json.load(f)
            
        # test with both Youden and F1 thresholds
        for threshold_type in ["youden", "f1"]:
            print(f"\nTesting with {threshold_type.upper()} thresholds...")
            
            # Extract thresholds from the JSON results
            thresholds = {}
            for class_name in class_names:
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
            output_dir = os.path.join(OUTPUT_PATH, "threshold_comparison", "cnn")
            os.makedirs(output_dir, exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"confusion_matrix_comparison_{threshold_type}.png"))
            plt.close()
            
            print(f"\nConfusion matrix comparison saved to {output_dir}")
    else:
        print(f"Threshold optimization results not found at {threshold_results_path}")
        print("Please run optimize_cnn() first to generate threshold values.")