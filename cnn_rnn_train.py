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

                # If we haven't reached the maximum samples for this class and outcome
                if cam_counters[status][true_class] < max_cam_samples:
                    # For incorrect predictions, use pred_class to visualize what the model actually saw
                    target_for_cam = pred_class if not is_correct else true_class

                    # Now we're passing all collected audio chunks for this ID
                    print(f"Processing CAM for audio_id: {audio_id} with {len(chunk_outputs)} chunks")
                    print(f"Number of audio chunks collected: {len(audio_tensors[audio_id])}")
                    
                    # Get first chunk as reference, but pass all chunks
                    first_chunk = audio_tensors[audio_id][0]
                    
                    visualize_cam(
                        audio=first_chunk,  # Pass the first chunk for reference
                        model=model,
                        target_class=target_for_cam,
                        save_path=cam_output_dir,
                        audio_id=f"{audio_id}_pred{pred_class}_true{true_class}",
                        correct=is_correct,
                        audio_paths_dir=os.path.join(cam_output_dir, "audio_paths"),
                        epoch=epoch,
                        audio_chunks=audio_tensors[audio_id],  # Pass all collected chunks
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
            audio_id = f"eval_sample_{i}"
            if "audio_id" in batch and batch["audio_id"] is not None:
                audio_id = batch["audio_id"][i]
                
            # Get file path if available
            file_path = None
            if "file_path" in batch:
                file_path = batch["file_path"][i] if isinstance(batch["file_path"], list) else batch["file_path"]
            elif hasattr(batch, "file_paths"):
                file_path = batch["file_paths"][i] if i < len(batch["file_paths"]) else None
            
            visualize_cam(
                audio=audio,
                model=model,
                target_class=target_for_cam,  # Use prediction for incorrect samples
                save_path=cam_output_dir,
                audio_id=f"{audio_id}_pred{pred_class}_true{true_class}",  # Clear filename
                correct=is_correct,
                audio_paths_dir=os.path.join(cam_output_dir, "audio_paths"),
                file_path=file_path,  # Pass the original file path
                show_time_domain=True  # Enable time-domain visualization
            )
            
            # Update counter
            cam_counters[status][true_class] += 1


def train_cnn_rnn_model(model, dataloaders, num_epochs=10):
    """Train the CNN+RNN model."""        
    from sklearn.utils.class_weight import compute_class_weight
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # From HPO:        
    hpo_max_lr = 0.0011885281702589529
    hpo_focal_loss_gamma = 1.3005369189225944
    hpo_weight_scaling_factor = 0.4364698824624799
    hpo_weight_decay = 4.7150495089938455e-05
    hpo_pct_start = 0.18662795601481053
    hpo_div_factor = 23.23664102515406
    hpo_final_div_factor = 271.49589180568336

    # Initialize wandb
    if not wandb.run:
        # Determine if we're doing binary or multi-class classification
        num_classes = model.classifier.out_features if hasattr(model, 'classifier') else 3
        model_type = "CNN+RNN-Binary" if num_classes == 2 else "CNN+RNN"
        
        wandb.init(
            project=myConfig.wandb_project,
            entity=myConfig.wandb_entity,
            name="cnn_rnn",
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
        if myConfig.wandb_watch_model:
            wandb.watch(model, log="all", log_freq=100)

    # Determine number of classes from model
    num_classes = model.classifier.out_features if hasattr(model, 'classifier') else 3
    
    # Calculate class weights with scaling factor
    if num_classes == 2:
        # Binary classification (Healthy vs Non-Healthy)
        classes = np.array([0, 1])
        # Adapt class counts for binary classification
        class_counts = {
            0: myConfig.num_samples_per_class.get(0, 0),  # Healthy
            1: myConfig.num_samples_per_class.get(1, 0) + myConfig.num_samples_per_class.get(2, 0)  # MCI + AD
        }
    else:
        # Original 3-class classification
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
        
    model.to(device)

    # Set up the optimizer with proper hyperparameters    
    initial_lr = hpo_max_lr / hpo_div_factor
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=initial_lr,
        weight_decay=hpo_weight_decay,  # L2 regularization
    )

    total_steps = len(dataloaders["train"]) * num_epochs
    
    # Create scheduler with optimized hyperparameters
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hpo_max_lr,
        total_steps=total_steps,
        pct_start=hpo_pct_start,
        div_factor=hpo_div_factor,
        final_div_factor=hpo_final_div_factor,
        anneal_strategy='cos',
        three_phase=False
    )    
    
    # Create output directory for CNN+RNN model
    cnn_rnn_output_dir = os.path.join(myConfig.training_args.output_dir, "cnn_rnn")
    if num_classes == 2:
        # For binary classification, use a different output directory
        cnn_rnn_output_dir = os.path.join(myConfig.training_args.output_dir, "cnn_rnn_binary")
    
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
            scheduler
        )
        
        # Validation phase
        val_loss, all_labels, all_preds = evaluate(
            model, 
            dataloaders["validation"], 
            criterion, 
            device,
            use_cam=False,                                      # Enable CAM visualization
            cam_output_dir=myConfig.OUTPUT_PATH+'/CAM_Validation'  ,   # Output directory
            max_cam_samples=10,                                 # Max samples per class/outcome
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
            
            """ # Also save in safetensors format if available
            try:
                from safetensors.torch import save_file
                safetensors_path = os.path.join(cnn_rnn_output_dir, "cnn_rnn_best.safetensors")
                save_file(model.state_dict(), safetensors_path)
                print(f"  Also saved model in safetensors format to {safetensors_path}")
            except ImportError:
                print("  safetensors not available, skipping safetensors format") """
    
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


def test_cnn_rnn_model(model, test_loader, use_cam=False, cam_output_dir=None, max_cam_samples=10):
    """
    Test the CNN+RNN model on the test set with optional CAM visualization.
    
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
        os.makedirs(os.path.join(cam_output_dir, 'LogMelSpecs', 'correct'), exist_ok=True)
        os.makedirs(os.path.join(cam_output_dir, 'LogMelSpecs', 'incorrect'), exist_ok=True)
        os.makedirs(os.path.join(cam_output_dir, 'CAMs', 'correct'), exist_ok=True)
        os.makedirs(os.path.join(cam_output_dir, 'CAMs', 'incorrect'), exist_ok=True)
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
            
            # Process CAM for chunked audio
            if use_cam and cam_output_dir and audio_id in audio_tensors:
                true_class = label.item()
                pred_class = pred.item()
                
                # Check if prediction is correct
                is_correct = pred_class == true_class
                status = 'correct' if is_correct else 'incorrect'
                target_for_cam = pred_class if not is_correct else true_class
                
                # If we haven't reached max samples for this class/outcome
                if cam_counters[status][true_class] < max_cam_samples:
                    # Now we're passing all collected audio chunks
                    print(f"Processing CAM for audio_id: {audio_id} with {len(chunk_outputs)} chunks")
                    print(f"Number of audio chunks collected: {len(audio_tensors[audio_id])}")
                    
                    # Get first chunk as reference, but pass all chunks
                    first_chunk = audio_tensors[audio_id][0]
                    
                    # Generate CAM visualization
                    visualize_cam(
                        audio=first_chunk,
                        model=model,
                        target_class=target_for_cam,
                        save_path=cam_output_dir,
                        audio_id=f"{audio_id}_pred{pred_class}_true{true_class}",
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
    report = classification_report(
        all_labels, all_preds, target_names=["Healthy", "MCI", "AD"]
    )
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    # Print CAM generation summary if enabled
    if use_cam:
        print("\nCAM Visualization Summary:")
        for status in ['correct', 'incorrect']:
            print(f"  {status.capitalize()} predictions:")
            for class_id, count in cam_counters[status].items():
                class_name = ["Healthy", "MCI", "AD"][class_id]
                print(f"    Class {class_name}: {count} samples")
        print(f"\nVisualizations saved to {cam_output_dir}")
    
    return test_accuracy, all_preds, all_labels


def main_cnn_rnn(use_prosodic_features=False, binary_classification=False):
    """Main function for the CNN+RNN pipeline.
    
    Args:
        use_prosodic_features: Whether to use prosodic features
        binary_classification: Whether to use binary classification (Healthy vs Non-Healthy)
    """
    from cnn_rnn_model import AugmentedDataset, DualPathAudioClassifier, CNN14Classifier, PretrainedDualPathAudioClassifier
    hpo_n_mels = 128
    
    if binary_classification:
        print("Running CNN+RNN model with binary classification (Healthy vs. Non-Healthy)")
    else:
        print("Running CNN+RNN model with 3-class classification (Healthy vs. MCI vs. AD)")
    
    # Load and prepare dataset using the dedicated cnn_rnn_data module
    # Apply binary classification conversion if requested
    dataset = prepare_cnn_rnn_dataset(binary_classification=binary_classification)
    
    # Get dataloaders optimized for CNN+RNN training
    dataloaders = get_cnn_rnn_dataloaders(
        dataset,         
        batch_size=96
    )
    
    # Best hyperparameters from previous optimization
    hpo_attention_dropout = 0.25320974179977257
    hpo_fusion_dropout = 0.2783466229854074
    hpo_prosodic_weight = 1.4943883706790098

    # Create model with the appropriate number of classes
    model = PretrainedDualPathAudioClassifier(
        num_classes=2 if binary_classification else 3,  # Binary or 3-class based on parameter
        sample_rate=16000,
        pretrained_cnn14_path=myConfig.checkpoint_dir+'/Cnn14_mAP=0.431.pth',    
        attention_dropout=hpo_attention_dropout,
        fusion_dropout=hpo_fusion_dropout,
        prosodic_weight=hpo_prosodic_weight                
    )

    print(f"Model created with {model.classifier.out_features} output classes!")
    
    # Train model
    print("Training model...")
    train_cnn_rnn_model(
        model, 
        dataloaders, 
        num_epochs=10
    )
    print("Training complete!")


def test_cnn_rnn(binary_classification=False):
    """Test function for the CNN+RNN pipeline.
    
    Args:
        binary_classification: Whether to use binary classification (Healthy vs Non-Healthy)
    """
    from cnn_rnn_model import PretrainedDualPathAudioClassifier
    hpo_n_mels = 128
    
    # Prepare data with appropriate classification mode
    if binary_classification:
        print("Testing CNN+RNN model with binary classification (Healthy vs. Non-Healthy)")
    else:
        print("Testing CNN+RNN model with 3-class classification (Healthy vs. MCI vs. AD)")
        
    dataset = prepare_cnn_rnn_dataset(binary_classification=binary_classification)
    
    # Get dataloaders
    dataloaders = get_cnn_rnn_dataloaders(
        dataset, 
        batch_size=96
    )
    
    # Best hyperparameters from previous optimization
    hpo_attention_dropout = 0.21790974595973722
    hpo_fusion_dropout = 0.3668445892921854
    hpo_prosodic_weight = 0.8587093661398519
    
    # Create model with appropriate number of classes
    model = PretrainedDualPathAudioClassifier(
        num_classes=2 if binary_classification else 3,
        sample_rate=16000,
        pretrained_cnn14_path=myConfig.checkpoint_dir+'/Cnn14_mAP=0.431.pth',    
        attention_dropout=hpo_attention_dropout,
        fusion_dropout=hpo_fusion_dropout,
        prosodic_weight=hpo_prosodic_weight
    )
    
    # Load the best model weights from the appropriate directory
    model_dir = "cnn_rnn_binary" if binary_classification else "cnn_rnn"
    model_path = os.path.join(myConfig.training_args.output_dir, model_dir, "cnn_rnn_best.pt")
    
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
    
    # Run evaluation with appropriate CAM output directory
    cam_dir_suffix = "Binary" if binary_classification else "ThreeClass" 
    test_cnn_rnn_model(
        model,
        dataloaders["test"],
        use_cam=True,
        cam_output_dir=f"/ProcessedFiles/CAM_Test_{cam_dir_suffix}",
        max_cam_samples=10
    )


def optimize_cnn_rnn(binary_classification=False):
    """Function to run threshold optimization for CNN+RNN model
    
    Args:
        binary_classification: Whether to use binary classification (Healthy vs Non-Healthy)
    """    
    hpo_n_mels = 128
    # Configure paths
    path_config = myConfig.configure_paths()
    for key, path in path_config.items():
        setattr(myConfig, key, path)
    
    if binary_classification:
        print("Running threshold optimization for CNN+RNN model with binary classification (Healthy vs. Non-Healthy)")
    else:
        print("Running threshold optimization for CNN+RNN model with 3-class classification (Healthy vs. MCI vs. AD)")
        
    # Prepare the dataset with appropriate classification mode
    dataset = prepare_cnn_rnn_dataset(binary_classification=binary_classification)
    
    # Create model with appropriate number of classes
    from cnn_rnn_model import PretrainedDualPathAudioClassifier
    
    # Best hyperparameters from previous optimization
    hpo_attention_dropout = 0.21790974595973722
    hpo_fusion_dropout = 0.3668445892921854
    hpo_prosodic_weight = 0.8587093661398519

    model = PretrainedDualPathAudioClassifier(
        num_classes=2 if binary_classification else 3,
        sample_rate=16000,
        pretrained_cnn14_path=myConfig.checkpoint_dir+'/Cnn14_mAP=0.431.pth',    
        attention_dropout=hpo_attention_dropout,
        fusion_dropout=hpo_fusion_dropout,
        prosodic_weight=hpo_prosodic_weight                
    )
    
    # Load the best model weights from the appropriate directory
    model_dir = "cnn_rnn_binary" if binary_classification else "cnn_rnn"
    model_path = os.path.join(myConfig.training_args.output_dir, model_dir, "cnn_rnn_best.pt")
    
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
    dataloader = get_cnn_rnn_dataloaders(
        dataset, 
        batch_size=96
    )["validation"]
    
    # Set output directory with appropriate name for binary/multi-class
    output_dir_name = "cnn_rnn_binary" if binary_classification else "cnn_rnn"
    output_dir = os.path.join(myConfig.OUTPUT_PATH, "threshold_optimization", output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set class names based on binary/multi-class
    if binary_classification:
        class_names = ["Healthy", "Non-Healthy"]
    else:
        class_names = ["Healthy", "MCI", "AD"]
    
    # Run optimization
    print(f"Running threshold optimization for {len(class_names)} classes: {class_names}...")
    optimize_thresholds_for_model(
        model=model,
        dataloader=dataloader,
        class_names=class_names,
        output_dir=output_dir,
        is_cnn_rnn=True, 
        log_to_wandb=not myConfig.running_offline
    )
    
    print(f"Threshold optimization completed. Results saved to {output_dir}")


def test_cnn_rnn_with_thresholds(binary_classification=False):
    """Test CNN+RNN model using the optimized thresholds
    
    Args:
        binary_classification: Whether to use binary classification (Healthy vs Non-Healthy)
    """
    import os
    import json
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    
    # Configure paths
    path_config = myConfig.configure_paths()
    for key, path in path_config.items():
        setattr(myConfig, key, path)
    
    # Prepare dataset with appropriate classification mode
    if binary_classification:
        print("Testing CNN+RNN model with binary classification (Healthy vs. Non-Healthy) using optimized thresholds")
    else:
        print("Testing CNN+RNN model with 3-class classification (Healthy vs. MCI vs. AD) using optimized thresholds")
        
    dataset = prepare_cnn_rnn_dataset(binary_classification=binary_classification)
    
    # Best hyperparameters from previous optimization
    hpo_attention_dropout = 0.21790974595973722
    hpo_fusion_dropout = 0.3668445892921854
    hpo_prosodic_weight = 0.8587093661398519

    # Create model with appropriate number of classes
    from cnn_rnn_model import PretrainedDualPathAudioClassifier
    model = PretrainedDualPathAudioClassifier(
        num_classes=2 if binary_classification else 3,
        sample_rate=16000,
        pretrained_cnn14_path=myConfig.checkpoint_dir+'/Cnn14_mAP=0.431.pth',    
        attention_dropout=hpo_attention_dropout,
        fusion_dropout=hpo_fusion_dropout,
        prosodic_weight=hpo_prosodic_weight                
    )
    
    # Load the best model weights from the appropriate directory
    model_dir = "cnn_rnn_binary" if binary_classification else "cnn_rnn"
    model_path = os.path.join(myConfig.training_args.output_dir, model_dir, "cnn_rnn_best.pt")
    
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
    dataloader = get_cnn_rnn_dataloaders(
        dataset, 
        batch_size=96
    )["test"]
    
    # Try to load threshold values from the optimization results
    threshold_results_path = os.path.join(
        myConfig.OUTPUT_PATH, 
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
            output_dir = os.path.join(myConfig.OUTPUT_PATH, "threshold_comparison", "cnn_rnn")
            os.makedirs(output_dir, exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"confusion_matrix_comparison_{threshold_type}.png"))
            plt.close()
            
            print(f"\nConfusion matrix comparison saved to {output_dir}")
    else:
        print(f"Threshold optimization results not found at {threshold_results_path}")
        print("Please run optimize_cnn_rnn() first to generate threshold values.")


def run_bayesian_optimization(n_trials=100, resume_study=False, n_folds=5, binary_classification=False):
    """Run Bayesian hyperparameter optimization for the CNN+RNN model with k-fold cross-validation.
    
    Args:
        n_trials: Number of trials for optimization
        resume_study: Whether to resume a previous study
        n_folds: Number of folds for cross-validation
        binary_classification: Whether to use binary classification (Healthy vs Non-Healthy)
    """
    from cnn_rnn_model import DualPathAudioClassifier, AugmentedDataset, CNN14Classifier, PretrainedDualPathAudioClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.model_selection import StratifiedKFold
    import json
    import joblib
    
    # Initialize wandb if not already running
    model_type = "CNN14-Binary" if binary_classification else "CNN14"
    if not wandb.run:
        wandb.init(
            project="CNN14-HPO",
            entity=myConfig.wandb_entity,
            name=f"hpo_{model_type}_{n_folds}fold",
            config={
                "model_type": f"{model_type} Classifier HPO with CV",
                "n_trials": n_trials,
                "optimization_type": "bayesian",
                "n_folds": n_folds,
                "binary_classification": binary_classification
            },
            tags=["hpo", "bayesian-optimization", model_type.lower(), "cross-validation"]
        )
    
    # Create output directory for hyperparameter optimization    
    output_dir = os.path.join(myConfig.OUTPUT_PATH, "hyperparameter_optimization")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare dataset with binary classification if requested
    print(f"Loading dataset for hyperparameter optimization with cross-validation ({model_type})...")
    dataset = prepare_cnn_rnn_dataset(binary_classification=binary_classification)
    
    # Combine train and validation for cross-validation
    combined_data = []
    combined_labels = []
    for split in ["train", "validation"]:
        for i in range(len(dataset[split])):
            combined_data.append(dataset[split][i])
            combined_labels.append(dataset[split][i]["label"])
    
    # Prepare indices for stratified k-fold splits to handle class imbalance
    indices = np.arange(len(combined_data))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def objective(trial):
        try:            
            torch.cuda.empty_cache()
            gc.collect()
                        
            # Core hyperparameters from previous optimization
            weight_scaling_factor = trial.suggest_float("weight_scaling_factor", 0.3, 0.7)
            focal_loss_gamma = trial.suggest_float("focal_loss_gamma", 0.5, 1.5)  
            weight_decay = trial.suggest_float("weight_decay", 5e-6, 5e-5, log=True)                      
            hpo_max_learning_rate = trial.suggest_float("learning_rate", 5e-4, 3e-3, log=True)
            hpo_pct_start = trial.suggest_float("pct_start", 0.15, 0.35)  
            hpo_div_factor = trial.suggest_float("div_factor", 20.0, 50.0)  
            hpo_final_div_factor = trial.suggest_float("final_div_factor", 200.0, 600.0)

            #hyperparameters specific to the Dual Path model
            attention_dropout = trial.suggest_float("attention_dropout", 0.1, 0.3)
            fusion_dropout = trial.suggest_float("fusion_dropout", 0.15, 0.4)                        
            prosodic_weight = trial.suggest_float("prosodic_weight", 0.7, 2.5)
            
            
            
            trial_name = f"trial_{trial.number}"            
            # Log trial parameters to wandb
            if wandb.run:
                wandb.log({
                    f"{trial_name}_params": {
                        param: value for param, value in trial.params.items()
                    }
                })
                        
            fold_f1_scores = []
            fold_val_losses = []
                        
            # Number of epochs for HPO cross-validation
            n_epochs = 8            
            print(f"\n--- Trial {trial.number}: Running {n_folds}-fold cross-validation ---")            
            # Store fold histories
            fold_histories = []
            
            # For each fold - using stratified k-fold for better class balance
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, combined_labels)):
                print(f"  Processing fold {fold_idx+1}/{n_folds} for trial {trial.number}")
                # Create fold-specific datasets
                fold_train = [combined_data[i] for i in train_idx]
                fold_val = [combined_data[i] for i in val_idx]

                # Create balanced training dataset for this fold
                fold_train_balanced = AugmentedDataset(
                    original_dataset=fold_train,            
                    num_classes=2 if binary_classification else 3
                )

                # Create fold dataset dictionary
                fold_dataset = {
                    "train": fold_train_balanced,
                    "validation": fold_val,
                    "test": dataset["test"]
                }

                # Get dataloaders for this fold 
                fold_dataloaders = get_cnn_rnn_dataloaders(
                    fold_dataset,
                    batch_size=96
                )
                fold_train_loader = fold_dataloaders["train"]
                fold_val_loader = fold_dataloaders["validation"]                
                
                # Create model with appropriate number of classes
                fold_model = PretrainedDualPathAudioClassifier(
                    num_classes=2 if binary_classification else 3,
                    sample_rate=16000,
                    pretrained_cnn14_path=myConfig.checkpoint_dir+'/Cnn14_mAP=0.431.pth',
                    attention_dropout=attention_dropout,
                    fusion_dropout=fusion_dropout,
                    prosodic_weight=prosodic_weight
                )
                fold_model.to(device)
                
                # Calculate class weights with scaling factor
                if binary_classification:
                    classes = np.array([0, 1])  
                    class_counts = {
                        0: myConfig.num_samples_per_class.get(0, 0),  # Healthy
                        1: myConfig.num_samples_per_class.get(1, 0) + myConfig.num_samples_per_class.get(2, 0)  # MCI + AD
                    }
                else:
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
                
                # Create optimizer with trial hyperparameters
                
                optimizer = torch.optim.Adam(
                    fold_model.parameters(),
                    lr=hpo_max_learning_rate/hpo_div_factor,
                    weight_decay=weight_decay,
                )
                total_steps = len(fold_dataloaders["train"]) * n_epochs
    
                # Create scheduler with optimized hyperparameters
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=hpo_max_learning_rate,
                    total_steps=total_steps,
                    pct_start=hpo_pct_start,
                    div_factor=hpo_div_factor,
                    final_div_factor=hpo_final_div_factor,
                    anneal_strategy='cos',
                    three_phase=False
                )
                # Train for specified epochs in each fold
                fold_history = []
                best_fold_f1 = 0.0
                patience = 3
                no_improvement = 0

                for epoch in range(n_epochs):
                    try:
                        # Train
                        train_loss = train_epoch(
                            fold_model, fold_train_loader, optimizer, 
                            criterion, device,scheduler
                        )

                        # Evaluate 
                        val_loss, val_labels, val_preds = evaluate(
                            fold_model, 
                            fold_val_loader, 
                            criterion, 
                            device,
                            use_cam=False
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
                        # Track best F1 for this fold with early stopping
                        if val_f1_macro > best_fold_f1:
                            best_fold_f1 = val_f1_macro
                            no_improvement = 0
                        else:
                            no_improvement += 1
                            if no_improvement >= patience:
                                print(f"    Early stopping at epoch {epoch} for fold {fold_idx+1}")
                                break
                        # Report intermediate value for pruning more aggressively
                        if epoch >= 2:  # Report earlier for faster pruning of poor trials
                            trial.report(val_f1_macro, epoch + fold_idx * n_epochs)
                            if trial.should_prune():
                                print(f"    Trial {trial.number} pruned at epoch {epoch} for fold {fold_idx+1}")
                                raise optuna.exceptions.TrialPruned()
                    except optuna.exceptions.TrialPruned:
                        raise
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
                del fold_train, fold_val
                gc.collect()
                torch.cuda.empty_cache()
            
            # Calculate average performance across all folds
            avg_f1_macro = np.mean(fold_f1_scores)
            avg_val_loss = np.mean(fold_val_losses)
            
            # 10% above max observed loss
            MAX_VAL_LOSS = max(np.max(fold_val_losses) * 1.1, 1.0)
            # Normalize validation loss to [0,1] scale (lower is better)
            norm_val_loss = max(0, 1 - (avg_val_loss / MAX_VAL_LOSS))
            
            # Combined objective: prioritize both high F1 and low validation loss
            #combined_score = avg_f1_macro * (0.8 + 0.2 * norm_val_loss)
            combined_score = avg_f1_macro
            
            print(f"Trial {trial.number} completed: Avg F1 across {n_folds} folds: {avg_f1_macro:.4f}, Combined Score: {combined_score:.4f}")
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
                "norm_val_loss": norm_val_loss,
                "combined_score": combined_score,
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
                    f"{trial_name}/norm_val_loss": norm_val_loss,
                    f"{trial_name}/combined_score": combined_score,
                    **{f"{trial_name}/fold_{i}_f1": score for i, score in enumerate(fold_f1_scores)},
                    **{f"{trial_name}/fold_{i}_val_loss": loss for i, loss in enumerate(fold_val_losses)}
                })
                wandb.run.summary[f"{trial_name}_avg_f1"] = avg_f1_macro
            
            # Store the average validation loss as trial user attribute
            trial.set_user_attr("val_loss", avg_val_loss)
            
            return combined_score  # Return combined score that balances performance and consistency
            
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
    study_type = "binary" if binary_classification else "multiclass"
    study_storage_path = os.path.join(
        myConfig.OUTPUT_PATH, 
        "hyperparameter_optimization",
        f"hpo_study_cnn_rnn_{study_type}_{n_folds}fold.pkl"
    )
    
    # Create study for maximizing F1-macro
    print(f"Running Bayesian optimization with {n_trials} trials using {n_folds}-fold cross-validation...")
    study_name = f"cnn_rnn_{study_type}_{n_folds}fold"
    
    # Initialize or resume study
    if resume_study and os.path.exists(study_storage_path):
        print(f"Resuming study from {study_storage_path}")
        study = joblib.load(study_storage_path)
    else:
        # More aggressive pruner to terminate underperforming trials earlier
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=3,  
            n_warmup_steps=2,    
            interval_steps=1
        )
        
        sampler = optuna.samplers.TPESampler(
            seed=42,
            multivariate=True,  
            n_startup_trials=5, 
            constant_liar=True   
        )

        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            pruner=pruner,
            sampler=sampler
        )
        
        # Best previous hyperparameters as a starting point
        previous_best = {
            # Core parameters from previous best run
            "learning_rate": 0.0010831279442946378,  
            "focal_loss_gamma": 1.2214471854780586, 
            "weight_scaling_factor": 0.42877749204715, 
            "weight_decay": 6.754417251186016e-05,            
            "pct_start": 0.24140038998213117,
            "div_factor": 26.312388937073905,
            "final_div_factor": 294.73021118648194,
            "attention_dropout": 0.21790974595973722,  
            "fusion_dropout": 0.3668445892921854,     
            "prosodic_weight": 0.8587093661398519
        }

        study.enqueue_trial(previous_best)
    
    wandb_callback = WandbCallback(metric_name="combined_score")

    study.optimize(objective, n_trials=n_trials, callbacks=[wandb_callback])
    
    # Get and print best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\n=== Bayesian Optimization Results with {n_folds}-fold CV ===")
    print(f"Best combined score across folds: {best_value:.4f}")
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
        "n_folds": n_folds,
        "binary_classification": binary_classification
    }
    
    model_type_str = "binary" if binary_classification else "multiclass"
    with open(os.path.join(output_dir, f"best_hyperparams_cnn_rnn_{model_type_str}_{n_folds}fold.json"), "w") as f:
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
            from optuna.visualization import plot_optimization_history, plot_param_importances
            
            # History plot
            fig1 = plot_optimization_history(study)
            fig1.write_image(os.path.join(output_dir, f"optimization_history_cnn_rnn_{model_type_str}_{n_folds}fold.png"))
            
            # Parameter importance plot
            fig2 = plot_param_importances(study)
            fig2.write_image(os.path.join(output_dir, f"param_importances_cnn_rnn_{model_type_str}_{n_folds}fold.png"))
            
            print(f"Optimization visualizations saved to {output_dir}")
        except Exception as e:
            print(f"Could not create visualizations: {str(e)}")
    
    print(f"Optimization results saved to {output_dir}")
    
    if wandb.run:
        # Log best parameters
        wandb.run.summary["best_combined_score"] = best_value
        wandb.run.summary["best_params"] = best_params
        wandb.run.summary["n_folds"] = n_folds
        wandb.run.summary["binary_classification"] = binary_classification
    
    # Save study state for possible resumption
    os.makedirs(os.path.dirname(study_storage_path), exist_ok=True)
    joblib.dump(study, study_storage_path)
    print(f"Study state saved to {study_storage_path}")
    
    if wandb.run and wandb.run.name.startswith("hpo_cnn_rnn"):
        wandb.finish()
    
    # Optionally train final model with more epochs for better results
    if input(f"Train final model with best hyperparameters from {n_folds}-fold CV? (y/n): ").lower() == "y":
        print("Training final model with 20 epochs instead of the default...")
        train_with_best_hyperparameters(dataset, best_params, binary_classification=binary_classification)
    
    return result


def train_with_best_hyperparameters(dataset, best_params, use_prosodic_features=True, binary_classification=False):
    """Train a final model using the best hyperparameters from Bayesian optimization.
    
    Args:
        dataset: Dataset dictionary with train, validation, test splits
        best_params: Dictionary of best hyperparameters from optimization
        use_prosodic_features: Whether to use prosodic features
        binary_classification: Whether to use binary classification (Healthy vs Non-Healthy)
    """
    from cnn_rnn_model import DualPathAudioClassifier, BalancedAugmentedDataset, PretrainedDualPathAudioClassifier
    
    classification_type = "Binary" if binary_classification else "3-class"
    print(f"\n=== Training with Best Hyperparameters ({classification_type}) ===")
    
    # Create balanced training dataset with appropriate number of classes
    print("Creating balanced training dataset...")
    balanced_train_dataset = BalancedAugmentedDataset(
        original_dataset=dataset["train"],
        total_target_samples=1000,
        num_classes=2 if binary_classification else 3
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
    
    # Parameters specific to PretrainedDualPathAudioClassifier
    attention_dropout = best_params.get("attention_dropout", 0.25)
    fusion_dropout = best_params.get("fusion_dropout", 0.35)
    prosodic_weight = best_params.get("prosodic_weight", 1.0)
    
    # Create dataloaders
    dataloaders = get_cnn_rnn_dataloaders(
        balanced_dataset, 
        batch_size=32,
        use_prosodic_features=use_prosodic_features
    )
    
    # Create model with optimized hyperparameters (using PretrainedDualPathAudioClassifier)
    model = PretrainedDualPathAudioClassifier(
        num_classes=2 if binary_classification else 3,
        sample_rate=16000,
        pretrained_cnn14_path=myConfig.checkpoint_dir+'/Cnn14_mAP=0.431.pth',
        attention_dropout=attention_dropout,
        fusion_dropout=fusion_dropout,
        prosodic_weight=prosodic_weight
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Calculate class weights with scaling factor
    weight_scaling_factor = best_params.get("weight_scaling_factor", 0.33)
    focal_loss_gamma = best_params.get("focal_loss_gamma", 1.2)
    
    from sklearn.utils.class_weight import compute_class_weight
    
    if binary_classification:
        classes = np.array([0, 1])
        # Adapt class counts for binary classification
        class_counts = {
            0: myConfig.num_samples_per_class.get(0, 0),  # Healthy
            1: myConfig.num_samples_per_class.get(1, 0) + myConfig.num_samples_per_class.get(2, 0)  # MCI + AD
        }
    else:
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
    
    # Create focal loss with optimized gamma and class weights
    criterion = FocalLoss(gamma=focal_loss_gamma, weight=weight_tensor)
    
    # Create optimizer with optimized hyperparameters
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Calculate total steps for full training (10 epochs)
    total_steps = len(dataloaders["train"]) * 10
    
    # Create scheduler with optimized hyperparameters
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr if max_lr else lr * 10,
        total_steps=total_steps,
        pct_start=pct_start,
        div_factor=25,
        final_div_factor=1000,
        anneal_strategy='cos',
        three_phase=False
    )
    
    # Create output directory for optimized CNN+RNN model - use different paths for binary/multiclass
    model_type_str = "binary" if binary_classification else "multiclass"
    optimized_output_dir = os.path.join(myConfig.OUTPUT_PATH, f"cnn_rnn_optimized_{model_type_str}")
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
            scheduler
        )
        
        # Validation phase
        val_loss, all_labels, all_preds = evaluate(
            model, 
            dataloaders["validation"], 
            criterion, 
            device,
            use_cam=False  # Disable CAM for hyperparameter optimization
        )
        
        # Calculate metrics
        total_val_recordings = len(all_labels)
        avg_val_loss = val_loss / total_val_recordings
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
            model_path = os.path.join(optimized_output_dir, f"cnn_rnn_{model_type_str}_optimized.pt")
            torch.save(model.state_dict(), model_path)
            print(f"  Saved new best model with F1-macro: {best_f1_macro:.4f} to {model_path}!")
    
    # Test the best model
    # Load the best model
    model.load_state_dict(torch.load(os.path.join(optimized_output_dir, f"cnn_rnn_{model_type_str}_optimized.pt")))
    
    # Test on test set
    print(f"\nTesting optimized {model_type_str} model on test set...")
    test_loss, test_labels, test_preds = evaluate(
        model, 
        dataloaders["test"], 
        criterion, 
        device,
        use_cam=True,
        cam_output_dir=os.path.join(myConfig.OUTPUT_PATH, f'CAM_Test_{model_type_str}_optimized'),
        max_cam_samples=10
    )
    
    # Determine class names based on classification mode
    if binary_classification:
        target_names = ["Healthy", "Non-Healthy"]
    else:
        target_names = ["Healthy", "MCI", "AD"]
    
    # Calculate test metrics
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_f1_macro = f1_score(test_labels, test_preds, average='macro')
    test_report = classification_report(test_labels, test_preds, target_names=target_names)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1-Macro: {test_f1_macro:.4f}")
    print("Classification Report:")
    print(test_report)
    
    # Save hyperparameters and test results
    results = {
        "hyperparameters": best_params,
        "test_accuracy": test_accuracy,
        "test_f1_macro": test_f1_macro,
        "val_f1_macro": best_f1_macro,
        "binary_classification": binary_classification
    }
    
    with open(os.path.join(optimized_output_dir, "results.json"), "w") as f:
        import json
        json.dump(results, f, indent=2)
    
    print(f"Optimized {model_type_str} model and results saved to {optimized_output_dir}")
    
    return model