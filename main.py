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
import myModel
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


def collate_fn_cnn_rnn(batch):
   
    audio_tensors = []
    for item in batch:
        if isinstance(item["audio"], list):
            audio_tensors.append(torch.tensor(item["audio"], dtype=torch.float32))
        else:            
            audio_tensors.append(item["audio"])
    
    # Stack on CPU
    audio = torch.stack(audio_tensors)
    labels = torch.tensor([item["label"] for item in batch])
    
    result = {
        "audio": audio,  # Keep on CPU
        "labels": labels  # Keep on CPU
    }
        
    if "prosodic_features" in batch[0]:
        features_list = []
        for item in batch:
            pf = item["prosodic_features"]
            if not isinstance(pf, torch.Tensor):
                pf = torch.tensor(pf, dtype=torch.float32)
            features_list.append(pf)
        
        result["prosodic_features"] = torch.stack(features_list)
    
    # Pass augmentation IDs if present
    if "augmentation_id" in batch[0]:
        result["augmentation_id"] = [item.get("augmentation_id") for item in batch]
    
    return result


def train_cnn_rnn_model(model, dataset, num_epochs=10, use_prosodic_features=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Optimize DataLoader for CPU memory 
    train_loader = DataLoader(
        dataset["train"], 
        batch_size=64,  # 
        collate_fn=collate_fn_cnn_rnn,
        num_workers=0,  # 
        pin_memory=True,  
        persistent_workers=False  
    )
    
    val_loader = DataLoader(
        dataset["validation"], 
        batch_size=64,  
        collate_fn=collate_fn_cnn_rnn,
        num_workers=0  
    )
    
    # Initialize wandb
    import wandb
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
                "batch_size": 16,
                "weight_decay": 5e-4,
                "prosodic_features_dim": len(myData.extracted_features) if use_prosodic_features else 0
            }
        )
        
        # Watch model parameters and gradients
        if myConfig.wandb_watch_model:
            wandb.watch(model, log="all", log_freq=100)

    # Calculate weights
    y_train = list(dataset["train"]["label"])
    class_weights = compute_class_weight(
        class_weight="balanced", 
        classes=numpy.unique(y_train), 
        y=y_train
    )
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = FocalLoss(gamma=0, weight=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    
    # Calculate total steps for 1cycle scheduler
    total_steps = len(train_loader) * num_epochs
    
    # 1cycle LR scheduler 
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4, 
        total_steps=total_steps,
        pct_start=0.3,  
        div_factor=25,  
        final_div_factor=1000  
    )
    
    # Create output directory for CNN+RNN model
    cnn_rnn_output_dir = os.path.join(myConfig.training_args.output_dir, "cnn_rnn")
    os.makedirs(cnn_rnn_output_dir, exist_ok=True)
    
    # Move model to device
    model.to(device)
    
    # Tracking variables
    best_f1_macro = 0.0  
    
    # 3. Modify the training loop with explicit garbage collection
    import gc

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(f"Epoch {epoch+1} start")
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
            # Move batch to GPU if available
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Zero gradients more efficiently
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass
            if use_prosodic_features and "prosodic_features" in batch:
                logits = model(batch["audio"], batch["prosodic_features"], batch.get("augmentation_id"))
            else:
                logits = model(batch["audio"], augmentation_id=batch.get("augmentation_id"))
                
            # Calculate loss                
            loss = criterion(logits, batch["labels"].to(device))
            
            # Backpropagation
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Update LR
            scheduler.step()
            
            # Track loss (use item() to detach from graph)
            train_loss += loss.item()
            
            # Explicit cleanup of tensors - with detach to ensure no graph references 
            logits_detached = logits.detach()
            loss_value = loss.item()
            del logits, loss, batch
            del logits_detached
            
            # Periodic garbage collection 
            if i % 10 == 0:  # Less frequent to reduce overhead
                gc.collect()
                
                # Monitor memory every 50 batches
                if i % 50 == 0:
                    log_memory_usage(f"Epoch {epoch+1}, batch {i}")
                
        # Explicit cleanup after training phase
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(f"Epoch {epoch+1} end")
        
        # Calculate average loss
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                if use_prosodic_features:
                    # Forward pass
                    logits = model(batch["audio"].to(device), batch["prosodic_features"].to(device))
                else:
                    # Forward pass
                    logits = model(batch["audio"].to(device))
                # Calculate loss
                loss = criterion(logits, batch["labels"].to(device))
                # Track loss
                val_loss += loss.item()
                # Get predictions
                preds = torch.argmax(logits, dim=-1)
                # Track predictions and labels
                all_preds.extend(preds.cpu().numpy())
                # Convert labels to numpy and extend
                all_labels.extend(batch["labels"].cpu().numpy())
        
        # Calculate metrics
        avg_val_loss = val_loss / len(val_loader)
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
            tp[i] = sum((numpy.array(all_preds) == i) & (numpy.array(all_labels) == i))
            fp[i] = sum((numpy.array(all_preds) == i) & (numpy.array(all_labels) != i))
            tn[i] = sum((numpy.array(all_preds) != i) & (numpy.array(all_labels) != i))
            fn[i] = sum((numpy.array(all_preds) != i) & (numpy.array(all_labels) == i))
            
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
        
        
        # Simplify the validation visualization - only log at end of training
        if epoch == num_epochs - 1:  # Only in final epoch
            """ import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
            
            plt.figure(figsize=(6, 4), dpi=72)  # Even lower resolution
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Final Confusion Matrix')
            plt.colorbar()
            classes = ["Healthy", "MCI", "AD"]
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            
            # Log to wandb (if applicable)
            if wandb.run:
                wandb.log({
                    **log_dict,
                    "confusion_matrix": wandb.Image(plt)
                })
            
            # Close figure immediately
            plt.close('all') """
        else:
            # Just log metrics for all other epochs
            if wandb.run:
                wandb.log(log_dict)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")
        print(f"  Val F1-Macro: {val_f1_macro:.4f}")
        print(f"  Val F1 per class: {val_f1_per_class}")
        #print("  Confusion Matrix:")
        print(cm)
        
        # Save best model based on F1-macro instead of validation loss
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
            cnn_rnn_output_dir = os.path.join(myConfig.training_args.output_dir, "cnn_rnn")
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


def test_cnn_rnn_model(model, dataset, use_prosodic_features=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loader = DataLoader(dataset["test"], batch_size=8, collate_fn=collate_fn_cnn_rnn)
    
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if use_prosodic_features:
                logits = model(batch["audio"].to(device), batch["prosodic_features"].to(device))
            else:
                logits = model(batch["audio"].to(device))
                
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


def main_cnn_rnn(use_prosodic_features=False):    
    print(f"Running CNN+RNN model {'with' if use_prosodic_features else 'without'} manual features")
    
    # Load data
    myData.DownloadAndExtract()
    
    # Check if dataframe.csv exists in the Data directory
    data_file_path = os.path.join(myConfig.DATA_DIR, "dataframe.csv")   
    if os.path.exists(data_file_path):
        data_df = pd.read_csv(data_file_path)
        print(f"Loaded existing dataframe from {data_file_path}")
    else:
        # Create and process dataframe
        data_df = myFunctions.createDataframe()        
        data_df = myFunctions.featureEngineering(data_df)
        os.makedirs(os.path.dirname(data_file_path), exist_ok=True)        
        data_df.to_csv(data_file_path, index=False)
    

    if not os.path.exists(myConfig.OUTPUT_PATH) or (os.path.exists(myConfig.OUTPUT_PATH) and len(os.listdir(myConfig.OUTPUT_PATH)) == 0):
        # Data splits
        train_df, val_df, test_df = myData.datasetSplit(data_df)
        # Apply standard scaling to the splits
        train_df, val_df, test_df = myData.ScaleDatasets(train_df, val_df, test_df)
        # Create HF's dataset
        print("Creating HF's dataset...")
        myData.createHFDatasets(train_df, val_df, test_df)    
    
    # Load HF's dataset
    print("Loading HF's dataset...")
    dataset = myData.loadHFDataset()
    print("Preparing data for CNN+RNN model...")
    dataset = dataset.map(myData.prepare_for_cnn_rnn)
    
    # Create balanced training dataset
    print("Creating balanced training dataset with augmentations...")
    from myCnnRnnModel import BalancedAugmentedDataset
    balanced_train_dataset = BalancedAugmentedDataset(
        original_dataset=dataset["train"],
        total_target_samples=1000,
        num_classes=3
    )
    
    # Replace the training set with the balanced version
    balanced_dataset = {
        "train": balanced_train_dataset,
        "validation": dataset["validation"],
        "test": dataset["test"]
    }
    
    print("Data preparation complete!")
    
    # Create model
    from myCnnRnnModel import DualPathAudioClassifier
    model = DualPathAudioClassifier(
        num_classes=3,
        sample_rate=16000,
        use_prosodic_features=use_prosodic_features,
        prosodic_features_dim=len(myData.extracted_features)
    )
    print("Model created!")
    
    # Train model
    print("Training model...")
    train_cnn_rnn_model(model, balanced_dataset, num_epochs=10, use_prosodic_features=use_prosodic_features)
    print("Training complete!")


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


def test_cnn_rnn(use_prosodic_features=True):
    # Import the CNN+RNN model
    from myCnnRnnModel import DualPathAudioClassifier

    # Load data
    myData.DownloadAndExtract()
    
    data_file_path = os.path.join(myConfig.DATA_DIR, "dataframe.csv")
    if os.path.exists(data_file_path):
        data_df = pd.read_csv(data_file_path)
        print(f"Loaded existing dataframe from {data_file_path}")
    else:
        data_df = myFunctions.createDataframe()
        data_df = myFunctions.featureEngineering(data_df)
        os.makedirs(os.path.dirname(data_file_path), exist_ok=True)
        data_df.to_csv(data_file_path, index=False)
        print(f"Created and saved dataframe to {data_file_path}")
    
    # Split and prepare data for CNN+RNN model
    train_df, val_df, test_df = myData.datasetSplit(data_df)
    train_df, val_df, test_df = myData.ScaleDatasets(train_df, val_df, test_df)
    dataset = myData.createHFDatasets(train_df, val_df, test_df)
    dataset = dataset.map(myData.prepare_for_cnn_rnn)
    
    # Create the CNN+RNN model
    model = DualPathAudioClassifier(
        num_classes=3,
        sample_rate=16000,
        use_prosodic_features=use_prosodic_features,
        prosodic_features_dim=len(myData.extracted_features)
    )
    
    # Evaluate the model using the existing test function for cnn_rnn
    test_cnn_rnn_model(model, dataset, use_prosodic_features=use_prosodic_features)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cognitive Impairment Detection Model")
    parser.add_argument("mode", choices=["train", "finetune", "test"],
                        help="Mode of operation: train (from scratch), finetune (existing model) or test (evaluate model)")
    parser.add_argument("--pipeline", choices=["wav2vec2", "cnn_rnn"], default="wav2vec2",
                        help="Specify the pipeline to use")
    parser.add_argument("--online", action="store_true", help="Run with online services (WandB logging)")
    parser.add_argument("--no_manual", action="store_true",
                        help="Disable manual features for cnn_rnn pipeline")
    
    args = parser.parse_args()
    
    myConfig.running_offline = not args.online

    if args.pipeline == "wav2vec2":
        if args.mode in ["train", "finetune"]:
            if args.mode == "train":
                myConfig.training_from_scratch = True
                print("Starting training from scratch (Wav2Vec2 pipeline)...")
            else:
                myConfig.training_from_scratch = False
                print("Starting fine-tuning (Wav2Vec2 pipeline)...")
            main_fn()
        elif args.mode == "test":
            myConfig.training_from_scratch = False
            print("Running model evaluation (Wav2Vec2 pipeline)...")
            test()
    elif args.pipeline == "cnn_rnn":
        use_manual = not args.no_manual
        if args.mode in ["train", "finetune"]:
            if args.mode == "train":
                myConfig.training_from_scratch = True
                print("Starting training from scratch (CNN+RNN pipeline)...")
            else:
                myConfig.training_from_scratch = False
                print("Starting fine-tuning (CNN+RNN pipeline)...")
            # Import the CNN+RNN model locally
            from myCnnRnnModel import DualPathAudioClassifier
            main_cnn_rnn(use_prosodic_features=use_manual)
        elif args.mode == "test":
            myConfig.training_from_scratch = False
            print("Running model evaluation (CNN+RNN pipeline)...")
            test_cnn_rnn(use_prosodic_features=use_manual)
