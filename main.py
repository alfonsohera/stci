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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torch import nn


# Suppress specific warnings
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized from the model checkpoint.*")
logging.set_verbosity_error()  # Set transformers logging to show only errors


def collate_fn_cnn_rnn(batch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    audio = torch.stack([item["audio"] for item in batch]).to(device)
    manual_features = torch.stack([item["manual_features"] for item in batch]).to(device)
    labels = torch.tensor([item["label"] for item in batch]).to(device)
    
    return {
        "audio": audio,
        "manual_features": manual_features,
        "labels": labels
    }


def train_cnn_rnn_model(model, dataset, num_epochs=10, use_manual_features=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = DataLoader(dataset["train"], batch_size=8, collate_fn=collate_fn_cnn_rnn)
    val_loader = DataLoader(dataset["validation"], batch_size=8, collate_fn=collate_fn_cnn_rnn)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Calculate total steps for 1cycle scheduler
    total_steps = len(train_loader) * num_epochs
    
    # 1cycle LR scheduler instead of ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,  # Peak learning rate
        total_steps=total_steps,
        pct_start=0.3,  # Percentage of steps used for warmup
        div_factor=25,  # Initial LR = max_lr / div_factor
        final_div_factor=1000  # Final LR = initial_lr / final_div_factor
    )
    
    model.to(device)
    
    # Tracking variables
    best_f1_macro = 0.0  # Track best F1 macro instead of validation loss
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            optimizer.zero_grad()
            
            if use_manual_features:
                logits = model(batch["audio"], batch["manual_features"])
            else:
                logits = model(batch["audio"])
                
            loss = criterion(logits, batch["labels"])
            
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update scheduler after each batch with 1cycle
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                if use_manual_features:
                    logits = model(batch["audio"], batch["manual_features"])
                else:
                    logits = model(batch["audio"])
                
                loss = criterion(logits, batch["labels"])
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
        
        # Calculate metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1_macro = f1_score(all_labels, all_preds, average='macro')
        val_f1_per_class = f1_score(all_labels, all_preds, average=None)
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")
        print(f"  Val F1-Macro: {val_f1_macro:.4f}")
        print(f"  Val F1 per class: {val_f1_per_class}")
        print("  Confusion Matrix:")
        print(cm)
        
        # Save best model based on F1-macro instead of validation loss
        if val_f1_macro > best_f1_macro:
            best_f1_macro = val_f1_macro
            torch.save(model.state_dict(), os.path.join(myConfig.training_args.output_dir, "cnn_rnn_best.pt"))
            print(f"  Saved new best model with F1-macro: {best_f1_macro:.4f}!")


def test_cnn_rnn_model(model, dataset, use_manual_features=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loader = DataLoader(dataset["test"], batch_size=8, collate_fn=collate_fn_cnn_rnn)
    
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if use_manual_features:
                logits = model(batch["audio"], batch["manual_features"])
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


def main_cnn_rnn(use_manual_features=True):    
    print(f"Running CNN+RNN model {'with' if use_manual_features else 'without'} manual features")
    
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
    
    # Split and prepare data
    train_df, val_df, test_df = myData.datasetSplit(data_df)
    train_df, val_df, test_df = myData.ScaleDatasets(train_df, val_df, test_df)
    
    # Process data for CNN+RNN model
    dataset = myData.createHFDatasets(train_df, val_df, test_df)
    dataset = dataset.map(myData.prepare_for_cnn_rnn)
    
    # Create model
    model = DualPathAudioClassifier(
        num_classes=3,
        sample_rate=16000,
        use_manual_features=use_manual_features,
        manual_features_dim=len(myData.extracted_features)
    )
    
    # Train model
    train_cnn_rnn_model(model, dataset, num_epochs=10, use_manual_features=use_manual_features)
    
    # Test model
    test_cnn_rnn_model(model, dataset, use_manual_features=use_manual_features)


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


def test_cnn_rnn(use_manual_features=True):
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
        use_manual_features=use_manual_features,
        manual_features_dim=len(myData.extracted_features)
    )
    
    # Evaluate the model using the existing test function for cnn_rnn
    test_cnn_rnn_model(model, dataset, use_manual_features=use_manual_features)


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
            main_cnn_rnn(use_manual_features=use_manual)
        elif args.mode == "test":
            myConfig.training_from_scratch = False
            print("Running model evaluation (CNN+RNN pipeline)...")
            test_cnn_rnn(use_manual_features=use_manual)
