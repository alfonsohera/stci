import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import gc

# Local imports
import myConfig
import myData
import myFunctions
import myAudio


class CNNRNNDataset(Dataset):
    """Dataset wrapper specifically for CNN+RNN model requirements"""
    def __init__(self, hf_dataset, use_prosodic_features=True):
        self.dataset = hf_dataset
        self.use_prosodic_features = use_prosodic_features
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Ensure audio is a tensor with proper dimensions
        if not isinstance(item["audio"], torch.Tensor):
            item["audio"] = torch.tensor(item["audio"], dtype=torch.float32)
            
        # Add channel dimension if missing (for CNN input)
        if len(item["audio"].shape) == 1:
            item["audio"] = item["audio"].unsqueeze(0)  # [C, L]
            
        result = {
            "audio": item["audio"],
            "label": item["label"]
        }
        
        # Include prosodic features only if needed
        if self.use_prosodic_features:
            if isinstance(item["prosodic_features"], list):
                result["prosodic_features"] = torch.tensor(item["prosodic_features"], dtype=torch.float32)
            else:
                result["prosodic_features"] = item["prosodic_features"]
                
        return result


def collate_fn_cnn_rnn(batch):
    """
    Collate function that handles variable-length audio more effectively.
    """
    # Get original lengths and convert to tensors
    audio_tensors = []
    audio_lengths = []
    
    for item in batch:
        audio = item["audio"]
        if isinstance(audio, list):
            audio = torch.tensor(audio, dtype=torch.float32)
        
        # Ensure 2D format [C, T]
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
            
        audio_tensors.append(audio)
        audio_lengths.append(audio.shape[1])
    
    # Find max length in this batch only
    max_len = max(audio_lengths)
    
    # Cap maximum length to avoid OOM (e.g., 60 seconds at 16kHz)
    max_allowed_len = 16000 * 60  
    max_len = min(max_len, max_allowed_len)
    
    # Pad to max length in this batch (or cap)
    padded_tensors = []
    
    # Use enumeration to get index directly
    for i, (audio, orig_len) in enumerate(zip(audio_tensors, audio_lengths)):
        if audio.shape[1] > max_len:
            # Trim audio to max allowed length
            padded_tensors.append(audio[:, :max_len])
            # Update length using the index directly
            audio_lengths[i] = max_len
        elif audio.shape[1] < max_len:
            padding = torch.zeros((audio.shape[0], max_len - audio.shape[1]), dtype=audio.dtype)
            padded = torch.cat([audio, padding], dim=1)
            padded_tensors.append(padded)
        else:
            padded_tensors.append(audio)
    
    # Stack tensors
    audio = torch.stack(padded_tensors)
    labels = torch.tensor([item["label"] for item in batch])
    
    result = {
        "audio": audio,
        "labels": labels,
        "audio_lengths": torch.tensor(audio_lengths)
    }
    
    # Add prosodic features if present
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


def get_cnn_rnn_dataloaders(dataset_dict, batch_size=64, use_prosodic_features=True):
    """
    Creates appropriate dataloaders for CNN+RNN model training.
    
    Args:
        dataset_dict: Dictionary of datasets with train/validation/test splits
        batch_size: Batch size for training and evaluation
        use_prosodic_features: Whether to include prosodic features
        
    Returns:
        Dictionary with train, validation, and test dataloaders
    """
    # Datasets are already processed by prepare_cnn_rnn_dataset() 
    # or are custom datasets like BalancedAugmentedDataset
    processed_dataset = dataset_dict
    
    # Wrap the datasets in a custom dataset class
    train_dataset = CNNRNNDataset(processed_dataset["train"], use_prosodic_features)
    val_dataset = CNNRNNDataset(processed_dataset["validation"], use_prosodic_features)
    test_dataset = CNNRNNDataset(processed_dataset["test"], use_prosodic_features)
    
    # Create dataloaders with appropriate settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn_cnn_rnn,
        shuffle=True,
        num_workers=2,  
        pin_memory=True  
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn_cnn_rnn,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn_cnn_rnn,
        shuffle=False,
        num_workers=0
    )
    
    return {
        "train": train_loader,
        "validation": val_loader,
        "test": test_loader
    }


def prepare_cnn_rnn_dataset():
    """Prepare the dataset specifically for CNN+RNN model"""
    myData.DownloadAndExtract()
    
    # Check if dataframe.csv exists
    data_file_path = os.path.join(myConfig.DATA_DIR, "dataframe.csv")
    if os.path.exists(data_file_path):
        data_df = pd.read_csv(data_file_path)
        print(f"Loaded existing dataframe from {data_file_path}")
        
        # Check if paths are absolute and convert if needed
        if '/' in data_df['file_path'].iloc[0] and not data_df['file_path'].iloc[0].startswith(('Healthy', 'MCI', 'AD')):
            print("Converting absolute paths to relative paths...")
            data_df = myFunctions.convert_absolute_to_relative_paths(data_df)
            # Save the updated dataframe
            data_df.to_csv(data_file_path, index=False)
    else:
        # Create new dataframe with features
        data_df = myFunctions.createDataframe()
        data_df = myFunctions.featureEngineering(data_df)
        os.makedirs(os.path.dirname(data_file_path), exist_ok=True)
        data_df.to_csv(data_file_path, index=False)
        print(f"Created and saved dataframe to {data_file_path}")
    
    # Prepare dataset splits
    if not os.path.exists(myConfig.OUTPUT_PATH) or (os.path.exists(myConfig.OUTPUT_PATH) and len(os.listdir(myConfig.OUTPUT_PATH)) == 0):
        print("Creating dataset splits...")
        train_df, val_df, test_df = myData.datasetSplit(data_df)
        train_df, val_df, test_df = myData.ScaleDatasets(train_df, val_df, test_df)
        dataset = myData.createHFDatasets(train_df, val_df, test_df)
    else:
        print("Loading existing dataset...")
        dataset = myData.loadHFDataset()
    
    print("Preparing dataset for CNN+RNN model...")
    dataset = dataset.map(myData.prepare_for_cnn_rnn)
    
    return dataset


def load_waveform_from_file(file_path, target_sr=16000, max_duration=10.0):
    """
    Load and preprocess audio waveform from file for CNN+RNN model.
    Handles resampling, normalization, and padding/trimming.
    """
    # Load audio
    waveform, sr = myAudio.load_audio(file_path)
    
    # Convert to tensor
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform, dtype=torch.float32)
    
    # Add channel dimension if missing
    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)  # [C, L]
    
    return waveform