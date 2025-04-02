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


def prepare_for_cnn_rnn(example):
    """
    Convert dataset examples for CNN+RNN model format.
    Converts the audio from HuggingFace dataset format to tensor format for the CNN+RNN model.
    """
    # Extract raw audio array
    audio = example["audio"]["array"]
    
    # Convert to torch tensor if not already
    if not isinstance(audio, torch.Tensor):
        audio = torch.tensor(audio, dtype=torch.float32)
    
    # Create list of prosodic features
    prosodic_features = []
    for feature in myData.extracted_features:
        if feature in example:
            prosodic_features.append(example[feature])
    
    # Return updated example with properly formatted audio and prosodic features
    return {
        "audio": audio,
        "prosodic_features": prosodic_features,
        "label": example["label"]
    }


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
    Collate function for the CNN+RNN model data loader.
    Handles proper batching of audio tensors and optional prosodic features.
    """
    device = "cpu"  # Initially collect on CPU, will move to GPU in the model
    
    # Handle audio
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
        "audio": audio,
        "labels": labels
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
        num_workers=0,  
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
    dataset = dataset.map(prepare_for_cnn_rnn)
    
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