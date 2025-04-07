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


def chunk_audio(audio, chunk_size_seconds=5, sample_rate=16000):
    """
    Split audio into fixed-size chunks of specified duration
    
    Args:
        audio: Audio tensor of shape [C, L]
        chunk_size_seconds: Size of each chunk in seconds
        sample_rate: Sample rate of the audio
        
    Returns:
        List of audio chunks, each of shape [C, chunk_size*sample_rate]
    """
    chunk_size = int(chunk_size_seconds * sample_rate)
    chunks = []
    
    # Get audio length (sequence dimension)
    audio_length = audio.shape[1]
    
    # If audio is smaller than chunk size, pad it
    if audio_length < chunk_size:
        padding = torch.zeros((audio.shape[0], chunk_size - audio_length), dtype=audio.dtype)
        padded_audio = torch.cat([audio, padding], dim=1)
        return [padded_audio]
    
    # Split audio into chunks
    num_chunks = audio_length // chunk_size
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunks.append(audio[:, start:end])
    
    # Handle the remaining part if it exists and is significant
    remaining = audio_length % chunk_size
    if remaining > 0.5 * chunk_size:  # If remaining part is more than half the chunk size
        start = num_chunks * chunk_size
        padding = torch.zeros((audio.shape[0], chunk_size - remaining), dtype=audio.dtype)
        last_chunk = torch.cat([audio[:, start:], padding], dim=1)
        chunks.append(last_chunk)
    
    return chunks


class CNNRNNDataset(Dataset):
    """Dataset wrapper specifically for CNN+RNN model requirements with fixed chunk size"""
    def __init__(self, hf_dataset, chunk_size_seconds=5):
        self.dataset = hf_dataset        
        self.chunk_size_seconds = chunk_size_seconds
        
        # Always create chunk-based indices
        self._create_chunk_indices()
    
    def _create_chunk_indices(self):
        """Create mapping from chunk index to (dataset_idx, chunk_idx)"""
        self.chunk_indices = []
        
        for dataset_idx in range(len(self.dataset)):
            item = self.dataset[dataset_idx]
            
            # Ensure audio is a tensor
            if not isinstance(item["audio"], torch.Tensor):
                audio = torch.tensor(item["audio"], dtype=torch.float32)
            else:
                audio = item["audio"]
                
            # Add channel dimension if missing
            if len(audio.shape) == 1:
                audio = audio.unsqueeze(0)  # [C, L]
                
            # Chunk the audio
            chunks = chunk_audio(audio, self.chunk_size_seconds)
            
            # Add mapping for each chunk
            for chunk_idx in range(len(chunks)):
                self.chunk_indices.append((dataset_idx, chunk_idx))
    
    def __len__(self):
        return len(self.chunk_indices)
    
    def __getitem__(self, idx):
        # Get original item index and chunk index
        dataset_idx, chunk_idx = self.chunk_indices[idx]
        item = self.dataset[dataset_idx]
        
        # Ensure audio is a tensor
        if not isinstance(item["audio"], torch.Tensor):
            audio = torch.tensor(item["audio"], dtype=torch.float32)
        else:
            audio = item["audio"]
            
        # Add channel dimension if missing
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)  # [C, L]
            
        # Get the specified chunk
        chunks = chunk_audio(audio, self.chunk_size_seconds)
        audio = chunks[chunk_idx]
        
        result = {
            "audio": audio,
            "label": item["label"],
            "original_idx": dataset_idx
        }
                
        return result


def collate_fn_cnn_rnn(batch):
    """
    Collate function that handles chunked audio.
    All audio chunks should have the same length.
    """
    # For chunked audio, just stack them directly since they should be uniform size
    audio = torch.stack([item["audio"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    
    result = {
        "audio": audio,
        "labels": labels,
        # All chunks have same length
        "audio_lengths": torch.tensor([item["audio"].shape[1] for item in batch])
    }
    
    # Track original indices if available
    if "original_idx" in batch[0]:
        result["original_idx"] = [item["original_idx"] for item in batch]
    
    # Pass augmentation IDs if present
    if "augmentation_id" in batch[0]:
        result["augmentation_id"] = [item.get("augmentation_id") for item in batch]
    
    return result


def get_cnn_rnn_dataloaders(dataset_dict, batch_size=64, chunk_size_seconds=5):
    """
    Creates appropriate dataloaders for CNN+RNN model training with chunked audio.
    
    Args:
        dataset_dict: Dictionary of datasets with train/validation/test splits
        batch_size: Batch size for training and evaluation        
        chunk_size_seconds: Size of each chunk in seconds
        
    Returns:
        Dictionary with train, validation, and test dataloaders
    """
    # Datasets are already processed by prepare_cnn_rnn_dataset() 
    # or are custom datasets like BalancedAugmentedDataset
    processed_dataset = dataset_dict
    
    # Wrap the datasets in a custom dataset class
    train_dataset = CNNRNNDataset(
        processed_dataset["train"],         
        chunk_size_seconds=chunk_size_seconds
    )
    
    val_dataset = CNNRNNDataset(
        processed_dataset["validation"],         
        chunk_size_seconds=chunk_size_seconds
    )
    
    test_dataset = CNNRNNDataset(
        processed_dataset["test"],         
        chunk_size_seconds=chunk_size_seconds
    )
    
    # Create dataloaders with appropriate settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn_cnn_rnn,
        shuffle=True,
        num_workers=4,  
        pin_memory=True  
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn_cnn_rnn,
        shuffle=False,
        num_workers=4,  
        pin_memory=True 
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn_cnn_rnn,
        shuffle=False,
        num_workers=4,  
        pin_memory=True 
    )
    
    return {
        "train": train_loader,
        "validation": val_loader,
        "test": test_loader
    }


def prepare_cnn_rnn_dataset():
    """Prepare data for CNN-RNN model training."""
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


def load_waveform_from_file(file_path, target_sr=16000, max_duration=10.0, chunk_size_seconds=5):
    """
    Load and preprocess audio waveform from file for CNN+RNN model.
    Handles resampling, normalization, and chunks the audio into fixed-size segments.
    """
    # Load audio
    waveform, sr = myAudio.load_audio(file_path)
    
    # Convert to tensor
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform, dtype=torch.float32)
    
    # Add channel dimension if missing
    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)  # [C, L]
    
    # Always chunk the audio
    return chunk_audio(waveform, chunk_size_seconds, target_sr)