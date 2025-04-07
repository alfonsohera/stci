import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import librosa

# Local imports
import myConfig
import myData
import myFunctions
import myAudio


def chunk_audio(audio, chunk_size_seconds=10, sample_rate=16000):
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
    """Dataset wrapper specifically for CNN+RNN model with fixed chunk size
    that works with standard PyTorch datasets."""
    def __init__(self, dataset, chunk_size_seconds=10, sample_rate=16000):
        self.dataset = dataset        
        self.chunk_size_seconds = chunk_size_seconds
        self.sample_rate = sample_rate
        
        # Precompute chunk indices (how many chunks per file)
        self.chunk_indices = []
        print("Computing chunk indices...")
        for idx in tqdm(range(len(dataset)), desc="Calculating chunks"):
            item = dataset[idx]
            audio = item["audio"]
            
            # Get number of chunks for this audio
            chunks = chunk_audio(audio, self.chunk_size_seconds, self.sample_rate)
            
            # Store mapping from chunk index to dataset index
            for chunk_idx in range(len(chunks)):
                self.chunk_indices.append((idx, chunk_idx))
                
        print(f"Created dataset with {len(self.chunk_indices)} chunks from {len(dataset)} files")
    
    def __len__(self):
        return len(self.chunk_indices)
    
    def __getitem__(self, idx):
        # Get original item index and chunk index
        dataset_idx, chunk_idx = self.chunk_indices[idx]
        
        # Get the original data
        item = self.dataset[dataset_idx]
        audio = item["audio"]
        
        # Get the specified chunk
        chunks = chunk_audio(audio, self.chunk_size_seconds, self.sample_rate)
        audio_chunk = chunks[chunk_idx] if chunk_idx < len(chunks) else chunks[0]
        
        result = {
            "audio": audio_chunk,
            "label": item["label"],
            "original_idx": dataset_idx
        }
        
        # Add audio_id for chunk handling
        audio_id = f"{dataset_idx}_{chunk_idx}"
        result["audio_id"] = audio_id
                
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


def get_cnn_rnn_dataloaders(dataset_dict, batch_size=64, chunk_size_seconds=10):
    """
    Creates dataloaders using the PyTorch dataset approach instead of HuggingFace.
    
    Args:
        dataset_dict: Dictionary of PyTorch datasets with train/validation/test splits
        batch_size: Batch size for training and evaluation        
        chunk_size_seconds: Size of each chunk in seconds
        
    Returns:
        Dictionary with train, validation, and test dataloaders
    """
    # Wrap the datasets in the chunking dataset class
    train_dataset = CNNRNNDataset(
        dataset_dict["train"],         
        chunk_size_seconds=chunk_size_seconds
    )
    
    val_dataset = CNNRNNDataset(
        dataset_dict["validation"],         
        chunk_size_seconds=chunk_size_seconds
    )
    
    test_dataset = CNNRNNDataset(
        dataset_dict["test"],         
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


class PreloadedAudioDataset(Dataset):
    """PyTorch dataset that preloads all audio data in memory with a duration limit."""
    def __init__(self, dataframe, max_duration=100.0, sample_rate=16000):
        self.dataframe = dataframe
        self.samples = []
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        
        print(f"Preloading {len(dataframe)} audio files (max {max_duration}s each)...")
        for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Loading audio"):
            file_path = row["file_path"]
            label = row["label"]
            
            try:
                # Load audio directly with librosa, limiting duration
                audio_path = myFunctions.resolve_audio_path(file_path)
                
                # Use direct librosa loading with duration limit (matching wav2vec2 pipeline)
                audio, sr = self._load_audio_with_duration(audio_path)
                
                # Convert to tensor
                if not isinstance(audio, torch.Tensor):
                    audio = torch.tensor(audio, dtype=torch.float32)
                
                # Add channel dimension if missing
                if len(audio.shape) == 1:
                    audio = audio.unsqueeze(0)  # [C, L]
                
                # Store the processed sample
                self.samples.append({
                    "audio": audio,
                    "label": label,
                    "file_path": file_path
                })
            except Exception as e:
                print(f"Error loading audio {file_path}: {e}")
                # Create a dummy sample for failed audio
                dummy_audio = torch.zeros((1, 16000), dtype=torch.float32)  # 1 second of silence
                self.samples.append({
                    "audio": dummy_audio,
                    "label": label,
                    "file_path": file_path,
                    "error": str(e)
                })
    
    def _load_audio_with_duration(self, file_path):
        """Load audio with a duration limit to match previous implementation."""
        # Use librosa with duration parameter to limit length
        audio = librosa.load(
            file_path, 
            sr=self.sample_rate,
            duration=self.max_duration  # Limit duration to max_duration seconds
        )[0]
        
        # Convert to the expected format
        return np.array(audio, dtype=np.float32), self.sample_rate
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class CachedAudioDataset(Dataset):
    """PyTorch dataset that uses cached audio samples."""
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def save_pytorch_dataset(dataset_dict, save_path):
    """Save PyTorch datasets to disk."""
    os.makedirs(save_path, exist_ok=True)
    
    for split_name, dataset in dataset_dict.items():
        split_path = os.path.join(save_path, f"{split_name}.pt")
        print(f"Saving {split_name} split to {split_path}...")
        
        # Extract samples to avoid saving the whole dataset class
        samples = dataset.samples
        
        # Save samples to disk
        torch.save(samples, split_path)
        print(f"Saved {len(samples)} samples for {split_name} split")
    
    print(f"All splits saved to {save_path}")


def load_cached_pytorch_dataset(cache_path):
    """Load cached PyTorch datasets from disk."""
    datasets = {}
    
    for split_name in ["train", "validation", "test"]:
        split_path = os.path.join(cache_path, f"{split_name}.pt")
        if os.path.exists(split_path):
            print(f"Loading {split_name} split from {split_path}...")
            samples = torch.load(split_path)
            print(f"Loaded {len(samples)} samples for {split_name} split")
            
            # Create a dataset with the loaded samples
            dataset = CachedAudioDataset(samples)
            datasets[split_name] = dataset
        else:
            raise FileNotFoundError(f"Cannot find cached dataset at {split_path}")
    
    return datasets


def prepare_cnn_rnn_dataset():
    """Prepare data for CNN-RNN model training using PyTorch datasets instead of HF."""
    myData.DownloadAndExtract()
    
    # Check if cached PyTorch dataset exists
    pytorch_dataset_path = os.path.join(myConfig.DATA_DIR, "pytorch_dataset")
    if os.path.exists(pytorch_dataset_path) and len(os.listdir(pytorch_dataset_path)) > 0:
        print(f"Loading cached PyTorch dataset from {pytorch_dataset_path}")
        return load_cached_pytorch_dataset(pytorch_dataset_path)
    print("No cached dataset found. Creating new dataset...")
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
    
    # Perform dataset splits directly on the dataframe
    print("Creating dataset splits...")
    train_df, val_df, test_df = myData.datasetSplit(data_df)
    train_df, val_df, test_df = myData.ScaleDatasets(train_df, val_df, test_df)
    
    print("Creating PyTorch datasets with preloaded audio...")
    dataset = {
        "train": PreloadedAudioDataset(train_df, max_duration=100.0, sample_rate=16000),
        "validation": PreloadedAudioDataset(val_df, max_duration=100.0, sample_rate=16000),
        "test": PreloadedAudioDataset(test_df, max_duration=100.0, sample_rate=16000)
    }
    save_pytorch_dataset(dataset, pytorch_dataset_path)
    return dataset
