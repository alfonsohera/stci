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


def chunk_audio(audio, chunk_size_seconds=10, sample_rate=16000, min_segment_length=5):
    """
    Split audio into fixed-size chunks of specified duration
    
    Args:
        audio: Audio tensor of shape [C, L]
        chunk_size_seconds: Size of each chunk in seconds
        sample_rate: Sample rate of the audio
        min_segment_length: Minimum length in seconds to keep a segment (otherwise discard)
        
    Returns:
        List of audio chunks, each of shape [C, chunk_size*sample_rate]
    """
    chunk_size = int(chunk_size_seconds * sample_rate)
    min_segment_size = int(min_segment_length * sample_rate)
    chunks = []
    
    # Get audio length (sequence dimension)
    audio_length = audio.shape[1]
    
    # If audio is smaller than minimum segment size, pad it
    if audio_length < min_segment_size:
        padding = torch.zeros((audio.shape[0], chunk_size - audio_length), dtype=audio.dtype)
        padded_audio = torch.cat([audio, padding], dim=1)
        return [padded_audio]
    
    # Calculate how many whole chunks we can get from this audio
    num_chunks = audio_length // chunk_size
    
    # Split audio into full chunks
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunks.append(audio[:, start:end])
    
    # Handle the remaining part if it exists
    remaining = audio_length % chunk_size
    if remaining > 0:
        # Get the remaining audio
        start = num_chunks * chunk_size
        remaining_audio = audio[:, start:]
        
        # Only keep and pad segments that are at least min_segment_seconds long
        if remaining >= min_segment_size:
            padding = torch.zeros((audio.shape[0], chunk_size - remaining), dtype=audio.dtype)
            last_chunk = torch.cat([remaining_audio, padding], dim=1)
            chunks.append(last_chunk)
        # Otherwise, the remaining segment is discarded (less than min_segment_length seconds)
    
    return chunks


def debug_chunk_audio(audio_id="3", sample_rate=16000):
    """
    Debug function to trace the chunking process for a specific audio ID
    """
    import os
    import torch
    import librosa
    import numpy as np
    
    # Try to load the original audio file directly
    pytorch_dataset_path = os.path.join(myConfig.DATA_DIR, "pytorch_dataset")
    
    # Search in test split first
    split_path = os.path.join(pytorch_dataset_path, "test.pt")
    found_audio = None
    file_path = None
    
    try:
        if os.path.exists(split_path):
            samples = torch.load(split_path)
            if int(audio_id) < len(samples):
                i = int(audio_id)
                print(f"Found audio ID {audio_id} in test dataset at index {i}")
                found_audio = samples[i]["audio"]
                file_path = samples[i].get("file_path", f"test_{i}")
    except Exception as e:
        print(f"Error finding audio ID {audio_id}: {e}")
    
    if found_audio is not None:
        print(f"\nDEBUG CHUNKING FOR AUDIO ID {audio_id}")
        print(f"File path: {file_path}")
        print(f"Audio tensor shape: {found_audio.shape}")
        
        # Calculate original duration
        audio_length_samples = found_audio.shape[1]
        original_duration = audio_length_samples / sample_rate
        print(f"Original audio duration: {original_duration:.2f} seconds")
        
        # Trace the chunking process
        chunks = chunk_audio(found_audio, chunk_size_seconds=10, sample_rate=sample_rate, min_segment_length=5)
        print(f"Number of chunks created: {len(chunks)}")
        
        # Print details of each chunk
        for i, chunk in enumerate(chunks):
            chunk_duration = chunk.shape[1] / sample_rate
            # Check for zeros at the end (padding)
            non_zero = torch.where(torch.abs(chunk[0]) > 1e-6)[0]
            if len(non_zero) > 0:
                last_non_zero = non_zero[-1].item()
                padding_samples = chunk.shape[1] - last_non_zero - 1
                padding_seconds = padding_samples / sample_rate
                print(f"Chunk {i+1}: {chunk_duration:.2f}s duration with {padding_seconds:.2f}s padding at the end")
            else:
                print(f"Chunk {i+1}: {chunk_duration:.2f}s duration (empty chunk)")
        
        # Calculate total duration of all chunks
        total_chunk_duration = sum(chunk.shape[1] for chunk in chunks) / sample_rate
        print(f"Total duration of all chunks: {total_chunk_duration:.2f} seconds")
        
        # Check if any of the data versions have zeros in them
        print("\nChecking for zeros in the audio data:")
        # Check the original loaded audio
        zero_count = torch.sum(torch.abs(found_audio) < 1e-6).item()
        zero_percent = (zero_count / found_audio.numel()) * 100
        print(f"Original audio has {zero_count} zeros ({zero_percent:.2f}% of samples)")
        
        # Check if there might be a different version in the raw data directory
        try:
            import torchaudio
            data_dir = os.path.join(myConfig.ROOT_DIR, "Data")
            for subdir in ["Healthy", "MCI", "AD"]:
                for ext in ['.wav', '.flac']:
                    file_check_path = os.path.join(data_dir, subdir, f"{os.path.basename(file_path)}")
                    if os.path.exists(file_check_path):
                        print(f"Found original file at {file_check_path}")
                        waveform, sr = torchaudio.load(file_check_path)
                        raw_duration = waveform.shape[1] / sr
                        print(f"Raw audio duration: {raw_duration:.2f} seconds")
                        
                        # Check for zeros
                        raw_zero_count = torch.sum(torch.abs(waveform) < 1e-6).item()
                        raw_zero_percent = (raw_zero_count / waveform.numel()) * 100
                        print(f"Raw audio has {raw_zero_count} zeros ({raw_zero_percent:.2f}% of samples)")
                        break
        except Exception as e:
            print(f"Error checking raw audio: {e}")
        
        return chunks
    else:
        print(f"Could not find audio ID {audio_id}")
        return None


class CNNDataset(Dataset):
    """Dataset wrapper specifically for CNN model with fixed chunk size
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
        
        # Calculate chunk context information
        total_chunks = len(chunks)
        audio_length = audio.shape[1]
        chunk_length = audio_chunk.shape[1]
        
        # Relative position (0-1 range)
        rel_position = chunk_idx / max(1, total_chunks - 1)
        # Relative length compared to original
        rel_length = chunk_length / audio_length
        
        # Create chunk context tensor
        chunk_context = torch.tensor([rel_position, rel_length], dtype=torch.float32)
        
        result = {
            "audio": audio_chunk,
            "label": item["label"],
            "original_idx": dataset_idx,
            "chunk_context": chunk_context
        }
        
        # Pass prosodic features if available (same for all chunks from the same audio)
        if "prosodic_features" in item:
            result["prosodic_features"] = item["prosodic_features"]
        
        # Add audio_id for chunk handling
        audio_id = f"{dataset_idx}" 
        result["audio_id"] = audio_id
        
        # Preserve file path if available
        if "file_path" in item:
            result["file_path"] = item["file_path"]
                
        return result


def collate_fn_cnn(batch):
    """Collate function that handles chunked audio with prosodic features."""
    audio = torch.stack([item["audio"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    
    result = {
        "audio": audio,
        "labels": labels,
        "audio_lengths": torch.tensor([item["audio"].shape[1] for item in batch])
    }
    
    # Include prosodic features if available
    if "prosodic_features" in batch[0]:
        result["prosodic_features"] = torch.stack([item["prosodic_features"] for item in batch])
    
    # Include chunk context if available
    if "chunk_context" in batch[0]:
        result["chunk_context"] = torch.stack([item["chunk_context"] for item in batch])
    
    # Track original indices if available
    if "original_idx" in batch[0]:
        result["original_idx"] = [item["original_idx"] for item in batch]
        
    # Pass augmentation IDs if present
    if "augmentation_id" in batch[0]:
        result["augmentation_id"] = [item.get("augmentation_id") for item in batch]
    
    # Include audio_id for chunk aggregation
    if "audio_id" in batch[0]:
        result["audio_id"] = [item["audio_id"] for item in batch]

    # Include file paths if available - use consistent key name file_path (singular)
    if "file_path" in batch[0]:
        result["file_path"] = [item["file_path"] for item in batch]
        
    return result


def get_cnn_dataloaders(dataset_dict, batch_size=64, chunk_size_seconds=10):
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
    train_dataset = CNNDataset(
        dataset_dict["train"],         
        chunk_size_seconds=chunk_size_seconds
    )
    
    val_dataset = CNNDataset(
        dataset_dict["validation"],         
        chunk_size_seconds=chunk_size_seconds
    )
    
    test_dataset = CNNDataset(
        dataset_dict["test"],         
        chunk_size_seconds=chunk_size_seconds
    )
    
    # Create dataloaders with appropriate settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn_cnn,
        shuffle=True,
        num_workers=4,  
        pin_memory=True  
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn_cnn,
        shuffle=False,
        num_workers=4,  
        pin_memory=True 
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn_cnn,
        shuffle=False,
        num_workers=4,  
        pin_memory=True 
    )
    print(f"Original validation set size: {len(dataset_dict['validation'])} recordings")
    print(f"After chunking: {len(val_dataset)} chunks")
    return {
        "train": train_loader,
        "validation": val_loader,
        "test": test_loader
    }


class PreloadedAudioDataset(Dataset):
    """PyTorch dataset that preloads all audio data and prosodic features."""
    def __init__(self, dataframe, max_duration=100.0, sample_rate=16000):
        self.dataframe = dataframe
        self.samples = []
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        
        # Define which prosodic features to use
        self.prosodic_feature_columns = [
            "num_pauses",
            "total_pause_duration",
            "phonation_time",            
            "wer"
        ]
        
        print(f"Preloading {len(dataframe)} audio files with prosodic features...")
        for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Loading audio"):
            file_path = row["file_path"]
            label = row["label"]
            
            try:
                # Load audio directly with librosa, limiting duration
                audio_path = myFunctions.resolve_audio_path(file_path)
                audio, sr = self._load_audio_with_duration(audio_path)
                
                # Convert to tensor
                if not isinstance(audio, torch.Tensor):
                    audio = torch.tensor(audio, dtype=torch.float32)
                
                # Add channel dimension if missing
                if len(audio.shape) == 1:
                    audio = audio.unsqueeze(0)  # [C, L]
                
                # Extract prosodic features from dataframe
                prosodic_features = []
                for feature in self.prosodic_feature_columns:
                    if feature in row:
                        prosodic_features.append(float(row[feature]))
                    else:
                        prosodic_features.append(0.0)  # Default value if missing
                
                # Convert to tensor
                prosodic_tensor = torch.tensor(prosodic_features, dtype=torch.float32)
                
                # Store the processed sample with prosodic features
                self.samples.append({
                    "audio": audio,
                    "label": label,
                    "file_path": file_path,
                    "prosodic_features": prosodic_tensor
                })
            except Exception as e:
                print(f"Error loading audio {file_path}: {e}")
                # Create a dummy sample for failed audio
                dummy_audio = torch.zeros((1, 16000), dtype=torch.float32)
                dummy_prosodic = torch.zeros(len(self.prosodic_feature_columns), dtype=torch.float32)
                self.samples.append({
                    "audio": dummy_audio,
                    "label": label,
                    "file_path": file_path,
                    "prosodic_features": dummy_prosodic,
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
        
        # Trim leading and trailing silence
        # This addresses the issue where some processed audio files have excessive silence
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
        
        # Safety check to ensure we don't return empty audio after trimming
        if len(audio_trimmed) < self.sample_rate:  # Less than 1 second after trimming
            print(f"Warning: Audio {file_path} was trimmed too aggressively, using original")
            audio_trimmed = audio
        
        # Convert to the expected format
        return np.array(audio_trimmed, dtype=np.float32), self.sample_rate
    
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


def prepare_cnn_dataset(binary_classification=True):
    """Prepare data for CNN model training using PyTorch datasets instead of HF,
    excluding files specified in exclude_list.csv.
    
    Args:
        binary_classification: If True, convert to binary classification 
                              (0=Healthy, 1=Non-Healthy (MCI+AD))
    """
    myData.DownloadAndExtract()
    
    # Check if cached PyTorch dataset exists
    pytorch_dataset_path = os.path.join(myConfig.DATA_DIR, "pytorch_dataset")
    if os.path.exists(pytorch_dataset_path) and len(os.listdir(pytorch_dataset_path)) > 0:
        print(f"Loading cached PyTorch dataset from {pytorch_dataset_path}")
        dataset = load_cached_pytorch_dataset(pytorch_dataset_path)
        
        # Convert to binary classification if requested
        if binary_classification:
            print("Converting to binary classification (Healthy vs. Non-Healthy)...")
            for split in dataset:
                for i in range(len(dataset[split].samples)):
                    # Convert MCI (1) and AD (2) to Non-Healthy (1)
                    if dataset[split].samples[i]["label"] > 0:
                        dataset[split].samples[i]["label"] = 1
            print("Conversion to binary classification complete.")
        
        return dataset
    
    print("No cached dataset found. Creating new dataset...")
    
    # Load exclude list
    exclude_list_path = os.path.join(myConfig.ROOT_DIR, "exclude_list.csv")
    if os.path.exists(exclude_list_path):
        exclude_df = pd.read_csv(exclude_list_path)
        exclude_filenames = set(exclude_df['filename'].tolist())
        print(f"Loaded exclude list with {len(exclude_filenames)} files to exclude")
    else:
        print(f"Warning: Exclude list not found at {exclude_list_path}")
        exclude_filenames = set()
    
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
    
    # Apply the exclude list by filtering the dataframe
    # Extract the base filename from file_path for comparison with exclude list
    initial_count = len(data_df)
    
    # Create a function to extract base filename without directory and extension
    def extract_base_filename(file_path):
        # Get filename without directory
        filename = os.path.basename(file_path)
        # Remove extension
        base_name = os.path.splitext(filename)[0]
        return base_name
    
    # Apply filter to exclude files
    data_df['base_filename'] = data_df['file_path'].apply(extract_base_filename)
    filtered_df = data_df[~data_df['base_filename'].isin(exclude_filenames)]
    filtered_df = filtered_df.drop('base_filename', axis=1)  # Remove temporary column
    
    excluded_count = initial_count - len(filtered_df)
    print(f"Excluded {excluded_count} files from the dataset")
    
    # Perform dataset splits directly on the filtered dataframe
    print("Creating dataset splits...")
    train_df, val_df, test_df = myData.datasetSplit(filtered_df)
    train_df, val_df, test_df = myData.ScaleDatasets(train_df, val_df, test_df)
    
    # Convert to binary classification if requested
    if binary_classification:
        print("Converting to binary classification (Healthy vs. Non-Healthy)...")
        for df in [train_df, val_df, test_df]:
            # Convert MCI (1) and AD (2) to Non-Healthy (1)
            df.loc[df['label'] > 0, 'label'] = 1
        
        # Print new class distribution
        for name, df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
            healthy_count = (df['label'] == 0).sum()
            non_healthy_count = (df['label'] == 1).sum()
            total = len(df)
            print(f"{name} set: Healthy: {healthy_count} ({healthy_count/total*100:.1f}%), "
                  f"Non-Healthy: {non_healthy_count} ({non_healthy_count/total*100:.1f}%)")
    
    print("Creating PyTorch datasets with preloaded audio...")
    dataset = {
        "train": PreloadedAudioDataset(train_df, max_duration=120, sample_rate=16000),
        "validation": PreloadedAudioDataset(val_df, max_duration=120, sample_rate=16000),
        "test": PreloadedAudioDataset(test_df, max_duration=120, sample_rate=16000)
    }
    
    print(f"Final dataset sizes: Train: {len(dataset['train'])}, "
          f"Validation: {len(dataset['validation'])}, Test: {len(dataset['test'])}")
    
    save_pytorch_dataset(dataset, pytorch_dataset_path)
    return dataset


def get_original_audio_by_id(audio_id):
    """
    Retrieve the original full-length processed audio file for a given audio ID.    
    
    Args:
        audio_id (str): The ID of the audio file to retrieve
        
    Returns:
        tuple: (torch.Tensor, str) - The full audio tensor and the file path, or (None, None) if not found
    """
    import os
    import torch
    import torchaudio
        
    pytorch_dataset_path = os.path.join(myConfig.DATA_DIR, "pytorch_dataset")
    
    # Define the order to search splits - prioritize test split since CAM is typically used during testing
    splits_to_search = ["test", "validation", "train"]
    
    # Track whether we found a match in any split
    found_audio = None
    found_path = None
    found_split = None
    
    # Try to load the dataset info to find the correct file for the given ID
    try:
        print(f"Searching for audio ID: {audio_id} in pytorch dataset...")
        
        for split in splits_to_search:
            split_path = os.path.join(pytorch_dataset_path, f"{split}.pt")
            if os.path.exists(split_path):
                samples = torch.load(split_path)
                
                # Try to find a sample with matching ID (exact match)
                if audio_id.isdigit() and int(audio_id) < len(samples):
                    # Direct index access if ID is a valid index
                    i = int(audio_id)
                    print(f"Found matching processed audio in {split} dataset at index {i}")
                    audio_tensor = samples[i]["audio"]
                    file_path = samples[i].get("file_path", f"dataset_{split}_{i}")
                    
                    if split == "test":
                        print(f"Match found in test split, which is the correct split for CAM visualization.")
                        return audio_tensor, file_path
                    else:
                        # Save the match but keep looking for a test split match first
                        found_audio = audio_tensor
                        found_path = file_path
                        found_split = split
                        
                # If not a direct index match, search through all samples
                for i, sample in enumerate(samples):
                    if str(i) == str(audio_id) or (
                        "file_path" in sample and audio_id in str(sample["file_path"])):
                        print(f"Found matching processed audio in {split} dataset at index {i}")
                        # Use the audio data directly from the saved dataset (already processed)
                        audio_tensor = sample["audio"]
                        file_path = sample.get("file_path", f"dataset_{split}_{i}")
                        
                        if split == "test":
                            print(f"Match found in test split, which is the correct split for CAM visualization.")
                            return audio_tensor, file_path
                        else:
                            # Save the match but keep looking for a test split match first
                            found_audio = audio_tensor
                            found_path = file_path
                            found_split = split
        
        # If we found a match but not in the test split, warn the user and return that match
        if found_audio is not None:
            print(f"WARNING: Audio ID {audio_id} was found in the {found_split} split, not the test split!")
            print(f"This may lead to incorrect CAM visualization since you're analyzing a {found_split} sample.")
            return found_audio, found_path
                    
    except Exception as e:
        print(f"Error searching in pytorch dataset: {str(e)}")
        
    data_dir = os.path.join(myConfig.ROOT_DIR, "Data")
    print(f"Searching for audio ID: {audio_id} in Data directory: {data_dir}")
    
    try:
        # Try to find the file directly in the Data directory
        for ext in ['.wav', '.flac']:
            filepath = os.path.join(data_dir, f"{audio_id}{ext}")
            if os.path.exists(filepath):
                print(f"Found audio file in Data directory: {filepath}")
                waveform, sample_rate = torchaudio.load(filepath)
                # Convert to mono if needed
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                print("WARNING: This file was found in the raw data directory, not in a specific dataset split.")
                print("Make sure this is a test sample if you're using it for CAM visualization.")
                return waveform, filepath
        
        # Check class-specific subdirectories inside Data if they exist
        for subdir in ["Healthy", "MCI", "AD"]:
            subdir_path = os.path.join(data_dir, subdir)
            if os.path.exists(subdir_path):
                for ext in ['.wav', '.flac']:
                    filepath = os.path.join(subdir_path, f"{audio_id}{ext}")
                    if os.path.exists(filepath):
                        print(f"Found audio file in Data/{subdir} directory: {filepath}")
                        waveform, sample_rate = torchaudio.load(filepath)
                        # Convert to mono if needed
                        if waveform.shape[0] > 1:
                            waveform = waveform.mean(dim=0, keepdim=True)
                        print("WARNING: This file was found in the raw data directory, not in a specific dataset split.")
                        print("Make sure this is a test sample if you're using it for CAM visualization.")
                        return waveform, filepath
    except Exception as e:
        print(f"Error searching in Data directory: {str(e)}")
    
    # If we get here, we couldn't find the file in allowed locations
    print(f"Could not find audio file for ID: {audio_id}")
    print(f"Only searched in pytorch_dataset and /Data/ directory as specified.")
    return None, None


def analyze_dataset_chunking(dataset_dict, sample_rate=16000, trim_silence=True):
    """
    Analyze how audio files in the dataset are chunked with and without silence trimming.
    
    Args:
        dataset_dict: Dictionary of datasets with train/validation/test splits
        sample_rate: Sample rate of audio
        trim_silence: Whether to analyze with silence trimming or not
    
    Returns:
        Dictionary with analysis results
    """
    import os
    import torch
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    
    results = {
        "all": {"original_durations": [], "processed_durations": [], "zero_percentages": [], 
                "num_chunks_before": [], "num_chunks_after": []},
        "by_split": {}
    }
    
    for split_name in ["train", "validation", "test"]:
        if split_name not in dataset_dict:
            continue
            
        results["by_split"][split_name] = {
            "original_durations": [], 
            "processed_durations": [], 
            "zero_percentages": [],
            "num_chunks_before": [], 
            "num_chunks_after": []
        }
        
        split_data = dataset_dict[split_name]
        print(f"\nAnalyzing {split_name} split with {len(split_data)} samples...")
        
        for i, item in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
            if i >= 200:  # Limit to first 200 files per split for speed
                break
                
            audio = item["audio"]
            file_path = item.get("file_path", f"{split_name}_{i}")
            
            # Calculate original duration
            orig_duration = audio.shape[1] / sample_rate
            
            # Check for zeros in original audio
            zero_count = torch.sum(torch.abs(audio) < 1e-6).item()
            zero_percent = (zero_count / audio.numel()) * 100
            
            # Count chunks before trimming silence
            chunks_before = chunk_audio(audio, chunk_size_seconds=10, sample_rate=sample_rate, min_segment_length=5)
            num_chunks_before = len(chunks_before)
            
            # Process with silence trimming if requested
            if trim_silence:
                # Try to load original audio file to avoid double-trimming
                try:
                    data_dir = os.path.join(myConfig.DATA_DIR, "Data")
                    audio_path = None
                    
                    # Try to resolve path relative to the data directory
                    potential_paths = [
                        os.path.join(myConfig.ROOT_DIR, file_path),
                        os.path.join(myConfig.ROOT_DIR, "Data", file_path),
                        os.path.join(myConfig.ROOT_DIR, "Data", os.path.basename(file_path))
                    ]
                    
                    for path in potential_paths:
                        if os.path.exists(path):
                            audio_path = path
                            break
                    
                    if audio_path is None:
                        # Try to find in Healthy/MCI/AD subdirectories
                        for subdir in ["Healthy", "MCI", "AD"]:
                            path = os.path.join(myConfig.ROOT_DIR, "Data", subdir, os.path.basename(file_path))
                            if os.path.exists(path):
                                audio_path = path
                                break
                    
                    if audio_path:
                        # Load and trim original audio
                        y, sr = librosa.load(audio_path, sr=sample_rate)
                        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
                        
                        # Safety check to avoid over-trimming
                        if len(y_trimmed) < sample_rate:  # Less than 1 second after trimming
                            y_trimmed = y
                        
                        # Convert to tensor format
                        processed_audio = torch.tensor(y_trimmed, dtype=torch.float32).unsqueeze(0)
                    else:
                        # If original file not found, trim the tensor directly
                        np_audio = audio.numpy().squeeze()
                        trimmed, _ = librosa.effects.trim(np_audio, top_db=30)
                        processed_audio = torch.tensor(trimmed, dtype=torch.float32).unsqueeze(0)
                except Exception as e:
                    print(f"  Error trimming {file_path}: {e}")
                    processed_audio = audio  # Fallback to original
                    
                # Calculate processed duration
                proc_duration = processed_audio.shape[1] / sample_rate
                
                # Count chunks after trimming silence
                chunks_after = chunk_audio(processed_audio, chunk_size_seconds=10, sample_rate=sample_rate, min_segment_length=5)
                num_chunks_after = len(chunks_after)
            else:
                proc_duration = orig_duration
                num_chunks_after = num_chunks_before
            
            # Store results
            results["all"]["original_durations"].append(orig_duration)
            results["all"]["processed_durations"].append(proc_duration)
            results["all"]["zero_percentages"].append(zero_percent)
            results["all"]["num_chunks_before"].append(num_chunks_before)
            results["all"]["num_chunks_after"].append(num_chunks_after)
            
            results["by_split"][split_name]["original_durations"].append(orig_duration)
            results["by_split"][split_name]["processed_durations"].append(proc_duration)
            results["by_split"][split_name]["zero_percentages"].append(zero_percent)
            results["by_split"][split_name]["num_chunks_before"].append(num_chunks_before)
            results["by_split"][split_name]["num_chunks_after"].append(num_chunks_after)
    
    # Summarize results
    print("\n===== CHUNKING ANALYSIS SUMMARY =====")
    all_orig_durations = np.array(results["all"]["original_durations"])
    all_proc_durations = np.array(results["all"]["processed_durations"])
    all_zero_pcts = np.array(results["all"]["zero_percentages"])
    all_chunks_before = np.array(results["all"]["num_chunks_before"])
    all_chunks_after = np.array(results["all"]["num_chunks_after"])
    
    print(f"All splits combined:")
    print(f"  Original duration: {np.mean(all_orig_durations):.2f}s ± {np.std(all_orig_durations):.2f}s")
    print(f"  Processed duration: {np.mean(all_proc_durations):.2f}s ± {np.std(all_proc_durations):.2f}s")
    print(f"  Average reduction: {np.mean(all_orig_durations - all_proc_durations):.2f}s ({np.mean((all_orig_durations - all_proc_durations) / all_orig_durations) * 100:.1f}%)")
    print(f"  Zero percentage: {np.mean(all_zero_pcts):.2f}% ± {np.std(all_zero_pcts):.2f}%")
    print(f"  Average chunks before: {np.mean(all_chunks_before):.2f} ± {np.std(all_chunks_before):.2f}")
    print(f"  Average chunks after: {np.mean(all_chunks_after):.2f} ± {np.std(all_chunks_after):.2f}")
    print(f"  Samples with reduced chunks: {np.sum(all_chunks_after < all_chunks_before)} ({np.sum(all_chunks_after < all_chunks_before) / len(all_chunks_before) * 100:.1f}%)")
    
    for split_name in results["by_split"]:
        split_orig_durations = np.array(results["by_split"][split_name]["original_durations"]) 
        split_proc_durations = np.array(results["by_split"][split_name]["processed_durations"])
        split_zero_pcts = np.array(results["by_split"][split_name]["zero_percentages"])
        split_chunks_before = np.array(results["by_split"][split_name]["num_chunks_before"])
        split_chunks_after = np.array(results["by_split"][split_name]["num_chunks_after"])
        
        print(f"\n{split_name.capitalize()} split:")
        print(f"  Original duration: {np.mean(split_orig_durations):.2f}s ± {np.std(split_orig_durations):.2f}s")
        print(f"  Processed duration: {np.mean(split_proc_durations):.2f}s ± {np.std(split_proc_durations):.2f}s")
        print(f"  Average reduction: {np.mean(split_orig_durations - split_proc_durations):.2f}s ({np.mean((split_orig_durations - split_proc_durations) / split_orig_durations) * 100:.1f}%)")
        print(f"  Zero percentage: {np.mean(split_zero_pcts):.2f}% ± {np.std(split_zero_pcts):.2f}%")
        print(f"  Average chunks before: {np.mean(split_chunks_before):.2f} ± {np.std(split_chunks_before):.2f}")
        print(f"  Average chunks after: {np.mean(split_chunks_after):.2f} ± {np.std(split_chunks_after):.2f}")
        print(f"  Samples with reduced chunks: {np.sum(split_chunks_after < split_chunks_before)} ({np.sum(split_chunks_after < split_chunks_before) / len(split_chunks_before) * 100:.1f}%)")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(all_orig_durations, all_proc_durations, alpha=0.5)
    plt.plot([0, max(all_orig_durations)], [0, max(all_orig_durations)], 'r--')
    plt.xlabel('Original Duration (s)')
    plt.ylabel('Processed Duration (s)')
    plt.title('Duration Before vs After Silence Trimming')
    
    plt.subplot(2, 2, 2)
    plt.scatter(all_orig_durations, all_zero_pcts, alpha=0.5)
    plt.xlabel('Original Duration (s)')
    plt.ylabel('Zero Samples (%)')
    plt.title('Silence vs Duration')
    
    plt.subplot(2, 2, 3)
    plt.scatter(all_chunks_before, all_chunks_after, alpha=0.5)
    plt.plot([0, max(all_chunks_before)], [0, max(all_chunks_before)], 'r--')
    plt.xlabel('Chunks Before Trimming')
    plt.ylabel('Chunks After Trimming')
    plt.title('Number of Chunks Before vs After')
    
    plt.subplot(2, 2, 4)
    duration_diff = all_orig_durations - all_proc_durations
    plt.hist(duration_diff, bins=20)
    plt.xlabel('Duration Reduction (s)')
    plt.ylabel('Count')
    plt.title('Histogram of Duration Reduction')
    
    plt.tight_layout()
    
    # Create output directory
    output_dir = os.path.join(myConfig.OUTPUT_PATH, "chunking_analysis")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "chunking_analysis.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nAnalysis plot saved to {plot_path}")
    
    return results
