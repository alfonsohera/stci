import os
import requests
import shutil
from .Config import ROOT_DIR, DATA_DIR, LABEL_MAP, OUTPUT_PATH  
from .Functions import resolve_audio_path, chunk_input_sample  
import numpy as np
import psutil
import gc
import torch

from zipfile import ZipFile
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datasets import load_from_disk
from tqdm import tqdm
from datasets import Dataset, DatasetDict


extracted_features = [        
    "num_pauses",
    "total_pause_duration",
    "phonation_time",
    "shimmer_local",
    "skewness",
    "centre_of_gravity",
    "wer"
]

# Add memory monitoring function
def log_memory_usage(label):
    """Log current memory usage with a descriptive label"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Convert to GB for readability
    rss_gb = memory_info.rss / (1024 * 1024 * 1024)
    vms_gb = memory_info.vms / (1024 * 1024 * 1024)
    
    print(f"MEMORY [{label}] - RSS: {rss_gb:.2f} GB, VMS: {vms_gb:.2f} GB")
    
    # Log if we're approaching system limits
    total_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)
    percent_used = (memory_info.rss / psutil.virtual_memory().total) * 100
    print(f"MEMORY [{label}] - Using {percent_used:.1f}% of total system memory ({total_memory:.1f} GB)")
    
    return memory_info


def DownloadAndExtract():
    # Always use Data directory at script level
    data_dir = DATA_DIR
    
    # Define paths for category folders within data_dir
    healthy_dir = os.path.join(data_dir, "Healthy")
    mci_dir = os.path.join(data_dir, "MCI")
    ad_dir = os.path.join(data_dir, "AD")
    
    # Create data directory and category subdirectories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(healthy_dir, exist_ok=True)
    os.makedirs(mci_dir, exist_ok=True)
    os.makedirs(ad_dir, exist_ok=True)
    
    # Skip download if files already exist
    if all(os.path.exists(folder) and os.listdir(folder) 
           for folder in [healthy_dir, mci_dir, ad_dir]):
        print("Target folders already exist with files. Skipping download and extraction.")
        return
    
    urls = [
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc2.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc3.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc4.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc5.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc6.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc7.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc8.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc9.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc10.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc11.zip"
    ]

    # Create a temporary folder for extracted files within the data directory
    temp_folder = os.path.join(data_dir, "tmp_extracted")
    os.makedirs(temp_folder, exist_ok=True)

    # Download and extract each zip file
    for i, url in enumerate(urls):
        zip_filename = os.path.join(data_dir, f"downloaded_{i}.zip")  # Store downloads in data directory

        # Download the file
        print(f"Downloading from {url}...")
        response = requests.get(url)
        with open(zip_filename, "wb") as f:
            f.write(response.content)
        print(f"Saved {zip_filename}")

        # Extract all contents into the temp_folder
        print(f"Extracting {zip_filename}...")
        with ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(temp_folder)

        # Delete the ZIP file after extraction to save space
        os.remove(zip_filename)

    # Move files to their destinations, convert from mp3 to wav if needed
    for root, dirs, files in os.walk(temp_folder):
        for filename in files:
            full_path = os.path.join(root, filename)

            # Convert MP3 to WAV if needed
            if filename.endswith(".mp3"):
                wav_filename = filename.replace(".mp3", ".wav")
                wav_path = os.path.join(root, wav_filename)
                # Convert MP3 to WAV
                audio = AudioSegment.from_mp3(full_path)
                audio.export(wav_path, format="wav")
                # Remove the original MP3
                os.remove(full_path)
                # Update full_path to the new WAV file
                full_path = wav_path
                filename = wav_filename

            # Move to corresponding folder
            if filename.startswith("AD"):
                shutil.move(full_path, os.path.join(ad_dir, filename))
            elif filename.startswith("MCI"):
                shutil.move(full_path, os.path.join(mci_dir, filename))
            elif filename.startswith("HC"):
                shutil.move(full_path, os.path.join(healthy_dir, filename))
            else:
                print(f"File '{filename}' doesn't match AD/MCI/HC. Skipping or placing it elsewhere.")

    print("MP3 conversion and file moving completed.")
    # Delete temporary folder
    shutil.rmtree(temp_folder, ignore_errors=True)
    print("Temporary folder removed.")


def datasetSplit(data_df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Split dataset into training, validation and test sets with specified ratios.
    
    Args:
        data_df: Input dataframe
        train_ratio: Proportion for training set (default 0.6)
        val_ratio: Proportion for validation set (default 0.2)
        test_ratio: Proportion for test set (default 0.2)
    """
    # Verify ratios add up to 1.0
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    # First, Drop class feature (label already encodes this info) and Sex (The class is imbalanced)
    data_df = data_df.drop(columns=["class", "Sex"])

    # First split: separate out test set
    train_val_df, test_df = train_test_split(
        data_df,
        test_size=test_ratio,
        stratify=data_df["label"],
        random_state=42
    )

    # Second split: separate train and validation from the remaining data
    # Calculate the validation size relative to train+val data
    relative_val_size = val_ratio / (train_ratio + val_ratio)
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df["label"],
        random_state=42
    )

    # Print dataset sizes
    print(f"Training set: {len(train_df)} ({len(train_df) / len(data_df) * 100:.2f}%)")
    print(f"Validation set: {len(val_df)} ({len(val_df) / len(data_df) * 100:.2f}%)")
    print(f"Test set: {len(test_df)} ({len(test_df) / len(data_df) * 100:.2f}%)")

    return train_df, val_df, test_df


def loadHFDataset():
    dataset = load_from_disk(OUTPUT_PATH)  
    return dataset


def ScaleDatasets(train_df, val_df, test_df):
    # Initialize scaler
    scaler = StandardScaler()

    # Fit scaler on TRAIN numeric columns only
    scaler.fit(train_df[extracted_features])

    # Transform train, val, and test numeric columns
    train_df[extracted_features] = scaler.transform(train_df[extracted_features])
    val_df[extracted_features] = scaler.transform(val_df[extracted_features])
    test_df[extracted_features] = scaler.transform(test_df[extracted_features])
    return train_df, val_df, test_df


def process_data(df):
    # Import Audio module inside the function to avoid circular imports
    from . import Audio
    
    data = []
    for row in tqdm(df.itertuples(), total=len(df)):
        audio_file = row.file_path
        file_path = resolve_audio_path(audio_file)  
        label = row.label

        # Load processed audio
        audio, sr = Audio.load_audio(file_path)

        # Build a dictionary with everything you need
        # -> the audio array, sampling rate, path, label, plus numeric features
        data.append({
            "audio": {
                "array": np.array(audio, dtype=np.float32),
                "sampling_rate": sr,
                "path": file_path
            },
            "label": label,
            # For each numeric feature, store the standardized value            
            "num_pauses": row.num_pauses,
            "total_pause_duration": row.total_pause_duration,
            "phonation_time": row.phonation_time,
            "shimmer_local": row.shimmer_local,
            "skewness": row.skewness,
            "centre_of_gravity": row.centre_of_gravity,
            "wer": row.wer
        })
    return data


def createHFDatasets(train_df, val_df, test_df):
    log_memory_usage("Before process_data")
    train_data = process_data(train_df)
    log_memory_usage("After train process_data")
    val_data = process_data(val_df)
    log_memory_usage("After val process_data")
    test_data = process_data(test_df)
    log_memory_usage("After test process_data")

    # Build HF Datasets from lists
    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data),
    })
    
    log_memory_usage("Before chunking")
    # Try cleaning up before chunking
    del train_data, val_data, test_data
    gc.collect()
    log_memory_usage("After cleanup, before chunking")
    
    # Monitor chunking in batches to see where it fails
    try:
        dataset = dataset.map(
            chunk_input_sample,
            desc="Chunking audio samples",
            batch_size=8  # Process in smaller batches
        )
        log_memory_usage("After chunking")
    except Exception as e:
        log_memory_usage("Chunking failed")
        print(f"Chunking error: {e}")
        raise

    # Finally, save to disk    
    dataset.save_to_disk(OUTPUT_PATH)
    log_memory_usage("After saving dataset")
    print(f"Dataset saved to {OUTPUT_PATH}")
    
    return dataset


def prepare_for_cnn_rnn(example):
    """
    Convert dataset examples for CNN+RNN model format.    
    """
    # Extract raw audio array
    audio = example["audio"]["array"]
    
    # Convert to torch tensor if not already
    if not isinstance(audio, torch.Tensor):
        audio = torch.tensor(audio, dtype=torch.float32)
    
    # Standardize audio length - 100 seconds at 16kHz
    max_length = 16000 * 100  # 100 seconds max
    
    # Handle as 1D tensor as in the original implementation
    if len(audio) > max_length:
        audio = audio[:max_length]
    elif len(audio) < max_length:
        padding = torch.zeros(max_length - len(audio), dtype=audio.dtype)
        audio = torch.cat([audio, padding])
    
    # Create list of prosodic features
    prosodic_features = []
    for feature in extracted_features:
        if feature in example:
            prosodic_features.append(example[feature])
    
    # Return updated example with properly formatted audio and prosodic features
    return {
        "audio": audio,
        "prosodic_features": prosodic_features,
        "label": example["label"]
    }