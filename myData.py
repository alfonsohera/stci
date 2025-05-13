import os
import requests
import shutil
import myConfig
import myFunctions
import myAudio
import numpy as np
import psutil
import gc
import torch

from zipfile import ZipFile
#from google.colab import drive
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
    data_dir = myConfig.DATA_DIR
    
    # Define paths for category folders within data_dir
    healthy_dir = os.path.join(data_dir, "Healthy")
    mci_dir = os.path.join(data_dir, "MCI")
    ad_dir = os.path.join(data_dir, "AD")
    
    # Create data directory and category subdirectories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(healthy_dir, exist_ok=True)
    os.makedirs(mci_dir, exist_ok=True)
    os.makedirs(ad_dir, exist_ok=True)
    
    # Skip download if files already exist in offline mode
    if myConfig.running_offline and all(os.path.exists(folder) and os.listdir(folder) 
                                     for folder in [healthy_dir, mci_dir, ad_dir]):
        print("Running offline and target folders already exist with files. Skipping download and extraction.")
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


def datasetSplit(data_df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, similarity_threshold=0.95):
    """
    Split dataset into training, validation and test sets with specified ratios, while considering
    cosine similarity to prevent similar samples from appearing in different splits.
    
    Args:
        data_df: Input dataframe
        train_ratio: Proportion for training set (default 0.6)
        val_ratio: Proportion for validation set (default 0.2)
        test_ratio: Proportion for test set (default 0.2)
        similarity_threshold: Threshold above which samples are considered too similar (default 0.95)
    """
    import os
    import torch
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from panns_inference.panns_inference.models import Cnn14
    import librosa
    from tqdm import tqdm
    
    # Verify ratios add up to 1.0
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    print("Starting similarity-aware dataset split...")
    
    # First, Drop class feature (label already encodes this info) and Sex (The class is imbalanced)
    data_df = data_df.drop(columns=["class", "Sex"])
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load CNN14 feature extractor
    print("Loading CNN14 feature extractor...")
    def load_cnn14(checkpoint_path=myConfig.checkpoint_dir+'/Cnn14_mAP=0.431.pth'):
        model = Cnn14(classes_num=527, sample_rate=16000, mel_bins=64, hop_size=320, window_size=1024, fmin=50, fmax=8000)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.eval()
        return model.to(device)
    
    feature_extractor = load_cnn14()
    
    # Create feature extraction function
    def extract_features(audio_path, feature_extractor):
        def extract_embedding(model, sample_audio):
            with torch.no_grad():
                output = model(sample_audio)
                embedding = output['embedding']
            return embedding  # shape: [1, 2048]
                
        try:
            # Load audio
            sample_audio, sr = librosa.load(audio_path, sr=16000)
            # Convert to tensor format for CNN14
            sample_audio = torch.tensor(sample_audio).unsqueeze(0).to(device)  # [1, time]                     
            
            with torch.inference_mode():
                # Extract features using CNN14
                features = extract_embedding(feature_extractor, sample_audio).flatten().cpu().numpy()
                
            return features
        except Exception as e:
            print(f"Error processing audio file: {e}")
            return None
    
    # Extract features for all audio files
    print("Extracting features for all samples...")
    features_dict = {}  # {index: features}
    
    for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Extracting features"):
        try:
            file_path = myFunctions.resolve_audio_path(row['file_path'])
            features = extract_features(file_path, feature_extractor)
            if features is not None:
                features_dict[idx] = features
        except Exception as e:
            print(f"Error extracting features for {row['file_path']}: {e}")
    
    print(f"Successfully extracted features for {len(features_dict)} files")
    
    # Clean up GPU memory
    del feature_extractor
    torch.cuda.empty_cache()
    
    # Get samples with extracted features
    valid_indices = list(features_dict.keys())
    data_df_valid = data_df.loc[valid_indices].copy()
    
    # Group samples by class
    class_groups = {label: data_df_valid[data_df_valid['label'] == label].index.tolist() 
                    for label in data_df_valid['label'].unique()}
    
    # Step 1: Start with empty splits
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Step 2: Iterate through each class
    for label, indices in class_groups.items():
        print(f"Processing class {label} with {len(indices)} samples")
        
        # Calculate target counts for this class
        n_total = len(indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val
        
        # Extract features for this class
        class_features = np.vstack([features_dict[idx] for idx in indices])
        
        # Step 3: Start the split algorithm
        remaining_indices = indices.copy()
        current_train = []
        current_val = []
        current_test = []
        
        # Start with a random sample for each split
        if remaining_indices:
            # First sample for train
            idx = np.random.choice(remaining_indices)
            current_train.append(idx)
            remaining_indices.remove(idx)
        
        if remaining_indices:
            # First sample for validation - find least similar to first train sample
            train_features = np.vstack([features_dict[idx] for idx in current_train])
            candidate_features = np.vstack([features_dict[idx] for idx in remaining_indices])
            similarities = cosine_similarity(train_features, candidate_features)[0]
            # Choose the least similar sample
            least_similar_idx = remaining_indices[np.argmin(similarities)]
            current_val.append(least_similar_idx)
            remaining_indices.remove(least_similar_idx)
        
        if remaining_indices:
            # First sample for test - find least similar to both train and val
            existing_features = np.vstack([features_dict[idx] for idx in current_train + current_val])
            candidate_features = np.vstack([features_dict[idx] for idx in remaining_indices])
            similarities = cosine_similarity(existing_features, candidate_features)
            # Calculate mean similarity for each candidate
            mean_similarities = np.mean(similarities, axis=0)
            # Choose the least similar sample
            least_similar_idx = remaining_indices[np.argmin(mean_similarities)]
            current_test.append(least_similar_idx)
            remaining_indices.remove(least_similar_idx)
        
        # Step 4: Iteratively build each split
        while (len(current_train) < n_train or len(current_val) < n_val or len(current_test) < n_test) and remaining_indices:
            # Determine which split needs more samples
            next_split = None
            if len(current_train) < n_train:
                next_split = 'train'
            elif len(current_val) < n_val:
                next_split = 'val'
            elif len(current_test) < n_test:
                next_split = 'test'
            else:
                break
            
            if next_split == 'train':
                target_set = current_train
                avoid_sets = current_val + current_test
            elif next_split == 'val':
                target_set = current_val
                avoid_sets = current_train + current_test
            else:  # test
                target_set = current_test
                avoid_sets = current_train + current_val
            
            # Calculate similarity to target set (want similar) and avoid sets (want dissimilar)
            candidate_features = np.vstack([features_dict[idx] for idx in remaining_indices])
            
            # Similarity to samples in same split (want high similarity)
            if target_set:
                target_features = np.vstack([features_dict[idx] for idx in target_set])
                target_similarities = cosine_similarity(target_features, candidate_features)
                similarity_to_target = np.mean(target_similarities, axis=0)
            else:
                similarity_to_target = np.zeros(len(remaining_indices))
            
            # Similarity to samples in other splits (want low similarity)
            if avoid_sets:
                avoid_features = np.vstack([features_dict[idx] for idx in avoid_sets])
                avoid_similarities = cosine_similarity(avoid_features, candidate_features)
                similarity_to_avoid = np.mean(avoid_similarities, axis=0)
                
                # Check for samples that would cause data leakage
                max_cross_similarities = np.max(avoid_similarities, axis=0)
                leakage_risk = max_cross_similarities >= similarity_threshold
                
                # Filter out high-similarity candidates if possible
                if np.any(~leakage_risk) and len(remaining_indices) > 1:
                    # Only consider candidates that don't have high similarity to other splits
                    valid_candidates = ~leakage_risk
                    if np.sum(valid_candidates) > 0:
                        # Choose candidate with highest similarity to target from valid candidates
                        if np.sum(valid_candidates) == 1:
                            selected_idx = np.where(valid_candidates)[0][0]
                        else:
                            selected_idx = np.argmax(similarity_to_target * valid_candidates)
                        best_candidate_idx = remaining_indices[selected_idx]
                        target_set.append(best_candidate_idx)
                        remaining_indices.remove(best_candidate_idx)
                        continue
            else:
                similarity_to_avoid = np.zeros(len(remaining_indices))
            
            # Score each candidate by similarity to target and dissimilarity to avoid
            if avoid_sets:
                # Select candidate with highest similarity to target AND lowest similarity to avoid
                combined_score = similarity_to_target - similarity_to_avoid
                best_candidate_idx = remaining_indices[np.argmax(combined_score)]
            else:
                # Just use similarity to target if no avoid set
                best_candidate_idx = remaining_indices[np.argmax(similarity_to_target)]
            
            # Add best candidate to appropriate split
            target_set.append(best_candidate_idx)
            remaining_indices.remove(best_candidate_idx)
        
        # Handle remaining indices if any (should be rare)
        if remaining_indices:
            print(f"Warning: {len(remaining_indices)} samples couldn't be assigned optimally")
            if len(current_train) < n_train:
                n_needed = n_train - len(current_train)
                current_train.extend(remaining_indices[:n_needed])
                remaining_indices = remaining_indices[n_needed:]
            
            if remaining_indices and len(current_val) < n_val:
                n_needed = n_val - len(current_val)
                current_val.extend(remaining_indices[:n_needed])
                remaining_indices = remaining_indices[n_needed:]
            
            if remaining_indices:
                current_test.extend(remaining_indices)
        
        # Add this class's samples to the final splits
        train_indices.extend(current_train)
        val_indices.extend(current_val)
        test_indices.extend(current_test)
        
        print(f"Class {label} split: Train: {len(current_train)}, Val: {len(current_val)}, Test: {len(current_test)}")
    
    # Create the final dataframes
    train_df = data_df.loc[train_indices].copy()
    val_df = data_df.loc[val_indices].copy()
    test_df = data_df.loc[test_indices].copy()

    # Print dataset sizes
    print("\nFinal dataset split sizes:")
    print(f"Training set: {len(train_df)} ({len(train_df) / len(data_df) * 100:.2f}%)")
    print(f"Validation set: {len(val_df)} ({len(val_df) / len(data_df) * 100:.2f}%)")
    print(f"Test set: {len(test_df)} ({len(test_df) / len(data_df) * 100:.2f}%)")
    
    # Verify split ratios by class
    print("\nClass distribution in splits:")
    for label in data_df['label'].unique():
        n_train = len(train_df[train_df['label'] == label])
        n_val = len(val_df[val_df['label'] == label])
        n_test = len(test_df[test_df['label'] == label])
        n_total = n_train + n_val + n_test
        
        print(f"Class {label}:")
        print(f"  Train: {n_train} ({n_train/n_total*100:.2f}%)")
        print(f"  Val: {n_val} ({n_val/n_total*100:.2f}%)")
        print(f"  Test: {n_test} ({n_test/n_total*100:.2f}%)")
    
    # Check cross-split similarities
    print("\nChecking for potential data leakage across splits...")
    
    # Sample a subset of files from each split for cross-checking
    max_check = 100  # Limit checks for performance
    train_sample = np.random.choice(train_indices, min(max_check, len(train_indices)), replace=False)
    val_sample = np.random.choice(val_indices, min(max_check, len(val_indices)), replace=False)
    test_sample = np.random.choice(test_indices, min(max_check, len(test_indices)), replace=False)
    
    # Check train vs val
    train_features = np.vstack([features_dict[idx] for idx in train_sample])
    val_features = np.vstack([features_dict[idx] for idx in val_sample])
    similarities = cosine_similarity(train_features, val_features)
    max_similarity = np.max(similarities)
    print(f"Maximum train-val similarity: {max_similarity:.4f}")
    
    # Check train vs test
    test_features = np.vstack([features_dict[idx] for idx in test_sample])
    similarities = cosine_similarity(train_features, test_features)
    max_similarity = np.max(similarities)
    print(f"Maximum train-test similarity: {max_similarity:.4f}")
    
    # Check val vs test
    similarities = cosine_similarity(val_features, test_features)
    max_similarity = np.max(similarities)
    print(f"Maximum val-test similarity: {max_similarity:.4f}")
    
    return train_df, val_df, test_df


def loadHFDataset():
    dataset = load_from_disk(myConfig.OUTPUT_PATH)
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
    data = []
    for row in tqdm(df.itertuples(), total=len(df)):
        audio_file = row.file_path
        file_path = myFunctions.resolve_audio_path(audio_file)
        label = row.label

        # Load processed audio
        audio, sr = myAudio.load_audio(file_path)

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
            myFunctions.chunk_input_sample,
            desc="Chunking audio samples",
            batch_size=8  # Process in smaller batches
        )
        log_memory_usage("After chunking")
    except Exception as e:
        log_memory_usage("Chunking failed")
        print(f"Chunking error: {e}")
        raise

    # Finally, save to disk    
    dataset.save_to_disk(myConfig.OUTPUT_PATH)
    log_memory_usage("After saving dataset")
    print(f"Dataset saved to {myConfig.OUTPUT_PATH}")
    
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