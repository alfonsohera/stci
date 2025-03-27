import parselmouth
import torch
import pandas as pd
import numpy as np
import myConfig
import myAudio
import myModel
import my_Speech2text
import os
from parselmouth.praat import call
from sklearn.metrics import classification_report, confusion_matrix
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import gc

def preprocess_function(example):
    # Wav2Vec2 processing
    _, processor, _  = myModel.getModelDefinitions()
    inputs = processor(example["audio"]["array"], sampling_rate=example["audio"]["sampling_rate"], return_tensors="pt")

    extracted_features = myConfig.selected_features
    feats = [example[col] for col in extracted_features]
    inputs["prosodic_features"] = torch.tensor(feats, dtype=torch.float32)
    inputs["label"] = example["label"]
    return inputs


def chunk_input_sample(example, max_length=16000*100):  # 100 seconds max
    audio = example["audio"]
    if len(audio) > max_length:
        example["audio"] = audio[:max_length]
    return preprocess_function(example)


# Extract class labels from file names
def extract_class(file_path):    
    filename = file_path.split("/")[-1]  # Extract filename
    if filename.startswith("HC"):
        return "HC"  # Healthy
    elif filename.startswith("MCI"):
        return "MCI"  # Mild Cognitive Impairment
    elif filename.startswith("AD"):
        return "AD"  # Alzheimer's
    return "Unknown"


def extract_spectral_features(sound):
    """Extract spectral features from a Parselmouth Sound object"""
    power = 2.    
    spectrum = call(sound, "To Spectrum", "yes")
    
    # Get features from spectrum
    centre_of_gravity = call(spectrum, "Get centre of gravity", power)
    skewness = call(spectrum, "Get skewness", power)
    
    # Clean up spectrum object    
    del spectrum
    
    feature_values = [skewness, centre_of_gravity]
    return np.array(feature_values, dtype=np.float32)


def extract_jitter_shimmer(sound, audio_path):
    path_to_use = audio_path
    
    # Extract gender from filename to set appropriate pitch range
    filename = os.path.basename(path_to_use)
    parts = filename.replace(".wav", "").split("-")
    
    # Set gender-specific pitch ranges as recommended in the paper
    if len(parts) >= 2 and parts[1] == "M":  # Male
        pitch_floor = 75
        pitch_ceiling = 300
    elif len(parts) >= 2 and parts[1] == "W":  # Female
        pitch_floor = 100
        pitch_ceiling = 500
    else:  # Default if gender can't be determined
        pitch_floor = 75
        pitch_ceiling = 600
        
    try:                
        try:
            # Create pitch object with gender-specific settings
            pitch = parselmouth.praat.call(sound, "To Pitch", 0.0, pitch_floor, pitch_ceiling)
            
            # Create point process using the pitch object
            point_process = parselmouth.praat.call([sound, pitch], 
                                                  "To PointProcess (cc)")
            
            # Make sure we have enough points for analysis
            num_points = parselmouth.praat.call(point_process, "Get number of points")
            if num_points < 3:
                print(f"Warning: Too few voice points in {path_to_use} ({num_points} points)")
                return np.array([np.nan, np.nan], dtype=np.float32)
            
            # Extract jitter local - using correct parameters
            try:
                jitter_local = parselmouth.praat.call(point_process, 
                                                     "Get jitter (local)", 
                                                     0.0, 0.0, 0.0001, 0.02, 1.3)
            except Exception as e:
                print(f"Error getting jitter_local for {path_to_use}: {e}")
                jitter_local = np.nan
            
            # Extract shimmer APQ11 - use the sound and point_process
            try:
                shimmer_apq11 = parselmouth.praat.call([sound, point_process], 
                                                      "Get shimmer (local)", 
                                                      0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
            except Exception as e:
                print(f"Error getting shimmer_apq11 for {path_to_use}: {e}")
                shimmer_apq11 = np.nan
                
        except Exception as e:
            print(f"Error creating point process for {path_to_use}: {e}")
            return np.array([np.nan, np.nan], dtype=np.float32)
            
    except Exception as e:
        print(f"Error loading sound for {path_to_use}: {e}")
        return np.array([np.nan, np.nan], dtype=np.float32)

    # Check for invalid values
    feature_values = [jitter_local, shimmer_apq11]
    
    # Replace extreme values with NaN
    for i, val in enumerate(feature_values):
        if val is not None:
            if np.isinf(val) or np.isnan(val) or abs(val) > 1e10:
                feature_values[i] = np.nan
                
    # Convert to a NumPy array
    return np.array(feature_values, dtype=np.float32)

def extract_prosodic_features(audio_path):
    sound = parselmouth.Sound(audio_path)
    duration_sec = sound.duration

    # ------------------------------------------------------
    # 1) Phonation Time, Pauses, Voice Breaks
    # ------------------------------------------------------
    # Compute intensity (Praat method)
    intensity = sound.to_intensity()
    # **1) Dynamic Silence Threshold**
    # Use median intensity of voiced regions instead of a fixed 20 dB threshold
    mean_intensity = call(intensity, "Get mean", 0, 0)
    min_silence_db = mean_intensity - 15  # Adaptive threshold (-15 dB from avg.)
    # **2) Pause Detection Logic**
    min_pause_duration = 0.15  # Ignore pauses shorter than 150 ms
    time_step = 0.01  # Analysis step size (10 ms)
    silence_intervals = []
    is_silent = False
    start_silence = None

    # Iterate through intensity frames to detect pauses
    for t in np.arange(0, duration_sec, time_step):
        current_db = intensity.get_value(time=t) or 0.0

        if current_db < min_silence_db:
            if not is_silent:
                is_silent = True
                start_silence = t  # Pause starts
        else:
            if is_silent:
                end_silence = t  # Pause ends
                pause_duration = end_silence - start_silence

                if pause_duration >= min_pause_duration:
                    silence_intervals.append((start_silence, end_silence))

                is_silent = False

    # **If the file ends in silence**
    if is_silent and (duration_sec - start_silence) >= min_pause_duration:
        silence_intervals.append((start_silence, duration_sec))

    # **3) Compute Final Pause Features**
    num_pauses = len(silence_intervals)  # Number of pauses
    total_pause_duration = np.sum([end - start for start, end in silence_intervals])  # Total pause time
    phonation_time = duration_sec - total_pause_duration  # Phonated (spoken) time

    # ------------------------------------------------------
    # 2) Speech Rate
    # ------------------------------------------------------
    known_syllable_count = 126
    speech_rate = known_syllable_count / phonation_time if phonation_time > 0 else 0.0

    # ------------------------------------------------------
    # 3) Speech Energy & Intensity
    # ------------------------------------------------------    
    # Approximate LTAS:
    spectrum = sound.to_spectrum()
    n_bands = 100
    freq_step = 5000 / n_bands
    band_energies = []
    for b in range(n_bands):
        low_freq = b * freq_step
        hi_freq = (b + 1) * freq_step
        band_energy = call(spectrum, "Get band energy", low_freq, hi_freq)
        band_energies.append(band_energy)
    ltas_mean = np.mean(band_energies)
    ltas_std = np.std(band_energies)

    # Collect features into a list (or tuple) in a consistent order:
    feature_values = [
        num_pauses,
        total_pause_duration,
        phonation_time,
        speech_rate,
        mean_intensity,
        #ltas_mean,
        #ltas_std,
    ]

    # Convert to a NumPy array (float32 for compactness/speed)
    return np.array(feature_values, dtype=np.float32)


def get_data_dir():
    """Helper function to get the consistent data directory path"""
    return myConfig.DATA_DIR


def createDataframe():
    audio_files = []
    labels = []
    relative_paths = []  # Store relative paths
    
    # Always use the Data directory from myConfig
    data_dir = get_data_dir()
    
    for category in myConfig.LABEL_MAP.keys():
        category_path = os.path.join(data_dir, category)
        if not os.path.exists(category_path):
            print(f"Warning: Category directory '{category_path}' not found")
            continue
        
        for file in os.listdir(category_path):
            if file.endswith(".wav"):
                audio_files.append(os.path.join(category_path, file))
                # Store only the relative path (category/filename)
                relative_paths.append(os.path.join(category, file))
                labels.append(myConfig.LABEL_MAP[category])

    if not audio_files:
        print(f"Warning: No audio files found in the data directory")
    
    df = pd.DataFrame({"file_path": relative_paths, "label": labels})
    return df


def extract_sex_age(file_path):    
    try:
        filename = file_path.split("/")[-1]  # Get filename from path
        parts = filename.replace(".wav", "").split("-")  # Split by '-'

        if len(parts) >= 3:  # Ensure we have at least "Class-Sex-Age"
            sex = parts[1]  # Extract sex (M or W)
            age = int(parts[2])  # Convert age to integer
            return sex, age
    except Exception as e:
        print(f"Error extracting from {file_path}: {e}")

    return None, None  # Default if extraction fails


def featureEngineering(data_df):
    """Feature engineering with proper CPU/GPU separation"""
    # Initialize feature columns
    for feature_list in [myConfig.features, myConfig.jitter_shimmer_features, 
                         myConfig.spectral_features, myConfig.speech2text_features]:
        for feature in feature_list:
            data_df[feature] = None
    
    # PHASE 1: CPU-only processing 
    extract_cpu_features(data_df)
    
    # PHASE 2: GPU-dependent processing 
    extract_gpu_features(data_df)
    
    # Post-processing
    # Fill NaN values with the mean for jitter/shimmer features
    for col in myConfig.jitter_shimmer_features:
        if data_df[col].isna().any():
            mean_val = data_df[col].mean()
            data_df[col].fillna(mean_val, inplace=True)
    
    # Drop duplicate columns
    data_df = data_df.loc[:, ~data_df.columns.duplicated()]
    return data_df

def cpu_worker(file_info):
    idx, file_path = file_info
    result = {}
    # Resolve the audio path passed to the extraction functions
    file_path = resolve_audio_path(file_path)

    try:
        # 1. Extract metadata features
        result['duration'] = myAudio.compute_audio_length(file_path)
        result['class'] = extract_class(file_path)
        sex, age = extract_sex_age(file_path)
        result['Sex'] = sex
        result['Age'] = age
                        
        # 2. Process the audio file (if needed)
        base_file_path = file_path.replace("_original", "")
        marker_file = f"{base_file_path}.processed"
        if not os.path.exists(marker_file):
            myAudio.process_audio(file_path)        
        
        # 3. Create sound object once and use for both feature extractions
        sound = parselmouth.Sound(file_path)
        
        # Extract jitter and shimmer 
        jitter_shimmer = extract_jitter_shimmer(sound, file_path)
        for i, feature in enumerate(myConfig.jitter_shimmer_features):
            result[feature] = jitter_shimmer[i]
        
        # Extract spectral features using the same sound object
        spectral = extract_spectral_features(sound)
        for i, feature in enumerate(myConfig.spectral_features):
            result[feature] = spectral[i]
            
        # Explicitly delete sound object to free memory
        del sound
        
    except Exception as e:
        print(f"Error processing CPU features for {file_path}: {str(e)}")
        # Set default values for failed features
        for feature_list in [myConfig.jitter_shimmer_features, myConfig.spectral_features]:
            for feature in feature_list:
                result[feature] = np.nan
        
    return idx, result


def extract_cpu_features(data_df):
    """Extract CPU features"""
    import time
    import signal
    from functools import partial
    
    # Create work items
    work_items = [(idx, path) for idx, path in enumerate(data_df['file_path'])]
    num_workers = max(2, multiprocessing.cpu_count()//4)  
    total_files = len(work_items)
    
    print(f"Extracting CPU-only features using {num_workers} worker processes")
    
    # Set a global timeout handler for worker processes
    def timeout_handler(signum, frame):
        raise TimeoutError("Processing timed out")
    
    # Process in even larger batches to minimize pool creation/destruction
    batch_size = 100
    for batch_start in range(0, total_files, batch_size):
        batch_end = min(batch_start + batch_size, total_files)
        print(f"Processing batch {batch_start//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}")
        
        # Extract just this batch
        batch_items = work_items[batch_start:batch_end]
        
        # Process this batch with a separate pool
        processed_count = 0
        results = []
        
        # Create a fresh pool for this batch
        try:
            # Use the more robust multiprocessing.Pool
            with multiprocessing.Pool(processes=num_workers) as pool:
                # Process files one by one with timeout
                for item in batch_items:
                    try:
                        # Apply with 60 second timeout per file
                        result = pool.apply_async(cpu_worker, (item,))
                        results.append(result.get(timeout=60))  # Wait for this result with timeout
                        
                        # Update progress after each file
                        processed_count += 1
                        if processed_count % 5 == 0:
                            print(f"Processed {processed_count}/{len(batch_items)} in current batch")
                            
                    except Exception as e:
                        print(f"Error processing file: {str(e)}")
                        
                # Explicit cleanup
                pool.close()
                pool.join()
                
        except Exception as e:
            print(f"Pool error: {str(e)}")
                
        # Update dataframe after pool is closed
        for idx, result in results:
            for key, value in result.items():
                data_df.at[idx, key] = value
        
        # Force garbage collection after batch
        gc.collect()
        time.sleep(1)  # Pause between batches to allow OS to reclaim resources
        print(f"Completed batch {batch_start//batch_size + 1}, {batch_end}/{total_files} files processed")
    
    return data_df

def extract_gpu_features(data_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print("Extracting GPU-dependent features:")
    
    """ # 0. Load audio separation model
    model = myAudio.load_demucs_model()
    print("Applying voice separation with Demucs...")
    
    # Optimize batch size based on GPU memory - smaller batches might be faster
    demucs_batch_size = 8  
    total_files = len(data_df)
    
    # Track processing time for performance analysis
    import time
    start_time = time.time()
    total_processed = 0 """
    
    """ # Process files in batches with Demucs
    for i in range(0, total_files, demucs_batch_size):
        batch_start_time = time.time()
        batch_end = min(i + demucs_batch_size, total_files)
        batch_files = data_df.iloc[i:batch_end]
        print(f"Processing Demucs batch {i//demucs_batch_size + 1}/{(total_files + demucs_batch_size - 1)//demucs_batch_size}")
        
        # Pre-resolve all paths at once
        batch_paths = [resolve_audio_path(row['file_path']) for _, row in batch_files.iterrows()]
        
        # Process batch with Demucs
        myAudio.process_batch_with_demucs(batch_paths, model, device)
        
        # Update progress and timing information
        batch_time = time.time() - batch_start_time
        total_processed += len(batch_paths)
        avg_time = batch_time / len(batch_paths)
        elapsed = time.time() - start_time
        remaining = (avg_time * (total_files - total_processed)) if total_processed > 0 else 0
        
        print(f"Batch completed in {batch_time:.1f}s ({avg_time:.1f}s per file)")
        print(f"Overall progress: {total_processed}/{total_files} files")
        print(f"Estimated remaining time: {remaining/60:.1f} minutes")
        
        # Clear CUDA cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache() """
    # 1. Extract VAD-based prosodic features
    print("Extracting VAD-based prosodic features...")
    for idx, row in data_df.iterrows():
        file_path = row['file_path']
        # Resolve the audio path passed to the extraction functions
        file_path = resolve_audio_path(file_path)
        try:
            prosodic = myAudio.extract_prosodic_features_vad(file_path)
            for i, feature in enumerate(myConfig.features):
                data_df.at[idx, feature] = prosodic[i]
        except Exception as e:
            print(f"Error extracting prosodic features for {file_path}: {str(e)}")
            for feature in myConfig.features:
                data_df.at[idx, feature] = np.nan
    
    # 2. Load ASR model for speech-to-text
    print("Loading ASR model...")    
    model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
    asr_model, processor = my_Speech2text.load_asr_model(model_name, device)
    
    # Set model to evaluation mode
    asr_model.eval()
    
    # 3. Extract speech-to-text features in small batches
    print("Extracting speech-to-text features...")
    batch_size = 16  # Ajdust based on GPU memory
    total_files = len(data_df)
    
    with torch.no_grad():
        for i in range(0, total_files, batch_size):
            batch_files = data_df.iloc[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}")
            
            for idx, row in batch_files.iterrows():
                file_path = row['file_path']
                # Resolve the audio path passed to the extraction functions
                file_path = resolve_audio_path(file_path)
                try:
                    wer, transcript = my_Speech2text.extract_speechFromtext(file_path, asr_model, processor)
                    data_df.at[idx, 'wer'] = wer
                    data_df.at[idx, 'transcript'] = transcript
                except Exception as e:
                    print(f"Error in speech-to-text: {str(e)}")
                    data_df.at[idx, 'wer'] = -1
                    data_df.at[idx, 'transcript'] = f"ERROR: {str(e)}"
            
            # Free GPU memory between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def setWeightedCELoss():
    # Compute original class weights
    total_samples = sum(myConfig.num_samples_per_class.values())
    num_classes = len(myConfig.num_samples_per_class)
    class_weights = {cls: total_samples / (num_classes * count) for cls, count in myConfig.num_samples_per_class.items()}
    
    # Manually increase MCI weight
    class_weights[1] *= 2  # Double MCI weight
    class_weights[2] *= 2  # Double AD weight
    
    # Normalize class weights (to prevent excessive imbalance)
    max_weight = max(class_weights.values())
    class_weights = {cls: weight / max_weight for cls, weight in class_weights.items()}
    
    # Convert to PyTorch tensor - but don't move to device yet
    weights_tensor = torch.tensor([class_weights[0], class_weights[1], class_weights[2]], dtype=torch.float)
    
    # Let the Trainer move this to the appropriate device with the model
    return None, weights_tensor  # Return None for criterion as it will be created in CustomTrainer


def createAgeSexStats(data_df):
    # Count the number of men (M) and women (W)
    total_men = (data_df["Sex"] == "M").sum()
    total_women = (data_df["Sex"] == "W").sum()

    print(f"Total number of men: {total_men}")
    print(f"Total number of women: {total_women}")

    most_prevalent_sex = data_df.groupby("class")["Sex"].agg(lambda x: x.mode()[0])
    print("Most prevalent sex per class")
    print(most_prevalent_sex)

def resolve_audio_path(relative_path):
    """
    Convert relative audio file path to absolute path based on environment
    
    Args:
        relative_path (str): Relative path like 'Healthy/HC-W-79-180.wav'
        
    Returns:
        str: Absolute path to the audio file
    """
    data_dir = get_data_dir()
    return os.path.join(data_dir, relative_path)

def convert_absolute_to_relative_paths(dataframe):
    """
    Convert absolute paths in dataframe to relative paths
    
    Args:
        dataframe (pd.DataFrame): Dataframe with absolute paths
        
    Returns:
        pd.DataFrame: Dataframe with relative paths
    """
    # Function to extract relative path
    def extract_relative_path(abs_path):
        # Look for the pattern "Data/Category/filename.wav"
        parts = abs_path.split('/')
        try:
            # Find "Data" directory in the path
            data_idx = parts.index("Data")
            # Return "Category/filename.wav"
            return '/'.join(parts[data_idx+1:])
        except ValueError:
            # If "Data" not found, just return the filename
            return os.path.basename(abs_path)
    
    # Make a copy to avoid modifying the original
    df_copy = dataframe.copy()
    # Apply conversion to file_path column
    df_copy['file_path'] = df_copy['file_path'].apply(extract_relative_path)
    
    return df_copy

