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
from torch.nn.utils.rnn import pad_sequence


def preprocess_function(example):
    # Wav2Vec2 processing
    _, processor, _  = myModel.getModelDefinitions()
    inputs = processor(example["audio"]["array"], sampling_rate=example["audio"]["sampling_rate"], return_tensors="pt")
    
    # Build prosodic features including jitter and shimmer
    numeric_cols = ["Age", "duration", "num_pauses", "total_pause_duration", 
                   "phonation_time", "speech_rate", "dynamic_range_db", 
                   "jitter_local", "shimmer_apq11", "skewness", "centre_of_gravity"]  
                
                   
    feats = [example[col] for col in numeric_cols]
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


def extract_spectral_features(audio_path):
    power = 2.
    sound = parselmouth.Sound(audio_path)
    spectrum = call(sound, "To Spectrum", "yes")    
    centre_of_gravity = call(spectrum, "Get centre of gravity",power)
    skewness = call(spectrum, "Get skewness",power)
    feature_values = [skewness, centre_of_gravity]
    return np.array(feature_values, dtype=np.float32)


def extract_jitter_shimmer(audio_path):
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
        sound = parselmouth.Sound(path_to_use)        
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


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    # Ensure predictions and labels are NumPy arrays
    preds = torch.argmax(torch.tensor(logits), dim=-1).numpy()

    # Compute classification report
    report = classification_report(labels, preds, target_names=["Healthy", "MCI", "AD"], output_dict=True)

    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)

    # Extract TP, FP, TN, FN per class
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    # Compute Specificity (TNR) and Negative Predictive Value (NPV) per class
    specificity = TN / (TN + FP + 1e-10)  # True Negative Rate
    npv = TN / (TN + FN + 1e-10)  # Negative Predictive Value

    # Store per-class and macro-average results
    results = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_specificity": np.mean(specificity),
        "macro_npv": np.mean(npv),
        "f1_healthy": report["Healthy"]["f1-score"],
        "f1_mci": report["MCI"]["f1-score"],
        "f1_ad": report["AD"]["f1-score"],
        "specificity_healthy": specificity[0],
        "specificity_mci": specificity[1],
        "specificity_ad": specificity[2],
        "npv_healthy": npv[0],
        "npv_mci": npv[1],
        "npv_ad": npv[2]
    }
    return results


def data_collator_fn(processor, features):
    waveforms = [torch.tensor(f["audio"]["array"]) for f in features]
    prosodic_features = torch.stack([
        torch.tensor(f["prosodic_features"], dtype=torch.float) for f in features
    ])  # Now each prosodic_features is converted to a tensor
    labels = torch.tensor([f["label"] for f in features])

    input_values = pad_sequence(waveforms, batch_first=True, padding_value=0)

    inputs = processor(
        input_values.numpy(),
        sampling_rate=16000,
        padding=True,
        return_tensors="pt"
    )
    inputs["labels"] = labels
    inputs["prosodic_features"] = prosodic_features  # Add prosodic features
    return inputs


def get_data_dir():
    """Helper function to get the consistent data directory path"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")


def createDataframe():
    audio_files = []
    labels = []
    
    # Always use the Data directory at script level
    data_dir = get_data_dir()
    
    for category in myConfig.LABEL_MAP.keys():
        category_path = os.path.join(data_dir, category)
        if not os.path.exists(category_path):
            print(f"Warning: Category directory '{category_path}' not found")
            continue
        
        for file in os.listdir(category_path):
            if file.endswith(".wav"):
                audio_files.append(os.path.join(category_path, file))
                labels.append(myConfig.LABEL_MAP[category])

    if not audio_files:
        print(f"Warning: No audio files found in the data directory")
    
    df = pd.DataFrame({"file_path": audio_files, "label": labels})
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
    # Add duration and class columns to the dataframe
    data_df["duration"] = data_df["file_path"].apply(myAudio.compute_audio_length)
    data_df["class"] = data_df["file_path"].apply(extract_class)
    # Extract and add Sex and Age columns to the dataframe
    data_df["Sex"], data_df["Age"] = zip(*data_df["file_path"].apply(extract_sex_age))
    # Remove possible duplicates
    data_df = data_df.loc[:, ~data_df.columns.duplicated()]
    # Extract jitter and shimmer features with unprocessed audio
    jitter_shimmer_features = data_df["file_path"].apply(extract_jitter_shimmer)    
    # Apply noise filtering and normalization to all audio files 
    data_df["file_path"].apply(myAudio.process_audio)
    # Extract prosodic features    
    prosodic_features = data_df["file_path"].apply(myAudio.extract_prosodic_features_vad)    
    # Extract spectral features
    spectral_features = data_df["file_path"].apply(extract_spectral_features)
    # Convert the extracted prosodic feature arrays into separate columns
    prosodic_df = pd.DataFrame(prosodic_features.tolist(), columns=myConfig.features)
    # Convert the extracted jitter and shimmer feature arrays into separate columns
    jitter_shimmer_df = pd.DataFrame(jitter_shimmer_features.tolist(), columns=myConfig.jitter_shimmer_features)
    # Convert the extracted spectral feature arrays into separate columns
    spectral_df = pd.DataFrame(spectral_features.tolist(), columns=myConfig.spectral_features)
    # Call speech2text to extract text from audio and calculate WER (potentially also speech rate)
    speech2text_df = data_df["file_path"].apply(my_Speech2text.extract_speechFromtext)
    speech2text_df = pd.DataFrame(speech2text_df.tolist(),columns=["wer", "transcript"])
    # Merge the extracted features with the main DataFrame
    data_df = pd.concat([data_df, prosodic_df, jitter_shimmer_df, spectral_df, speech2text_df], axis=1)    
    
    # Fill NaN values with the mean for jitter/shimmer features
    for col in myConfig.jitter_shimmer_features:
        if data_df[col].isna().any():
            mean_val = data_df[col].mean()
            data_df[col].fillna(mean_val, inplace=True)
    
    # Drop duplicate columns (keep only one occurrence of each feature)
    data_df = data_df.loc[:, ~data_df.columns.duplicated()]    
    return data_df


def setWeightedCELoss():
    # Compute original class weights
    total_samples = sum(myConfig.num_samples_per_class.values())
    num_classes = len(myConfig.num_samples_per_class)
    class_weights = {cls: total_samples / (num_classes * count) for cls, count in myConfig.num_samples_per_class.items()}
    #  Manually increase MCI weight
    class_weights[1] *= 2  # Double MCI weight
    class_weights[2] *= 2  # Double AD weight
    #  Normalize class weights (to prevent excessive imbalance)
    max_weight = max(class_weights.values())
    class_weights = {cls: weight / max_weight for cls, weight in class_weights.items()}
    # Convert to PyTorch tensor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_tensor = torch.tensor([class_weights[0], class_weights[1], class_weights[2]], dtype=torch.float).to(device)
    # Use in CrossEntropyLoss
    criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor)
    return criterion, weights_tensor


def createAgeSexStats(data_df):
    # Count the number of men (M) and women (W)
    total_men = (data_df["Sex"] == "M").sum()
    total_women = (data_df["Sex"] == "W").sum()

    print(f"Total number of men: {total_men}")
    print(f"Total number of women: {total_women}")

    most_prevalent_sex = data_df.groupby("class")["Sex"].agg(lambda x: x.mode()[0])
    print("Most prevalent sex per class")
    print(most_prevalent_sex)

