import os
import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
from pyannote.audio import Pipeline
import parselmouth
from parselmouth.praat import call
import torch

# Load the pretrained voice activity detection model
vad = Pipeline.from_pretrained("pyannote/voice-activity-detection")
# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vad.to(device)


def load_audio(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=target_sr)
    return np.array(audio, dtype=np.float32), sr  # Ensure float32 output


# Function to compute audio duration
def compute_audio_length(file_path):
    y, sr = librosa.load(file_path, sr=None)  # Load audio
    return len(y) / sr  # Compute duration in seconds


def apply_noise_reduction(audio, sr=16000):
    # Parameters tuned for elderly voice recordings with background noise
    # Use stationary noise reduction for consistent background noise
    # Higher prop_decrease for stronger noise reduction
    # Smaller chunk size to better capture variations in voice patterns
    return nr.reduce_noise(
        y=audio, 
        sr=sr,
        stationary=True,
        prop_decrease=0.75,
        n_std_thresh_stationary=1.5,
        chunk_size=4096,
        padding=1024,
        use_tqdm=True, 
        n_jobs=-1
    )


def process_audio(file_path, sr=16000):
    """
    Applies noise reduction and normalization to the audio file
    Creates both processed and original versions
    Uses marker files to prevent repeated processing
    """
    # Standardize path handling
    base_file_path = file_path.replace("_original", "")
    
    # Check if this file has already been processed
    marker_file = f"{base_file_path}.processed"    
    
    # If marker exists, assume already processed
    if os.path.exists(marker_file) :
        return
    
    # Load the audio from original path
    audio, _ = load_audio(file_path, sr)
    
    # Apply processing
    target_rms = 0.05
    noise_reduced_audio = apply_noise_reduction(audio, sr)
    rms = np.sqrt(np.mean(noise_reduced_audio**2))
    audio_normalized = noise_reduced_audio * (target_rms / rms)
    
    # Always write to the base filename without suffixes
    sf.write(base_file_path, audio_normalized, sr)
    
    # Create marker file
    with open(marker_file, 'w') as f:
        f.write("1")


def pyannoteExtractProsodic(speech_segments):
    phonation_time = sum(seg.end - seg.start for seg in speech_segments.get_timeline())
    pauses = [(start, end) for (start, end) in speech_segments.get_timeline().gaps()]
    pause_durations = [end-start for start, end in pauses]
    
    denominator = phonation_time + sum(pause_durations)
    if denominator > 0:
        speech_rate = phonation_time / denominator
    else:
        speech_rate = 0.0
        
    return phonation_time, len(pauses), sum(pause_durations), speech_rate


def extract_prosodic_features_vad(audio_path):
    sound = parselmouth.Sound(audio_path)
    rms_amplitude = sound.get_root_mean_square()
    
    # Handle if rms_amplitude is an array
    if isinstance(rms_amplitude, np.ndarray) and rms_amplitude.size > 1:
        rms_amplitude = np.mean(rms_amplitude)
    
    # Use np.max for safe handling of arrays
    max_amplitude = np.max(np.abs(sound.values))
    
    # Now both should be scalars
    if rms_amplitude == 0.0 or max_amplitude == 0.0:
        crestFactor_dB = 0.0
    else:
        crestFactor_dB = 20 * np.log10(max_amplitude / rms_amplitude)
        
    speech_segments = vad(audio_path)
    phonation_time, pauses, pause_durations, speech_rate = pyannoteExtractProsodic(speech_segments)

    feature_values = [
        pauses,
        pause_durations,
        phonation_time,
        speech_rate,
        crestFactor_dB,
    ]
    # Convert to a NumPy array (float32 for compactness/speed)
    return np.array(feature_values, dtype=np.float32)

# diarization = pipeline("audio_normalized.wav")
""" vad = Pipeline.from_pretrained("pyannote/voice-activity-detection")
speech_segments = vad('/home/bosh/Documents/ML/zz_PP/00_SCTI/Repo/Data/MCI/MCI-W-85-58.wav')
phonation_time, pauses, pause_durations = pyannoteExtractProsodic(speech_segments)
print(f"Phonation Time: {phonation_time:.2f} seconds")
# Total number of pauses and the total pause duration
print(f"Total Number of Pauses: {len(pauses)}")
print(f"Total Pause Duration: {sum(pause_durations):.2f} seconds") """
