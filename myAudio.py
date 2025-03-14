import librosa
import noisereduce as nr
import soundfile as sf
import functions
import numpy as np


def load_audio(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=target_sr)
    return np.array(audio, dtype=np.float32), sr  # Ensure float32 output


def chunk_audio(example, max_length=16000*100):  # 100 seconds max
    audio = example["audio"]
    if len(audio) > max_length:
        example["audio"] = audio[:max_length]
    return myFunctions.preprocess_function(example)


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


# Applies noise reduction and normalization to the audio file
def process_audio(file_path, sr=16000):
    audio, _ = load_audio(file_path, sr)
    target_rms = 0.05
    noise_reduced_audio = apply_noise_reduction(audio, sr)
    rms = np.sqrt(np.mean(noise_reduced_audio**2))
    audio_normalized = noise_reduced_audio * (target_rms / rms)
    sf.write(file_path, audio_normalized, sr)
