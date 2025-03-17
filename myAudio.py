import librosa
import noisereduce as nr
import soundfile as sf
import myConfig
import numpy as np
from pyannote.audio import Pipeline
#import torch


#pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=myConfig.hf_token)
vad_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token={myConfig.hf_token})

#pipeline.to(torch.device("cuda"))
#vad_pipeline.to(torch.device("cuda"))


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


# Applies noise reduction and normalization to the audio file
def process_audio(file_path, sr=16000):
    audio, _ = load_audio(file_path, sr)
    target_rms = 0.05
    noise_reduced_audio = apply_noise_reduction(audio, sr)
    rms = np.sqrt(np.mean(noise_reduced_audio**2))
    audio_normalized = noise_reduced_audio * (target_rms / rms)
    sf.write(file_path, audio_normalized, sr)


def pyannoteExtractProsodic(speech_segments):
    phonation_time = sum(seg.end - seg.start for seg in speech_segments.get_timeline())
    pauses = [(start, end) for (start, end) in speech_segments.get_timeline().gaps()]
    pause_durations = [end-start for start, end in pauses]
    return phonation_time, pauses, pause_durations


# diarization = pipeline("audio_normalized.wav")
vad = Pipeline.from_pretrained("pyannote/voice-activity-detection")
speech_segments = vad('/home/bosh/Documents/ML/zz_PP/00_SCTI/Repo/Data/MCI/MCI-W-85-58.wav')
phonation_time, pauses, pause_durations, speech_rate = pyannoteExtractProsodic(speech_segments)
print(f"Phonation Time: {phonation_time:.2f} seconds")
# Total number of pauses and the total pause duration
print(f"Total Number of Pauses: {len(pauses)}")
print(f"Total Pause Duration: {sum(pause_durations):.2f} seconds")
