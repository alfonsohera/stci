import os
import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
from pyannote.audio import Pipeline
import parselmouth
from parselmouth.praat import call
import torch
import myPlots
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio

# Load the Hugging Face authentication token from the environment (The variable needs to be set first!)
hf_token = os.environ.get('HF_TOKEN')
# Load the pretrained voice activity detection model
vad = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=hf_token)
# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vad.to(device)


def load_audio(file_path, target_sr=16000):        
    audio, sr = librosa.load(file_path, sr=target_sr)
    return np.array(audio, dtype=np.float32), sr  # Ensure float32 output


def load_audio_segment(file_path, start_time, end_time, target_sr=16000):
    """
    Load a specific segment of an audio file between start_time and end_time.
    
    Args:
        file_path (str): Path to the audio file
        start_time (float): Start time of the segment in seconds
        end_time (float): End time of the segment in seconds
        target_sr (int): Target sample rate
    
    Returns:
        numpy.ndarray: Audio segment as float32 NumPy array
    """
    # Load the complete audio file    
    audio, sr = load_audio(file_path, target_sr)
    
    # Calculate segment indices
    start_idx = int(start_time * sr)
    end_idx = int(end_time * sr)
    
    # Ensure indices are within bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(audio), end_idx)
    
    # Extract the segment
    segment = audio[start_idx:end_idx]
    
    return segment  # Return just the array as the tokenizer expects a single array


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
    intensity = sound.to_intensity(time_step=0.01)
    # --- Retrieve min/max dB (using Parabolic interpolation) ---
    min_intensity = call(intensity, "Get minimum", 0, 0, "Parabolic")
    max_intensity = call(intensity, "Get maximum", 0, 0, "Parabolic")
    # --- Compute dynamic range in dB ---
    dynamic_range_db = max_intensity - min_intensity
        
    speech_segments = vad(audio_path)
    phonation_time, pauses, pause_durations, speech_rate = pyannoteExtractProsodic(speech_segments)

    feature_values = [
        pauses,
        pause_durations,
        phonation_time,
        speech_rate,
        dynamic_range_db,
    ]
    # Convert to a NumPy array (float32 for compactness/speed)
    return np.array(feature_values, dtype=np.float32)


def load_demucs_model():
    model = get_model("htdemucs_ft")
    model.to(device)
    return model


def separate_with_demucs(file_path, model, device, target_sr=16000):
    """Use Facebook's Demucs for better voice separation."""    
    
    # Load audio
    waveform, sr = torchaudio.load(file_path)
    
    # Convert mono to stereo if needed
    if waveform.size(0) == 1:  # If mono (1 channel)
        waveform = waveform.repeat(2, 1)  # Duplicate the channel to make stereo
    
    if sr != model.samplerate:
        waveform = torchaudio.functional.resample(waveform, sr, model.samplerate)
    
    # Separate sources
    with torch.inference_mode():
        sources = apply_model(model, waveform.unsqueeze(0), device=device)[0]
    
    # Get just the vocals
    vocals = sources[model.sources.index('vocals')]
    
    # Convert to target sample rate if needed
    if model.samplerate != target_sr:
        vocals = torchaudio.functional.resample(vocals, model.samplerate, target_sr)
    
    # Save to file        
    vocals_numpy = vocals.cpu().numpy().mean(axis=0)  # Convert to mono
    sf.write(file_path, vocals_numpy, target_sr)
    
    return 


def process_batch_with_demucs(file_paths, model, device, target_sr=16000):
    """Process multiple audio files in parallel with Demucs"""
    import os
    
    # Group files by similar length to minimize padding waste
    file_lengths = []
    for file_path in file_paths:
        try:
            info = torchaudio.info(file_path)
            file_lengths.append((file_path, info.num_frames))
        except Exception as e:
            print(f"Error getting info for {file_path}: {e}")
            file_lengths.append((file_path, 0))
    
    # Sort by length and process in sub-batches of similar length files
    file_lengths.sort(key=lambda x: x[1])
    
    # Use smaller sub-batches for better memory efficiency
    sub_batch_size = 2  # Process just 2 files at a time for better memory management
    
    for i in range(0, len(file_lengths), sub_batch_size):
        sub_batch_files = [f[0] for f in file_lengths[i:i+sub_batch_size]]
        print(f"  Processing sub-batch {i//sub_batch_size + 1}/{(len(file_lengths) + sub_batch_size - 1)//sub_batch_size}")
        
        # Load sub-batch of files
        waveforms = []
        for file_path in sub_batch_files:
            try:
                waveform, sr = torchaudio.load(file_path)
                
                # Convert mono to stereo if needed
                if waveform.size(0) == 1:
                    waveform = waveform.repeat(2, 1)
                    
                if sr != model.samplerate:
                    waveform = torchaudio.functional.resample(waveform, sr, model.samplerate)
                    
                waveforms.append(waveform)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if not waveforms:
            continue
            
        try:
            # Pad to same length if needed (within this smaller sub-batch)
            max_length = max(w.shape[1] for w in waveforms)
            padded_waveforms = []
            
            for waveform in waveforms:
                if waveform.shape[1] < max_length:
                    padding = torch.zeros(2, max_length - waveform.shape[1], device=waveform.device)
                    padded_waveform = torch.cat([waveform, padding], dim=1)
                    padded_waveforms.append(padded_waveform)
                else:
                    padded_waveforms.append(waveform)
            
            # Stack sub-batch
            batch_waveforms = torch.stack(padded_waveforms).to(device)
            
            # Use mixed precision for faster processing
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    sources = apply_model(model, batch_waveforms, device=device)
            else:
                sources = apply_model(model, batch_waveforms, device=device)
                
            # Process results and save files
            for j, file_path in enumerate(sub_batch_files):
                if j < len(sources):
                    vocals = sources[j][model.sources.index('vocals')]
                    
                    # Convert to target sample rate if needed
                    if model.samplerate != target_sr:
                        vocals = torchaudio.functional.resample(vocals, model.samplerate, target_sr)
                    
                    # Save the separated vocals (convert to mono)
                    vocals_numpy = vocals.cpu().numpy().mean(axis=0)
                    sf.write(file_path, vocals_numpy, target_sr)
                    
            # Explicitly free memory
            del batch_waveforms, sources, padded_waveforms
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error in sub-batch processing: {e}")
            # Fall back to individual processing
            for file_path in sub_batch_files:
                try:
                    separate_with_demucs(file_path, model, device)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    file = "MCI-W-67-123"
    ext = ".wav"
    suffix = "_demucs"
    orig_file_path = "/home/bosh/Documents/ML/zz_PP/00_SCTI/Repo/Data/MCI/" + file + ext
    proc_file_path = "/home/bosh/Documents/ML/zz_PP/00_SCTI/Repo/Data/MCI/separated/" + file + suffix + ext
    separate_with_demucs(orig_file_path)
    myPlots.visualize_audio_comparison(orig_file_path, proc_file_path)


    # diarization = pipeline("audio_normalized.wav")
    """ vad = Pipeline.from_pretrained("pyannote/voice-activity-detection")
    speech_segments = vad('/home/bosh/Documents/ML/zz_PP/00_SCTI/Repo/Data/MCI/MCI-W-85-58.wav')
    phonation_time, pauses, pause_durations = pyannoteExtractProsodic(speech_segments)
    print(f"Phonation Time: {phonation_time:.2f} seconds")
    # Total number of pauses and the total pause duration
    print(f"Total Number of Pauses: {len(pauses)}")
    print(f"Total Pause Duration: {sum(pause_durations):.2f} seconds") """
