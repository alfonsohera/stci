from pyannote.audio import Pipeline
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import jiwer
import myAudio
import pandas as pd
import numpy as np
from Levenshtein import distance
# Install editdistance: pip install editdistance
import editdistance

vad = Pipeline.from_pretrained("pyannote/voice-activity-detection")
# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vad.to(device)

def compute_wer(audio_file, reference_text, asr_model, processor):
    """Compute Word Error Rate between reference text and ASR output."""
    try:
        # 1) Voice Activity Detection
        speech_segments = vad(audio_file)
        
        # 2) Extract each speech segment as WAV and run ASR
        transcript = []
        
        # Check if we have any valid segments
        segment_count = 0
        for segment, _, _ in speech_segments.itertracks(yield_label=True):
            segment_count += 1
            start_time = segment.start
            end_time = segment.end
            
            # Skip very short segments (likely noise)
            if end_time - start_time < 0.1:
                continue
                
            try:
                # Load audio segment
                audio_chunk = myAudio.load_audio_segment(audio_file, start_time, end_time)
                
                # Skip empty or very small segments
                if len(audio_chunk) < 160:  # Less than 10ms at 16kHz
                    continue
                    
                # Process the audio with the processor - DON'T move to device here
                input_values = processor(
                    audio_chunk, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).input_values
                
                # Move to device only once before passing to model
                input_values = input_values.to(device)
                
                # Process through model
                with torch.no_grad():
                    logits = asr_model(input_values).logits
                    
                # Decode predictions
                predicted_ids = torch.argmax(logits, dim=-1)
                predicted_text = processor.batch_decode(predicted_ids)[0]
                
                # Only add non-empty text
                if predicted_text.strip():
                    transcript.append(predicted_text)
            except Exception as e:
                print(f"Error processing segment {start_time:.2f}-{end_time:.2f}: {str(e)}")
                continue
        
        # If no valid transcription was generated
        if not transcript or segment_count == 0:
            print(f"No valid speech segments found in {audio_file}")
            return 1.0  # Return worst possible WER
            
        # Combine segments
        full_hypothesis = " ".join(transcript).lower()
        reference_clean = reference_text.lower()
        
        # Handle empty transcription
        if not full_hypothesis.strip():
            return 1.0
            
        # 3) Compute WER with word-based split
        try:
            # First, clean up the text by lowercasing and removing extra spaces
            reference_clean = reference_clean.lower().strip()
            full_hypothesis = full_hypothesis.lower().strip()
            
            # Apply manual cleaning to both strings
            import re
            def clean_text(text):
                # Remove punctuation and normalize spaces
                text = re.sub(r'[^\w\s]', '', text)
                text = re.sub(r'\s+', ' ', text)
                return text.strip()
            
            reference_clean = clean_text(reference_clean)
            full_hypothesis = clean_text(full_hypothesis)
            
            # Split into words
            reference_words = reference_clean.split()
            hypothesis_words = full_hypothesis.split()                                    
            
            # Compute Levenshtein distance (minimum number of single-character edits)
            word_edits = editdistance.eval(reference_words, hypothesis_words)
            wer = word_edits / len(reference_words)
            # This will be between 0.0 and potentially >1.0, so cap it:
            wer = min(wer, 1.0)
            return min(wer, 1.0)  # Cap at 1.0 (100% error)
        except Exception as e:
            print(f"WER calculation error: {str(e)}")
            return 1.0  # Return worst possible WER on error
    except Exception as e:
        print(f"Overall WER computation error: {str(e)}")
        return 1.0  # Return worst possible WER on error

def compute_wer_with_transcript(audio_file, reference_text, asr_model, processor):
    """Compute Word Error Rate and return transcript between reference text and ASR output."""
    try:
        # 1) Voice Activity Detection
        speech_segments = vad(audio_file)
        
        # 2) Extract each speech segment as WAV and run ASR
        transcript = []
        
        # Check if we have any valid segments
        segment_count = 0
        for segment, _, _ in speech_segments.itertracks(yield_label=True):
            segment_count += 1
            start_time = segment.start
            end_time = segment.end
            
            # Skip very short segments (likely noise)
            if end_time - start_time < 0.1:
                continue
                
            try:
                # Load audio segment
                audio_chunk = myAudio.load_audio_segment(audio_file, start_time, end_time)
                
                # Skip empty or very small segments
                if len(audio_chunk) < 160:  # Less than 10ms at 16kHz
                    continue
                    
                # Process the audio with the processor - DON'T move to device here
                input_values = processor(
                    audio_chunk, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).input_values
                
                # Move to device only once before passing to model
                input_values = input_values.to(device)
                
                # Process through model
                with torch.no_grad():
                    logits = asr_model(input_values).logits
                    
                # Decode predictions
                predicted_ids = torch.argmax(logits, dim=-1)
                predicted_text = processor.batch_decode(predicted_ids)[0]
                
                # Only add non-empty text
                if predicted_text.strip():
                    transcript.append(predicted_text)
            except Exception as e:
                print(f"Error processing segment {start_time:.2f}-{end_time:.2f}: {str(e)}")
                continue
        
        # If no valid transcription was generated
        if not transcript or segment_count == 0:
            print(f"No valid speech segments found in {audio_file}")
            return 1.0, ""  # Return worst possible WER and empty transcript
            
        # Combine segments
        full_hypothesis = " ".join(transcript).lower()
        reference_clean = reference_text.lower()
        
        # Handle empty transcription
        if not full_hypothesis.strip():
            return 1.0, ""
            
        # 3) Compute WER with word-based split
        try:
            # First, clean up the text by lowercasing and removing extra spaces
            reference_clean = reference_clean.lower().strip()
            full_hypothesis = full_hypothesis.lower().strip()
            
            # Apply manual cleaning to both strings
            import re
            def clean_text(text):
                # Remove punctuation and normalize spaces
                text = re.sub(r'[^\w\s]', '', text)
                text = re.sub(r'\s+', ' ', text)
                return text.strip()
            
            reference_clean = clean_text(reference_clean)
            full_hypothesis = clean_text(full_hypothesis)
            
            # Split into words
            reference_words = reference_clean.split()
            hypothesis_words = full_hypothesis.split()                                    
            
            # Compute Levenshtein distance (minimum number of single-character edits)
            word_edits = editdistance.eval(reference_words, hypothesis_words)
            wer = word_edits / len(reference_words)
            # This will be between 0.0 and potentially >1.0, so cap it:
            wer = min(wer, 1.0)
            return min(wer, 1.0), full_hypothesis  # Cap at 1.0 (100% error) and return transcript
        except Exception as e:
            print(f"WER calculation error: {str(e)}")
            return 1.0, ""  # Return worst possible WER and empty transcript on error
    except Exception as e:
        print(f"Overall WER computation error: {str(e)}")
        return 1.0, ""  # Return worst possible WER and empty transcript on error

reference_text = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, " \
    "no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, " \
    "rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, " \
    "duelos y quebrantos los sábados, lantejas los viernes, algún palomino de añadidura los domingos, " \
    "consumían las tres partes de su hacienda"

# Load ASR model and move it to the same device
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
asr_model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)  # Add .to(device)
processor = Wav2Vec2Processor.from_pretrained(model_name)

data_df = pd.read_csv('/home/bosh/Documents/ML/zz_PP/00_SCTI/Repo/Data/dataframe.csv')

# Process each audio file
scores = []
# Create new columns for WER scores and transcripts
data_df['wer_score'] = None
data_df['transcript'] = None

#count = 0
for index, row in data_df.iterrows():
    audio_file = row['file_path']
    print(f"Processing file {index+1}/{len(data_df)}: {audio_file}")
    try:
        # Modified compute_wer function to return both WER and transcript
        reading_wer, transcript = compute_wer_with_transcript(audio_file, reference_text, asr_model, processor)
        # Add WER score and transcript to the dataframe
        data_df.at[index, 'wer_score'] = reading_wer
        data_df.at[index, 'transcript'] = transcript
        scores.append((audio_file, reading_wer))
    except Exception as e:
        print(f"Error processing {audio_file}: {str(e)}")
        data_df.at[index, 'wer_score'] = -1  # Mark as error
        data_df.at[index, 'transcript'] = "ERROR: " + str(e)
    
"""     # Increment the counter and check if we've processed 10 samples
    count += 1
    if count >= 10:
        break
 """
# Save the updated dataframe with WER scores and transcripts
output_path = '/home/bosh/Documents/ML/zz_PP/00_SCTI/Repo/Data/dataframe_with_transcripts.csv'
data_df.to_csv(output_path, index=False)
print(f"\nSaved dataframe with WER scores and transcripts to {output_path}")

# Sort scores by WER (ascending)
scores.sort(key=lambda x: x[1])

# Print the results
print("\n=== WER Results (sorted by performance) ===")
for file, wer in scores:
    print(f"File: {file}, WER: {wer:.4f}")
