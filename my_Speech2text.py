from pyannote.audio import Pipeline
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import myAudio
import myConfig
import myFunctions
import pandas as pd
import editdistance
import re

vad = Pipeline.from_pretrained("pyannote/voice-activity-detection")
# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vad.to(device)


def load_asr_model(model_name, device):
    # Load ASR model   
    asr_model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    return asr_model, processor


def clean_text(text):
    # Remove punctuation and normalize spaces
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def compute_wer_with_transcript(file_path, reference_text, asr_model, processor):
    """Compute Word Error Rate and return transcript between reference text and ASR output."""    
    
    try:
        # 1) Voice Activity Detection
        speech_segments = vad(file_path)                
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
                audio_chunk = myAudio.load_audio_segment(file_path, start_time, end_time)                
                # Skip empty or very small segments
                if len(audio_chunk) < 160:  # Less than 10ms at 16kHz
                    continue                    
                # Process the audio with the model's processor
                input_values = processor(
                    audio_chunk,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_values                
                # Move to device 
                input_values = input_values.to(device)                
                # Process through ASR model
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
            print(f"No valid speech segments found in {file_path}")
            return 1.0, ""  # Return worst possible WER and empty transcript            
        # Combine segments
        audio_transcript = " ".join(transcript).lower()
        reference_clean = reference_text.lower()
        # Handle empty transcription
        if not audio_transcript.strip():
            return 1.0, ""
        # Do word-based split
        try:
            # First, clean up the text by lowercasing and removing extra spaces
            reference_clean = reference_clean.lower().strip()
            audio_transcript = audio_transcript.lower().strip()
            # Apply manual cleaning to both strings
            reference_clean = clean_text(reference_clean)
            audio_transcript = clean_text(audio_transcript)
            # Split into words
            reference_words = reference_clean.split()
            hypothesis_words = audio_transcript.split()
            # Compute word-level edit distance
            word_edits = editdistance.eval(reference_words, hypothesis_words)
            wer = word_edits / len(reference_words)
            # This will be between 0.0 and potentially >1.0, so cap it:
            wer = min(wer, 1.0)
            return min(wer, 1.0), audio_transcript  # Cap at 1.0 (100% error) and return transcript
        except Exception as e:
            print(f"WER calculation error: {str(e)}")
            return 1.0, ""  # Return worst possible WER and empty transcript on error
    except Exception as e:
        print(f"Overall WER computation error: {str(e)}")
        return 1.0, ""  # Return worst possible WER and empty transcript on error


def extract_speechFromtext(audio_path, asr_model, processor):        
    try:
        wer, transcript = compute_wer_with_transcript(
            audio_path,
            myConfig.reference_text,
            asr_model,
            processor
        )
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        wer = -1
        transcript = f"ERROR: {str(e)}"
    return wer, transcript


if __name__ == "__main__":

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
        file_path = myFunctions.resolve_audio_path(audio_file)
        print(f"Processing file {index+1}/{len(data_df)}: {file_path}")
        try:
            # Modified compute_wer function to return both WER and transcript
            reading_wer, transcript = compute_wer_with_transcript(file_path, myConfig.reference_text, asr_model, processor)
            # Add WER score and transcript to the dataframe
            data_df.at[index, 'wer_score'] = reading_wer
            data_df.at[index, 'transcript'] = transcript
            scores.append((file_path, reading_wer))
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
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
