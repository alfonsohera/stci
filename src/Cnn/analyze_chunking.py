import os
import torch
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import csv

# Local imports
import myConfig
from cnn_rnn_data import chunk_audio, load_cached_pytorch_dataset

def analyze_all_files(dataset_path=None, backup_dataset_path=None, sample_rate=16000):
    """
    Analyze all audio files in the dataset and output results to CSV
    
    Args:
        dataset_path: Path to the dataset directory (default: None, uses current dataset)
        backup_dataset_path: Path to the backup dataset directory (default: None, uses current backup dataset)
        sample_rate: Sample rate of the audio
    """
    # Configure paths
    myConfig.configure_paths()
    
    # Load cached PyTorch dataset
    if dataset_path is None:
        pytorch_dataset_path = os.path.join(myConfig.DATA_DIR, "pytorch_dataset")
    else:
        pytorch_dataset_path = dataset_path
        
    print(f"Loading dataset from {pytorch_dataset_path}")
    dataset = load_cached_pytorch_dataset(pytorch_dataset_path)
    
    # Create output directory
    output_dir = os.path.join(myConfig.OUTPUT_PATH, "chunking_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a unique name for the CSV file based on the path
    if dataset_path is None:
        csv_filename = "chunking_analysis_current_dataset.csv"
    else:
        # Extract last part of the path for naming
        path_parts = dataset_path.rstrip('/').split('/')
        csv_filename = f"chunking_analysis_{path_parts[-1]}.csv"
    
    csv_path = os.path.join(output_dir, csv_filename)
    
    # Fields for CSV
    fields = [
        "split", "index", "file_path", "label", 
        "original_duration", "zero_percentage",
        "num_samples", "num_zeros",
        "num_chunks_min5s", "num_chunks_min2s", "num_chunks_no_min",
        "silent_chunk_count"
    ]
    
    print(f"Analyzing all files and writing results to {csv_path}")
    
    # Open CSV file for writing
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        
        # Process each split
        for split_name in ["train", "validation", "test"]:
            if split_name not in dataset:
                continue
                
            split_data = dataset[split_name]
            print(f"\nAnalyzing {split_name} split with {len(split_data)} samples...")
            
            for i, item in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
                audio = item["audio"]
                file_path = item.get("file_path", f"{split_name}_{i}")
                label = item.get("label", -1)
                
                # Calculate original duration
                audio_length_samples = audio.shape[1]
                orig_duration = audio_length_samples / sample_rate
                
                # Check for zeros in original audio
                zero_count = torch.sum(torch.abs(audio) < 1e-6).item()
                zero_percent = (zero_count / audio.numel()) * 100
                
                # Count chunks with different minimum segment lengths
                chunks_min5s = chunk_audio(audio, chunk_size_seconds=10, 
                                          sample_rate=sample_rate, min_segment_length=5)
                
                chunks_min2s = chunk_audio(audio, chunk_size_seconds=10,
                                          sample_rate=sample_rate, min_segment_length=2)
                
                chunks_no_min = chunk_audio(audio, chunk_size_seconds=10,
                                          sample_rate=sample_rate, min_segment_length=0)
                
                # Count "silent" chunks (>50% zeros)
                silent_chunks = 0
                for chunk in chunks_no_min:
                    chunk_zero_count = torch.sum(torch.abs(chunk) < 1e-6).item()
                    chunk_zero_percent = (chunk_zero_count / chunk.numel()) * 100
                    if chunk_zero_percent > 50:
                        silent_chunks += 1
                
                # Write row to CSV
                writer.writerow({
                    "split": split_name,
                    "index": i,
                    "file_path": file_path,
                    "label": label,
                    "original_duration": round(orig_duration, 2),
                    "zero_percentage": round(zero_percent, 2),
                    "num_samples": audio_length_samples,
                    "num_zeros": zero_count,
                    "num_chunks_min5s": len(chunks_min5s),
                    "num_chunks_min2s": len(chunks_min2s),
                    "num_chunks_no_min": len(chunks_no_min),
                    "silent_chunk_count": silent_chunks
                })
                
    print(f"\nAnalysis complete! Results saved to {csv_path}")
    return csv_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze audio file chunking')
    parser.add_argument('--dataset-path', type=str, default=None, 
                        help='Path to a previous PyTorch dataset directory')
    parser.add_argument('--backup-dataset-path', type=str, default=None, 
                        help='Path to a backup PyTorch dataset directory')
    args = parser.parse_args()
    
    # Run analysis
    csv_path = analyze_all_files(args.dataset_path, args.backup_dataset_path)
    
    # Load and display summary statistics
    df = pd.read_csv(csv_path)
    print("\nSummary Statistics:")
    print(f"Total files analyzed: {len(df)}")
    print(f"Files with >5% zeros: {len(df[df['zero_percentage'] > 5])}")
    print(f"Files with >10% zeros: {len(df[df['zero_percentage'] > 10])}")
    print(f"Files with >25% zeros: {len(df[df['zero_percentage'] > 25])}")
    print(f"Files with >40% zeros: {len(df[df['zero_percentage'] > 40])}")
    print(f"Files with >50% zeros: {len(df[df['zero_percentage'] > 50])}")
    
    print("\nChunk Count Statistics:")
    print(f"Avg chunks with 5s min: {df['num_chunks_min5s'].mean():.2f} ± {df['num_chunks_min5s'].std():.2f}")
    print(f"Avg chunks with 2s min: {df['num_chunks_min2s'].mean():.2f} ± {df['num_chunks_min2s'].std():.2f}")
    print(f"Avg chunks with no min: {df['num_chunks_no_min'].mean():.2f} ± {df['num_chunks_no_min'].std():.2f}")
    
    print("\nFiles with silent chunks:")
    print(f"Files with 1+ silent chunks: {len(df[df['silent_chunk_count'] > 0])}")
    print(f"Files with 2+ silent chunks: {len(df[df['silent_chunk_count'] > 1])}")
    print(f"Files with 3+ silent chunks: {len(df[df['silent_chunk_count'] > 2])}")
    
    # Find top 10 files with highest zero percentage
    print("\nTop 10 files with highest zero percentage:")
    top_zeros = df.sort_values(by='zero_percentage', ascending=False).head(10)
    for _, row in top_zeros.iterrows():
        print(f"{row['file_path']}: {row['zero_percentage']:.2f}% zeros, {row['original_duration']:.2f}s duration, {row['num_chunks_min5s']} chunks")
        
    # Look for files with substantially different chunk counts based on minimum segment length
    print("\nFiles with large differences in chunk counts based on minimum segment length:")
    df['chunk_count_diff'] = df['num_chunks_no_min'] - df['num_chunks_min5s']
    top_diff = df.sort_values(by='chunk_count_diff', ascending=False).head(10)
    for _, row in top_diff.iterrows():
        print(f"{row['file_path']}: {row['chunk_count_diff']} more chunks with no min, Duration: {row['original_duration']:.2f}s")
    
    # Generate additional CSV with problematic files
    problem_files = df[(df['zero_percentage'] > 10) | (df['silent_chunk_count'] > 0) | (df['chunk_count_diff'] > 2)]
    if len(problem_files) > 0:
        problem_csv_path = os.path.join(os.path.dirname(csv_path), 
            f"problematic_files_{os.path.basename(csv_path)}")
        problem_files.to_csv(problem_csv_path, index=False)
        print(f"\nList of {len(problem_files)} problematic files saved to {problem_csv_path}")
        
    # Look specifically for HC-W-84-107.wav
    target_file = df[df['file_path'].str.contains('HC-W-84-107', case=False, na=False)]
    if not target_file.empty:
        print("\nDetails for HC-W-84-107.wav:")
        for _, row in target_file.iterrows():
            print(f"Split: {row['split']}, Index: {row['index']}")
            print(f"Duration: {row['original_duration']:.2f}s")
            print(f"Zero percentage: {row['zero_percentage']:.2f}%")
            print(f"Chunks (min5s/min2s/no_min): {row['num_chunks_min5s']}/{row['num_chunks_min2s']}/{row['num_chunks_no_min']}")
            print(f"Silent chunks: {row['silent_chunk_count']}")
    else:
        print("\nFile HC-W-84-107.wav not found in the dataset.")