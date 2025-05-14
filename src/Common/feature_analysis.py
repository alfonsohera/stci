import pandas as pd
import numpy as np
import os
import argparse
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from .Config import ROOT_DIR, LABEL_MAP
from .Data import DownloadAndExtract, extracted_features
from .Functions import get_data_dir, createDataframe, featureEngineering
from .Plots import plotProsodicFeatures, histogramProsodicFeatures 
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def check_data_exists():
    """Check if required data directories exist."""
    data_dir = get_data_dir()
    healthy_dir = os.path.join(data_dir, "Healthy")
    mci_dir = os.path.join(data_dir, "MCI")
    ad_dir = os.path.join(data_dir, "AD")
    
    if not all(os.path.exists(folder) and os.listdir(folder) 
               for folder in [healthy_dir, mci_dir, ad_dir]):
        print("Error: Required data not found in Data directory.")
        print("Please run myData.DownloadAndExtract() first to download and extract the audio files.")
        return False
    return True


def createDataframe_filtered():
    """
    Create a dataframe of audio files excluding _original files to prevent duplicates.
    """
    audio_files = []
    labels = []
    
    # Always use the Data directory at script level
    data_dir = get_data_dir()
    
    for category in LABEL_MAP.keys():
        category_path = os.path.join(data_dir, category)
        if not os.path.exists(category_path):
            print(f"Warning: Category directory '{category_path}' not found")
            continue
        
        for file in os.listdir(category_path):
            # Only include primary wav files (exclude _original files)
            if file.endswith(".wav") and "_original" not in file:
                audio_files.append(os.path.join(category_path, file))
                labels.append(LABEL_MAP[category])

    if not audio_files:
        print(f"Warning: No audio files found in the data directory")
    
    df = pd.DataFrame({"file_path": audio_files, "label": labels})
    return df


def extract_and_report_features():
    """
    Extract prosodic and acoustic features from audio files and 
    generate a statistical report of features across classes,
    excluding files specified in exclude_list.csv.
    """
    DownloadAndExtract()    
    # Check if dataframe.csv exists in the Data directory
    data_file_path = os.path.join(ROOT_DIR, "..", "Data", "dataframe.csv")
    if os.path.exists(data_file_path):
        # Load existing dataframe
        data_df = pd.read_csv(data_file_path)
        print(f"Loaded existing dataframe from {data_file_path}")
    else:
        # Create dataframe and save it
        data_df = createDataframe()
        data_df = featureEngineering(data_df)
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(data_file_path), exist_ok=True)        
        # Save dataframe
        data_df.to_csv(data_file_path, index=False)
        print(f"Created and saved dataframe to {data_file_path}")
        print("Extracting prosodic and acoustic features...")
    
    # Load exclude list
    exclude_list_path = os.path.join(ROOT_DIR, "exclude_list.csv")
    if os.path.exists(exclude_list_path):
        exclude_df = pd.read_csv(exclude_list_path)
        exclude_filenames = set(exclude_df['filename'].tolist())
        print(f"Loaded exclude list with {len(exclude_filenames)} files to exclude")
    else:
        print(f"Warning: Exclude list not found at {exclude_list_path}")
        exclude_filenames = set()
    
    # Apply the exclude list by filtering the dataframe
    initial_count = len(data_df)
    
    # Create a function to extract base filename without directory and extension
    def extract_base_filename(file_path):
        # Get filename without directory
        filename = os.path.basename(file_path)
        # Remove extension
        base_name = os.path.splitext(filename)[0]
        return base_name
    
    # Apply filter to exclude files
    data_df['base_filename'] = data_df['file_path'].apply(extract_base_filename)
    filtered_df = data_df[~data_df['base_filename'].isin(exclude_filenames)]
    filtered_df = filtered_df.drop('base_filename', axis=1)  # Remove temporary column
    
    excluded_count = initial_count - len(filtered_df)
    print(f"Excluded {excluded_count} files from the dataset for feature statistics")
                
    # Create a combined list of features for reporting
    all_features = extracted_features
    
    # Group by class and compute statistics for all features using filtered data
    stats = filtered_df.groupby('class')[all_features].agg(['mean', 'std'])
    
    print("\n===== Feature Statistics Report (Filtered Dataset) =====")
    print("\nFeature statistics by class (Healthy, MCI, AD):")
    print(stats)
    
    # Print overall statistics
    print("\nOverall feature statistics (Filtered Dataset):")
    overall_stats = filtered_df[all_features].describe()
    print(overall_stats)
    
    #Audio duration stats
    print("\nAudio duration statistics (Filtered Dataset):")
    plotProsodicFeatures(filtered_df)
    return filtered_df, stats


def plot_feature_histograms():
    """
    Extract prosodic and acoustic features and create histogram plots,
    excluding files specified in exclude_list.csv.
    """
    # Extract and report features
    data_df, _ = extract_and_report_features()
    
    # Load exclude list
    exclude_list_path = os.path.join(ROOT_DIR, "exclude_list.csv")
    if os.path.exists(exclude_list_path):
        exclude_df = pd.read_csv(exclude_list_path)
        exclude_filenames = set(exclude_df['filename'].tolist())
        print(f"Loaded exclude list with {len(exclude_filenames)} files to exclude")
    else:
        print(f"Warning: Exclude list not found at {exclude_list_path}")
        exclude_filenames = set()
    
    # Apply the exclude list by filtering the dataframe
    initial_count = len(data_df)
    
    # Create a function to extract base filename without directory and extension
    def extract_base_filename(file_path):
        # Get filename without directory
        filename = os.path.basename(file_path)
        # Remove extension
        base_name = os.path.splitext(filename)[0]
        return base_name
    
    # Apply filter to exclude files
    data_df['base_filename'] = data_df['file_path'].apply(extract_base_filename)
    filtered_df = data_df[~data_df['base_filename'].isin(exclude_filenames)]
    filtered_df = filtered_df.drop('base_filename', axis=1)  # Remove temporary column
    
    excluded_count = initial_count - len(filtered_df)
    print(f"Excluded {excluded_count} files from the dataset for histogram plots")
    
    print("Generating histogram plots for prosodic features...")
    histogramProsodicFeatures(filtered_df)
    
    print("Histogram plots generated successfully using filtered data.")
    return filtered_df


def main():
    parser = argparse.ArgumentParser(description="Prosodic and Acoustic Feature Analysis")
    parser.add_argument('action', choices=['report', 'plot'], 
                      help="Action to perform: 'report' for feature statistics or 'plot' for histograms")
    parser.add_argument('--offline', action='store_true', 
                      help="Run in offline mode (assumes data is already downloaded)")
    
    args = parser.parse_args()
    
    if args.action == 'report':
        extract_and_report_features()
    elif args.action == 'plot':
        plot_feature_histograms()


if __name__ == "__main__":
    main()