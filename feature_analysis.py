import os
import sys
import argparse
import pandas as pd
import numpy as np
# Local imports
import myConfig
import myData
import myFunctions
import myPlots


def check_data_exists():
    """Check if required data directories exist."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
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
    data_dir = myFunctions.get_data_dir()
    
    for category in myConfig.LABEL_MAP.keys():
        category_path = os.path.join(data_dir, category)
        if not os.path.exists(category_path):
            print(f"Warning: Category directory '{category_path}' not found")
            continue
        
        for file in os.listdir(category_path):
            # Only include primary wav files (exclude _original files)
            if file.endswith(".wav") and "_original" not in file:
                audio_files.append(os.path.join(category_path, file))
                labels.append(myConfig.LABEL_MAP[category])

    if not audio_files:
        print(f"Warning: No audio files found in the data directory")
    
    df = pd.DataFrame({"file_path": audio_files, "label": labels})
    return df


def extract_and_report_features():
    """
    Extract prosodic and acoustic features from audio files and 
    generate a statistical report of features across classes.
    """

    myData.DownloadAndExtract()    
    # Check if dataframe.csv exists in the Data directory
    data_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "dataframe.csv")   
    if os.path.exists(data_file_path):
        # Load existing dataframe
        data_df = pd.read_csv(data_file_path)
        print(f"Loaded existing dataframe from {data_file_path}")
    else:
        # Create dataframe and save it
        data_df = myFunctions.createDataframe()
        data_df = myFunctions.featureEngineering(data_df)
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(data_file_path), exist_ok=True)        
        # Save dataframe
        data_df.to_csv(data_file_path, index=False)
        print(f"Created and saved dataframe to {data_file_path}")
        print("Extracting prosodic and acoustic features...")                
    # Create a combined list of features for reporting
    all_features = myData.numeric_cols + myConfig.jitter_shimmer_features + myConfig.spectral_features + myConfig.speech2text_features
    
    # Group by class and compute statistics for all features
    stats = data_df.groupby('class')[all_features].agg(['mean', 'std'])
    
    print("\n===== Feature Statistics Report =====")
    print("\nFeature statistics by class (Healthy, MCI, AD):")
    print(stats)
    
    # Print overall statistics
    print("\nOverall feature statistics:")
    overall_stats = data_df[all_features].describe()
    print(overall_stats)
    
    return data_df, stats


def plot_feature_histograms():
    """
    Extract prosodic and acoustic features and create histogram plots.
    """
    data_df, _ = extract_and_report_features()
    
    print("Generating histogram plots for prosodic features...")
    myPlots.histogramProsodicFeatures(data_df)
    
    print("Histogram plots generated successfully.")
    return data_df


def main():
    parser = argparse.ArgumentParser(description="Prosodic and Acoustic Feature Analysis")
    parser.add_argument('action', choices=['report', 'plot'], 
                      help="Action to perform: 'report' for feature statistics or 'plot' for histograms")
    parser.add_argument('--offline', action='store_true', 
                      help="Run in offline mode (assumes data is already downloaded)")
    
    args = parser.parse_args()
    
    # Set offline mode if specified
    #myConfig.running_offline = args.offline
    myConfig.running_offline = True
    
    if args.action == 'report':
        extract_and_report_features()
    elif args.action == 'plot':
        plot_feature_histograms()


if __name__ == "__main__":
    main()