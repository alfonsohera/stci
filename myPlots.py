import matplotlib.pyplot as plt
import seaborn as sns
import myConfig
import librosa
import librosa.display
import numpy as np
import pandas as pd
import os   
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
from plotly.offline import plot
import os
from cnn_rnn_model import PretrainedDualPathAudioClassifier

def plotAgeDistribution(data_df):
    plt.style.use("default")
    
    mean_values = data_df.groupby("class")["Age"].mean()
    std_values = data_df.groupby("class")["Age"].std()
    
    # Get unique classes
    fig, axes = plt.subplots(len(myConfig.classes), 1, figsize=(8, 5 * len(myConfig.classes)), sharex=True)

    for i, cls in enumerate(myConfig.classes):
        ax = axes[i] if len(myConfig.classes) > 1 else axes  # Handle single-class case
        subset = data_df[data_df["class"] == cls]["Age"]

        sns.histplot(subset, bins=15, kde=False, ax=ax)

        # Correct way to access values
        mean_age = mean_values[cls]
        std_age = std_values[cls]

        # Plot mean and standard deviation lines
        ax.axvline(mean_age, color='r', linestyle='dashed', linewidth=2, label=f"Mean: {mean_age:.2f}")
        ax.axvline(mean_age - std_age, color='g', linestyle='dashed', linewidth=2, label=f'-1 Std: {mean_age - std_age:.2f}')
        ax.axvline(mean_age + std_age, color='g', linestyle='dashed', linewidth=2, label=f'+1 Std: {mean_age + std_age:.2f}')

        # Titles and labels
        ax.set_title(f"Age Distribution - {cls}")
        ax.set_xlabel("Age")
        ax.set_ylabel("Frequency")
        ax.legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

def plotProsodicFeatures(data_df):
    # Compute statistics
    mean_duration = data_df["duration"].mean()
    std_duration = data_df["duration"].std()

    # Print statistics
    print(f"Mean Duration: {mean_duration:.2f} seconds")
    print(f"Standard Deviation: {std_duration:.2f} seconds")

    # Plot a single histogram for all durations
    plt.figure(figsize=(10, 5))
    plt.hist(data_df["duration"], bins=40, edgecolor="black", alpha=0.7)
    plt.axvline(mean_duration, color='red', linestyle='dashed', linewidth=2, label=f"Mean: {mean_duration:.2f}s")
    plt.axvline(mean_duration + std_duration, color='green', linestyle='dashed', linewidth=2, label=f"+1 Std: {mean_duration + std_duration:.2f}s")
    plt.axvline(mean_duration - std_duration, color='green', linestyle='dashed', linewidth=2, label=f"-1 Std: {mean_duration - std_duration:.2f}s")
    plt.axvline(mean_duration + 2*std_duration, color='blue', linestyle='dashed', linewidth=2, label=f"+2 Std: {mean_duration + 2*std_duration:.2f}s")
    plt.axvline(mean_duration + 3*std_duration, color='black', linestyle='dashed', linewidth=2, label=f"+3 Std: {mean_duration + 3*std_duration:.2f}s")

    plt.xlabel("Duration (seconds)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Audio File Durations")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Compute mean and std deviation of duration per class
    heatmap_data = data_df.groupby("class")["duration"].agg(["mean", "std"]).reset_index()
    # Convert to a matrix format suitable for a heatmap
    heatmap_matrix = heatmap_data.set_index("class").T

    # Plot heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(heatmap_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

    plt.title("Heatmap of Audio Duration by Cognitive Class")
    plt.xlabel("Cognitive Condition")
    plt.ylabel("Statistic (Mean/Std)")
    plt.show()


    ## Box plot
    # Create a box plot for audio duration per class
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="class", y="duration", data=data_df)
    # Formatting the plot
    plt.title("Box Plot of Audio Duration by Cognitive Class")
    plt.xlabel("Cognitive Condition")
    plt.ylabel("Audio Duration (seconds)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def histogramProsodicFeatures(data_df):
    # Define all features to plot (prosodic + jitter/shimmer)
    all_features = myConfig.features + myConfig.jitter_shimmer_features + myConfig.spectral_features + myConfig.speech2text_features
    
    # Calculate mean and std values for all features
    mean_values = data_df.groupby("class")[all_features].mean()
    std_values = data_df.groupby("class")[all_features].std()
    
    # Create one figure per feature, with subplots for each class
    for feature in all_features:
        # Create figure with 3 subplots (one for each class)
        fig, axes = plt.subplots(len(myConfig.classes), 1, figsize=(10, 12), sharex=True)
        
        # Set figure title for the entire plot
        fig.suptitle(f"Distribution of {feature} across Cognitive Classes", fontsize=16)
        
        # Generate histograms for each class
        for i, cls in enumerate(myConfig.classes):
            # Get current axis (handle case of single class)
            ax = axes[i] if len(myConfig.classes) > 1 else axes
            
            # Get class data for this feature
            class_data = data_df[data_df["class"] == cls][feature]
            
            # Skip if no valid data
            if len(class_data.dropna()) == 0:
                ax.text(0.5, 0.5, f"No valid data for {cls}", ha='center', va='center')
                ax.set_title(f"{cls}")
                continue
            
            # Plot histogram
            sns.histplot(
                data=class_data,
                bins=20,
                kde=True,
                color=myConfig.CLASS_COLORS.get(cls, f"C{i}"),
                ax=ax
            )
            
            # Get mean and std values
            mean_val = mean_values.loc[cls, feature]
            std_val = std_values.loc[cls, feature]
            
            # Add vertical lines for mean and std deviation
            ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f"Mean: {mean_val:.2f}")
            ax.axvline(mean_val - std_val, color='blue', linestyle='dashed', linewidth=1, label=f"-1 SD: {mean_val - std_val:.2f}")
            ax.axvline(mean_val + std_val, color='blue', linestyle='dashed', linewidth=1, label=f"+1 SD: {mean_val + std_val:.2f}")
            
            # Add text annotations for clarity
            ax.legend(loc='upper right')
            
            # Set subplot title and labels
            ax.set_title(f"{cls} - {len(class_data.dropna())} samples")
            ax.set_ylabel("Frequency")
            
            # Only add x-label for the bottom subplot
            if i == len(myConfig.classes) - 1:
                ax.set_xlabel(feature)
        
        # Adjust layout for clarity
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Make room for suptitle
        plt.show()
    
    # Print summary statistics
    print("\nMean and Standard Deviation per Feature per Class:\n")
    for cls in myConfig.classes:
        print(f"\nClass: {cls}")
        # Print standard prosodic features
        print("  -- Prosodic Features --")
        for feature in myConfig.features:
            mean_val = mean_values.loc[cls, feature]
            std_val = std_values.loc[cls, feature]
            print(f"  Feature: {feature} | Mean: {mean_val:.4f} | Std Dev: {std_val:.4f}")
        
        # Print jitter and shimmer features
        print("\n  -- Voice Quality Features --")
        for feature in myConfig.jitter_shimmer_features:
            mean_val = mean_values.loc[cls, feature]
            std_val = std_values.loc[cls, feature]
            print(f"  Feature: {feature} | Mean: {mean_val:.4f} | Std Dev: {std_val:.4f}")
        # Print spectral features
        print("\n  -- Spectral Features --")
        for feature in myConfig.spectral_features:
            mean_val = mean_values.loc[cls, feature]
            std_val = std_values.loc[cls, feature]
            print(f"  Feature: {feature} | Mean: {mean_val:.4f} | Std Dev: {std_val:.4f}")
    print("\n")
    

def plot_waveform(audio_path, ax, title):
    """Plots the waveform of an audio file."""
    y, sr = librosa.load(audio_path, sr=None)
    ax.plot(np.linspace(0, len(y)/sr, num=len(y)), y, color='b')
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")


def plot_mel_spectrogram(audio_path, ax, title):
    """Plots the Mel spectrogram of an audio file."""
    y, sr = librosa.load(audio_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title(title)
    return img


def plot_difference_spectrogram(original_path, processed_path, ax):
    """Plots the difference between two Mel spectrograms."""
    y_orig, sr_orig = librosa.load(original_path, sr=None)
    y_proc, sr_proc = librosa.load(processed_path, sr=None)

    mel_orig = librosa.feature.melspectrogram(y=y_orig, sr=sr_orig, n_mels=128, fmax=8000)
    mel_proc = librosa.feature.melspectrogram(y=y_proc, sr=sr_proc, n_mels=128, fmax=8000)

    # Get the smaller time dimension
    min_time_dim = min(mel_orig.shape[1], mel_proc.shape[1])

    # Truncate both to the shorter length
    mel_orig = mel_orig[:, :min_time_dim]
    mel_proc = mel_proc[:, :min_time_dim]

    mel_diff = librosa.power_to_db(mel_proc, ref=np.max) - librosa.power_to_db(mel_orig, ref=np.max)
    img = librosa.display.specshow(mel_diff, sr=sr_orig, x_axis='time', y_axis='mel', ax=ax, cmap='RdBu_r')
    ax.set_title("Difference (Processed - Original) Mel Spectrogram")
    return img


def visualize_audio_comparison(original_path, processed_path):
    """Plots waveform, Mel spectrograms, and difference spectrogram for two audio files."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Waveforms
    plot_waveform(original_path, axes[0, 0], "Original Waveform")
    plot_waveform(processed_path, axes[0, 1], "Processed Waveform")

    # Mel Spectrograms
    img1 = plot_mel_spectrogram(original_path, axes[1, 0], "Original Mel Spectrogram")
    img2 = plot_mel_spectrogram(processed_path, axes[1, 1], "Processed Mel Spectrogram")

    # Difference Spectrogram
    #img3 = plot_difference_spectrogram(original_path, processed_path, axes[2, 0])
    
    fig.colorbar(img1, ax=axes[1, 0])
    fig.colorbar(img2, ax=axes[1, 1])
    #fig.colorbar(img3, ax=axes[2, 0])

    axes[1, 1].axis("off")  # Empty placeholder

    plt.tight_layout()
    plt.show()


def visualize_spectrogram_augmentations(data_df, audio_root_path):
    """
    Selects one random sample from each class, creates log mel spectrograms,
    applies augmentations, and visualizes the original and augmented spectrograms.
    
    Args:
        data_df: DataFrame containing your dataset information
        audio_root_path: Root path to your audio files
    """
    import random
    import torch
    from torch.utils.data import Dataset
    from cnn_rnn_model import SpecAugment  # Import your augmentation class
    
    # Print column names for debugging
    print("Available DataFrame columns:", data_df.columns.tolist())
    
    # Initialize the SpecAugment augmentation
    spec_augment = SpecAugment(
        freq_mask_param=30,
        time_mask_param=30,
        n_freq_masks=1,
        n_time_masks=1,
        apply_prob=1.0  # Always apply for visualization purposes
    )
    
    # Create figure
    fig = plt.figure(figsize=(15, 5 * len(myConfig.classes)))
    
    # Select one random sample from each class
    for i, cls in enumerate(myConfig.classes):
        # Filter data by class
        class_data = data_df[data_df["class"] == cls]
        
        if len(class_data) == 0:
            print(f"No samples found for class: {cls}")
            continue
            
        # Randomly select one sample
        sample_row = class_data.sample(n=1).iloc[0]
        
        # Get the file path (using 'file_path' instead of 'audio_path')
        if 'file_path' in sample_row:
            sample_path = sample_row['file_path']
        else:
            print(f"Could not find file path column for class {cls}")
            print("Available columns:", sample_row.index.tolist())
            continue
        
        # Construct full path
        full_path = os.path.join(audio_root_path, sample_path)
        
        # Check if file exists
        if not os.path.exists(full_path):
            print(f"File not found: {full_path}")
            # Try a direct path without joining
            if os.path.exists(sample_path):
                full_path = sample_path
                print(f"Using direct path: {full_path}")
            else:
                # Try to find the file by basename
                basename = os.path.basename(sample_path)
                for root, _, files in os.walk(audio_root_path):
                    if basename in files:
                        full_path = os.path.join(root, basename)
                        print(f"Found file at: {full_path}")
                        break
                else:
                    print(f"Could not find file for class {cls}: {sample_path}")
                    continue
        
        try:
            # Load audio
            y, sr = librosa.load(full_path, sr=None)
            
            # Create log mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Convert to torch tensor for augmentation
            log_mel_tensor = torch.from_numpy(log_mel_spec).unsqueeze(0)  # Add batch dimension
            
            # Apply augmentation
            augmented_spec = spec_augment(log_mel_tensor).squeeze(0).numpy()
            
            # Create subplot for this class (3 columns)
            ax1 = fig.add_subplot(len(myConfig.classes), 3, i*3 + 1)
            ax2 = fig.add_subplot(len(myConfig.classes), 3, i*3 + 2)
            ax3 = fig.add_subplot(len(myConfig.classes), 3, i*3 + 3)
            
            # Plot waveform
            times = np.linspace(0, len(y)/sr, num=len(y))
            ax1.plot(times, y)
            ax1.set_title(f"{cls} - Original Waveform - {os.path.basename(sample_path)}")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Amplitude")
            
            # Plot original mel spectrogram
            img1 = librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel', ax=ax2)
            ax2.set_title(f"{cls} - Original Log-Mel Spectrogram")
            fig.colorbar(img1, ax=ax2, format="%+2.0f dB")
            
            # Plot augmented mel spectrogram
            img2 = librosa.display.specshow(augmented_spec, sr=sr, x_axis='time', y_axis='mel', ax=ax3)
            ax3.set_title(f"{cls} - Augmented Log-Mel Spectrogram")
            fig.colorbar(img2, ax=ax3, format="%+2.0f dB")
            
        except Exception as e:
            print(f"Error processing file for class {cls}: {e}")
            continue
    
    plt.tight_layout()
    plt.show()


def visualize_augmentation_examples(data_df, audio_root_path, n_examples=3):
    """
    Creates a detailed visualization of several spectrogram augmentations for a
    single random sample, to show variations in the augmentation process.
    
    Args:
        data_df: DataFrame containing your dataset information
        audio_root_path: Root path to your audio files
        n_examples: Number of augmentation examples to show
    """
    import random
    import torch
    import os
    from cnn_rnn_model import SpecAugment
    
    # Initialize the SpecAugment augmentation
    spec_augment = SpecAugment(
        freq_mask_param=30,
        time_mask_param=30,
        n_freq_masks=1,
        n_time_masks=1,
        apply_prob=1.0  # Always apply for visualization
    )
    
    # Randomly select a class and then a sample from that class
    selected_class = random.choice(myConfig.classes)
    class_data = data_df[data_df["class"] == selected_class]
    
    if len(class_data) == 0:
        print(f"No samples available for class {selected_class}")
        return
        
    sample_row = class_data.sample(n=1).iloc[0]
    
    # Get the file path (using 'file_path' instead of 'audio_path')
    if 'file_path' in sample_row:
        sample_path = sample_row['file_path']
    else:
        print(f"Could not find file path column for class {selected_class}")
        print("Available columns:", sample_row.index.tolist())
        return
    
    # Construct full path
    full_path = os.path.join(audio_root_path, sample_path)
    
    # Check if file exists
    if not os.path.exists(full_path):
        print(f"File not found: {full_path}")
        # Try a direct path without joining
        if os.path.exists(sample_path):
            full_path = sample_path
            print(f"Using direct path: {full_path}")
        else:
            # Try to find the file by basename
            basename = os.path.basename(sample_path)
            for root, _, files in os.walk(audio_root_path):
                if basename in files:
                    full_path = os.path.join(root, basename)
                    print(f"Found file at: {full_path}")
                    break
            else:
                print(f"Could not find file: {sample_path}")
                return
    
    try:
        # Load audio
        y, sr = librosa.load(full_path, sr=None)
        
        # Create log mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Convert to torch tensor for augmentation
        log_mel_tensor = torch.from_numpy(log_mel_spec).unsqueeze(0)  # Add batch dimension
        
        # Create figure with n_examples+1 rows (original + n augmented versions)
        fig = plt.figure(figsize=(12, 4 * (n_examples + 1)))
        
        # Plot original waveform and spectrogram
        ax_wave = fig.add_subplot(n_examples + 1, 2, 1)
        ax_spec = fig.add_subplot(n_examples + 1, 2, 2)
        
        # Plot waveform
        times = np.linspace(0, len(y)/sr, num=len(y))
        ax_wave.plot(times, y)
        ax_wave.set_title(f"Original Waveform - {os.path.basename(sample_path)}")
        ax_wave.set_xlabel("Time (s)")
        ax_wave.set_ylabel("Amplitude")
        
        # Plot original spectrogram
        img = librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel', ax=ax_spec)
        ax_spec.set_title("Original Log-Mel Spectrogram")
        fig.colorbar(img, ax=ax_spec, format="%+2.0f dB")
        
        # Generate and plot multiple augmentations
        for i in range(n_examples):
            # Apply augmentation with different random masks each time
            augmented_spec = spec_augment(log_mel_tensor).squeeze(0).numpy()
            
            # Plot augmented spectrogram
            ax = fig.add_subplot(n_examples + 1, 2, (i + 1) * 2 + 2)
            img = librosa.display.specshow(augmented_spec, sr=sr, x_axis='time', y_axis='mel', ax=ax)
            ax.set_title(f"Augmented Example #{i+1}")
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            
            # Add empty plot for alignment (or could add feature visualization here)
            ax_empty = fig.add_subplot(n_examples + 1, 2, (i + 1) * 2 + 1)
            ax_empty.axis('off')
        
        plt.tight_layout()
        plt.suptitle(f"Spectrogram Augmentation Examples - Class: {selected_class}", fontsize=16, y=1.0)
        plt.subplots_adjust(top=0.95)
        plt.show()
        
    except Exception as e:
        print(f"Error processing file: {e}")

def analyze_augmentation_diversity(data_df, audio_root_path, n_examples=5):
    """
    Analyzes the diversity of spectrogram augmentations using the actual CNN feature extractor
    from your model to measure similarity in embedding space.
    
    Args:
        data_df: DataFrame containing your dataset information
        audio_root_path: Root path to your audio files
        n_examples: Number of augmentation examples to generate
    """
    import random
    import torch
    import torch.nn.functional as F
    from sklearn.metrics.pairwise import cosine_similarity
    import matplotlib.pyplot as plt
    import seaborn as sns
    from cnn_rnn_model import SpecAugment, DualPathAudioClassifier
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the SpecAugment augmentation
    spec_augment = SpecAugment(
        freq_mask_param=50,
        time_mask_param=50,
        n_freq_masks=2,
        n_time_masks=2,
        apply_prob=1.0  # Always apply for visualization
    ).to(device)
    
    # Load the CNN extractor 
    model = DualPathAudioClassifier(
        num_classes=3,
        sample_rate=16000,        
        apply_specaugment=False
    ).to(device)
    model.eval()  # Set to evaluation mode
    
    # Randomly select a class and then a sample from that class
    selected_class = random.choice(myConfig.classes)
    class_data = data_df[data_df["class"] == selected_class]
    
    if len(class_data) == 0:
        print(f"No samples available for class {selected_class}")
        return
        
    sample_row = class_data.sample(n=1).iloc[0]
    
    # Get the file path
    if 'file_path' in sample_row:
        sample_path = sample_row['file_path']
    else:
        print(f"Could not find file path column for class {selected_class}")
        print("Available columns:", sample_row.index.tolist())
        return
    
    # Construct full path and verify file exists
    full_path = os.path.join(audio_root_path, sample_path)
    if not os.path.exists(full_path):
        print(f"File not found: {full_path}")
       
        if os.path.exists(sample_path):
            full_path = sample_path
            print(f"Using direct path: {full_path}")
        else:
            
            basename = os.path.basename(sample_path)
            for root, _, files in os.walk(audio_root_path):
                if basename in files:
                    full_path = os.path.join(root, basename)
                    print(f"Found file at: {full_path}")
                    break
            else:
                print(f"Could not find file: {sample_path}")
                return
    
    try:
        # Load audio
        y, sr = librosa.load(full_path, sr=16000)  
        
        # Create log mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Convert to tensor for augmentation
        log_mel_tensor = torch.from_numpy(log_mel_spec).unsqueeze(0).to(device)  # Add batch dimension [1, freq, time]
        
        # Generate multiple augmentations
        augmented_specs = []
        for i in range(n_examples):
            # Clone to avoid modifying the original
            aug_spec = spec_augment(log_mel_tensor.clone()).cpu().squeeze(0).numpy()
            augmented_specs.append(aug_spec)
        
        # Extract features using the CNN extractor
        print("Extracting features from original and augmented spectrograms...")
        
        with torch.inference_mode():
            # Prepare original spectrogram for the model (match model's preprocessing)
            # [1, 1, freq, time] - adding channel dimension expected by CNN
            orig_mel_db = log_mel_tensor.unsqueeze(0)  
            
            # Scale to [0,1] range and apply normalization as in your model
            orig_mel_db = (orig_mel_db - orig_mel_db.min()) / (orig_mel_db.max() - orig_mel_db.min() + 1e-6)
            
            #ToDo: remove these transformations
            # Apply the same transformations as in the model's forward pass
            # Resize to target size (ensure model.target_size is accessible)
            orig_mel_db = F.interpolate(orig_mel_db, size=(384, 384),  # Expected size for EfficientNetV2 
                                        mode='bilinear', align_corners=False)
            
            # Normalize with mean and standard deviation
            orig_mel_db = (orig_mel_db - orig_mel_db.mean()) / (orig_mel_db.std() + 1e-5)
            
            # Apply ImageNet normalization
            orig_mel_db = model.normalize(orig_mel_db)
            
            # Pass through CNN feature extractor only
            original_features = model.cnn_extractor(orig_mel_db).flatten().cpu().numpy()
            
            # Extract features for each augmented version
            augmented_features = []
            for aug_spec in augmented_specs:
                # Convert back to tensor
                aug_tensor = torch.from_numpy(aug_spec).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, freq, time]
                
                # Scale to [0,1] range
                aug_tensor = (aug_tensor - aug_tensor.min()) / (aug_tensor.max() - aug_tensor.min() + 1e-6)
                
                # Apply the same transformations as in the model's forward pass
                aug_tensor = F.interpolate(aug_tensor, size=model.target_size, 
                                          mode='bilinear', align_corners=False)
                aug_tensor = (aug_tensor - aug_tensor.mean()) / (aug_tensor.std() + 1e-5)
                aug_tensor = model.normalize(aug_tensor)
                
                # Extract features
                aug_features = model.cnn_extractor(aug_tensor).flatten().cpu().numpy()
                augmented_features.append(aug_features)
        
        # Calculate feature similarities and distances
        feature_similarities = []
        feature_distances = []
        
        for aug_feat in augmented_features:
            # Cosine similarity between original and augmented features
            cos_sim = cosine_similarity([original_features], [aug_feat])[0][0]
            feature_similarities.append(cos_sim)
            
            # Euclidean distance (normalized)
            euc_dist = np.linalg.norm(original_features - aug_feat) / np.sqrt(len(original_features))
            feature_distances.append(euc_dist)
        
        # Calculate pixel-wise metrics for comparison
        pixel_similarities = []
        mask_percentages = []
        
        for aug_spec in augmented_specs:
            # Flatten arrays for comparison
            orig_flat = log_mel_spec.flatten()
            aug_flat = aug_spec.flatten()
            
            # Cosine similarity at pixel level
            cos_sim = cosine_similarity([orig_flat], [aug_flat])[0][0]
            pixel_similarities.append(cos_sim)
            
            # Calculate percentage of masked values
            mask = (aug_spec == 0) & (log_mel_spec != 0)
            mask_percent = np.sum(mask) / mask.size * 100
            mask_percentages.append(mask_percent)
        
        # Pairwise similarity matrix between augmentations in feature space
        feature_similarity_matrix = np.zeros((n_examples, n_examples))
        for i in range(n_examples):
            for j in range(n_examples):
                feature_similarity_matrix[i, j] = cosine_similarity(
                    [augmented_features[i]], [augmented_features[j]]
                )[0, 0]
        
        # Visualization
        # 1. Create grid for spectrograms and difference maps
        fig = plt.figure(figsize=(15, 10))
        
        # Plot original spectrogram
        ax0 = fig.add_subplot(n_examples + 1, 2, 1)
        img0 = librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel', ax=ax0)
        ax0.set_title("Original Spectrogram")
        fig.colorbar(img0, ax=ax0, format="%+2.0f dB")
        
        # Plot feature similarity metrics
        ax0_right = fig.add_subplot(n_examples + 1, 2, 2)
        ax0_right.text(0.5, 0.5, "Feature Space Analysis", ha='center', va='center', fontsize=14)
        ax0_right.text(0.5, 0.3, f"CNN Model: EfficientNetV2", ha='center', va='center')
        ax0_right.text(0.5, 0.1, f"Feature Dimension: {len(original_features)}", ha='center', va='center')
        ax0_right.axis('off')
        
        # Plot augmented spectrograms with their similarity metrics
        for i in range(n_examples):
            # Plot augmented spectrogram
            ax = fig.add_subplot(n_examples + 1, 2, (i+1)*2 + 1)
            img = librosa.display.specshow(augmented_specs[i], sr=sr, x_axis='time', y_axis='mel', ax=ax)
            ax.set_title(f"Augmented #{i+1} | Feature Sim: {feature_similarities[i]:.4f}")
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            
            # Plot difference map
            ax_right = fig.add_subplot(n_examples + 1, 2, (i+1)*2 + 2)
            diff = log_mel_spec - augmented_specs[i]
            img_diff = librosa.display.specshow(diff, sr=sr, x_axis='time', y_axis='mel', 
                                               ax=ax_right, cmap='coolwarm')
            ax_right.set_title(f"Difference Map | Masked: {mask_percentages[i]:.2f}%")
            fig.colorbar(img_diff, ax=ax_right)
        
        plt.tight_layout()
        plt.suptitle(f"CNN Feature Space Analysis - {selected_class}", fontsize=16, y=0.99)
        plt.subplots_adjust(top=0.95)
        plt.show()
        
        # 2. Plot similarity and distance metrics
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        
        # Feature similarity plot
        x = range(n_examples)
        ax[0].bar(x, feature_similarities, color='skyblue')
        ax[0].axhline(y=np.mean(feature_similarities), color='r', linestyle='--', 
                     label=f'Mean: {np.mean(feature_similarities):.4f}')
        ax[0].set_title("CNN Feature Similarity to Original")
        ax[0].set_xlabel("Augmentation #")
        ax[0].set_ylabel("Cosine Similarity")
        ax[0].set_ylim(0, 1)
        ax[0].legend()
        
        # Feature distance plot
        ax[1].bar(x, feature_distances, color='lightgreen')
        ax[1].axhline(y=np.mean(feature_distances), color='r', linestyle='--',
                     label=f'Mean: {np.mean(feature_distances):.4f}')
        ax[1].set_title("CNN Feature Distance from Original")
        ax[1].set_xlabel("Augmentation #")
        ax[1].set_ylabel("Normalized Euclidean Distance")
        ax[1].legend()
        
        plt.tight_layout()
        plt.show()
        
        # 3. Plot heatmap of feature similarity between augmentations
        plt.figure(figsize=(10, 8))
        sns.heatmap(feature_similarity_matrix, annot=True, fmt=".3f", cmap="YlGnBu", 
                   vmin=0, vmax=1, square=True)
        plt.title("Pairwise CNN Feature Similarity Between Augmentations")
        plt.tight_layout()
        plt.show()
        
        # 4. Compare pixel similarity vs feature similarity
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bars for both metrics
        width = 0.35
        x = np.arange(n_examples)
        ax.bar(x - width/2, pixel_similarities, width, label='Pixel Similarity', color='skyblue')
        ax.bar(x + width/2, feature_similarities, width, label='Feature Similarity', color='salmon')
        
        # Add mean lines
        ax.axhline(y=np.mean(pixel_similarities), color='blue', linestyle='--', 
                  label=f'Pixel Mean: {np.mean(pixel_similarities):.4f}')
        ax.axhline(y=np.mean(feature_similarities), color='red', linestyle='--',
                  label=f'Feature Mean: {np.mean(feature_similarities):.4f}')
        
        # Labels and formatting
        ax.set_ylabel('Cosine Similarity')
        ax.set_xlabel('Augmentation #')
        ax.set_title('Comparison of Pixel-wise vs CNN Feature Similarity')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Aug {i+1}' for i in range(n_examples)])
        ax.set_ylim(0, 1)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\nAugmentation Diversity Analysis for {selected_class}:")
        print(f"CNN Feature Similarity (cosine): {np.mean(feature_similarities):.4f} ± {np.std(feature_similarities):.4f}")
        print(f"CNN Feature Distance (euclidean): {np.mean(feature_distances):.4f} ± {np.std(feature_distances):.4f}")
        print(f"Pixel-wise Similarity (cosine): {np.mean(pixel_similarities):.4f} ± {np.std(pixel_similarities):.4f}")
        print(f"Masked Percentage: {np.mean(mask_percentages):.2f}% ± {np.std(mask_percentages):.2f}%")
        
        # Interpretation guidance
        print("\nInterpretation Guide:")
        feature_sim_mean = np.mean(feature_similarities)
        if feature_sim_mean > 0.95:
            print("⚠️ Augmentations appear very similar in feature space (>0.95 similarity)")
            print("   Consider increasing mask parameters or using multiple augmentations")
        elif feature_sim_mean < 0.5:
            print("⚠️ Augmentations appear extremely different (similarity <0.5)")
            print("   The masking might be too aggressive, consider reducing parameters")
        else:
            print("✓ Augmentations create good diversity in feature space")
        
        # Compare pixel vs feature similarity
        pixel_feature_diff = np.mean(pixel_similarities) - np.mean(feature_similarities)
        if (pixel_feature_diff > 0.3):
            print("✓ Feature similarity is much lower than pixel similarity")
            print("   This suggests the CNN is sensitive to the masked regions")
        elif (pixel_feature_diff < 0.1):
            print("⚠️ Feature similarity is close to pixel similarity")
            print("   The CNN might not be focusing enough on the masked regions")
            
    except Exception as e:
        print(f"Error during augmentation analysis: {e}")
        import traceback
        traceback.print_exc()

# Add these imports at the top of your file
import plotly.graph_objs as go
import plotly.io as pio
from plotly.offline import plot
import os

def create_interactive_3d_plot(embeddings_3d, color_field, unique_categories, color_mapping, title, output_filename):
    """
    Creates an interactive 3D plot with manual rotation control using Plotly.
    
    Args:
        embeddings_3d: The 3D t-SNE embeddings
        color_field: List of category labels for each point
        unique_categories: List of unique categories
        color_mapping: Dictionary mapping categories to colors
        title: Title for the plot
        output_filename: Filename to save the HTML file to
    """
    print(f"Creating interactive 3D visualization for {title}...")
    
    # Convert matplotlib colors to hex format for plotly
    def rgba_to_hex(rgba_color):
        r, g, b, a = rgba_color
        return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
    
    hex_color_mapping = {cat: rgba_to_hex(color_mapping[cat]) for cat in unique_categories}
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each category
    for category in unique_categories:
        indices = [i for i, cat in enumerate(color_field) if cat == category]
        if not indices:
            continue
            
        fig.add_trace(go.Scatter3d(
            x=embeddings_3d[indices, 0],
            y=embeddings_3d[indices, 1],
            z=embeddings_3d[indices, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=hex_color_mapping[category],
                opacity=0.7,
                line=dict(width=0.5, color='#FFFFFF')
            ),
            name=f"{category} (n={len(indices)})"
        ))
    
    # Set layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
            zaxis_title="t-SNE Dimension 3"
        ),
        legend=dict(
            title="Categories",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )
    
    # Save to HTML file
    html_filename = f"{output_filename.split('.')[0]}.html"
    plot(fig, filename=html_filename, auto_open=False)
    
    print(f"Saved interactive 3D visualization to {html_filename}")
    return html_filename



def create_rotating_3d_plot(embeddings_3d, color_field, unique_categories, color_mapping, title, output_filename):
    """
    Creates and saves a rotating 3D animation of the embeddings.
    
    Args:
        embeddings_3d: The 3D t-SNE embeddings
        color_field: List of category labels for each point
        unique_categories: List of unique categories
        color_mapping: Dictionary mapping categories to colors
        title: Title for the plot
        output_filename: Filename to save the animation to
    """
    print(f"Creating rotating 3D animation for {title}...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each category with its own color
    for category in unique_categories:
        indices = [i for i, cat in enumerate(color_field) if cat == category]
        if not indices:
            continue
        
        ax.scatter(
            embeddings_3d[indices, 0],
            embeddings_3d[indices, 1],
            embeddings_3d[indices, 2],
            c=[color_mapping[category]],
            label=f"{category} (n={len(indices)})",
            alpha=0.7,
            s=50
        )
    
    # Set labels and title
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_zlabel("t-SNE Dimension 3", fontsize=12)
    ax.legend(title="Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Function to update the view for animation
    def rotate(angle):
        ax.view_init(elev=20, azim=angle)
        return fig,
    
    # Create animation with 360-degree rotation
    anim = FuncAnimation(fig, rotate, frames=np.arange(0, 360, 5), interval=100)
    
    # Save as GIF
    anim.save(output_filename, writer='pillow', dpi=100)
    plt.close()
    print(f"Saved rotating 3D animation to {output_filename}")


def analyze_class_similarity(dataset_path, audio_root_path, similarity_threshold=0.95, binary_classification=False, exclusion_csv=None, output_prefix="class_similarity"):
    """
    Analyzes inter-class and intra-class similarities to identify problematic samples
    that are hard to classify due to high similarity with samples from other classes.
    
    Args:
        dataset_path: Path to the HuggingFace dataset directory 
        audio_root_path: Root path to audio files
        similarity_threshold: Cosine similarity threshold above which samples are considered too similar
                            (default: 0.95)
        binary_classification: If True, treats classes as binary (HC vs Non-HC)
        exclusion_csv: Path to CSV file listing samples to exclude
        output_prefix: Prefix for output files
    
    Returns:
        Dictionary with similarity statistics and lists of potentially problematic sample pairs
    """
    import torch
    import torch.nn.functional as F
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm
    import os
    import gc
    from datasets import load_from_disk
    import librosa
    from panns_inference.panns_inference.models import Cnn14
    from cnn_rnn_data import prepare_cnn_rnn_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load exclusion list if provided
    excluded_files = set()
    if exclusion_csv:
        try:
            exclusion_df = pd.read_csv(exclusion_csv)
            print(f"Loaded {len(exclusion_df)} files to exclude from analysis")
            
            # Extract the actual filenames (without split prefix)
            for filename in exclusion_df['filename']:
                # The actual filename starts after the second underscore
                parts = filename.split('_', 2)
                if len(parts) >= 3:
                    actual_filename = parts[2]
                    excluded_files.add(actual_filename)
                else:
                    # If the format is different, add the whole filename
                    excluded_files.add(filename+'.wav')
                    
            print(f"Extracted {len(excluded_files)} unique filenames to exclude")
        except Exception as e:
            print(f"Error loading exclusion CSV file: {e}")
            print("Continuing without exclusions")
            excluded_files = set()
    
    # Load dataset from disk
    dataset = prepare_cnn_rnn_dataset(binary_classification=binary_classification)
    
    # Verify dataset structure
    print(f"Dataset keys: {list(dataset.keys())}")
    
    # Print distribution of samples by split
    split_counts = {split: len(dataset[split]) for split in dataset.keys()}
    print(f"Dataset distribution:")
    for split, count in split_counts.items():
        print(f"  {split}: {count} samples")
    
    # Initialize class labels - adjust based on binary classification flag
    if binary_classification:
        print("Using binary classification mode: HC vs Non-HC")
        class_labels = ["Healthy", "Non-Healthy"]
    else:
        print("Using 3-way classification mode: HC vs MCI vs AD")
        class_labels = ["Healthy", "MCI", "AD"]
        
    print(f"Using class labels: {class_labels}")
    
    # Load feature extraction model (CNN14)
    print("Using CNN14 as feature extractor")

    def load_cnn14(checkpoint_path="/home/bosh/Documents/ML/zz_PP/00_SCTI/Repo/training_output/cnn_rnn_binary/cnn_rnn_best.pt"):
        model = Cnn14(classes_num=527, sample_rate=16000, mel_bins=64, hop_size=320, window_size=1024, fmin=50, fmax=8000)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_state = checkpoint
        if 'model' in checkpoint:
            model_state = checkpoint['model'] 
        elif 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
        # Create the full model with your architecture
        full_model = PretrainedDualPathAudioClassifier(
            num_classes=2,  # Binary classification
            pretrained_cnn14_path=myConfig.checkpoint_dir+'/Cnn14_mAP=0.431.pth'
        )
        
        # Load the fine-tuned weights
        try:
            full_model.load_state_dict(model_state)
            print("Successfully loaded fine-tuned model!")
        except Exception as e:
            print(f"Error loading model: {e}")
            # If that fails, attempt partial loading with strict=False
            try:
                full_model.load_state_dict(model_state, strict=False)
                print("Loaded model with some mismatched keys (non-strict loading)")
            except Exception as e:
                print(f"Error with non-strict loading: {e}")
        
        # Extract just the CNN14 part
        cnn14_model = full_model.cnn_extractor
        cnn14_model.eval()
        return cnn14_model
    
    feature_extractor = load_cnn14()
    feature_extractor.to(device)
    feature_extractor.eval()
    
    # Create feature extraction function
    def extract_features(audio_path, feature_extractor):
        def extract_embedding(model, sample_audio):
            with torch.no_grad():
                output = model(sample_audio)
                embedding = output['embedding']
            return embedding  # shape: [1, 2048]
                
        try:
            # Load audio
            sample_audio, sr = librosa.load(audio_path, sr=16000)
            # Convert to tensor format for CNN14
            sample_audio = torch.tensor(sample_audio).unsqueeze(0).to(device)  # [1, time]                     
            
            with torch.inference_mode():
                # Extract features using CNN14
                features = extract_embedding(feature_extractor, sample_audio).flatten().cpu().numpy()
                
            return features
        except Exception as e:
            print(f"Error processing audio file: {e}")
            return None
    
    # Helper function to resolve audio paths
    def resolve_audio_path(file_path):
        # First try direct path
        if os.path.exists(file_path):
            return file_path
        
        # Try with audio_root_path
        full_path = os.path.join(audio_root_path, file_path)
        if os.path.exists(full_path):
            return full_path
        
        # Try to find by basename
        basename = os.path.basename(file_path)
        for root, _, files in os.walk(audio_root_path):
            if basename in files:
                return os.path.join(root, basename)
        
        # Could not find file
        raise FileNotFoundError(f"Could not find file: {file_path}")
    
    # Process each file and extract features
    features_dict = {}  # Will store {file_id: features}
    file_info = {}      # Will store {file_id: {path, split, class_label}}
    
    # Dictionary to track class distributions
    class_counts = {split: {cls: 0 for cls in range(len(class_labels))} for split in dataset.keys()}
    
    print("Extracting features from all audio files...")
    skipped_files = 0
    for split in dataset.keys():
        print(f"Processing {split} split...")
        
        for idx, sample in enumerate(tqdm(dataset[split], desc=f"Processing {split}")):
            try:
                # Get file path from sample
                if 'file_path' in sample:
                    file_path = sample['file_path']
                elif 'audio' in sample and isinstance(sample['audio'], dict) and 'path' in sample['audio']:
                    file_path = sample['audio']['path']
                else:
                    potential_path_fields = [k for k in sample.keys() if 'path' in k.lower() or 'file' in k.lower()]
                    if potential_path_fields:
                        file_path = sample[potential_path_fields[0]]
                    else:
                        print(f"Cannot find file path in sample keys: {list(sample.keys())}")
                        continue
                
                # Check if this file should be excluded
                basename = os.path.basename(file_path)
                if basename in excluded_files:
                    skipped_files += 1
                    if skipped_files <= 5:
                        print(f"Skipping excluded file: {basename}")
                    elif skipped_files == 6:
                        print("Skipping more excluded files...")
                    continue
                
                # Create a unique file ID
                file_id = f"{split}_{idx}_{os.path.basename(file_path)}"
                
                # Get the label
                if 'label' in sample:
                    label = sample['label']
                elif 'labels' in sample:
                    label = sample['labels']
                else:
                    print(f"Cannot find label in sample keys: {list(sample.keys())}")
                    label = -1  # Unknown label
                
                # Apply binary classification mapping if needed
                if binary_classification and label > 0:  # If MCI or AD, map to Non-Healthy
                    label = 1  # Non-Healthy
                
                # Update class counts
                class_counts[split][label] += 1
                
                # Store file info
                file_info[file_id] = {
                    'path': file_path,
                    'split': split,
                    'class': label,
                    'filename': basename
                }
                
                # Resolve full path and extract features
                try:
                    full_path = resolve_audio_path(file_path)
                    features = extract_features(full_path, feature_extractor)
                    if features is not None:
                        features_dict[file_id] = features
                except FileNotFoundError as e:
                    print(f"File not found: {file_path}")
                    continue
                
            except Exception as e:
                print(f"Error processing sample {idx} in {split} split: {str(e)}")
                continue
            
            # Periodic memory cleanup
            if idx % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    print(f"Successfully extracted features for {len(features_dict)} files")
    print(f"Skipped {skipped_files} files from the exclusion list")
    
    # Print class distribution statistics
    print("\nClass distribution across splits:")
    for split in dataset.keys():
        print(f"  {split}:")
        for cls, count in class_counts[split].items():
            if cls >= 0 and cls < len(class_labels):
                print(f"    {class_labels[cls]}: {count}")
    
    # Clean up memory
    del feature_extractor
    torch.cuda.empty_cache()
    gc.collect()
    
    # Group files by classes
    class_files = {}
    for c in range(len(class_labels)):
        class_files[c] = [fid for fid, info in file_info.items() if info['class'] == c and fid in features_dict]
        print(f"Class {class_labels[c]}: {len(class_files[c])} files with extracted features")
    
    # Prepare to store similarity results
    intra_class_similarities = {c: [] for c in range(len(class_labels))}  # Similarities within same class
    inter_class_similarities = {}  # Similarities between different classes
    for c1 in range(len(class_labels)):
        for c2 in range(c1+1, len(class_labels)):
            inter_class_similarities[(c1, c2)] = []
    
    high_similarity_pairs = []  # Will store problematic pairs
    
    # Analyze intra-class similarities (within same class)
    print("\nAnalyzing intra-class similarities (within the same class)...")
    for c in range(len(class_labels)):
        files = class_files[c]
        print(f"Analyzing {len(files)} files in class {class_labels[c]}...")
        
        # Skip if too few samples
        if len(files) <= 1:
            print(f"  Skipping class {class_labels[c]} - too few samples")
            continue
            
        # Process in batches to avoid memory issues
        batch_size = 100
        for i in tqdm(range(0, len(files), batch_size), desc=f"Class {class_labels[c]} internal comparison"):
            batch1 = files[i:i+batch_size]
            features1 = np.vstack([features_dict[fid] for fid in batch1])
            
            for j in range(i, len(files), batch_size):
                batch2 = files[j:j+batch_size]
                features2 = np.vstack([features_dict[fid] for fid in batch2])
                
                # Compute similarity matrix
                sim_matrix = cosine_similarity(features1, features2)
                
                # Process similarity values
                for batch_i, fid1 in enumerate(batch1):
                    for batch_j, fid2 in enumerate(batch2):
                        # Skip self-comparisons
                        if i == j and batch_i == batch_j:
                            continue
                            
                        similarity = sim_matrix[batch_i, batch_j]
                        intra_class_similarities[c].append(similarity)
                        
                        # Record high similarity pairs within class
                        if similarity >= similarity_threshold:
                            high_similarity_pairs.append({
                                'file1': fid1,
                                'file2': fid2,
                                'split1': file_info[fid1]['split'],
                                'split2': file_info[fid2]['split'],
                                'class1': file_info[fid1]['class'],
                                'class2': file_info[fid2]['class'],
                                'similarity': similarity,
                                'comparison_type': 'intra-class',
                                'filename1': file_info[fid1]['filename'],
                                'filename2': file_info[fid2]['filename'],
                                'class1_name': class_labels[file_info[fid1]['class']],
                                'class2_name': class_labels[file_info[fid2]['class']]
                            })

    # Analyze inter-class similarities (between different classes)
    print("\nAnalyzing inter-class similarities (between different classes)...")
    for c1 in range(len(class_labels)):
        for c2 in range(c1+1, len(class_labels)):
            files1 = class_files[c1]
            files2 = class_files[c2]
            
            print(f"Comparing {len(files1)} files from class {class_labels[c1]} with {len(files2)} files from class {class_labels[c2]}...")
            
            # Process in batches to avoid memory issues
            batch_size = 100
            for i in tqdm(range(0, len(files1), batch_size), desc=f"{class_labels[c1]} vs {class_labels[c2]}"):
                batch1 = files1[i:i+batch_size]
                features1 = np.vstack([features_dict[fid] for fid in batch1])
                
                for j in range(0, len(files2), batch_size):
                    batch2 = files2[j:j+batch_size]
                    features2 = np.vstack([features_dict[fid] for fid in batch2])
                    
                    # Compute similarity matrix
                    sim_matrix = cosine_similarity(features1, features2)
                    
                    # Process similarity values
                    for batch_i, fid1 in enumerate(batch1):
                        for batch_j, fid2 in enumerate(batch2):
                            similarity = sim_matrix[batch_i, batch_j]
                            inter_class_similarities[(c1, c2)].append(similarity)
                            
                            # Record high similarity pairs between different classes - these are problematic
                            if similarity >= similarity_threshold:
                                high_similarity_pairs.append({
                                    'file1': fid1,
                                    'file2': fid2,
                                    'split1': file_info[fid1]['split'],
                                    'split2': file_info[fid2]['split'],
                                    'class1': file_info[fid1]['class'],
                                    'class2': file_info[fid2]['class'],
                                    'similarity': similarity,
                                    'comparison_type': 'inter-class',
                                    'filename1': file_info[fid1]['filename'],
                                    'filename2': file_info[fid2]['filename'],
                                    'class1_name': class_labels[file_info[fid1]['class']],
                                    'class2_name': class_labels[file_info[fid2]['class']]
                                })

    # Sort high similarity pairs by similarity (highest first)
    high_similarity_pairs.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Generate summary report
    print("\n===== Class Similarity Analysis Report =====")
    print(f"Total files analyzed: {len(features_dict)}")
    print(f"High similarity pairs (>= {similarity_threshold}): {len(high_similarity_pairs)}")
    print(f"  Intra-class high similarity pairs: {sum(1 for p in high_similarity_pairs if p['comparison_type'] == 'intra-class')}")
    print(f"  Inter-class high similarity pairs: {sum(1 for p in high_similarity_pairs if p['comparison_type'] == 'inter-class')}")
    
    # Generate statistics for intra-class similarities
    print("\nIntra-class similarity statistics:")
    for c in range(len(class_labels)):
        similarities = intra_class_similarities[c]
        if similarities:
            print(f"  {class_labels[c]}:")
            print(f"    Mean: {np.mean(similarities):.4f}")
            print(f"    Std dev: {np.std(similarities):.4f}")
            print(f"    Min: {np.min(similarities):.4f}")
            print(f"    Max: {np.max(similarities):.4f}")
            print(f"    High similarity pairs: {sum(1 for s in similarities if s >= similarity_threshold)}")
            
    # Generate statistics for inter-class similarities
    print("\nInter-class similarity statistics:")
    for (c1, c2), similarities in inter_class_similarities.items():
        if similarities:
            print(f"  {class_labels[c1]} vs {class_labels[c2]}:")
            print(f"    Mean: {np.mean(similarities):.4f}")
            print(f"    Std dev: {np.std(similarities):.4f}")
            print(f"    Min: {np.min(similarities):.4f}")
            print(f"    Max: {np.max(similarities):.4f}")
            print(f"    High similarity pairs: {sum(1 for s in similarities if s >= similarity_threshold)}")
    
    # Display top high similarity pairs
    print("\nTop high similarity problematic pairs (different classes):")
    inter_class_pairs = [p for p in high_similarity_pairs if p['comparison_type'] == 'inter-class']
    n_to_show = min(10, len(inter_class_pairs))
    for i, pair in enumerate(inter_class_pairs[:n_to_show]):
        print(f"  {i+1}. {pair['filename1']} ({pair['class1_name']}, {pair['split1']}) ↔ "
              f"{pair['filename2']} ({pair['class2_name']}, {pair['split2']}): "
              f"{pair['similarity']:.4f}")
    
    # Display top high similarity pairs within same class but different splits
    print("\nTop high similarity pairs (same class but different splits):")
    cross_split_same_class_pairs = [p for p in high_similarity_pairs 
                                   if p['comparison_type'] == 'intra-class' and p['split1'] != p['split2']]
    n_to_show = min(10, len(cross_split_same_class_pairs))
    for i, pair in enumerate(cross_split_same_class_pairs[:n_to_show]):
        print(f"  {i+1}. {pair['filename1']} ({pair['split1']}) ↔ "
              f"{pair['filename2']} ({pair['split2']}): "
              f"{pair['similarity']:.4f}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. Plot histogram of similarities
    plt.figure(figsize=(10, 6))
    
    # Combine all similarity values by type for plotting
    all_intra_similarities = []
    for c, similarities in intra_class_similarities.items():
        all_intra_similarities.extend(similarities)
    
    all_inter_similarities = []
    for _, similarities in inter_class_similarities.items():
        all_inter_similarities.extend(similarities)
    
    # Plot histograms for both types
    plt.hist(all_intra_similarities, bins=50, alpha=0.7, label=f'Intra-class (n={len(all_intra_similarities)})', color='skyblue')
    plt.hist(all_inter_similarities, bins=50, alpha=0.7, label=f'Inter-class (n={len(all_inter_similarities)})', color='salmon')
    
    plt.axvline(similarity_threshold, color='red', linestyle='--', 
                label=f'Threshold ({similarity_threshold})')
    plt.axvline(np.mean(all_intra_similarities), color='blue', linestyle='-', 
                label=f'Intra-class Mean ({np.mean(all_intra_similarities):.4f})')
    plt.axvline(np.mean(all_inter_similarities), color='darkred', linestyle='-', 
                label=f'Inter-class Mean ({np.mean(all_inter_similarities):.4f})')
    
    plt.title('Distribution of Feature Similarities')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'{output_prefix}_similarity_distribution.png', dpi=300)
    plt.close()
    
        
    # Add 2D clustering visualization with t-SNE
    print("\nCreating 2D clustered representation of embeddings...")
    
    # Collect all feature vectors and metadata
    all_feature_vectors = []
    all_splits = []
    all_classes = []
    all_file_ids = []
    
    for file_id, features in tqdm(features_dict.items(), desc="Preparing for t-SNE"):
        all_feature_vectors.append(features)
        all_splits.append(file_info[file_id]['split'])
        all_classes.append(file_info[file_id]['class'])
        all_file_ids.append(file_id)
    
    # Convert to numpy arrays
    feature_matrix = np.array(all_feature_vectors)
    
    # Apply dimensionality reduction for visualization
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    # First use PCA to reduce dimensionality before applying t-SNE
    print("Reducing dimensionality with PCA...")
    n_components = min(50, feature_matrix.shape[0], feature_matrix.shape[1])
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(feature_matrix)
    
    print(f"Explained variance by PCA: {sum(pca.explained_variance_ratio_):.3f}")
    
    # Apply t-SNE
    print("Applying t-SNE for 2D visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feature_matrix)-1))
    embeddings_2d = tsne.fit_transform(features_pca)
    
    # Add 3D t-SNE visualization 
    print("Applying t-SNE for 3D visualization...")
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=min(30, len(feature_matrix)-1))
    embeddings_3d = tsne_3d.fit_transform(features_pca)
    
    # Create visualizations for clustering by split and by class
    for viz_type in ['split', 'class']:
        plt.figure(figsize=(12, 10))
        
        if viz_type == 'split':
            # Get unique splits and assign colors
            unique_categories = sorted(set(all_splits))
            title = 'Audio Embeddings Clustered by Dataset Split'
            color_field = all_splits
        else:
            # Get unique classes and assign colors
            unique_categories = sorted(set(all_classes))
            title = 'Audio Embeddings Clustered by Class'
            color_field = all_classes
        
        # Create a colormap
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
        color_mapping = {cat: colors[i] for i, cat in enumerate(unique_categories)}
        
        # Plot each category with its own color (2D plot)
        for category in unique_categories:
            indices = [i for i, cat in enumerate(color_field) if cat == category]
            if not indices:
                continue
            plt.scatter(
                embeddings_2d[indices, 0], 
                embeddings_2d[indices, 1],
                c=[color_mapping[category]],
                label=f"{category} (n={len(indices)})",
                alpha=0.7,
                s=50
            )
        
        plt.title(f"2D t-SNE Visualization of {title}", fontsize=14)
        plt.xlabel("t-SNE Dimension 1", fontsize=12)
        plt.ylabel("t-SNE Dimension 2", fontsize=12)
        plt.legend(title="Categories")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save the 2D plot
        filename = f"{output_prefix}_embeddings_by_{viz_type}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved 2D visualization to {filename}")
        
        # Create static 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each category in 3D
        for category in unique_categories:
            indices = [i for i, cat in enumerate(color_field) if cat == category]
            if not indices:
                continue
            
            ax.scatter(
                embeddings_3d[indices, 0],
                embeddings_3d[indices, 1],
                embeddings_3d[indices, 2],
                c=[color_mapping[category]],
                label=f"{category} (n={len(indices)})",
                alpha=0.7,
                s=50
            )
        
        ax.set_title(f"3D t-SNE Visualization of {title}", fontsize=14)
        ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
        ax.set_zlabel("t-SNE Dimension 3", fontsize=12)
        ax.legend(title="Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save the 3D static plot
        filename_3d = f"{output_prefix}_embeddings_3d_by_{viz_type}.png"
        plt.tight_layout()
        plt.savefig(filename_3d, dpi=300)
        plt.close()
        print(f"Saved static 3D visualization to {filename_3d}")
        
        # Create rotating 3D animation for this visualization type
        gif_filename = f"{output_prefix}_embeddings_3d_rotating_{viz_type}.gif"
        create_rotating_3d_plot(
            embeddings_3d, 
            color_field, 
            unique_categories, 
            color_mapping, 
            f"3D t-SNE Visualization of {title}",
            gif_filename
        )
        
        # Create interactive 3D visualization for each viz_type
        html_filename = create_interactive_3d_plot(
            embeddings_3d,
            color_field,
            unique_categories,
            color_mapping,
            f"Interactive 3D t-SNE Visualization of {title}",
            f"{output_prefix}_interactive_3d_{viz_type}.html"
        )
    
    # Create visualization highlighting high similarity pairs
    if high_similarity_pairs:
        print("Creating visualization for high similarity pairs...")
        
        # Create interactive 3D visualization for high similarity pairs
        from plotly.offline import plot
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # First determine which classes to use
        if viz_type == 'split':
            unique_categories = sorted(set(all_splits))
            color_field = all_splits
        else:
            unique_categories = sorted(set(all_classes))
            color_field = all_classes
            
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
        color_mapping = {cat: colors[i] for i, cat in enumerate(unique_categories)}
        
        # Add traces for each category - background points with lower opacity
        for category in unique_categories:
            indices = [i for i, cat in enumerate(all_classes) if cat == category]
            if not indices:
                continue
                
            # Convert matplotlib colors to hex
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(color_mapping[category][0] * 255),
                int(color_mapping[category][1] * 255),
                int(color_mapping[category][2] * 255)
            )
            
            fig.add_trace(go.Scatter3d(
                x=embeddings_3d[indices, 0],
                y=embeddings_3d[indices, 1],
                z=embeddings_3d[indices, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=hex_color,
                    opacity=0.3,
                    line=dict(width=0, color='#FFFFFF')
                ),
                name=f"{category}",
                hoverinfo="skip"  # Hide hover for background points
            ))
        
        # Add high similarity connections for problematic pairs (inter-class)
        problematic_pairs = [p for p in high_similarity_pairs if p['comparison_type'] == 'inter-class']
        for i, pair in enumerate(problematic_pairs[:50]):  # Limit to 50 pairs
            try:
                idx1 = all_file_ids.index(pair['file1'])
                idx2 = all_file_ids.index(pair['file2'])
                
                # Add line trace
                fig.add_trace(go.Scatter3d(
                    x=[embeddings_3d[idx1, 0], embeddings_3d[idx2, 0]],
                    y=[embeddings_3d[idx1, 1], embeddings_3d[idx2, 1]],
                    z=[embeddings_3d[idx1, 2], embeddings_3d[idx2, 2]],
                    mode='lines',
                    line=dict(color='rgba(255,0,0,0.5)', width=2),
                    showlegend=False,
                    hoverinfo="text",
                    hovertext=f"Similarity: {pair['similarity']:.4f}, {pair['class1_name']} vs {pair['class2_name']}"
                ))
                
                # Add highlighted points
                for idx, file_id, split_name, class_name in [(idx1, pair['file1'], pair['split1'], pair['class1_name']), 
                                                        (idx2, pair['file2'], pair['split2'], pair['class2_name'])]:
                    cat = all_classes[idx]
                    hex_color = "#{:02x}{:02x}{:02x}".format(
                        int(color_mapping[cat][0] * 255),
                        int(color_mapping[cat][1] * 255),
                        int(color_mapping[cat][2] * 255)
                    )
                    
                    fig.add_trace(go.Scatter3d(
                        x=[embeddings_3d[idx, 0]],
                        y=[embeddings_3d[idx, 1]],
                        z=[embeddings_3d[idx, 2]],
                        mode='markers',
                        marker=dict(
                            size=7,
                            color=hex_color,
                            opacity=1,
                            line=dict(width=1, color='#000000')
                        ),
                        showlegend=False,
                        hoverinfo="text",
                        hovertext=f"File: {os.path.basename(file_id)}<br>Split: {split_name}<br>Class: {class_name}"
                    ))
                
            except ValueError:
                continue
        
        # Set layout
        fig.update_layout(
            title=f"High Similarity Pairs (sim ≥ {similarity_threshold}) - Different Classes",
            scene=dict(
                xaxis_title="t-SNE Dimension 1",
                yaxis_title="t-SNE Dimension 2",
                zaxis_title="t-SNE Dimension 3"
            ),
            legend=dict(
                title="Classes",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            scene_camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        )
        
        # Save to HTML file
        html_filename = f"{output_prefix}_high_similarity_3d_interactive.html"
        plot(fig, filename=html_filename, auto_open=False)
        print(f"Saved interactive 3D visualization to {html_filename}")
        
        # Also create visualization for cross-split pairs (data leakage)
        cross_split_pairs = [p for p in high_similarity_pairs if p['split1'] != p['split2']]
        if len(cross_split_pairs) > 0:
            fig = go.Figure()
            
            # Create a separate color mapping for splits
            split_categories = sorted(set(all_splits))
            split_colors = plt.cm.Paired(np.linspace(0, 1, len(split_categories)))
            split_color_mapping = {split: split_colors[i] for i, split in enumerate(split_categories)}
            
            # Add background points
            for category in split_categories:
                indices = [i for i, split in enumerate(all_splits) if split == category]
                if not indices:
                    continue
                    
                hex_color = "#{:02x}{:02x}{:02x}".format(
                    int(split_color_mapping[category][0] * 255),
                    int(split_color_mapping[category][1] * 255),
                    int(split_color_mapping[category][2] * 255)
                )
                
                fig.add_trace(go.Scatter3d(
                    x=embeddings_3d[indices, 0],
                    y=embeddings_3d[indices, 1],
                    z=embeddings_3d[indices, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=hex_color,
                        opacity=0.3,
                        line=dict(width=0, color='#FFFFFF')
                    ),
                    name=f"{category}",
                    hoverinfo="skip"
                ))
            
            # Add cross-split connections
            for i, pair in enumerate(cross_split_pairs[:50]):  # Limit to 50 pairs
                try:
                    idx1 = all_file_ids.index(pair['file1'])
                    idx2 = all_file_ids.index(pair['file2'])
                    
                    # Add line trace
                    fig.add_trace(go.Scatter3d(
                        x=[embeddings_3d[idx1, 0], embeddings_3d[idx2, 0]],
                        y=[embeddings_3d[idx1, 1], embeddings_3d[idx2, 1]],
                        z=[embeddings_3d[idx1, 2], embeddings_3d[idx2, 2]],
                        mode='lines',
                        line=dict(color='rgba(255,140,0,0.7)', width=2),
                        showlegend=False,
                        hoverinfo="text",
                        hovertext=f"Similarity: {pair['similarity']:.4f}, {pair['split1']} ↔ {pair['split2']}"
                    ))
                    
                    # Add highlighted points
                    for idx, file_id, split_name in [(idx1, pair['file1'], pair['split1']), 
                                                (idx2, pair['file2'], pair['split2'])]:
                        split_val = all_splits[idx]
                        hex_color = "#{:02x}{:02x}{:02x}".format(
                            int(split_color_mapping[split_val][0] * 255),
                            int(split_color_mapping[split_val][1] * 255),
                            int(split_color_mapping[split_val][2] * 255)
                        )
                        
                        class_name = class_labels[all_classes[idx]] if all_classes[idx] < len(class_labels) else f"Category {all_classes[idx]}"
                        
                        fig.add_trace(go.Scatter3d(
                            x=[embeddings_3d[idx, 0]],
                            y=[embeddings_3d[idx, 1]],
                            z=[embeddings_3d[idx, 2]],
                            mode='markers',
                            marker=dict(
                                size=7,
                                color=hex_color,
                                opacity=1,
                                line=dict(width=1, color='#000000')
                            ),
                            showlegend=False,
                            hoverinfo="text",
                            hovertext=f"File: {os.path.basename(file_id)}<br>Split: {split_name}<br>Class: {class_name}"
                        ))
                    
                except ValueError:
                    continue
            
            # Set layout
            fig.update_layout(
                title=f"Cross-Split High Similarity Pairs (sim ≥ {similarity_threshold}) - Potential Data Leakage",
                scene=dict(
                    xaxis_title="t-SNE Dimension 1",
                    yaxis_title="t-SNE Dimension 2",
                    zaxis_title="t-SNE Dimension 3"
                ),
                legend=dict(
                    title="Splits",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                scene_camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            )
            
            # Save to HTML file
            html_filename = f"{output_prefix}_cross_split_3d_interactive.html"
            plot(fig, filename=html_filename, auto_open=False)
            print(f"Saved cross-split interactive 3D visualization to {html_filename}")
    
    # Save high similarity pairs to CSV for further analysis
    if high_similarity_pairs:
        # Full results
        df = pd.DataFrame(high_similarity_pairs)
        df.to_csv(f'{output_prefix}_high_similarity_pairs.csv', index=False)
        print(f"Saved {len(high_similarity_pairs)} high similarity pairs to {output_prefix}_high_similarity_pairs.csv")
        
        # Just problematic inter-class pairs
        df_problematic = pd.DataFrame([p for p in high_similarity_pairs if p['comparison_type'] == 'inter-class'])
        df_problematic.to_csv(f'{output_prefix}_problematic_pairs.csv', index=False)
        print(f"Saved {len(df_problematic)} problematic inter-class pairs to {output_prefix}_problematic_pairs.csv")
        
        # Cross-split pairs (for data leakage prevention)
        df_cross_split = pd.DataFrame([p for p in high_similarity_pairs if p['split1'] != p['split2']])
        df_cross_split.to_csv(f'{output_prefix}_cross_split_pairs.csv', index=False)
        print(f"Saved {len(df_cross_split)} cross-split pairs to {output_prefix}_cross_split_pairs.csv")
    
    # Return compiled results
    results = {
        'high_similarity_pairs': high_similarity_pairs,
        'intra_class_statistics': {
            c: {
                'mean': np.mean(intra_class_similarities[c]) if intra_class_similarities[c] else None,
                'std': np.std(intra_class_similarities[c]) if intra_class_similarities[c] else None,
                'min': np.min(intra_class_similarities[c]) if intra_class_similarities[c] else None,
                'max': np.max(intra_class_similarities[c]) if intra_class_similarities[c] else None,
                'threshold_count': sum(1 for s in intra_class_similarities[c] if s >= similarity_threshold) if intra_class_similarities[c] else 0,
                'total_count': len(intra_class_similarities[c])
            } for c in range(len(class_labels))
        },
        'inter_class_statistics': {
            f"{c1}_{c2}": {
                'class1': class_labels[c1],
                'class2': class_labels[c2],
                'mean': np.mean(inter_class_similarities[(c1, c2)]) if inter_class_similarities[(c1, c2)] else None,
                'std': np.std(inter_class_similarities[(c1, c2)]) if inter_class_similarities[(c1, c2)] else None,
                'min': np.min(inter_class_similarities[(c1, c2)]) if inter_class_similarities[(c1, c2)] else None,
                'max': np.max(inter_class_similarities[(c1, c2)]) if inter_class_similarities[(c1, c2)] else None,
                'threshold_count': sum(1 for s in inter_class_similarities[(c1, c2)] if s >= similarity_threshold) if inter_class_similarities[(c1, c2)] else 0,
                'total_count': len(inter_class_similarities[(c1, c2)])
            } for c1, c2 in inter_class_similarities.keys()
        },
        'problematic_files': sorted(list(set(
            [pair['file1'] for pair in high_similarity_pairs if pair['comparison_type'] == 'inter-class'] + 
            [pair['file2'] for pair in high_similarity_pairs if pair['comparison_type'] == 'inter-class']
        ))),
        'cross_split_files': sorted(list(set(
            [pair['file1'] for pair in high_similarity_pairs if pair['split1'] != pair['split2']] + 
            [pair['file2'] for pair in high_similarity_pairs if pair['split1'] != pair['split2']]
        )))
    }
    
    print("\nAnalysis complete! Review the generated CSV files and visualizations for detailed results.")
    
    # Provide recommendations
    print("\n===== RECOMMENDATIONS =====")
    
    # Calculate some metrics for recommendations
    total_inter_class_pairs = sum(len(similarities) for similarities in inter_class_similarities.values())
    high_inter_class_pairs = sum(1 for p in high_similarity_pairs if p['comparison_type'] == 'inter-class')
    high_inter_class_ratio = high_inter_class_pairs / total_inter_class_pairs if total_inter_class_pairs > 0 else 0
    
    # Recommendation based on prevalence of high similarity inter-class pairs
    print(f"1. Inter-class similarity issues: {high_inter_class_pairs} high similarity pairs found between different classes")
    if high_inter_class_ratio > 0.02:
        print("   ⚠️ WARNING: High proportion of inter-class similar samples")
        print("   This suggests significant overlap between classes which may affect model performance")
        print("   Consider reviewing these samples manually and potentially excluding them")
    else:
        print("   ✓ The proportion of inter-class similar samples is relatively low")
    
    # Recommendation for different splits
    cross_split_pairs = [p for p in high_similarity_pairs if p['split1'] != p['split2']]
    if len(cross_split_pairs) > 0:
        print(f"\n2. Data leakage risk: {len(cross_split_pairs)} high similarity pairs found across different splits")
        print("   ⚠️ WARNING: Potential data leakage between train/validation/test splits")
        print("   Consider reorganizing your dataset splits to avoid similar samples in different splits")
        print(f"   Review {output_prefix}_cross_split_pairs.csv for details")
    else:
        print("\n2. Data leakage risk: No high similarity pairs found across different splits")
        print("   ✓ Good split integrity - no data leakage detected")
    
    # Class-specific recommendations
    print("\n3. Class-specific recommendations:")
    for (c1, c2), similarities in inter_class_similarities.items():
        high_sim_count = sum(1 for s in similarities if s >= similarity_threshold)
        if high_sim_count > 0:
            ratio = high_sim_count / len(similarities) if similarities else 0
            if ratio > 0.01:
                print(f"   ⚠️ {class_labels[c1]} vs {class_labels[c2]}: {high_sim_count} high similarity pairs ({ratio:.2%})")
                print(f"      This pair of classes shows higher confusion, which may affect classification performance")
    
    return results

if __name__ == "__main__": 

    data_file_path = os.path.join(myConfig.DATA_DIR, "dataframe.csv")   
    dataset_path = myConfig.OUTPUT_PATH
    data_df = pd.read_csv(data_file_path) 

    # To visualize one sample per class:
    #visualize_spectrogram_augmentations(data_df, "/h   ome/bosh/Documents/ML/zz_PP/00_SCTI/Repo/Data")

    # To visualize multiple augmentations of a single sample:
    #visualize_augmentation_examples(data_df, "/home/bosh/Documents/ML/zz_PP/00_SCTI/Repo/Data", n_examples=3)

    # Analyze how different your augmentations are in feature space
    #analyze_augmentation_diversity(data_df, "/home/bosh/Documents/ML/zz_PP/00_SCTI/Repo/Data", n_examples=5)
   
    """ results = analyze_dataset_split_similarity(
        dataset_path=dataset_path,  # Path to your HF dataset
        audio_root_path=myConfig.DATA_DIR,  # Root path to audio files
        model_path=None,  # Set to your model path if a trained model is available
        similarity_threshold=0.95,
        exclusion_csv="/home/bosh/Documents/ML/zz_PP/00_SCTI/Repo/exclude_list.csv"  # Path to exclusion list CSV
    ) """

    results_binary = analyze_class_similarity(
    dataset_path=os.path.join(myConfig.DATA_DIR, "pytorch_dataset"),
    audio_root_path=myConfig.DATA_DIR,
    similarity_threshold=0.95,
    binary_classification=True, 
    output_prefix="binary_analysis",
    #exclusion_csv="/home/bosh/Documents/ML/zz_PP/00_SCTI/Repo/exclude_list.csv"  # Path to exclusion list CSV
)    

    # Example usage of the new function:
    """
    # For 3-way classification (HC/MCI/AD)
    results_3way = analyze_class_similarity(
        dataset_path=myConfig.OUTPUT_PATH,
        audio_root_path=myConfig.DATA_DIR,
        similarity_threshold=0.95,
        binary_classification=False,
        output_prefix="3way_analysis"
    )
    
    # For binary classification (HC vs Non-HC)
    results_binary = analyze_class_similarity(
        dataset_path=myConfig.OUTPUT_PATH,
        audio_root_path=myConfig.DATA_DIR,
        similarity_threshold=0.95,
        binary_classification=True, 
        output_prefix="binary_analysis"
    )
    """

    # Example usage
    """ original_audio_path = "/home/bosh/Documents/ML/zz_PP/00_SCTI/01_ExtractedFiles/MCI/MCI-W-50-205.wav"
    processed_audio_path = "/home/bosh/Documents/ML/zz_PP/00_SCTI/Repo/Data/MCI/MCI-W-50-205.wav"
    visualize_audio_comparison(original_audio_path, processed_audio_path) """
