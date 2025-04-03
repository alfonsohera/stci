import matplotlib.pyplot as plt
import seaborn as sns
import myConfig
import librosa
import librosa.display
import numpy as np
import pandas as pd
import os   

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
    
    # Load the same CNN extractor that your model uses
    model = DualPathAudioClassifier(
        num_classes=3,
        sample_rate=16000,
        use_prosodic_features=False,  # Not needed for feature extraction
        apply_specaugment=False  # We'll apply augmentation manually
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
        # Try to find the file using alternative methods as in your existing code
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
        y, sr = librosa.load(full_path, sr=16000)  # Use your model's expected sample rate
        
        # Create log mel spectrogram (similar to your model's preprocessing)
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

def analyze_dataset_split_similarity(dataset_path, audio_root_path, model_path=None, similarity_threshold=0.95):
    """
    Analyzes the similarity between samples across dataset splits (train, valid, test)
    to detect potential data leakage.
    
    Args:
        dataset_path: Path to the HuggingFace dataset directory 
        audio_root_path: Root path to audio files
        model_path: Optional path to a saved model, otherwise uses a fresh model
        similarity_threshold: Cosine similarity threshold above which samples are considered too similar
                            (default: 0.95)
    
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
    from cnn_rnn_model import DualPathAudioClassifier
    import librosa
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset from disk
    print(f"Loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    # Verify dataset structure
    print(f"Dataset keys: {list(dataset.keys())}")
    
    # Print distribution of samples by split
    split_counts = {split: len(dataset[split]) for split in dataset.keys()}
    print(f"Dataset distribution:")
    for split, count in split_counts.items():
        print(f"  {split}: {count} samples")
    
    # Verify data format by examining first sample from each split
    for split in dataset.keys():
        if len(dataset[split]) > 0:
            sample = dataset[split][0]
            print(f"Sample structure in {split} split: {list(sample.keys())}")
    
    # Load or create model
    print("Loading feature extraction model...")
    model = DualPathAudioClassifier(
        num_classes=3,  # Using 3 as defined in your config
        sample_rate=16000,
        use_prosodic_features=False,
        apply_specaugment=False
    ).to(device)
    
    if model_path and os.path.exists(model_path):
        print(f"Loading model weights from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Using model with random weights")
    
    model.eval()  # Set to evaluation mode
    
    # Create feature extraction function
    def extract_features(audio_path):
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Create log mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Convert to tensor
            log_mel_tensor = torch.from_numpy(log_mel_spec).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, freq, time]
            
            with torch.inference_mode():
                # Normalize as in model preprocessing
                log_mel_tensor = (log_mel_tensor - log_mel_tensor.min()) / (log_mel_tensor.max() - log_mel_tensor.min() + 1e-6)
                
                # Resize to model's expected input size
                log_mel_tensor = F.interpolate(log_mel_tensor, size=(384, 384), 
                                             mode='bilinear', align_corners=False)
                
                # Apply standard normalization
                log_mel_tensor = (log_mel_tensor - log_mel_tensor.mean()) / (log_mel_tensor.std() + 1e-5)
                
                # Apply ImageNet normalization
                log_mel_tensor = model.normalize(log_mel_tensor)
                
                # Extract features using CNN
                features = model.cnn_extractor(log_mel_tensor).flatten().cpu().numpy()
                
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
    
    print("Extracting features from all audio files...")
    for split in dataset.keys():
        print(f"Processing {split} split...")
        
        for idx, sample in enumerate(tqdm(dataset[split], desc=f"Processing {split}")):
            try:
                # Check if we have a file_path field directly
                if 'file_path' in sample:
                    file_path = sample['file_path']
                # If not, check if it's in an audio field
                elif 'audio' in sample and isinstance(sample['audio'], dict) and 'path' in sample['audio']:
                    file_path = sample['audio']['path']
                # Last resort, check for any field that might contain a path
                else:
                    potential_path_fields = [k for k in sample.keys() if 'path' in k.lower() or 'file' in k.lower()]
                    if potential_path_fields:
                        file_path = sample[potential_path_fields[0]]
                    else:
                        print(f"Cannot find file path in sample keys: {list(sample.keys())}")
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
                
                # Store file info
                file_info[file_id] = {
                    'path': file_path,
                    'split': split,
                    'class': label
                }
                
                # Resolve full path
                try:
                    full_path = resolve_audio_path(file_path)
                    
                    # Extract features
                    features = extract_features(full_path)
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
    
    # Clean up memory
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Create pairs of files to compare across different splits
    split_comparisons = []
    for i, split1 in enumerate(dataset.keys()):
        for split2 in list(dataset.keys())[i+1:]:
            split_comparisons.append((split1, split2))
    
    print(f"Will compare the following split pairs: {split_comparisons}")
    
    # Store similarity results
    high_similarity_pairs = []
    all_cross_similarities = []
    
    # For each split comparison
    for split1, split2 in split_comparisons:
        print(f"Comparing {split1} vs {split2} split...")
        
        # Get file IDs for each split
        split1_files = [fid for fid, info in file_info.items() if info['split'] == split1 and fid in features_dict]
        split2_files = [fid for fid, info in file_info.items() if info['split'] == split2 and fid in features_dict]
        
        if not split1_files or not split2_files:
            print(f"Skipping comparison - missing files in {split1} or {split2}")
            continue
        
        print(f"  Comparing {len(split1_files)} files from {split1} with {len(split2_files)} files from {split2}")
        
        # Process in smaller batches to avoid memory issues
        batch_size = 100
        
        for i in tqdm(range(0, len(split1_files), batch_size), desc=f"{split1}-{split2} comparison"):
            batch_split1 = split1_files[i:i+batch_size]
            split1_features = np.vstack([features_dict[fid] for fid in batch_split1])
            
            for j in range(0, len(split2_files), batch_size):
                batch_split2 = split2_files[j:j+batch_size]
                split2_features = np.vstack([features_dict[fid] for fid in batch_split2])
                
                # Compute cosine similarity between all pairs in this batch
                similarity_matrix = cosine_similarity(split1_features, split2_features)
                
                # Find high similarity pairs
                for batch_i, fid1 in enumerate(batch_split1):
                    for batch_j, fid2 in enumerate(batch_split2):
                        similarity = similarity_matrix[batch_i, batch_j]
                        all_cross_similarities.append(similarity)
                        
                        if similarity >= similarity_threshold:
                            high_similarity_pairs.append({
                                'file1': fid1,
                                'file2': fid2,
                                'split1': split1,
                                'split2': split2,
                                'class1': file_info[fid1]['class'],
                                'class2': file_info[fid2]['class'],
                                'similarity': similarity
                            })
    
    # Analyze within-split similarity for thoroughness (optional but useful)
    within_split_similarities = {}
    for split in dataset.keys():
        split_files = [fid for fid, info in file_info.items() if info['split'] == split and fid in features_dict]
        
        if len(split_files) <= 1:
            continue
            
        print(f"Analyzing within-split similarity for {split}...")
        within_split_similarities[split] = []
        
        # Process in smaller batches
        batch_size = 100
        for i in tqdm(range(0, len(split_files), batch_size), desc=f"{split} internal comparison"):
            batch1 = split_files[i:i+batch_size]
            features1 = np.vstack([features_dict[fid] for fid in batch1])
            
            for j in range(i, len(split_files), batch_size):
                batch2 = split_files[j:j+batch_size]
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
                        within_split_similarities[split].append(similarity)
                        
                        # Record high similarity pairs within split
                        if similarity >= similarity_threshold:
                            high_similarity_pairs.append({
                                'file1': fid1,
                                'file2': fid2,
                                'split1': split,
                                'split2': split,
                                'class1': file_info[fid1]['class'],
                                'class2': file_info[fid2]['class'],
                                'similarity': similarity
                            })
    
    # Sort high similarity pairs by similarity (highest first)
    high_similarity_pairs.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Generate summary report
    print("\n===== Dataset Split Similarity Analysis Report =====")
    print(f"Total files analyzed: {len(features_dict)}")
    print(f"High similarity pairs (>= {similarity_threshold}): {len(high_similarity_pairs)}")
    
    # Cross-split similarity statistics
    if all_cross_similarities:
        print(f"\nCross-split similarity statistics:")
        print(f"  Mean: {np.mean(all_cross_similarities):.4f}")
        print(f"  Std dev: {np.std(all_cross_similarities):.4f}")
        print(f"  Min: {np.min(all_cross_similarities):.4f}")
        print(f"  Max: {np.max(all_cross_similarities):.4f}")
    
    # Within-split similarity statistics
    print(f"\nWithin-split similarity statistics:")
    for split, similarities in within_split_similarities.items():
        if similarities:
            print(f"  {split}:")
            print(f"    Mean: {np.mean(similarities):.4f}")
            print(f"    Std dev: {np.std(similarities):.4f}")
            print(f"    Min: {np.min(similarities):.4f}")
            print(f"    Max: {np.max(similarities):.4f}")
    
    # Display top high similarity pairs
    n_to_show = min(10, len(high_similarity_pairs))
    if high_similarity_pairs:
        print(f"\nTop {n_to_show} highest similarity pairs:")
        for i, pair in enumerate(high_similarity_pairs[:n_to_show]):
            print(f"  {i+1}. {pair['file1']} ({pair['split1']}, class {pair['class1']}) ↔ "
                  f"{pair['file2']} ({pair['split2']}, class {pair['class2']}): "
                  f"{pair['similarity']:.4f}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. Plot histogram of cross-split similarities
    plt.figure(figsize=(10, 6))
    plt.hist(all_cross_similarities, bins=50, alpha=0.7, color='skyblue')
    plt.axvline(similarity_threshold, color='red', linestyle='--', 
                label=f'Threshold ({similarity_threshold})')
    plt.axvline(np.mean(all_cross_similarities), color='green', linestyle='-', 
                label=f'Mean ({np.mean(all_cross_similarities):.4f})')
    plt.title('Distribution of Cross-Split Feature Similarities')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('cross_split_similarity_histogram.png')
    plt.close()
    
    # 2. Plot distribution of within-split similarities for each split
    plt.figure(figsize=(12, 6))
    for split, similarities in within_split_similarities.items():
        if similarities:
            sns.kdeplot(similarities, label=f'{split} (mean={np.mean(similarities):.4f})')
    plt.axvline(similarity_threshold, color='red', linestyle='--', 
                label=f'Threshold ({similarity_threshold})')
    plt.title('Distribution of Within-Split Feature Similarities')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('within_split_similarity_distribution.png')
    plt.close()
    
    # Return results as a dictionary
    results = {
        'high_similarity_pairs': high_similarity_pairs,
        'cross_split_statistics': {
            'mean': np.mean(all_cross_similarities) if all_cross_similarities else None,
            'std': np.std(all_cross_similarities) if all_cross_similarities else None,
            'min': np.min(all_cross_similarities) if all_cross_similarities else None,
            'max': np.max(all_cross_similarities) if all_cross_similarities else None,
        },
        'within_split_statistics': {
            split: {
                'mean': np.mean(similarities) if similarities else None,
                'std': np.std(similarities) if similarities else None,
                'min': np.min(similarities) if similarities else None,
                'max': np.max(similarities) if similarities else None,
            } for split, similarities in within_split_similarities.items()
        },
        'problematic_files': sorted(list(set(
            [pair['file1'] for pair in high_similarity_pairs] + 
            [pair['file2'] for pair in high_similarity_pairs]
        )))
    }
    
    # Save high similarity pairs to CSV for further analysis
    if high_similarity_pairs:
        df = pd.DataFrame(high_similarity_pairs)
        df.to_csv('high_similarity_pairs.csv', index=False)
        print(f"Saved {len(high_similarity_pairs)} high similarity pairs to high_similarity_pairs.csv")
    
    return results

# Example usage (commented out)


""" data_file_path = os.path.join(myConfig.DATA_DIR, "dataframe.csv")   
dataset_path = myConfig.OUTPUT_PATH
data_df = pd.read_csv(data_file_path) """

# To visualize one sample per class:
#visualize_spectrogram_augmentations(data_df, "/home/bosh/Documents/ML/zz_PP/00_SCTI/Repo/Data")

# To visualize multiple augmentations of a single sample:
#visualize_augmentation_examples(data_df, "/home/bosh/Documents/ML/zz_PP/00_SCTI/Repo/Data", n_examples=3)

# Analyze how different your augmentations are in feature space
#analyze_augmentation_diversity(data_df, "/home/bosh/Documents/ML/zz_PP/00_SCTI/Repo/Data", n_examples=5)


""" results = analyze_dataset_split_similarity(
    dataset_path=dataset_path,  # Path to your HF dataset
    audio_root_path=myConfig.DATA_DIR,  # Root path to audio files
    model_path=None,  # Set to your model path if a trained model is available
    similarity_threshold=0.95
)

# Access results
print(f"Found {len(results['problematic_files'])} potentially problematic files")
 """

# Example usage
""" original_audio_path = "/home/bosh/Documents/ML/zz_PP/00_SCTI/01_ExtractedFiles/MCI/MCI-W-50-205.wav"
processed_audio_path = "/home/bosh/Documents/ML/zz_PP/00_SCTI/Repo/Data/MCI/MCI-W-50-205.wav"
visualize_audio_comparison(original_audio_path, processed_audio_path) """
