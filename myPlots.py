import matplotlib.pyplot as plt
import seaborn as sns
import myConfig
import librosa
import librosa.display
import numpy as np


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
    all_features = myConfig.features + myConfig.jitter_shimmer_features
    
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


# Example usage
""" original_audio_path = "/home/bosh/Documents/ML/zz_PP/00_SCTI/01_ExtractedFiles/MCI/MCI-W-50-205.wav"
processed_audio_path = "/home/bosh/Documents/ML/zz_PP/00_SCTI/Repo/Data/MCI/MCI-W-50-205.wav"
visualize_audio_comparison(original_audio_path, processed_audio_path) """
