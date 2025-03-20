import os
import requests
import shutil
import myConfig
import myFunctions
import myAudio
import numpy as np

from zipfile import ZipFile
#from google.colab import drive
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datasets import load_from_disk
from tqdm import tqdm
from datasets import Dataset, DatasetDict
#drive.mount('/content/drive')
#drive_path = '/content/drive/MyDrive/ModelCheckpoints'
#os.makedirs(drive_path, exist_ok=True)


numeric_cols = [
    "Age",
    "duration",
    "num_pauses",
    "total_pause_duration",
    "phonation_time",
    "speech_rate",
    "dynamic_range_db"
]


def DownloadAndExtract():
    # Always use Data directory at script level
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
    
    # Define paths for category folders within data_dir
    healthy_dir = os.path.join(data_dir, "Healthy")
    mci_dir = os.path.join(data_dir, "MCI")
    ad_dir = os.path.join(data_dir, "AD")
    
    # Create data directory and category subdirectories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(healthy_dir, exist_ok=True)
    os.makedirs(mci_dir, exist_ok=True)
    os.makedirs(ad_dir, exist_ok=True)
    
    # Skip download if files already exist in offline mode
    if myConfig.running_offline and all(os.path.exists(folder) and os.listdir(folder) 
                                     for folder in [healthy_dir, mci_dir, ad_dir]):
        print("Running offline and target folders already exist with files. Skipping download and extraction.")
        return
    
    urls = [
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc2.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc3.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc4.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc5.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc6.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc7.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc8.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc9.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc10.zip",
        "https://ars.els-cdn.com/content/image/1-s2.0-S0885230821001340-mmc11.zip"
    ]

    # Create a temporary folder for extracted files within the data directory
    temp_folder = os.path.join(data_dir, "tmp_extracted")
    os.makedirs(temp_folder, exist_ok=True)

    # Download and extract each zip file
    for i, url in enumerate(urls):
        zip_filename = os.path.join(data_dir, f"downloaded_{i}.zip")  # Store downloads in data directory

        # Download the file
        print(f"Downloading from {url}...")
        response = requests.get(url)
        with open(zip_filename, "wb") as f:
            f.write(response.content)
        print(f"Saved {zip_filename}")

        # Extract all contents into the temp_folder
        print(f"Extracting {zip_filename}...")
        with ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(temp_folder)

        # Delete the ZIP file after extraction to save space
        os.remove(zip_filename)

    # Move files to their destinations, convert from mp3 to wav if needed
    for root, dirs, files in os.walk(temp_folder):
        for filename in files:
            full_path = os.path.join(root, filename)

            # Convert MP3 to WAV if needed
            if filename.endswith(".mp3"):
                wav_filename = filename.replace(".mp3", ".wav")
                wav_path = os.path.join(root, wav_filename)
                # Convert MP3 to WAV
                audio = AudioSegment.from_mp3(full_path)
                audio.export(wav_path, format="wav")
                # Remove the original MP3
                os.remove(full_path)
                # Update full_path to the new WAV file
                full_path = wav_path
                filename = wav_filename

            # Move to corresponding folder
            if filename.startswith("AD"):
                shutil.move(full_path, os.path.join(ad_dir, filename))
            elif filename.startswith("MCI"):
                shutil.move(full_path, os.path.join(mci_dir, filename))
            elif filename.startswith("HC"):
                shutil.move(full_path, os.path.join(healthy_dir, filename))
            else:
                print(f"File '{filename}' doesn't match AD/MCI/HC. Skipping or placing it elsewhere.")

    print("MP3 conversion and file moving completed.")
    # Delete temporary folder
    shutil.rmtree(temp_folder, ignore_errors=True)
    print("Temporary folder removed.")


def datasetSplit(data_df, test_size):
    #First, Drop class feature (label already encodes this info) and Sex (The class is imbalanced)
    data_df = data_df.drop(columns=["class", "Sex"])

    test_size = test_size  # Adjust as needed (0.10 to 0.15)

    # Split data into train/test
    train_df, test_df = train_test_split(
        data_df,
        test_size=test_size,
        stratify=data_df["label"],  # keep class balance
        random_state=42
    )

    # Define new validation size (10-12% of total dataset, relative to new train size)
    val_size = 0.12 / (1 - test_size)  # Convert to fraction of remaining train data

    # Split the train again for validation
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        stratify=train_df["label"],
        random_state=42
    )

    # Print dataset sizes
    print(f"Training set: {len(train_df)} ({len(train_df) / len(data_df) * 100:.2f}%)")
    print(f"Validation set: {len(val_df)} ({len(val_df) / len(data_df) * 100:.2f}%)")
    print(f"Test set: {len(test_df)} ({len(test_df) / len(data_df) * 100:.2f}%)")

    return train_df, val_df, test_df


def loadHFDataset():
    dataset = load_from_disk(myConfig.OUTPUT_PATH)
    return dataset


def ScaleDatasets(train_df, val_df, test_df):
    # Initialize scaler
    scaler = StandardScaler()

    # Fit scaler on TRAIN numeric columns only
    scaler.fit(train_df[numeric_cols])

    # Transform train, val, and test numeric columns
    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
    return train_df, val_df, test_df


def process_data(df):
    data = []
    for row in tqdm(df.itertuples(), total=len(df)):
        audio_file = row.file_path
        label = row.label

        # Load processed audio
        audio, sr = myAudio.load_audio(audio_file)

        # Build a dictionary with everything you need
        # -> the audio array, sampling rate, path, label, plus numeric features
        data.append({
            "audio": {
                "array": np.array(audio, dtype=np.float32),
                "sampling_rate": sr,
                "path": audio_file
            },
            "label": label,
            # For each numeric feature, store the standardized value
            "Age": row.Age,
            "duration": row.duration,
            "num_pauses": row.num_pauses,
            "total_pause_duration": row.total_pause_duration,
            "phonation_time": row.phonation_time,
            "speech_rate": row.speech_rate,
            "dynamic_range_db": row.dynamic_range_db
        })
    return data


def createHFDatasets(train_df, val_df, test_df):
    train_data = process_data(train_df)
    val_data = process_data(val_df)
    test_data = process_data(test_df)

    # Build HF Datasets from lists
    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data),
    })

    # Example chunking or any other map-based transforms
    dataset = dataset.map(myFunctions.chuchunk_input_sample)

    # Finally, save to disk    
    dataset.save_to_disk(myConfig.OUTPUT_PATH)
    print(f"Dataset saved to {myConfig.OUTPUT_PATH}")
