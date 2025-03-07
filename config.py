import os
## Misc definitions
training_from_scratch = False
ROOT_DIR = os.getcwd()
DATASET_PATH = ROOT_DIR  # Root folder where audio files are stored
OUTPUT_PATH = os.path.join(ROOT_DIR, "ProcessedFiles")
os.makedirs(OUTPUT_PATH, exist_ok=True)
LABEL_MAP = {"Healthy": 0, "MCI": 1, "AD": 2}

classes = ["HC", "MCI", "AD"]
#Extracted features
features = [
        "num_pauses",
        "total_pause_duration",
        "phonation_time",
        "speech_rate",
        "mean_intensity"
    ]
## Definitions needed for the weighted cross entropy loss function
#Sample weights calculations to compensate for imbalancedd dataset
num_samples_per_class = {
    0: 197,  # Healthy (HC)
    1: 90,   # MCI
    2: 74    # AD
}