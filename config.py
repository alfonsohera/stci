import os
from transformers import TrainingArguments


## Misc definitions
training_from_scratch = False
ROOT_DIR = os.getcwd()
DATASET_PATH = ROOT_DIR  # Root folder where audio files are stored
OUTPUT_PATH = os.path.join(ROOT_DIR, "ProcessedFiles")
os.makedirs(OUTPUT_PATH, exist_ok=True)
LABEL_MAP = {"Healthy": 0, "MCI": 1, "AD": 2}
checkpoint_dir = '/content/drive/MyDrive/ModelCheckpoints'

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

training_args = TrainingArguments(
    output_dir="./wav2vec2_classification",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.2,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    fp16=True,
    remove_unused_columns=False,
    gradient_accumulation_steps=4,
    load_best_model_at_end=True,
    auto_find_batch_size=True
)