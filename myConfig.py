import os
from transformers import TrainingArguments

# Define data directory path
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")

running_offline = False
## Misc definitions
training_from_scratch = True
ROOT_DIR = os.getcwd()
DATASET_PATH = DATA_DIR  # Root folder where audio files are stored
OUTPUT_PATH = os.path.join(ROOT_DIR, "ProcessedFiles")
os.makedirs(OUTPUT_PATH, exist_ok=True)
LABEL_MAP = {"Healthy": 0, "MCI": 1, "AD": 2}
checkpoint_dir = '/content/drive/MyDrive/ModelCheckpoints'
total_words = 66 # Total number of words in the script

reference_text = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, " \
    "no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, " \
    "rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, " \
    "duelos y quebrantos los sábados, lantejas los viernes, algún palomino de añadidura los domingos, " \
    "consumían las tres partes de su hacienda"

hf_token = "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"

# Define the classes and features to be used
classes = ["HC", "MCI", "AD"]
#Extracted prosodic features
features = [
        "num_pauses",
        "total_pause_duration",
        "phonation_time",
        "speech_rate",
        "dynamic_range_db"
    ]
#extracted jitter and shimmer features
jitter_shimmer_features = ["jitter_local", "shimmer_local"]
spectral_features = ["skewness", "centre_of_gravity"]
speech2text_features = ["wer"]
## Definitions needed for the weighted cross entropy loss function
#Sample weights calculations to compensate for imbalancedd dataset
num_samples_per_class = {
    0: 197,  # Healthy (HC)
    1: 90,   # MCI
    2: 74    # AD
}

# Color mapping for visualization
CLASS_COLORS = {
    "HC": "#2E86C1",   # Blue for Healthy Control
    "MCI": "#F39C12",  # Orange for Mild Cognitive Impairment
    "AD": "#C0392B"    # Red for Alzheimer's Disease
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