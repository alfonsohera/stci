import os
import wandb
from transformers import TrainingArguments

# Dynamic path configuration
def configure_paths():
    """Set up all path variables based on current environment"""
    # Find the root directory (repository root)
    current_file_path = os.path.abspath(__file__)
    ROOT_DIR = os.path.dirname(current_file_path)
    
    # Define consistent data paths relative to ROOT_DIR
    DATA_DIR = os.path.join(ROOT_DIR, "Data")
    OUTPUT_PATH = os.path.join(ROOT_DIR, "ProcessedFiles")
    checkpoint_dir = os.path.join(ROOT_DIR, "checkpoints")
    
    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return {
        "ROOT_DIR": ROOT_DIR,
        "DATA_DIR": DATA_DIR,
        "OUTPUT_PATH": OUTPUT_PATH,
        "checkpoint_dir": checkpoint_dir
    }

# Initialize with default values
running_offline = True  # Default to offline mode
paths = configure_paths()
ROOT_DIR = paths["ROOT_DIR"]
DATA_DIR = paths["DATA_DIR"] 
OUTPUT_PATH = paths["OUTPUT_PATH"]
checkpoint_dir = paths["checkpoint_dir"]

# Other config variables
training_from_scratch = True
DATASET_PATH = DATA_DIR
LABEL_MAP = {"Healthy": 0, "MCI": 1, "AD": 2}
total_words = 66

reference_text = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, " \
    "no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, " \
    "rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, " \
    "duelos y quebrantos los sábados, lantejas los viernes, algún palomino de añadidura los domingos, " \
    "consumían las tres partes de su hacienda"

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
# The following features were selected after doing feature importance analysis
selected_features = [
"num_pauses", "total_pause_duration", "phonation_time",
"shimmer_local", "skewness", "centre_of_gravity", "wer"
]
num_extracted_features = len(selected_features)


# CNN-RNN model hyperparameters - centralized configuration
cnn_rnn_hyperparams = {
    # Core training hyperparameters
    "max_lr": 0.0010536492254066062,
    "focal_loss_gamma": 1.049466688019162,
    "weight_scaling_factor": 0.41170027130643716,
    "weight_decay": 4.8738206362704404e-05,
    "pct_start": 0.15289459538117114,
    "div_factor": 21.541110287967083,
    "final_div_factor": 264.94546459665844,
    
    # Model-specific hyperparameters
    "attention_dropout":0.21990537650935826,
    "fusion_dropout": 0.25477743209947934,
    "prosodic_weight": 1.4696607599726637,
    "learning_rate_cnn": 0.0001,
    "weight_decay_cnn": 4.8738e-4
}

## Definitions needed for the weighted cross entropy loss function
#Sample weights calculations to compensate for imbalancedd dataset
num_samples_per_class = {
    0: 137,  # Healthy (HC)
    1: 75,   # MCI
    2: 64    # AD
}

# Color mapping for visualization
CLASS_COLORS = {
    "HC": "#2E86C1",   # Blue for Healthy Control
    "MCI": "#F39C12",  # Orange for Mild Cognitive Impairment
    "AD": "#C0392B"    # Red for Alzheimer's Disease
}

# Read batch sizes from environment variables if available
per_device_train_batch_size = int(os.environ.get('BATCH_SIZE_TRAIN', 2))
per_device_eval_batch_size = int(os.environ.get('BATCH_SIZE_EVAL', 2))
train_epochs_num = int(os.environ.get('TRAIN_EPOCHS_VAL', 10))

# Existing wandb configuration
wandb_project = os.environ.get('WANDB_PROJECT', 'cognitive-classifier')
wandb_entity = os.environ.get('WANDB_ENTITY', None)
wandb_run_name = os.environ.get('WANDB_RUN_NAME', f"wav2vec2-prosodic-cls-{os.environ.get('RUN_ID', 'default')}")

# Add new wandb configuration options
wandb_log_model = os.environ.get('WANDB_LOG_MODEL', 'true').lower() == 'true'
wandb_watch_model = os.environ.get('WANDB_WATCH', 'false').lower() == 'true'

# Update training arguments to include model logging
training_args = TrainingArguments(
    output_dir="./training_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    learning_rate=5e-5,
    max_grad_norm=1.0,
    dataloader_num_workers=2,
    dataloader_pin_memory=True, 
    num_train_epochs=train_epochs_num,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    report_to="wandb",
    fp16=True,
    remove_unused_columns=False,
    gradient_accumulation_steps=4,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    auto_find_batch_size=False,
)