import torch
import torch.nn as nn
import numpy as np
import myConfig
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
    Trainer,
    get_scheduler,
    AutoConfig
)
from transformers.modeling_outputs import SequenceClassifierOutput
from safetensors.torch import load_file
from bitsandbytes.optim import Adam8bit
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import classification_report, confusion_matrix

class Wav2Vec2ProsodicClassifier(nn.Module):
    def __init__(self, base_model, num_labels, config=None, prosodic_dim=12): # prosodic_dim needs to match the length of myData.extracted_features!
        super().__init__()
        self.wav2vec2 = Wav2Vec2ForSequenceClassification.from_pretrained(
            base_model,
            num_labels=num_labels
        )
        if myConfig.training_from_scratch:
            self.config = self.wav2vec2.config  # base model config
        else:
            self.config = config or self.wav2vec2.config
            
        self.prosody_mlp = nn.Sequential(
            nn.Linear(prosodic_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        hidden_size = self.wav2vec2.config.hidden_size
        self.fc_combined = nn.Linear(hidden_size + 16, num_labels)
        self.dropout = nn.Dropout(0.1)
    
    def freeze_feature_extractor(self):
        """Freeze the feature extractor part of the model"""
        for param in self.wav2vec2.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
        print("Feature extractor frozen")
            
    def freeze_encoder_layers(self, num_layers_to_freeze):
        """Freeze a specified number of encoder layers from the bottom"""
        if num_layers_to_freeze <= 0:
            return
            
        total_layers = len(self.wav2vec2.wav2vec2.encoder.layers)
        freeze_until = min(num_layers_to_freeze, total_layers)
        
        for i in range(freeze_until):
            for param in self.wav2vec2.wav2vec2.encoder.layers[i].parameters():
                param.requires_grad = False
                
        print(f"First {freeze_until} encoder layers frozen out of {total_layers} total layers")
    
    def freeze_base_model(self, freeze_feature_extractor=True, num_encoder_layers_to_freeze=0):
        """Freeze parts of the base model"""
        if freeze_feature_extractor:
            self.freeze_feature_extractor()
            
        if num_encoder_layers_to_freeze > 0:
            self.freeze_encoder_layers(num_encoder_layers_to_freeze)

    def forward(self, input_values, prosodic_features, attention_mask=None, labels=None, **kwargs):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        wav_embeddings = outputs.hidden_states[-1].mean(dim=1)  # (Batch, hidden_size)

        # Pass prosodic features through MLP
        prosodic_embeddings = self.prosody_mlp(prosodic_features)  # (Batch, 16)

        # Concatenate both feature sets
        combined_features = torch.cat((wav_embeddings, prosodic_embeddings), dim=-1)
        combined_features = self.dropout(combined_features)

        logits = self.fc_combined(combined_features)  # (Batch, num_labels)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
    
    def gradient_checkpointing_enable(self):
        self.wav2vec2.gradient_checkpointing_enable()


def getModelDefinitions():
    if myConfig.training_from_scratch:
        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        base_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3
        )
    else:
        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
        processor = Wav2Vec2Processor.from_pretrained(myConfig.checkpoint_dir)
        base_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3
        )
    return model_name, processor, base_model


def data_collator_fn(features):
    _, processor, _  = getModelDefinitions()
    waveforms = [torch.tensor(f["audio"]["array"]) for f in features]
    prosodic_features = torch.stack([
        torch.tensor(f["prosodic_features"], dtype=torch.float) for f in features
    ])  # Now each prosodic_features is converted to a tensor
    labels = torch.tensor([f["label"] for f in features])

    input_values = pad_sequence(waveforms, batch_first=True, padding_value=0)

    inputs = processor(
        input_values.numpy(),
        sampling_rate=16000,
        padding=True,
        return_tensors="pt"
    )
    inputs["labels"] = labels
    inputs["prosodic_features"] = prosodic_features  # Add prosodic features
    return inputs


def loadModel(model_name):
    if myConfig.training_from_scratch:
        model = Wav2Vec2ProsodicClassifier(model_name, num_labels=3)
    else:
        model_config = AutoConfig.from_pretrained(myConfig.checkpoint_dir)
        model = Wav2Vec2ProsodicClassifier(model_name, num_labels=3, config=model_config)                
        # Load trained weights from .safetensors
        state_dict = load_file(f"{myConfig.checkpoint_dir}/model.safetensors")
        model.load_state_dict(state_dict)
        
    # Apply freezing strategy
    model.freeze_base_model(
        freeze_feature_extractor=True,  # Freeze feature extractor
        #num_encoder_layers_to_freeze=12  # Freeze first 12 encoder layers (12/24)
    )
    
    model.gradient_checkpointing_enable()
    optimizer = Adam8bit(model.parameters(), lr=2e-5)
    return model, optimizer


class CustomTrainer(Trainer):
    def __init__(self, model, args, train_dataset, eval_dataset, data_collator, optimizers, class_weights, compute_metrics=None):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            optimizers=optimizers,
            compute_metrics=compute_metrics
        )
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)  # Set the weighted loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        prosodic_features = inputs.pop("prosodic_features")
        outputs = model(input_values=inputs["input_values"], prosodic_features=prosodic_features, labels=labels)
        logits = outputs.logits
        # Use the weighted loss function
        loss = self.criterion(logits, labels)
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        labels = inputs.pop("labels", None)  # Remove labels from inputs
        prosodic_features = inputs.pop("prosodic_features")

        with torch.no_grad():
            outputs = model(input_values=inputs["input_values"], prosodic_features=prosodic_features, labels=labels)

        loss = outputs.loss if outputs.loss is not None else None
        logits = outputs.logits

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    # Ensure predictions and labels are NumPy arrays
    preds = torch.argmax(torch.tensor(logits), dim=-1).numpy()

    # Compute classification report
    report = classification_report(labels, preds, target_names=["Healthy", "MCI", "AD"], output_dict=True)

    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)

    # Extract TP, FP, TN, FN per class
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    # Compute Specificity (TNR) and Negative Predictive Value (NPV) per class
    specificity = TN / (TN + FP + 1e-10)  # True Negative Rate
    npv = TN / (TN + FN + 1e-10)  # Negative Predictive Value

    # Store per-class and macro-average results
    results = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_specificity": np.mean(specificity),
        "macro_npv": np.mean(npv),
        "f1_healthy": report["Healthy"]["f1-score"],
        "f1_mci": report["MCI"]["f1-score"],
        "f1_ad": report["AD"]["f1-score"],
        "specificity_healthy": specificity[0],
        "specificity_mci": specificity[1],
        "specificity_ad": specificity[2],
        "npv_healthy": npv[0],
        "npv_mci": npv[1],
        "npv_ad": npv[2]
    }
    return results


def createTrainer(model, optimizer, dataset, weights_tensor):
    # Define the learning rate scheduler
    num_training_steps = myConfig.training_args.num_train_epochs * len(dataset["train"]) // myConfig.training_args.gradient_accumulation_steps

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=100,  # Gradual warmup phase
        num_training_steps=num_training_steps
    )

    # Update Trainer initialization
    trainer = CustomTrainer(
        model=model,
        args=myConfig.training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator_fn,
        optimizers=(optimizer, lr_scheduler),
        class_weights=weights_tensor  # Pass class weights
    )
    return trainer

