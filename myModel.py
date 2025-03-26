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
from transformers.integrations import WandbCallback
import wandb
from pathlib import Path
from torch.optim.lr_scheduler import OneCycleLR


class Wav2Vec2ProsodicClassifier(nn.Module):
    def __init__(self, base_model, num_labels, config=None, prosodic_dim=None):
        super().__init__()
        self.wav2vec2 = Wav2Vec2ForSequenceClassification.from_pretrained(
            base_model,
            num_labels=num_labels
        )
        if myConfig.training_from_scratch:
            self.config = self.wav2vec2.config  # base model config
        else:
            self.config = config or self.wav2vec2.config
        if prosodic_dim is None:
            prosodic_dim = myConfig.num_extracted_features
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
    ])
    labels = torch.tensor([f["label"] for f in features])

    input_values = pad_sequence(waveforms, batch_first=True, padding_value=0)

    inputs = processor(
        input_values.numpy(),
        sampling_rate=16000,
        padding=True,
        return_tensors="pt"
    )
    inputs["labels"] = labels
    inputs["prosodic_features"] = prosodic_features
    
    # We don't move tensors to device here - Trainer will handle this later
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.freeze_base_model(
        freeze_feature_extractor=True)    
    
    model.gradient_checkpointing_enable()
    lr = myConfig.training_args.learning_rate
    optimizer = Adam8bit(model.parameters(), lr=lr)
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, optimizer


class CustomTrainer(Trainer):
    def __init__(self, model, args, train_dataset, eval_dataset, data_collator, optimizers, class_weights, compute_metrics=None, callbacks=None):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            optimizers=optimizers,
            compute_metrics=compute_metrics,
            callbacks=callbacks  
        )
        # Ensure class_weights are on same device as model
        device = next(model.parameters()).device
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Get model's device
        device = next(model.parameters()).device
        
        # Extract and move inputs to correct device
        labels = inputs.pop("labels").to(device)
        prosodic_features = inputs.pop("prosodic_features").to(device)
        input_values = inputs["input_values"].to(device)
        
        # Forward pass with all tensors on same device
        outputs = model(input_values=input_values, prosodic_features=prosodic_features, labels=labels)
        logits = outputs.logits
        
        # Use the weighted loss function
        loss = self.criterion(logits, labels)
        
        # Explicitly clear some tensors to free memory
        del outputs.hidden_states, outputs.attentions
        torch.cuda.empty_cache()
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        # Get model's device
        device = next(model.parameters()).device
        
        # Extract and move inputs to correct device
        labels = inputs.pop("labels", None)
        if labels is not None:
            labels = labels.to(device)
            
        prosodic_features = inputs.pop("prosodic_features").to(device)
        input_values = inputs["input_values"].to(device)

        with torch.no_grad():
            outputs = model(input_values=input_values, prosodic_features=prosodic_features, labels=labels)

        loss = outputs.loss if outputs.loss is not None else None
        logits = outputs.logits

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    # Use CPU for metrics computation
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu()
        
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
    
    # Get the default learning rate from training_args (default is 5e-5 if not set)
    base_lr = myConfig.training_args.learning_rate
        
    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=base_lr * 10,  # Peak at 10x base learning rate
        total_steps=num_training_steps,
        pct_start=0.3,  # Use 30% of training for warmup
        div_factor=25,  # Initial LR = max_lr/25 (starts low)
        final_div_factor=1e4,  # Final LR = max_lr/10000 (ends very low)
        anneal_strategy='cos',
        three_phase=False
    )

    if not myConfig.running_offline and "wandb" in myConfig.training_args.report_to:
        import wandb
        wandb.init(
            project=myConfig.wandb_project,
            entity=myConfig.wandb_entity,
            name=myConfig.wandb_run_name,
            config={
                "model_name": model.__class__.__name__,
                "total_params": sum(p.numel() for p in model.parameters()),
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "batch_size": myConfig.training_args.per_device_train_batch_size * myConfig.training_args.gradient_accumulation_steps,
                "num_epochs": myConfig.training_args.num_train_epochs,
            }
        )
        # Create custom wandb callback
        wandb_callback = CustomWandbCallback()
        callbacks = [wandb_callback]
    else:
        callbacks = []

    # Update Trainer initialization
    trainer = CustomTrainer(
        model=model,
        args=myConfig.training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator_fn,
        optimizers=(optimizer, lr_scheduler),
        class_weights=weights_tensor,  # Pass class weights
        callbacks=callbacks  
    )
    return trainer


class CustomWandbCallback(WandbCallback):
    """Enhanced W&B callback with model artifact logging"""
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Initialize wandb run and watch model if configured"""
        if self._wandb is None:
            return
        
        # Initialize wandb if not already done
        if not wandb.run:
            self._init_wandb(args, state, model)
            
        # Watch model architecture and gradients
        if myConfig.wandb_watch_model and model is not None:
            wandb.watch(model, log="all", log_freq=100)
            
        # Log hyperparameters
        config = {
            "batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "epochs": args.num_train_epochs,
            "model_name": model.__class__.__name__,
            "optimizer": kwargs.get("optimizer", "Adam8bit"),
        }
        wandb.config.update(config)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation metrics and best model if applicable"""
        super().on_evaluate(args, state, control, metrics, **kwargs)
        
        # Check if this is the best model so far
        if state.best_model_checkpoint is not None and metrics and metrics.get(args.metric_for_best_model, 0) == state.best_metric:
            # Log the best model checkpoint as an artifact
            self._log_best_model(state.best_model_checkpoint)
    
    def _log_best_model(self, checkpoint_dir):
        """Log the best model as a wandb artifact"""
        if not myConfig.wandb_log_model:
            return
            
        # Create model artifact
        artifact_name = f"model-{wandb.run.id}"
        artifact = wandb.Artifact(
            artifact_name, 
            type="model", 
            description=f"Best model checkpoint with {myConfig.training_args.metric_for_best_model}={wandb.run.summary.get(f'eval/{myConfig.training_args.metric_for_best_model}', 0):.4f}"
        )
            
        # Add files to artifact
        for path in Path(checkpoint_dir).glob("**/*"):
            if path.is_file():
                artifact.add_file(str(path), name=str(path.relative_to(checkpoint_dir)))
            
        # Log the artifact
        wandb.log_artifact(artifact)
        wandb.run.summary["best_model_artifact"] = artifact_name

