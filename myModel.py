import torch
import torch.nn as nn
import config
import functions
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


class Wav2Vec2ProsodicClassifier(nn.Module):
    def __init__(self, base_model, num_labels, config, prosodic_dim=7):
        super().__init__()
        self.wav2vec2 = Wav2Vec2ForSequenceClassification.from_pretrained(
            base_model,
            num_labels=num_labels
        )
        if config.training_from_scratch:
            self.config = self.wav2vec2.config  # base model config
        else:
            self.config = config
        self.prosody_mlp = nn.Sequential(
            nn.Linear(prosodic_dim, 32),  # Increase intermediate representation
            nn.ReLU(),
            nn.Linear(32, 16),  # Final projection before concatenation
            nn.ReLU()
        )

        hidden_size = self.wav2vec2.config.hidden_size
        self.fc_combined = nn.Linear(hidden_size + 16, num_labels)

        self.dropout = nn.Dropout(0.1)  # Regularization

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
    if config.training_from_scratch:
        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        base_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3
        )
    else:
        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
        processor = Wav2Vec2Processor.from_pretrained(config.checkpoint_dir)
        base_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3
        )
    return model_name, processor, base_model


def loadModel(model_name):
    if config.training_from_scratch:
        model = Wav2Vec2ProsodicClassifier(model_name, num_labels=3)
    else:
        model_config = AutoConfig.from_pretrained(config.checkpoint_dir)
        model = Wav2Vec2ProsodicClassifier(model_name, num_labels=3, config=model_config)                
        # Load trained weights from .safetensors
        state_dict = load_file(f"{config.checkpoint_dir}/model.safetensors")
        model.load_state_dict(state_dict)
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


def createTrainer(model, optimizer, dataset, weights_tensor):
    # Define the learning rate scheduler
    num_training_steps = config.training_args.num_train_epochs * len(dataset["train"]) // config.training_args.gradient_accumulation_steps

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=100,  # Gradual warmup phase
        num_training_steps=num_training_steps
    )

    # Update Trainer initialization
    trainer = CustomTrainer(
        model=model,
        args=config.training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=functions.compute_metrics,
        data_collator=functions.data_collator_fn,
        optimizers=(optimizer, lr_scheduler),
        class_weights=weights_tensor  # Pass class weights
    )
    return trainer

