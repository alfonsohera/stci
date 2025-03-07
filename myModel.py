import torch.nn as nn
from transformers import(
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor
)
from transformers.modeling_outputs import SequenceClassifierOutput

class Wav2Vec2ProsodicClassifier(nn.Module):
    def __init__(self, base_model, num_labels, prosodic_dim=7):
        super().__init__()
        self.wav2vec2 = Wav2Vec2ForSequenceClassification.from_pretrained(
            base_model,
            num_labels=num_labels
        )
        self.config = self.wav2vec2.config #base model config
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
if training_from_scratch:
  model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
  processor = Wav2Vec2Processor.from_pretrained(model_name)
  base_model = Wav2Vec2ForSequenceClassification.from_pretrained(
      model_name,
      num_labels=3
  )
else:
  model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
  processor = Wav2Vec2Processor.from_pretrained(checkpoint_dir)
  base_model = Wav2Vec2ForSequenceClassification.from_pretrained(
      model_name,
      num_labels=3
  )
