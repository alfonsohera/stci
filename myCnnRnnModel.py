import torch
import torch.nn as nn
import torchaudio
import torchvision.models as vision_models
import random
import numpy as np
from typing import Optional, Tuple
from torch.utils.data import Dataset
from collections import Counter
import torch.nn.functional as F
from torchvision import transforms

class SpecAugment(nn.Module):
    """SpecAugment for mel spectrograms as described in the paper:
    "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"
    """
    def __init__(
        self,
        freq_mask_param: int = 30,
        time_mask_param: int = 30,
        n_freq_masks: int = 1,
        n_time_masks: int = 1,
        apply_prob: float = 0.8,
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.apply_prob = apply_prob
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
    
    def forward(self, spec: torch.Tensor, force_apply: bool = False) -> torch.Tensor:
        """Apply SpecAugment to a batch of spectrograms
        
        Args:
            spec: Input spectrogram
            force_apply: If True, always apply augmentation (ignore apply_prob)
        """
        # Skip augmentation with probability 1-apply_prob, unless force_apply is True
        if not force_apply and random.random() > self.apply_prob:
            return spec            
        augmented = spec.clone()
        
        # Apply frequency masks
        for _ in range(self.n_freq_masks):
            augmented = self.freq_mask(augmented)
            
        # Apply time masks
        for _ in range(self.n_time_masks):
            augmented = self.time_mask(augmented)            
        return augmented

class DualPathAudioClassifier(nn.Module):
    def __init__(self, num_classes=3, sample_rate=16000, n_mels=128, 
                 use_prosodic_features=True, prosodic_features_dim=7,
                 apply_specaugment=True):
        super().__init__()
        
        # Spectrogram transform for CNN path
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.target_size = (384, 384)  # Expected size for EfficientNetV2
        self.normalize = transforms.Normalize(mean=[0.485], std=[0.229])  # Single channel version
        
        # SpecAugment for training (only applied when in training mode)
        self.apply_specaugment = apply_specaugment
        if apply_specaugment:
            self.spec_augment = SpecAugment()
        
        # CNN path (using EfficientNetV2-S pretrained)
        self.cnn_extractor = vision_models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        # Modify first layer for single-channel input
        self.cnn_extractor.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), 
                                                      stride=(2, 2), padding=(1, 1), bias=False)
        # Remove classifier
        self.cnn_extractor.classifier = nn.Identity()
        # Freeze CNN weights for feature extraction
        for param in self.cnn_extractor.parameters():
            param.requires_grad = False
            
        # Dimensionality reduction of the outputs of the feature extractor
        self.cnn_dim_reducer = nn.Linear(1280, 256)
        
        # RNN path for raw audio
        self.audio_downsample = nn.Conv1d(1, 8, kernel_size=50, stride=50)  
        self.rnn = nn.GRU(
            input_size=8,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Prosodic features processing
        self.use_prosodic_features = use_prosodic_features
        if use_prosodic_features:
            self.prosodic_feature_mlp = nn.Sequential(
                nn.Linear(prosodic_features_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32)
            )
            fusion_input_dim = 256 + 256 + 32  # CNN + RNN + prosodic
        else:
            fusion_input_dim = 256 + 256  # CNN + RNN only
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, audio, prosodic_features=None, augmentation_id=None):
        """
        Forward pass with optional augmentation ID to control augmentation.
        
        Args:
            audio: Audio input tensor
            prosodic_features: Optional prosodic features
            augmentation_id: Optional ID to control augmentation type/seed
        """
        # Start with processing the audio through the CNN
        # If the audio is just a single channel, expand to [B, 1, T]
        if len(audio.shape) == 2:
            audio = audio.unsqueeze(1)  # Add channel dimension
        
        # Generate mel spectrogram
        mel = self.mel_spec(audio.squeeze(1))
        mel_db = self.amplitude_to_db(mel).unsqueeze(1)  # Add channel dim back
        
        # Apply SpecAugment based on augmentation_id
        if self.training and self.apply_specaugment:
            # Determine which samples in the batch need augmentation
            if isinstance(augmentation_id, list):
                # Handle batched augmentation IDs
                batch_size = mel_db.size(0)
                for i in range(batch_size):
                    # Get the current sample's augmentation ID
                    current_id = augmentation_id[i]
                    if current_id is not None:
                        # Set random seed for reproducible augmentations
                        random.seed(current_id)
                        torch.manual_seed(current_id)
                        # Force augmentation for samples with augmentation_id
                        mel_db[i:i+1] = self.spec_augment(mel_db[i:i+1], force_apply=True)
                        # Reset seed to avoid affecting other randomness
                        random.seed()
                        torch.seed()
                    else:
                        # Apply regular probabilistic augmentation to original samples
                        mel_db[i:i+1] = self.spec_augment(mel_db[i:i+1], force_apply=False)
            elif augmentation_id is not None:
                # Single non-batched augmentation ID
                random.seed(augmentation_id)
                torch.manual_seed(augmentation_id)
                # Force augmentation for all samples in this non-batched case
                mel_db = self.spec_augment(mel_db, force_apply=True)
                random.seed()
                torch.seed()
            else:
                # No augmentation IDs provided, apply regular probabilistic augmentation
                mel_db = self.spec_augment(mel_db, force_apply=False)
        
        # First apply SpecAugment, then resize and normalize to match EfficientNetV2 input
        mel_db = F.interpolate(mel_db, size=self.target_size, mode='bilinear', align_corners=False)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)  # Scale to [0,1]
        mel_db = self.normalize(mel_db)  # Apply ImageNet normalization

        # Pass through CNN 
        cnn_features = self.cnn_extractor(mel_db)
        cnn_features = self.cnn_dim_reducer(cnn_features)
        
        # Process audio for RNN path
        audio_downsampled = self.audio_downsample(audio)
        audio_downsampled = audio_downsampled.transpose(1, 2)  # B, T, C
        
        # Pass through RNN
        rnn_output, _ = self.rnn(audio_downsampled)
        rnn_features = rnn_output[:, -1, :]  
        
        # Combine CNN and RNN features
        combined_features = torch.cat([cnn_features, rnn_features], dim=1)
        
        # Add prosodic features if available and enabled
        if self.use_prosodic_features:
            if prosodic_features is None:
                raise ValueError("Prosodic features are enabled but not provided to the model")
            processed = self.prosodic_feature_mlp(prosodic_features)
            combined_features = torch.cat([combined_features, processed], dim=1)
        
        # Final classification
        output = self.fusion(combined_features)
        return output

class BalancedAugmentedDataset(Dataset):
    """
    A dataset wrapper that balances classes by augmenting underrepresented samples
    with different CNN spectrograms while keeping RNN and prosodic features intact.
    """
    def __init__(self, original_dataset, target_samples_per_class=1000, num_classes=3, 
                 augmentation_variants=3):
        """
        Args:
            original_dataset: The dataset to balance
            target_samples_per_class: Target number of samples per class
            num_classes: Number of classes in the dataset
            augmentation_variants: How many different augmentation variants to create
                                   for each sample when balancing
        """
        self.original_dataset = original_dataset
        self.target_samples_per_class = target_samples_per_class
        self.num_classes = num_classes
        self.augmentation_variants = augmentation_variants
        
        # Count samples per class in the original dataset
        import numpy as np

        # Collect all labels first
        all_labels = np.array([
            sample["label"] if isinstance(sample, dict) else sample[1] 
            for sample in original_dataset
        ])        
        self.class_indices = [np.where(all_labels == i)[0].tolist() for i in range(num_classes)]
        
        # Calculate original class counts
        self.original_class_counts = [len(indices) for indices in self.class_indices]
        print(f"Original class distribution: {self.original_class_counts}")
        
        # Generate augmentation mapping
        # Pre-allocate lists based on final expected size
        total_samples = len(original_dataset) + sum(max(0, target_samples_per_class - count) 
                                                for count in self.original_class_counts)
        self.sample_indices = [0] * total_samples
        self.augmentation_ids = [None] * total_samples

        # Then use indices to fill them instead of append
        idx = 0
        # Add original samples first
        for i in range(len(original_dataset)):
            self.sample_indices[idx] = i
            self.augmentation_ids[idx] = None
            idx += 1

        # Add augmented samples to balance the dataset
        for class_idx in range(num_classes):
            original_count = self.original_class_counts[class_idx]
            if original_count < target_samples_per_class:
                samples_needed = target_samples_per_class - original_count
                
                # Calculate how many of each original sample to use
                repeats_per_sample = (samples_needed + len(self.class_indices[class_idx]) - 1) // len(self.class_indices[class_idx])
                
                # Generate all augmentation indices and IDs in one go
                original_indices = np.repeat(self.class_indices[class_idx], repeats_per_sample)[:samples_needed]
                
                # Generate augmentation IDs vectorized
                aug_variants = np.arange(len(original_indices)) % augmentation_variants
                aug_ids = (class_idx * 100000) + (np.arange(len(original_indices)) * 1000) + aug_variants
                
                # Extend our lists
                self.sample_indices.extend(original_indices.tolist())
                self.augmentation_ids.extend(aug_ids.tolist())
        
        # Calculate final class distribution
        final_distribution = Counter()
        for idx in self.sample_indices:
            sample = original_dataset[idx]
            label = sample["label"] if isinstance(sample, dict) else sample[1]
            final_distribution[label] += 1
        
        # Count how many samples have augmentation IDs
        augmented_count = sum(1 for aid in self.augmentation_ids if aid is not None)
        
        print(f"Balanced class distribution: {[final_distribution[i] for i in range(num_classes)]}")
        print(f"Total samples after balancing: {len(self.sample_indices)}")
        print(f"Original samples: {len(original_dataset)}, Augmented samples: {augmented_count}")
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        # Cache mechanism for frequently accessed augmented samples
        if not hasattr(self, 'sample_cache'):
            self.sample_cache = {}
        
        cache_key = (self.sample_indices[idx], self.augmentation_ids[idx])
        if cache_key in self.sample_cache:
            return self.sample_cache[cache_key]
        
        # Get the original sample
        original_idx = self.sample_indices[idx]
        augmentation_id = self.augmentation_ids[idx]
        
        # Get the sample from the original dataset
        sample = self.original_dataset[original_idx]
        
        # If it's a dictionary, add augmentation_id
        if isinstance(sample, dict):
            sample = sample.copy()  # Create a copy to avoid modifying the original
            sample["augmentation_id"] = augmentation_id
        # If it's a tuple (like audio, label), add augmentation_id
        else:
            # Assuming format is (audio, label) or (audio, prosodic, label)
            if len(sample) == 2:  # (audio, label)
                audio, label = sample
                sample = (audio, label, augmentation_id)
            elif len(sample) == 3:  # (audio, prosodic, label)
                audio, prosodic, label = sample
                sample = (audio, prosodic, label, augmentation_id)
        
        # Add result to cache before returning if it's an augmented sample
        if self.augmentation_ids[idx] is not None:
            self.sample_cache[cache_key] = sample
        
        return sample

    @property
    def class_distribution(self):
        if not hasattr(self, '_class_distribution'):
            self._class_distribution = Counter()
            for idx in self.sample_indices:
                sample = self.original_dataset[idx]
                label = sample["label"] if isinstance(sample, dict) else sample[1]
                self._class_distribution[label] += 1
        return self._class_distribution