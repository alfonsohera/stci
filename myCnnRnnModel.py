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
    def __init__(self, original_dataset, target_samples_per_class=None, total_target_samples=1000, 
                 num_classes=3, augmentation_variants=3, cache_size=1000):
        """
        Args:
            original_dataset: The dataset to balance
            target_samples_per_class: Target number of samples per class (if None, derived from total_target_samples)
            total_target_samples: Total target samples for the dataset (ignored if target_samples_per_class is set)
            num_classes: Number of classes in the dataset
            augmentation_variants: How many different augmentation variants to create
                                   for each sample when balancing
            cache_size: Maximum number of augmented samples to cache (0 = no cache)
        """
        import numpy as np
        
        self.original_dataset = original_dataset
        self.num_classes = num_classes
        self.augmentation_variants = augmentation_variants
        self.max_cache_size = cache_size
        self.sample_cache = {}  # Initialize cache
        
        # Extract label information
        all_labels = np.array([
            sample["label"] if isinstance(sample, dict) else sample[1] 
            for sample in original_dataset
        ])
        
        # Create class indices using numpy for speed
        self.class_indices = [np.where(all_labels == i)[0].tolist() for i in range(self.num_classes)]
        
        # Calculate original class counts
        self.original_class_counts = [len(indices) for indices in self.class_indices]
        original_total = sum(self.original_class_counts)
        
        # Determine target samples per class
        if target_samples_per_class is None:
            # Set target per class based on total samples divided across classes
            samples_per_class = total_target_samples // num_classes
            self.target_samples_per_class = [samples_per_class] * num_classes
            
            # Ensure we don't create a dataset beyond the total_target_samples
            if sum(self.target_samples_per_class) > total_target_samples:
                # Reduce the last class slightly if needed for exact fit
                self.target_samples_per_class[-1] -= (sum(self.target_samples_per_class) - total_target_samples)
        else:
            # Use specified target per class but ensure we don't exceed total_target_samples
            self.target_samples_per_class = [min(target_samples_per_class, total_target_samples // num_classes)] * num_classes
        
        # Build sample mapping 
        self._build_sample_mapping()
        
        # Log information about the dataset
        self._log_dataset_info()
    
    def _build_sample_mapping(self):
        """Build sample indices and augmentation IDs efficiently"""
        import numpy as np
        
        # Calculate total samples (just use the original calculation)
        total_samples = sum(self.target_samples_per_class)
        
        # Pre-allocate arrays for indices and augmentation IDs 
        self.sample_indices = np.zeros(total_samples, dtype=np.int32)
        self.augmentation_ids = np.array([None] * total_samples, dtype=object)
        
        # Fill arrays class by class (more efficient implementation)
        current_idx = 0
        for class_idx in range(self.num_classes):
            original_count = self.original_class_counts[class_idx]
            target_count = self.target_samples_per_class[class_idx]
            source_indices = np.array(self.class_indices[class_idx])
            
            if original_count >= target_count:
                # If we have enough original samples, randomly select subset
                selected_indices = np.random.choice(source_indices, target_count, replace=False)
                end_idx = current_idx + target_count
                self.sample_indices[current_idx:end_idx] = selected_indices
                # self.augmentation_ids already initialized to None
            else:
                # Add all original samples
                end_idx = current_idx + original_count
                self.sample_indices[current_idx:end_idx] = source_indices
                
                # Add augmented samples if needed
                samples_needed = target_count - original_count
                if samples_needed > 0:
                    # Generate augmentation
                    repeats = (samples_needed + original_count - 1) // original_count
                    aug_indices = np.repeat(source_indices, repeats)[:samples_needed]
                    aug_variants = np.arange(samples_needed) % self.augmentation_variants
                    aug_ids = (class_idx * 100000) + (np.arange(samples_needed) * 1000) + aug_variants
                    
                    # Store in arrays
                    aug_end_idx = end_idx + samples_needed
                    self.sample_indices[end_idx:aug_end_idx] = aug_indices
                    self.augmentation_ids[end_idx:aug_end_idx] = aug_ids
                    end_idx = aug_end_idx
            
            current_idx = end_idx
        
        # Trim arrays if needed
        if current_idx < total_samples:
            self.sample_indices = self.sample_indices[:current_idx]
            self.augmentation_ids = self.augmentation_ids[:current_idx]
    
    def _log_dataset_info(self):
        """Log information about the dataset with detailed class breakdown"""
        import numpy as np
        
        # Calculate final class distribution
        class_counts = np.zeros(self.num_classes, dtype=np.int32)
        original_per_class = np.zeros(self.num_classes, dtype=np.int32)
        augmented_per_class = np.zeros(self.num_classes, dtype=np.int32)
        
        # Count original samples per class
        for i in range(self.num_classes):
            original_per_class[i] = self.original_class_counts[i]
        
        # Count augmented samples per class
        non_none_mask = self.augmentation_ids != None
        if np.any(non_none_mask):
            valid_ids = self.augmentation_ids[non_none_mask]
            for i in range(self.num_classes):
                class_mask = valid_ids // 100000 == i
                augmented_per_class[i] = np.sum(class_mask)
        
        # Calculate total counts
        for i in range(self.num_classes):
            class_counts[i] = original_per_class[i] + augmented_per_class[i]
        
        # Calculate percentages
        total_samples = len(self.sample_indices)
        original_total = len(self.original_dataset)
        augmented_total = np.sum(self.augmentation_ids != None)
        
        # Print detailed information
        print("=" * 70)
        print("DATASET BALANCING REPORT")
        print("=" * 70)
        print(f"Original dataset: {original_total} samples")
        print(f"Balanced dataset: {total_samples} samples")
        print(f"Added {augmented_total} augmented samples")
        print("-" * 70)
        print("CLASS BREAKDOWN:")
        
        # Create a nice table format
        print(f"{'Class':<10}{'Original':<12}{'Augmented':<12}{'Total':<12}{'Percentage':<12}")
        print("-" * 70)
        for i in range(self.num_classes):
            percentage = (class_counts[i] / total_samples) * 100
            print(f"{i:<10}{original_per_class[i]:<12}{augmented_per_class[i]:<12}"
                  f"{class_counts[i]:<12}{percentage:.1f}%")
        
        print("-" * 70)
        print(f"Class distribution before: {self.original_class_counts}")
        print(f"Class distribution after:  {class_counts.tolist()}")
        print("=" * 70)
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        """Get an item from the dataset with proper index handling."""
        import numpy as np
        from datasets.arrow_dataset import Dataset as ArrowDataset
        
        # Handle dictionary access pattern (dataset["train"]["label"])
        if isinstance(idx, str):
            if idx in ["train", "validation", "test"]:
                # Return self to maintain chaining
                return self
            elif idx == "label":
                # Fast path for HuggingFace dataset labels
                if isinstance(self.original_dataset, ArrowDataset):
                    # Extract all labels at once from the original dataset
                    all_original_indices = [int(i) for i in self.sample_indices]
                    try:
                        all_labels = self.original_dataset.select(all_original_indices)["label"]
                        return list(all_labels)  # Convert to Python list for compatibility
                    except Exception as e:
                        print(f"Error extracting labels from HuggingFace dataset: {e}")
                        # Fall back to slow path if this fails
                
                # Slower path for other dataset types
                labels = []
                for i in range(len(self.sample_indices)):
                    try:
                        original_idx = int(self.sample_indices[i])
                        sample = self.original_dataset[original_idx]
                        if isinstance(sample, dict):
                            label = sample["label"]
                        elif hasattr(sample, '__getitem__') and len(sample) > 1:
                            label = sample[1]
                        else:
                            raise ValueError(f"Unexpected sample format: {type(sample)}")
                        labels.append(label)
                    except Exception as e:
                        print(f"Error processing label at index {i}: {str(e)}")
                        labels.append(-1)  # Add placeholder for error cases
                return labels
            else:
                # Fast path for HuggingFace dataset attributes
                if isinstance(self.original_dataset, ArrowDataset) and idx in self.original_dataset.column_names:
                    all_original_indices = [int(i) for i in self.sample_indices]
                    try:
                        all_values = self.original_dataset.select(all_original_indices)[idx]
                        return list(all_values)
                    except Exception as e:
                        print(f"Error extracting {idx} from HuggingFace dataset: {e}")
                        # Fall back to slow path if this fails
                
                # Slower path for other dataset types or attributes
                all_values = []
                for i in range(len(self.sample_indices)):
                    try:
                        original_idx = int(self.sample_indices[i])
                        sample = self.original_dataset[original_idx]
                        if isinstance(sample, dict) and idx in sample:
                            all_values.append(sample[idx])
                        else:
                            all_values.append(None)
                    except Exception as e:
                        all_values.append(None)
                return all_values
        
        # Regular integer index access for DataLoader
        try:
            if isinstance(idx, (torch.Tensor, np.ndarray)):
                idx = idx.item() if hasattr(idx, 'item') else int(idx)
            elif not isinstance(idx, int):
                idx = int(idx)
            
            # Check if index is in bounds
            if idx < 0 or idx >= len(self.sample_indices):
                raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.sample_indices)}")
            
            # Get data from original dataset with augmentation ID
            original_idx = int(self.sample_indices[idx])
            augmentation_id = self.augmentation_ids[idx]
            
            # Return cached result if available (for augmented samples)
            if augmentation_id is not None:
                cache_key = (original_idx, augmentation_id)
                if cache_key in self.sample_cache:
                    return self.sample_cache[cache_key]
            
            # Get the sample from the original dataset
            sample = self.original_dataset[original_idx]
            
            # Process the sample based on its format
            if isinstance(sample, dict):
                # Handle dictionary format
                result = dict(sample)  # Create a copy to avoid modifying the original
                result["augmentation_id"] = augmentation_id
            else:
                # Handle tuple format
                if len(sample) == 2:  # (audio, label)
                    audio, label = sample
                    result = (audio, label, augmentation_id)
                elif len(sample) == 3:  # (audio, prosodic, label)
                    audio, prosodic, label = sample
                    result = (audio, prosodic, label, augmentation_id)
                else:
                    raise ValueError(f"Unexpected sample format with {len(sample)} elements")
            
            # Cache result if it's an augmented sample
            if augmentation_id is not None and self.max_cache_size > 0:
                cache_key = (original_idx, augmentation_id)
                self.sample_cache[cache_key] = result
                
                # Trim cache if it gets too large
                if len(self.sample_cache) > self.max_cache_size:
                    # Remove oldest items (approximation of LRU)
                    for _ in range(len(self.sample_cache) - self.max_cache_size):
                        self.sample_cache.pop(next(iter(self.sample_cache)))
            
            return result
        except Exception as e:
            print(f"Error in __getitem__ with index {idx}: {str(e)}")
            raise

    @property
    def class_distribution(self):
        """Calculate class distribution on demand"""
        import numpy as np
        
        if not hasattr(self, '_cached_distribution'):
            # Count samples by class
            counts = {i: 0 for i in range(self.num_classes)}
            
            for i in range(len(self.sample_indices)):
                idx = self.sample_indices[i]
                sample = self.original_dataset[idx]
                label = sample["label"] if isinstance(sample, dict) else sample[1]
                counts[label] += 1
                
            self._cached_distribution = counts
            
        return self._cached_distribution