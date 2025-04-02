import torch
import torch.nn as nn
import torchaudio
import torchvision.models as vision_models
import random
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms

class SpecAugment(nn.Module):
    """SpecAugment for mel spectrograms as described in the paper:
    "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"
    """
    def __init__(
        self,
        freq_mask_param: int = 50,
        time_mask_param: int = 50,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        apply_prob: float = 0.5,
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
        
        # Get the device from input
        device = spec.device
        
        # Move transforms to device if needed        
        freq_mask_device = next(self.freq_mask.parameters(), torch.empty(0)).device
        time_mask_device = next(self.time_mask.parameters(), torch.empty(0)).device
        
        if freq_mask_device != device:
            self.freq_mask = self.freq_mask.to(device)
        if time_mask_device != device:
            self.time_mask = self.time_mask.to(device)
        
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
            hop_length=128,
            n_mels=n_mels
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # SpecAugment for training (only applied when in training mode)
        self.apply_specaugment = apply_specaugment
        if apply_specaugment:
            self.spec_augment = SpecAugment()
        
        # CNN path for mel spectrograms
        self.cnn_extractor = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            
            # Fourth convolutional block 
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            
            # Global average pooling instead of flattening
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # Feature dimension reducer
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
       
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
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Pre-initialize device tracking to avoid device checks at runtime
        self._device = None
        self._initialized_for_device = False
        
    def forward(self, audio, prosodic_features=None, augmentation_id=None):
        """
        Forward pass with optional augmentation ID to control augmentation.
        
        Args:
            audio: Audio input tensor
            prosodic_features: Optional prosodic features
            augmentation_id: Optional ID to control augmentation type/seed
        """
        # Get device from input tensor
        device = audio.device
                
        # Only perform one-time device initialization
        if not self._initialized_for_device or self._device != device:
            self.mel_spec = self.mel_spec.to(device)
            self.amplitude_to_db = self.amplitude_to_db.to(device)
            self.audio_downsample = self.audio_downsample.to(device)
            self.rnn = self.rnn.to(device)            
            self.fusion = self.fusion.to(device)
            
            if self.use_prosodic_features:
                self.prosodic_feature_mlp = self.prosodic_feature_mlp.to(device)
                
            if self.apply_specaugment:
                self.spec_augment = self.spec_augment.to(device)
                
            self._device = device
            self._initialized_for_device = True
        
        # Start with processing the audio through the CNN
        # If the audio is just a single channel, expand to [B, 1, T]
        # Process the entire batch at once
        if len(audio.shape) == 2:
            audio = audio.unsqueeze(1)  # Add channel dimension
               
        mel = self.mel_spec(audio.squeeze(1))
        mel_db = self.amplitude_to_db(mel).unsqueeze(1)  # Add channel dim back
        
        # Apply SpecAugment based on augmentation_id
        if self.training and self.apply_specaugment:            
            if isinstance(augmentation_id, list) and any(aid is not None for aid in augmentation_id):
                # Create a mask of which samples to augment with deterministic seeds
                to_augment = [i for i, aid in enumerate(augmentation_id) if aid is not None]
                no_aug = [i for i, aid in enumerate(augmentation_id) if aid is None]
                
                # Process samples with deterministic seeds
                if to_augment:
                    # Set random seeds once
                    deterministic_batch = mel_db[to_augment]
                    # Force apply augmentation to this entire subset
                    deterministic_batch = self.spec_augment(deterministic_batch, force_apply=True)
                    # Put back into the full batch
                    for i, orig_idx in enumerate(to_augment):
                        mel_db[orig_idx] = deterministic_batch[i]
                    
                # Process samples without deterministic seeds
                if no_aug:
                    # Regular probabilistic augmentation
                    regular_batch = mel_db[no_aug]
                    regular_batch = self.spec_augment(regular_batch, force_apply=False)
                    # Put back into the full batch
                    for i, orig_idx in enumerate(no_aug):
                        mel_db[orig_idx] = regular_batch[i]
            else:
                # Process entire batch with standard augmentation
                mel_db = self.spec_augment(mel_db, force_apply=False)
        
        # Normalization
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-5)
        
        cnn_features = self.cnn_extractor(mel_db)
        
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
    def __init__(self, original_dataset, total_target_samples=1000, num_classes=3):
        self.original_dataset = original_dataset
        self.num_classes = num_classes
        self.target_samples_per_class = total_target_samples // num_classes
        
        # Don't pre-generate augmentations, just store indices
        self.class_indices = {}
        self.class_counts = {}
        
        # Count original samples per class
        for i, item in enumerate(self.original_dataset):
            class_id = item["label"]
            if class_id not in self.class_indices:
                self.class_indices[class_id] = []
                self.class_counts[class_id] = 0
            self.class_indices[class_id].append(i)
            self.class_counts[class_id] += 1
        
        # Calculate needed augmentations
        self.augmentations_needed = {}
        for class_id in range(num_classes):
            orig_count = self.class_counts.get(class_id, 0)
            self.augmentations_needed[class_id] = max(0, self.target_samples_per_class - orig_count)
        
        # Create sample indices and augmentation IDs upfront
        self.sample_indices = []
        self.augmentation_ids = []

        # Add original samples
        for class_id in range(self.num_classes):
            if class_id in self.class_indices:
                for idx in self.class_indices[class_id]:
                    self.sample_indices.append(idx)
                    self.augmentation_ids.append(None)  # No augmentation for original samples

        # Add augmented samples
        for class_id in range(self.num_classes):
            if class_id in self.class_indices and self.augmentations_needed.get(class_id, 0) > 0:
                source_indices = self.class_indices[class_id]
                for aug_id in range(self.augmentations_needed[class_id]):
                    # Pick source sample deterministically
                    source_idx = source_indices[aug_id % len(source_indices)]
                    self.sample_indices.append(source_idx)
                    self.augmentation_ids.append(aug_id)  # Augmentation ID for deterministic augmentation

        # Create small, efficient sample cache
        self.max_cache_size = 100  # Limit cache size to control memory
        self.sample_cache = {}  # For caching augmented samples

        # Create small cache for frequently accessed items
        from collections import OrderedDict
        self.cache = OrderedDict()
        self.cache_size = 50  # Very small cache
        
        # Initialize RNG for reproducible augmentations
        import numpy as np
        self.rng = np.random.RandomState(42)

    def __len__(self):
        total_len = 0
        # Original samples
        for class_id in range(self.num_classes):
            total_len += self.class_counts.get(class_id, 0)
        # Augmented samples
        for class_id in range(self.num_classes):
            total_len += self.augmentations_needed.get(class_id, 0)
        return total_len

    def __getitem__(self, idx):        
        """Get an item from the dataset with proper index handling."""
        from datasets.arrow_dataset import Dataset as ArrowDataset
        
        # Handle dictionary access pattern (dataset["train"]["label"])
        if isinstance(idx, str):
            if idx in ["train", "validation", "test"]:
                # Return self to maintain chaining
                return self
            elif idx == "label":
                # Fast path for HuggingFace dataset labels
                if isinstance(self.original_dataset, ArrowDataset):
                    # Only get as many labels as needed for class weights
                    # Extract smaller batch of labels at a time to avoid memory issues                    
                    batch_size = 500  # Process in smaller batches
                    all_labels = []
                    
                    for start_idx in range(0, len(self.sample_indices), batch_size):
                        end_idx = min(start_idx + batch_size, len(self.sample_indices))
                        batch_indices = self.sample_indices[start_idx:end_idx]
                        try:
                            batch_original_indices = [int(i) for i in batch_indices]
                            batch_labels = self.original_dataset.select(batch_original_indices)["label"]
                            all_labels.extend(list(batch_labels))
                        except Exception as e:
                            print(f"Error extracting labels batch {start_idx}:{end_idx}: {e}")
                            # Fall back for this batch using individual access
                            for i in batch_indices:
                                try:
                                    all_labels.append(self.original_dataset[int(i)]["label"])
                                except:
                                    all_labels.append(-1)  # Mark errors
                    
                    return all_labels
                
                # Existing slower path for other dataset types
                return [self.original_dataset[int(i)]["label"] for i in self.sample_indices]
            
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
        """Calculate class distribution without actually accessing samples"""        
        
        # Use the class counts and augmentations needed to directly calculate
        # the final distribution 
        
        counts = {}
        for class_id in range(self.num_classes):
            original_count = self.class_counts.get(class_id, 0)
            augmented_count = self.augmentations_needed.get(class_id, 0)
            counts[class_id] = original_count + augmented_count
                
        return counts

    def print_distribution_stats(self):
        """Print class distribution statistics before and after augmentation."""
        print("\n=== Class Distribution Statistics ===")
        
        print("Original distribution:")
        total_orig = sum(self.class_counts.values())
        for class_id in sorted(self.class_counts.keys()):
            count = self.class_counts.get(class_id, 0)
            percentage = (count / total_orig * 100) if total_orig > 0 else 0
            print(f"  Class {class_id}: {count} samples ({percentage:.1f}%)")
        
        print("\nAfter augmentation:")
        final_dist = self.class_distribution  
        total_final = sum(final_dist.values())
        for class_id in sorted(final_dist.keys()):
            orig_count = self.class_counts.get(class_id, 0)
            final_count = final_dist[class_id]
            aug_count = final_count - orig_count
            percentage = (final_count / total_final * 100) if total_final > 0 else 0
            print(f"  Class {class_id}: {final_count} samples ({percentage:.1f}%) - Added {aug_count} augmented samples")
        
        print(f"\nTotal samples: {total_orig} → {total_final} (+{total_final-total_orig})")
        print("=====================================\n")