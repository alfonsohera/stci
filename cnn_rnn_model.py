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

class SelfAttention(nn.Module):
    """Self-attention module to replace GRU"""
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and normalization
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual connection and normalization
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x


class DualPathAudioClassifier(nn.Module):
    def __init__(self, num_classes=3, sample_rate=16000, n_mels=128, apply_specaugment=True):
        super(DualPathAudioClassifier, self).__init__()        
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.apply_specaugment = apply_specaugment
        
        # Create mel spectrogram converter
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels
        )
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # SpecAugment for data augmentation during training
        if apply_specaugment:
            self.spec_augment = SpecAugment(
                time_mask_param=30,
                freq_mask_param=20,
                p=0.5
            )
    
        # CNN path
        self.cnn_extractor = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            # Fourth convolutional block 
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            # Global average pooling 
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Downsample raw audio for attention path - keeping original dimensions
        self.audio_downsample = nn.Conv1d(1, 8, kernel_size=50, stride=25)
        
        # Attention layers instead of RNN
        self.position_embedding = nn.Parameter(torch.randn(1, 128, 8))  # Max seq length 128, feature dim 8
        
        # Self-attention layers
        self.attention_layers = nn.ModuleList([
            SelfAttention(embed_dim=8, num_heads=2, dropout=0.1)
            for _ in range(2)  # 2 attention layers
        ])
        
        # Attention output processing
        self.attention_pooling = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Fusion layer (CNN features + Attention features)
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256, 256),  # CNN (256) + Attention (256)
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Separate classifier layer
        self.classifier = nn.Linear(128, num_classes)
        
        # Pre-initialize device tracking to avoid device checks at runtime
        self._device = None
        self._initialized_for_device = False

    def forward(self, audio, audio_lengths=None, augmentation_id=None):
        """
        Forward pass for chunked audio processing
        
        Args:
            audio: Input audio tensor [B, C, T]
            audio_lengths: Tensor of actual audio lengths [B]
            augmentation_id: Optional IDs for deterministic augmentation
            
        Returns:
            Class logits [B, num_classes]
        """
        # Initialize for specific device if needed
        device = audio.device
        
        if not self._initialized_for_device or self._device != device:
            self.mel_spec = self.mel_spec.to(device)
            self.amplitude_to_db = self.amplitude_to_db.to(device)
            
            if self.apply_specaugment:
                self.spec_augment = self.spec_augment.to(device)
                
            self._device = device
            self._initialized_for_device = True
        
        # Start with processing the audio through the CNN
        # If the audio is just a single channel, expand to [B, 1, T]
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
        
        # CNN feature extraction
        cnn_features = self.cnn_extractor(mel_db)  # [B, 256]
        
        # Process audio for Attention path
        audio_downsampled = self.audio_downsample(audio.squeeze(1))  # [B, 8, T/25]
        audio_downsampled = audio_downsampled.transpose(1, 2)  # [B, T/25, 8]
        
        # Add positional embeddings
        seq_len = audio_downsampled.shape[1]
        audio_downsampled = audio_downsampled + self.position_embedding[:, :seq_len, :]
        
        # Create attention mask if audio_lengths is provided
        attention_mask = None
        if audio_lengths is not None:
            # Convert to downsampled lengths
            downsampled_lengths = (audio_lengths / 25).long()
            max_len = audio_downsampled.shape[1]
            
            # Create padding mask (True for padding positions)
            attention_mask = torch.arange(max_len, device=device)[None, :] >= downsampled_lengths[:, None]
        
        # Process through attention layers
        attn_output = audio_downsampled
        for attention_layer in self.attention_layers:
            attn_output = attention_layer(attn_output, mask=attention_mask)
        
        # Pool attention outputs - use mean pooling over sequence dimension
        if attention_mask is not None:
            # Apply mask before pooling (set padded positions to 0)
            mask_expanded = (~attention_mask).float().unsqueeze(-1)
            attn_output = attn_output * mask_expanded
            # Mean over non-padded positions
            attn_features = attn_output.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            # Simple mean pooling if no mask
            attn_features = attn_output.mean(dim=1)
        
        # Process attention features through linear layer
        attn_features = self.attention_pooling(attn_features)  # [B, 256]
        
        # Combine CNN and Attention features
        combined_features = torch.cat([cnn_features, attn_features], dim=1)  # [B, 512]
        
        # Final classification
        fusion_output = self.fusion(combined_features)  # [B, 128]
        output = self.classifier(fusion_output)  # [B, num_classes]
        
        return output
    
    def aggregate_chunk_predictions(self, chunk_outputs):
        """
        Aggregate predictions from multiple chunks of the same audio
        
        Args:
            chunk_outputs: List of prediction tensors, each of shape [num_classes]
            
        Returns:
            Aggregated prediction of shape [num_classes]
        """
        # Simple mean aggregation
        return torch.stack(chunk_outputs).mean(dim=0)


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

        # Add attributes for HuggingFace compatibility
        self._supports_huggingface_api = True
        from datasets.arrow_dataset import Dataset as ArrowDataset
        self._is_hf_dataset = isinstance(original_dataset, ArrowDataset)
        
        # Cache the column_names for faster access
        if hasattr(original_dataset, 'column_names'):
            self._column_names = original_dataset.column_names
        else:
            # Try to determine column names from the first item
            if len(original_dataset) > 0:
                first_item = original_dataset[0]
                if isinstance(first_item, dict):
                    self._column_names = list(first_item.keys())
                else:
                    # Default column names if we can't determine them
                    self._column_names = ["audio", "label"]
            else:
                self._column_names = ["audio", "label"]

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
                        if self._is_hf_dataset:
                            # Use select for HuggingFace datasets
                            sample = self.original_dataset.select([original_idx])[0]
                        else:
                            # Use direct indexing for other dataset types
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
            if self._is_hf_dataset:
                # Use select method for HuggingFace datasets which is more reliable
                sample = self.original_dataset.select([original_idx])[0]
            else:
                # Use direct indexing for other dataset types
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

    @property
    def column_names(self):
        """Return the column names to be compatible with HF datasets."""
        return self._column_names
    
    @property
    def features(self):
        """Return the features dictionary from the original dataset if available."""
        if hasattr(self.original_dataset, 'features'):
            return self.original_dataset.features
        return None
    
    def select(self, indices):
        """Implement select method for HF datasets compatibility."""
        from datasets import Dataset as HFDataset
        
        # Convert indices to list if it's not already
        if not isinstance(indices, list):
            indices = [indices]
        
        # Map indices to our internal sample_indices
        selected_indices = [int(self.sample_indices[i]) for i in indices 
                           if i >= 0 and i < len(self.sample_indices)]
        
        # Get augmentation IDs for these indices
        selected_aug_ids = [self.augmentation_ids[i] for i in indices 
                           if i >= 0 and i < len(self.augmentation_ids)]
        
        # Create a new dataset with these indices
        selected_data = []
        for idx, aug_id in zip(selected_indices, selected_aug_ids):
            sample = self.original_dataset[idx]
            if isinstance(sample, dict):
                result = dict(sample)  # Make a copy
                if aug_id is not None:
                    result["augmentation_id"] = aug_id
                selected_data.append(result)
                
        # If using HF datasets, try to convert back to an HFDataset
        if self._is_hf_dataset:
            try:
                return HFDataset.from_dict({k: [item[k] for item in selected_data] 
                                           for k in self.column_names if all(k in item for item in selected_data)})
            except:
                # Fallback to returning the list of dictionaries
                return selected_data
        
        return selected_data
    
    def map(self, function, batched=False, batch_size=None, **kwargs):
        """Implement map method for HF datasets compatibility."""
        from datasets import Dataset as HFDataset
        
        # Apply the function to each item or batch
        if batched:
            batch_size = batch_size or 1000
            results = []
            
            for i in range(0, len(self), batch_size):
                batch_indices = list(range(i, min(i + batch_size, len(self))))
                batch_data = [self[idx] for idx in batch_indices]
                batch_results = function(batch_data)
                results.extend(batch_results)
        else:
            results = [function(self[i]) for i in range(len(self))]
            
        # If using HF datasets, try to convert back to an HFDataset
        if self._is_hf_dataset:
            try:
                return HFDataset.from_dict({k: [item[k] for item in results] 
                                           for k in results[0].keys()})
            except:
                # Fallback to returning the processed data
                return results
                
        return results
    
    # Add compatibility check method
    def is_huggingface_compatible(self):
        """Check if this dataset is compatible with HuggingFace API."""
        return hasattr(self, '_supports_huggingface_api') and self._supports_huggingface_api