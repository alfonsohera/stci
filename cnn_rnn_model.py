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
        freq_mask_param=55,    # From HPO
        time_mask_param=22,    # From HPO
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


class ImprovedSelfAttention(nn.Module): #TDO: test instead of SelfAttention
    """Improved self-attention module with pre-LN and GELU"""
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Improved feed-forward with GELU and different expansion ratio
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),  # Smaller for efficiency
            nn.GELU(),  # GELU often performs better than ReLU
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x, mask=None):
        # Pre-LN architecture (more stable training)
        x_norm = self.norm1(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + self.dropout(attn_output)
        
        x_norm = self.norm2(x)
        ff_output = self.ff(x_norm)
        x = x + self.dropout(ff_output)
        
        return x


class DualPathAudioClassifier(nn.Module):
    def __init__(self, num_classes=3, sample_rate=16000, n_mels=128, 
                 apply_specaugment=True, use_prosodic_features=True, 
                 prosodic_feature_dim=7):
        super(DualPathAudioClassifier, self).__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.apply_specaugment = apply_specaugment
        
        # Create mel spectrogram converter
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=128,
            n_mels=n_mels
        )
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # SpecAugment for data augmentation during training
        if apply_specaugment:
            self.spec_augment = SpecAugment()
    
        # Reduce CNN path complexity
        self.cnn_extractor = nn.Sequential(
            # First block
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),
            
            # Second block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            
            # Third block 
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Downsample raw audio for attention path
        self.audio_downsample = nn.Conv1d(1, 8, kernel_size=50, stride=120)        
        self.position_embedding = nn.Parameter(torch.randn(1, 128, 8))  # Max seq length 128, feature dim 8
        
        # Self-attention layers
        self.attention_layers = nn.ModuleList([
            ImprovedSelfAttention(embed_dim=8, num_heads=2, dropout=0.2)
        ])
        
        # Attention output processing
        self.attention_pooling = nn.Sequential(
            nn.Linear(8, 64),  # Reduced from 256
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Add prosodic feature processing
        self.use_prosodic_features = use_prosodic_features        
        if use_prosodic_features:
            # Prosodic feature encoder
            self.prosodic_encoder = nn.Sequential(
                nn.Linear(prosodic_feature_dim, 64),
                nn.LayerNorm(64),  # Use LayerNorm for better generalization
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 128)
            )
            
            # Chunk context encoder (relative position in the audio)
            self.chunk_context = nn.Sequential(
                nn.Linear(2, 32),  # 2 inputs: relative position and chunk length
                nn.ReLU(),
                nn.Linear(32, 32)
            )
            
            # Modified fusion to include prosodic features
            self.fusion = nn.Sequential(
                nn.Linear(64 + 64 + 128 + 32, 128),  
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),  
                nn.Linear(128, 64),  
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        else:
            # Fusion layer (CNN features + Attention features)
            self.fusion = nn.Sequential(
                nn.Linear(64 + 64, 128),  
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),  
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2)
            )

        # Separate classifier layer
        self.classifier = nn.Linear(64, num_classes)
        
        # Pre-initialize device tracking to avoid device checks at runtime
        self._device = None
        self._initialized_for_device = False

        

    def forward(self, audio, audio_lengths=None, augmentation_id=None, 
                prosodic_features=None, chunk_context=None):
        """
        Forward pass for chunked audio processing
        
        Args:
            audio: Input audio tensor [B, C, T]
            audio_lengths: Tensor of actual audio lengths [B]
            augmentation_id: Optional IDs for deterministic augmentation
            prosodic_features: Optional prosodic features [B, prosodic_feature_dim]
            chunk_context: Optional chunk context [B, 2]
            
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
        # Ensure audio has shape [B, 1, T] for Conv1d
        if len(audio.shape) == 3 and audio.shape[1] > 1:
            # If we have [B, C, T] where C > 1, convert to mono by averaging
            audio_mono = audio.mean(dim=1, keepdim=True)  # [B, 1, T]
        elif len(audio.shape) == 2:
            # If we have [B, T], add channel dimension
            audio_mono = audio.unsqueeze(1)  # [B, 1, T]
        else:
            # Already in correct format [B, 1, T]
            audio_mono = audio
        # Downsample audio for attention path
        audio_downsampled = self.audio_downsample(audio_mono)  # [B, 8, T/120]
        # Apply adaptive pooling to ensure fixed sequence length
        audio_downsampled = F.adaptive_avg_pool1d(audio_downsampled, 128)  # [B, 8, 128]
        # Transpose to match positional embedding dimensions
        audio_downsampled = audio_downsampled.transpose(1, 2)  # [B, 128, 8]
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
        
        # Process prosodic features if available and enabled
        if self.use_prosodic_features and prosodic_features is not None:
            # Encode prosodic features
            prosodic_encoded = self.prosodic_encoder(prosodic_features)  # [B, 128]
            
            # Process chunk context if available
            if chunk_context is not None:
                # chunk_context contains: [relative_position, relative_length]
                chunk_encoding = self.chunk_context(chunk_context)  # [B, 32]
            else:
                # Default to zeros if no context provided
                chunk_encoding = torch.zeros(prosodic_features.size(0), 32, 
                                             device=prosodic_features.device)
            
            # Combine all features
            combined_features = torch.cat([
                cnn_features,          # [B, 256] - Spectral/frequency features
                attn_features,         # [B, 256] - Temporal dynamics features
                prosodic_encoded,      # [B, 128] - Global prosodic features
                chunk_encoding         # [B, 32]  - Chunk position context
            ], dim=1)
        else:
            # Original feature combination without prosodic features
            combined_features = torch.cat([cnn_features, attn_features], dim=1)
        
        # Final classification
        fusion_output = self.fusion(combined_features)
        output = self.classifier(fusion_output)
        
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
    Designed to work with standard PyTorch datasets.
    """
    def __init__(self, original_dataset, total_target_samples=1000, num_classes=3):
        self.original_dataset = original_dataset
        self.num_classes = num_classes
        self.target_samples_per_class = total_target_samples // num_classes
        
        # Don't pre-generate augmentations, just store indices
        self.class_indices = {}
        self.class_counts = {}
        
        # Count original samples per class
        for i in range(len(self.original_dataset)):
            item = self.original_dataset[i]
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
        
        # Initialize RNG for reproducible augmentations
        import numpy as np
        self.rng = np.random.RandomState(42)
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset, with optional augmentation."""
        source_idx = self.sample_indices[idx]
        augmentation_id = self.augmentation_ids[idx]
        
        # Get the original sample
        sample = self.original_dataset[source_idx].copy()
        
        # Apply augmentation if needed
        if augmentation_id is not None:
            # Apply deterministic augmentation on the audio
            sample["augmentation_id"] = augmentation_id
        else:
            sample["augmentation_id"] = None
        
        # Ensure prosodic features are passed through
        if "prosodic_features" in sample:
            # No augmentation for prosodic features, just pass them through
            pass
        
        return sample
    
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