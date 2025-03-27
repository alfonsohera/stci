import torch
import torch.nn as nn
import torchaudio
import torchvision.models as vision_models


class DualPathAudioClassifier(nn.Module):
    def __init__(self, num_classes=3, sample_rate=16000, n_mels=128, 
                 use_manual_features=True, manual_features_dim=7):
        super().__init__()
        
        # Spectrogram transform for CNN path
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
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
            
        # Dimensionality reduction for CNN features
        self.cnn_dim_reducer = nn.Linear(1280, 256)
        
        # RNN path for raw audio
        self.audio_downsample = nn.Conv1d(1, 8, kernel_size=50, stride=50)  # Change output channels to 8
        self.rnn = nn.GRU(
            input_size=8,  # Changed from sample_rate//50 to match Conv1d output channels
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Manual features processing
        self.use_manual_features = use_manual_features
        if use_manual_features:
            self.manual_feature_mlp = nn.Sequential(
                nn.Linear(manual_features_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32)
            )
            fusion_input_dim = 256 + 256 + 32  # CNN + RNN + Manual
        else:
            fusion_input_dim = 256 + 256  # CNN + RNN only
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, audio, manual_features=None):
        # Start with processing the audio through the CNN
        # If the audio is just a single channel, expand to [B, 1, T]
        if len(audio.shape) == 2:
            audio = audio.unsqueeze(1)  # Add channel dimension
        
        # Generate mel spectrogram
        mel = self.mel_spec(audio.squeeze(1))
        mel_db = self.amplitude_to_db(mel).unsqueeze(1)  # Add channel dim back
        
        # Pass through CNN backbone
        cnn_features = self.cnn_extractor(mel_db)
        cnn_features = self.cnn_dim_reducer(cnn_features)
        
        # Process audio for RNN path
        audio_downsampled = self.audio_downsample(audio)
        audio_downsampled = audio_downsampled.transpose(1, 2)  # B, T, C
        
        # Pass through RNN
        rnn_output, _ = self.rnn(audio_downsampled)
        rnn_features = rnn_output[:, -1, :]  # Take final state
        
        # Combine CNN and RNN features
        combined_features = torch.cat([cnn_features, rnn_features], dim=1)
        
        # Add manual features if available and enabled
        if manual_features is not None and self.use_manual_features:
            manual_features_processed = self.manual_feature_mlp(manual_features)
            combined_features = torch.cat([combined_features, manual_features_processed], dim=1)
        
        # Final classification
        output = self.fusion(combined_features)
        return output