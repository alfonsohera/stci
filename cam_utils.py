import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.nn import functional as F
import torchaudio
import torch.nn as nn
from skimage.transform import resize


def print_model_structure(model, indent=0):
    """Print the structure of a model to help debug layer access issues"""
    for name, module in model.named_children():
        print(' ' * indent + f"└─ {name}: {module.__class__.__name__}")
        if hasattr(module, 'named_children'):
            submodules = list(module.named_children())
            if submodules:
                print_model_structure(module, indent + 4)


class GradCAM:
    """
    Generic class for Grad-CAM visualization
    Adapted to work with the CNN-RNN models (CNN14 and DualPath)
    """
    def __init__(self, model, target_layer, use_cuda=True):
        self.model = model
        self.target_layer = target_layer
        self.cuda = use_cuda and torch.cuda.is_available()
        
        # Set model to eval mode to ensure consistent behavior
        self.model.eval()
        
        # Save original requires_grad state
        self.original_grad_states = {}
        for name, param in self.model.named_parameters():
            self.original_grad_states[name] = param.requires_grad
            param.requires_grad = True  # Enable gradients for all parameters
        
        # Initialize attributes
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.handle_activation = target_layer.register_forward_hook(self._get_activation)
        self.handle_gradients = target_layer.register_full_backward_hook(self._get_gradients)
        
        # Debug info
        self.target_layer_name = None
        for name, module in model.named_modules():
            if module is target_layer:
                self.target_layer_name = name
                break
    
    def _get_activation(self, module, input, output):
        # Make a deep copy to ensure we don't modify the original
        self.activations = output.clone()
        
        # For additional debug info
        #print(f"Activation shape from {self.target_layer_name}: {self.activations.shape}")
    
    def _get_gradients(self, module, grad_input, grad_output):
        # Make a deep copy to ensure we don't modify the original
        if grad_output and grad_output[0] is not None:
            self.gradients = grad_output[0].clone()
            #print(f"Gradient shape from {self.target_layer_name}: {self.gradients.shape}")
        else:
            self.gradients = None
            print(f"No gradients received from {self.target_layer_name}")
    
    def __call__(self, input_tensor, target_category=None):
        """Generate CAM for input tensor"""
        # Reset gradients and activations
        self.activations = None
        self.gradients = None
        
        # Zero model gradients
        self.model.zero_grad()
        
        # Enable gradients even if in inference mode
        with torch.set_grad_enabled(True):
            # Prepare tensor
            input_tensor = input_tensor.clone().detach().requires_grad_(True)
            
            # Model type check
            model_type = "cnn14" if hasattr(self.model, "feature_extractor") else "dualpath"
            
            # Prepare input shape based on the model type and current tensor dimensions
            if model_type == "cnn14":
                # CNN14 expects shape [B, T] for processing
                if len(input_tensor.shape) == 1:  # [T]
                    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension -> [1, T]
                elif len(input_tensor.shape) == 3 and input_tensor.shape[1] == 1:  # [B, 1, T]
                    input_tensor = input_tensor.squeeze(1)  # Remove channel dim -> [B, T]
                # For [B, T] shape, we keep as is
            else:
                # DualPath models expect shape [B, 1, T] (with channel dim)
                if len(input_tensor.shape) == 1:  # [T]
                    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # -> [1, 1, T]
                elif len(input_tensor.shape) == 2:  # [B, T]
                    input_tensor = input_tensor.unsqueeze(1)  # Add channel dim -> [B, 1, T]
                # For [B, C, T] shape, we keep as is
            
            if self.cuda:
                input_tensor = input_tensor.cuda()
            
            # Debug tensor shape
            # print(f"Input tensor shape for {model_type} model: {input_tensor.shape}")
            
            # Forward pass
            if model_type == "cnn14" and hasattr(self.model, "classifier"):
                output = self.model(input_tensor)
            else:
                # For dual-path models
                audio_lengths = torch.tensor([input_tensor.shape[-1]], device=input_tensor.device)
                
                if hasattr(self.model, "prosodic_encoder"):
                    # Find prosodic dimension
                    if hasattr(self.model, "prosodic_encoder") and isinstance(self.model.prosodic_encoder, torch.nn.Sequential):
                        first_layer = next(iter(self.model.prosodic_encoder.children()))
                        prosodic_dim = first_layer.in_features if hasattr(first_layer, "in_features") else 4
                    else:
                        prosodic_dim = 4
                    
                    prosodic_features = torch.zeros((1, prosodic_dim), device=input_tensor.device, requires_grad=True)
                    output = self.model(
                        input_tensor, 
                        audio_lengths=audio_lengths,
                        prosodic_features=prosodic_features
                    )
                else:
                    output = self.model(input_tensor, audio_lengths=audio_lengths)
            
            # Get predicted class if target_category is not specified
            if target_category is None:
                target_category = output.argmax(dim=1).item()
            
            # Create one hot vector for target category
            target = torch.zeros_like(output)
            target[0, target_category] = 1
            
            # Backward pass
            output.backward(gradient=target)
            
            # Check if gradients were properly captured
            if self.gradients is None:
                print(f"Warning: No gradients captured for {self.target_layer_name}. Using dummy CAM.")
                # Try debugging with more info
                model_types = []
                for name, m in self.model.named_modules():
                    model_types.append(f"{name}: {type(m).__name__}")
                print(f"Model layers: {', '.join(model_types[:10])}...")
                
                # Use activations for fallback visualization
                if self.activations is not None:
                    cam = torch.mean(self.activations, dim=1, keepdim=True)
                    cam = F.relu(cam)
                    cam = cam - cam.min()
                    if cam.max() > 0:
                        cam = cam / cam.max()
                    cam = cam.detach().squeeze(0).squeeze(0).cpu().numpy()
                else:
                    print("Warning: No activations captured. Returning uniform CAM.")
                    cam_shape = (32, 32)
                    cam = np.ones(cam_shape)
                
                return cam, output, target_category
            
            # Get weights from gradients - ensure proper dimensions
            if len(self.gradients.shape) == 4:  # [B, C, H, W]
                weights = torch.mean(self.gradients, dim=(2, 3))
            elif len(self.gradients.shape) == 3:  # [B, C, L]
                weights = torch.mean(self.gradients, dim=2)
            else:
                raise ValueError(f"Unexpected gradient shape: {self.gradients.shape}")
            
            # Apply weights to activations
            # Create a correctly sized output tensor
            if len(self.activations.shape) == 4:  # [B, C, H, W]
                B, C, H, W = self.activations.shape
                cam = torch.zeros((B, H, W), device=self.activations.device)
                
                # Apply weights to activations channel by channel
                for i, w in enumerate(weights[0]):
                    cam += (w * self.activations[0, i, :, :])
            else:  # [B, C, L]
                B, C, L = self.activations.shape
                cam = torch.zeros((B, L), device=self.activations.device)
                
                # Apply weights to activations channel by channel
                for i, w in enumerate(weights[0]):
                    cam += (w * self.activations[0, i, :])
            
            # ReLU
            cam = F.relu(cam)
            
            # Normalize
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
            
            # Convert to numpy
            cam = cam.detach().squeeze().cpu().numpy()
            
            return cam, output, target_category

    def restore_grad_states(self):
        """Restore original requires_grad states"""
        for name, param in self.model.named_parameters():
            if name in self.original_grad_states:
                param.requires_grad = self.original_grad_states[name]
    
    def remove_hooks(self):
        """Remove all hooks."""
        if hasattr(self, 'handle_activation') and self.handle_activation:
            self.handle_activation.remove()
        if hasattr(self, 'handle_gradients') and self.handle_gradients:
            self.handle_gradients.remove()
        self.restore_grad_states()
    
    def __del__(self):
        try:
            self.remove_hooks()
        except:
            pass  # Ignore errors during cleanup


def get_model_target_layers(model):
    """
    Get the appropriate target layer for CAM based on model architecture
    
    Args:
        model: CNN14Classifier, PretrainedDualPathAudioClassifier, or DualPathAudioClassifier
    
    Returns:
        target_layer: The layer to use for CAM
    """
    model_name = model.__class__.__name__
    
    if model_name == "CNN14Classifier":
        # For CNN14, we need to access the last conv layer inside the conv_block6        
        return model.cnn_extractor.conv_block6.conv2
    
    elif model_name == "PretrainedDualPathAudioClassifier":
        # For PretrainedDualPathAudioClassifier, access the last conv layer in conv_block6
        
        return model.cnn_extractor.conv_block6.conv2
    
    elif model_name == "DualPathAudioClassifier":
        # For DualPathAudioClassifier, use the last convolutional layer in CNN path
        # Assuming the structure: conv_block -> batch_norm -> relu -> pool -> dropout
        # We need to find the last conv layer
        for i in range(len(model.cnn_extractor) - 1, -1, -1):
            layer = model.cnn_extractor[i]
            if isinstance(layer, nn.Conv2d):
                return layer
    
    raise ValueError(f"Unsupported model type: {model_name}")


def generate_spectrogram(audio, model, sr=16000):
    """
    Generate spectrogram from audio based on model type
    
    Args:
        audio: Audio tensor [B, T] or [B, 1, T]
        model: Model instance
        sr: Sample rate
    
    Returns:
        spectrogram: Spectrogram as numpy array
    """
    # Get the device from the audio tensor
    device = audio.device
    
    # Make sure audio is detached and doesn't require grad for spectrogram generation
    audio = audio.detach()
    
    model_name = model.__class__.__name__
    
    # CNN14 uses a specific spectrogram configuration
    if model_name == "CNN14Classifier" or model_name == "PretrainedDualPathAudioClassifier":
        # CNN14 uses log mel spectrogram with these parameters
        window_size = 1024
        hop_size = 320
        mel_bins = 64
        fmin = 50
        fmax = 8000
        
        # Convert to mono if needed
        if len(audio.shape) == 3 and audio.shape[1] > 1:
            audio = audio.mean(dim=1)
        if len(audio.shape) == 3:
            audio = audio.squeeze(1)
        
        # Move audio to CPU for spectrogram extraction
        # This is safer than moving the spectrogram extractor to GPU
        audio_cpu = audio.cpu()
        
        # Create spectrogram extractor
        spectrogram_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=window_size,
            win_length=window_size,
            hop_length=hop_size,
            f_min=fmin,
            f_max=fmax,
            n_mels=mel_bins,
            power=2
        )
        
        # Generate spectrogram
        mel_spec = spectrogram_extractor(audio_cpu)
        
        # Convert to log scale
        eps = 1e-10
        log_mel_spec = torch.log10(torch.clamp(mel_spec, min=eps))
    
    else:  # For DualPathAudioClassifier
        # Convert to mono if needed
        if len(audio.shape) == 3 and audio.shape[1] > 1:
            audio = audio.mean(dim=1)
        if len(audio.shape) == 2:
            audio = audio.unsqueeze(1)
        
        # Move audio to CPU for consistency
        audio_cpu = audio.cpu()
        
        # Get the mel spectrogram configuration from the model (also ensure on CPU)
        with torch.no_grad():
            # Temporarily move any necessary model components to CPU
            if hasattr(model, 'mel_spec'):
                original_device = next(model.mel_spec.parameters()).device
                model.mel_spec = model.mel_spec.cpu()
                mel_spec = model.mel_spec(audio_cpu.squeeze(1))
                log_mel_spec = model.amplitude_to_db(mel_spec)
                model.mel_spec = model.mel_spec.to(original_device)
            else:
                # Fallback to standard spectrogram extraction
                spectrogram_extractor = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sr,
                    n_fft=1024,
                    win_length=1024, 
                    hop_length=320,
                    f_min=50,
                    f_max=8000,
                    n_mels=64,
                    power=2
                )
                mel_spec = spectrogram_extractor(audio_cpu)
                log_mel_spec = torch.log10(torch.clamp(mel_spec, min=1e-10))
    
    return log_mel_spec.numpy()


def combine_audio_chunks(audio_chunks_list, sample_rate=16000, chunk_size_seconds=10):
    """
    Combine audio chunks into a single waveform, properly handling padding in the last chunk.
    
    Args:
        audio_chunks_list: List of audio chunks as tensors
        sample_rate: Audio sample rate (default: 16000)
        chunk_size_seconds: Size of each chunk in seconds (default: 10)
        
    Returns:
        tuple: (numpy array of combined waveform, list of original lengths, actual duration in seconds)
    """
    if not audio_chunks_list:
        return np.zeros(sample_rate), [0], 1.0  # Return 1 second of silence as fallback
    
    # Convert all chunks to numpy arrays in the same format
    waveforms = []
    for chunk in audio_chunks_list:
        if len(chunk.shape) == 3:  # [batch, channel, time]
            waveform = chunk.squeeze(0).squeeze(0).cpu().numpy()
        elif len(chunk.shape) == 2:  # [batch, time]
            waveform = chunk.squeeze(0).cpu().numpy()
        else:  # [time]
            waveform = chunk.cpu().numpy()
        waveforms.append(waveform)
    
    # Check if the last chunk might have padding
    chunk_size_samples = int(chunk_size_seconds * sample_rate)
    original_lengths = [len(w) for w in waveforms]
    
    # Only try to detect padding in the last chunk
    if len(waveforms) > 1 and len(waveforms[-1]) == chunk_size_samples:
        last_chunk = waveforms[-1]
        # Detect significant content vs. padding
        threshold = 1e-5
        non_zero_indices = np.where(np.abs(last_chunk) > threshold)[0]
        
        # If we found non-zero values and they end well before the end of the chunk
        if len(non_zero_indices) > 0:
            last_non_zero_idx = non_zero_indices[-1]
            # Check if there's significant padding (at least 0.5 seconds of silence at the end)
            padding_threshold_samples = int(0.5 * sample_rate)
            
            if last_non_zero_idx < len(last_chunk) - padding_threshold_samples:
                # Estimate actual audio length (with a small buffer for decay)
                actual_end = min(last_non_zero_idx + int(0.1 * sample_rate), len(last_chunk))
                print(f"Detected padding in last chunk. Original length: {len(last_chunk)}, Trimmed length: {actual_end}")
                
                # Trim the last chunk
                waveforms[-1] = last_chunk[:actual_end]
                original_lengths[-1] = actual_end
    
    # Now combine all waveforms (some may have been trimmed)
    combined_waveform = np.concatenate(waveforms)
    
    # Calculate actual duration in seconds
    actual_duration = len(combined_waveform) / sample_rate
    
    return combined_waveform, original_lengths, actual_duration


def visualize_cam(audio, model, target_class=None, true_class=None, save_path=None, audio_id=None, correct=None, audio_paths_dir=None, epoch=None,                   
                  audio_chunks=None, chunk_outputs=None, show_time_domain=False, file_path=None):
    """
    Visualize CAM for an audio input, with support for chunked processing
    
    Args:
        audio: Audio tensor or a single chunk of audio 
        model: Model to analyze
        target_class: Target class for CAM generation (typically the predicted class)
        true_class: True class label for the audio (ground truth)
        save_path: Directory to save visualizations
        audio_id: ID of the audio sample
        correct: Whether prediction is correct
        audio_paths_dir: Directory to save audio paths
        epoch: Current epoch number
        audio_chunks: Optional list of already chunked audio or audio_id to fetch original audio
        chunk_outputs: Optional list of chunk outputs (logits) from model
        show_time_domain: Whether to display the time-domain representation (waveform) with CAM overlay
        file_path: Original audio file path (if available)
    """
    # Get model's device
    device = next(model.parameters()).device
    
    # Move audio to device
    audio = audio.to(device)
    
    # Make a clone of the input tensor
    audio_clone = audio.detach().clone()
    
    # Store original training state
    training_state = model.training
    
    # Set model to eval mode
    model.eval()
    
    # Get target layer for CAM
    target_layer = get_model_target_layers(model)
    
    # Extract a clean display name from file_path or audio_id - do this early so we can use it everywhere
    display_filename = "Unknown"
    if file_path:
        # Extract just the filename without the path
        display_filename = os.path.basename(file_path)
    elif audio_id and isinstance(audio_id, str):
        display_filename = audio_id
    
    # Always try to fetch original file path from audio_id if provided
    # This ensures we get the file path even when audio_chunks is a list
    if audio_id is not None:
        try:
            # Import the necessary function to fetch original audio
            from cnn_rnn_data import get_original_audio_by_id
            
            # Use the audio_id to fetch original file path
            original_audio_id = str(audio_id)
            print(f"Fetching original file path for ID: {original_audio_id}")
            _, original_file_path = get_original_audio_by_id(original_audio_id)
            
            # If we found an original file path, use it
            if original_file_path is not None:
                # Update display_filename and file_path
                file_path = original_file_path
                display_filename = os.path.basename(original_file_path)
                print(f"Found original file: {file_path}")
        except (ImportError, Exception) as e:
            print(f"Couldn't fetch original file path for ID {audio_id}: {str(e)}")
    
    # Process audio chunks based on their type
    if isinstance(audio_chunks, dict) and audio_id in audio_chunks and isinstance(audio_chunks[audio_id], list):
        audio_chunks_list = audio_chunks[audio_id]
    # If audio_chunks is already a list of audio chunks
    elif isinstance(audio_chunks, list):
        audio_chunks_list = audio_chunks
    # Fallback
    else:
        print("Using provided audio chunk")
        audio_chunks_list = [audio_clone]
    
    # Extract chunk outputs (logits) - similar logic as above
    if isinstance(chunk_outputs, dict) and audio_id in chunk_outputs:
        chunk_outputs_list = chunk_outputs[audio_id]
    elif isinstance(chunk_outputs, list):
        chunk_outputs_list = chunk_outputs
    else:
        # If no chunk outputs provided, we need to run the model on the chunks
        chunk_outputs_list = []
        with torch.no_grad():
            for chunk in audio_chunks_list:
                # Calculate audio lengths for this chunk
                chunk_lengths = torch.tensor([chunk.shape[-1]], device=chunk.device)
                
                # Check if model uses prosodic features
                if hasattr(model, "prosodic_encoder"):
                    # Find proper prosodic feature size
                    if hasattr(model.prosodic_encoder, "0") and hasattr(model.prosodic_encoder[0], "in_features"):
                        prosodic_dim = model.prosodic_encoder[0].in_features
                    else:
                        prosodic_dim = 4  # Default fallback
                    
                    # Create dummy prosodic features
                    prosodic_features = torch.zeros((1, prosodic_dim), device=chunk.device)
                    
                    # Get chunk prediction
                    chunk_logits = model(
                        chunk, 
                        audio_lengths=chunk_lengths,
                        prosodic_features=prosodic_features
                    )
                else:
                    # For models without prosodic features
                    chunk_logits = model(
                        chunk,
                        audio_lengths=chunk_lengths
                    )
                
                # Store chunk output
                chunk_outputs_list.append(chunk_logits[0])
    
    print(f"Processing {len(audio_chunks_list)} audio chunks with {len(chunk_outputs_list)} output chunks")

    # Now aggregate the predictions from all chunks - exactly like evaluate()
    aggregated_output = model.aggregate_chunk_predictions(chunk_outputs_list)
    actual_pred_class = torch.argmax(aggregated_output).item()
    actual_probs = F.softmax(aggregated_output, dim=-1)
    actual_pred_prob = actual_probs[actual_pred_class].item()
    
    # Generate CAM for each chunk
    all_cams = []
    all_specs = []
    
    # Store the raw audio waveforms for time-domain visualization
    all_waveforms = []
    
    # Process each chunk for CAM visualization
    for i, chunk in enumerate(audio_chunks_list):
        # Check if the chunk is too small for STFT processing
        # CNN14 requires at least 2*n_fft samples (typically 2*1024 = 2048)
        min_required_length = 2048
        
        # Get the actual length of the chunk (handling different tensor shapes)
        chunk_length = chunk.shape[-1]
        
        # Skip chunks that are too small to be processed by CNN14's spectrogram extractor
        if chunk_length < min_required_length:
            print(f"WARNING: Skipping chunk {i} - length {chunk_length} is smaller than minimum required length {min_required_length}")
            continue
            
        # Proceed with normal processing for chunks of valid size
        grad_cam = GradCAM(model, target_layer, use_cuda=(device.type == 'cuda'))
        # Use the target class for visualization
        chunk_cam, _, _ = grad_cam(chunk, target_class)
        all_cams.append(chunk_cam)
        
        # Generate spectrogram for this chunk
        chunk_spec = generate_spectrogram(chunk, model)
        all_specs.append(chunk_spec)
        
        # Store raw waveform for time-domain visualization
        # Ensure the chunk is in the right format [samples]
        if len(chunk.shape) == 3:  # [batch, channel, time]
            waveform = chunk.squeeze(0).squeeze(0).cpu().numpy()
        elif len(chunk.shape) == 2:  # [batch, time]
            waveform = chunk.squeeze(0).cpu().numpy()
        else:  # [time]
            waveform = chunk.cpu().numpy()
            
        # Check if this is the last chunk and potentially has padding
        if i == len(audio_chunks_list) - 1:
            # Try to detect and remove padding in the last chunk
            # Padding is typically zeros at the end of the audio
            # We use a threshold to determine where actual audio ends
            threshold = 1e-5  # Adjust this threshold based on your audio characteristics
            non_zero_indices = np.where(np.abs(waveform) > threshold)[0]
            
            sample_rate = 16000  # Assuming a sample rate of 16kHz
            # If we found non-zero values and the last non-zero value is significantly before the end
            if len(non_zero_indices) > 0 and non_zero_indices[-1] < len(waveform) - 100:  # At least 100 zeros at the end
                # Add a small buffer to include decay
                waveform_end = min(non_zero_indices[-1] + int(sample_rate * 0.1), len(waveform))
                print(f"Detected padding in last chunk. Original length: {len(waveform)}, Trimmed length: {waveform_end}")
                waveform = waveform[:waveform_end]
                
                # Important: Also trim the CAM for this chunk to match the new waveform length
                # We need to scale the CAM to match the new length
                if len(chunk_cam.shape) == 1:  # 1D CAM
                    cam_ratio = waveform_end / len(waveform)
                    cam_end = int(len(chunk_cam) * cam_ratio)
                    all_cams[-1] = chunk_cam[:cam_end]
                    
                # Update the spectrogram if needed
                # Not updating spectrogram since this is more complex and may cause issues
        
        all_waveforms.append(waveform)
        
        # Clean up GradCAM hooks for this chunk
        grad_cam.remove_hooks()

    # Check if we have any valid CAMs
    if not all_cams:
        print("WARNING: No valid chunks available for CAM visualization. Creating dummy CAM.")
        dummy_cam = np.ones((64, 64))  # Create a dummy CAM
        all_cams = [dummy_cam]  # Create a list with the dummy CAM
        all_specs = [np.zeros((64, 64))]  # Create a dummy spectrogram
        all_waveforms = [np.zeros(16000)]  # Create a dummy waveform of 1 second
    
    # Concatenate CAMs to show the full audio - FIX
    if len(all_cams) > 1:
        # Check CAM shape and determine how to concatenate
        if len(all_cams[0].shape) == 1:  # 1D CAM (time dimension only)
            cam = np.concatenate(all_cams)
        elif len(all_cams[0].shape) == 2:  # 2D CAM (time x features)
            # Concatenate along time dimension (axis 1)
            cam = np.concatenate(all_cams, axis=1)
        else:
            # For any other shape, use first CAM
            print(f"WARNING: Unexpected CAM shape {all_cams[0].shape}, using only first chunk.")
            cam = all_cams[0]
    else:
        # Only one CAM
        cam = all_cams[0]
    
    # Concatenate specs along time dimension (same as with CAMs)
    if all_specs:
        if len(all_specs[0].shape) == 3:  # [C, F, T]
            spectrogram = np.concatenate(all_specs, axis=2)
        else:  # [F, T]
            spectrogram = np.concatenate(all_specs, axis=1)
    else:
        # Fallback
        spectrogram = generate_spectrogram(audio, model)
    
    # Concatenate waveforms
    full_waveform = np.concatenate(all_waveforms)

    # Map class indices to human-readable labels based on model's number of classes
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'out_features'):
        num_classes = model.classifier.out_features
    else:
        # Default to 3 classes if we can't determine from the model
        num_classes = 3
    
    if num_classes == 2:
        # Binary classification
        class_names = ["Healthy", "Non-Healthy"]
    else:
        # 3-class classification
        class_names = ["Healthy", "MCI", "AD"]

    # Extract the prediction and true class indices from the audio_id if available
    pred_class_idx = actual_pred_class  # Default to actual prediction
    true_class_idx = true_class if true_class is not None else None  # Default to provided true class

    # If audio_id contains pred and true info (e.g., "11_pred2_true2"), extract them
    if audio_id is not None and 'pred' in audio_id and 'true' in audio_id:
        try:
            # Extract pred and true indices from the audio_id
            parts = audio_id.split('_')
            for part in parts:
                if part.startswith('pred'):
                    pred_class_idx = int(part[4:])
                elif part.startswith('true'):
                    true_class_idx = int(part[4:])
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not extract class indices from audio_id: {audio_id}, {e}")

    # Now map to human-readable labels using the extracted indices
    pred_label = class_names[pred_class_idx]
    true_label = class_names[true_class_idx] if true_class_idx is not None else "Unknown"

    # Create filename
    if audio_id is not None and ('pred' in audio_id or 'true' in audio_id):
        file_id = audio_id  
    else:
        file_id = audio_id if audio_id is not None else f"sample_{actual_pred_class}"
        # Use true_class parameter if available, otherwise fall back to target_class
        actual_true_class = true_class if true_class is not None else target_class
        if actual_pred_class is not None and actual_true_class is not None:
            file_id += f"_pred{actual_pred_class}_true{actual_true_class}"

    # Handle single-channel vs multi-channel spectrogram
    if len(spectrogram.shape) == 3:
        spec_for_plot = spectrogram[0]  # Take first channel
    else:
        spec_for_plot = spectrogram

    # Calculate actual frequency and time values
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    # Define axis scales
    f_min, f_max = 50, 8000  # From your spectrogram settings
    n_mels = spec_for_plot.shape[0]
    hop_length = 320  # From your settings
    sr = 16000  # Sample rate
    
    # Generate frequency ticks (Hz) from mel scale
    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels)
    freq_hz = [int(mel_to_hz(m)) for m in mel_points]

    # Select a few frequency points for a cleaner y-axis
    freq_ticks = np.linspace(0, n_mels-1, 8, dtype=int)
    freq_labels = [f"{freq_hz[i]}" for i in freq_ticks]

    # Create time axis in seconds - FIXED to show total duration of all chunks
    time_frames = spec_for_plot.shape[1]  
    total_chunks = len(all_specs)
    chunk_size_seconds = 10  # Each chunk is 10 seconds
    
    # Fix the time labels to show the full duration
    if total_chunks > 1:
        # Create time labels that span across all chunks
        total_duration = total_chunks * chunk_size_seconds
        time_sec = np.linspace(0, total_duration, time_frames)
    else:
        # Standard calculation for single chunk
        time_sec = np.arange(time_frames) * hop_length / sr
        
    # Generate time ticks that span the entire duration
    time_ticks = np.linspace(0, time_frames-1, min(10, time_frames), dtype=int)
    time_labels = [f"{time_sec[i]:.1f}" for i in time_ticks]
    
    # For debugging
    #print(f"Total chunks: {total_chunks}, Time frames: {time_frames}, Full duration: {total_chunks * chunk_size_seconds}s")
    #print(f"Time labels: {time_labels}")
    
    # Extract filename from file_path for display in titles
    display_filename = "Unknown"
    if file_path:
        # Extract just the filename without the path
        display_filename = os.path.basename(file_path)
    else:
        # Try to extract filename from audio_id if possible
        if audio_id and isinstance(audio_id, str):
            # Remove any prediction/true label info that might be in audio_id
            base_id = audio_id.split('_')[0] if '_' in audio_id else audio_id
            display_filename = f"ID: {base_id}"

    # Figure 1: Spectrogram and CAM visualization
    plt.figure(figsize=(12, 9))
    
    # Plot 1: Spectrogram
    plt.subplot(2, 1, 1)
    plt.imshow(spec_for_plot, origin='lower', aspect='auto', cmap='viridis')
    title = f"Log-Mel Spectrogram - {display_filename}\nPred: {pred_label} ({actual_pred_prob:.4f}), True: {true_label}"
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (seconds)')
    plt.yticks(freq_ticks, freq_labels)
    plt.xticks(time_ticks, time_labels)
    
    # Plot 2: CAM overlay on spectrogram
    plt.subplot(2, 1, 2)
    plt.imshow(spec_for_plot, origin='lower', aspect='auto', alpha=0.6, cmap='viridis')

    # Handle different CAM shapes properly
    if len(cam.shape) == 1:
        # For 1D CAM (time dimension only)
        # Use proper interpolation to resize from 1D to 2D
        from scipy.ndimage import zoom
        # Calculate scaling factor
        scale_x = spec_for_plot.shape[1] / cam.shape[0]
        # Resize CAM to match spectrogram width
        cam_resized = zoom(cam, scale_x, order=1)
        # Replicate across frequency dimension
        cam_2d = np.zeros((spec_for_plot.shape[0], spec_for_plot.shape[1]))
        for i in range(spec_for_plot.shape[0]):
            cam_2d[i, :] = cam_resized[:spec_for_plot.shape[1]]
        
        # Plot the resized CAM WITHOUT TRANSPOSING
        plt.imshow(cam_2d, origin='lower', aspect='auto', alpha=0.4, cmap='inferno')
        
        # Save the 1D CAM for waveform visualization
        cam_for_waveform = cam_resized
    elif len(cam.shape) == 2:
        # For 2D CAM (time x features or similar)
        # Resize to match spectrogram dimensions
        cam_resized = resize(cam, (spec_for_plot.shape[0], spec_for_plot.shape[1]), 
                             anti_aliasing=True)
        # Plot the resized 2D CAM WITHOUT TRANSPOSING
        plt.imshow(cam_resized, origin='lower', aspect='auto', alpha=0.4, cmap='inferno')
        
        # For waveform visualization, average across frequency dimension
        cam_for_waveform = np.mean(cam_resized, axis=0)
    else:
        # For any other shape, just use the first channel/dimension
        print(f"Warning: Unexpected CAM shape {cam.shape}, using first dimension only")
        cam_1d = cam.reshape(-1)
        cam_for_waveform = cam_1d

    title = f"Class Activation Map - {display_filename}\nPred: {pred_label} ({actual_pred_prob:.4f}), True: {true_label}"
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.xticks(time_ticks, time_labels)
    plt.yticks(freq_ticks, freq_labels)
    
    # Finalize spectrogram figure
    plt.tight_layout()
    
    # Define the paths for saving
    spec_cam_path = None
    waveform_cam_path = None
    html_cam_path = None
    
    # Save spectrogram/CAM visualization if save_path is provided
    if save_path:
        # Initialize base subdirectories
        spec_cam_subdir = 'SpectrogramCAMs'
        waveform_subdir = 'WaveformCAMs'
        html_subdir = 'InteractiveCAMs'  # New directory for HTML visualizations
        
        # Add epoch to path if provided
        if epoch is not None:
            spec_cam_subdir = os.path.join(f"epoch_{epoch}", spec_cam_subdir)
            waveform_subdir = os.path.join(f"epoch_{epoch}", waveform_subdir)
            html_subdir = os.path.join(f"epoch_{epoch}", html_subdir)
        
        # Then add correct/incorrect status
        if correct is not None:
            status = "correct" if correct else "incorrect" 
            spec_cam_subdir = os.path.join(spec_cam_subdir, status)
            waveform_subdir = os.path.join(waveform_subdir, status)
            html_subdir = os.path.join(html_subdir, status)

        # Create directories if they don't exist
        os.makedirs(os.path.join(save_path, spec_cam_subdir), exist_ok=True)
        
        # Create file paths
        spec_cam_path = os.path.join(save_path, spec_cam_subdir, f"{file_id}_spectro_cam.png")
        
        # Add chunk count to filename if multiple chunks
        if total_chunks > 1:
            spec_cam_path = os.path.join(save_path, spec_cam_subdir, f"{file_id}_{total_chunks}chunks_spectro_cam.png")
        
        # Save spectro/CAM figure
        plt.savefig(spec_cam_path, dpi=150)

    # Close the spectrogram/CAM figure to free memory
    plt.close()
    
    # Figure 2: Waveform with CAM overlay (if enabled)
    if show_time_domain:
        plt.figure(figsize=(12, 6))
        
        # Use our new helper function to combine chunks properly
        full_waveform, original_lengths, actual_duration = combine_audio_chunks(
            audio_chunks_list, sample_rate=sr, chunk_size_seconds=10
        )
        
        # Time axis now uses actual_duration from the combined waveform
        waveform_duration = actual_duration
        
        # PERFORMANCE OPTIMIZATION: Downsample the waveform and CAM for plotting
        # This prevents hanging with very long audio files
        target_points = 10000  # Maximum number of points to plot
        
        if len(full_waveform) > target_points:
            # Calculate downsample factor
            downsample_factor = int(np.ceil(len(full_waveform) / target_points))
            
            # Downsample waveform
            waveform_downsampled = full_waveform[::downsample_factor]
            
            # Create time axis for downsampled waveform - using actual duration
            waveform_time = np.linspace(0, waveform_duration, len(waveform_downsampled))
            
            print(f"Downsampling waveform from {len(full_waveform)} to {len(waveform_downsampled)} points")
        else:
            # Use original waveform if it's already small enough
            waveform_downsampled = full_waveform
            waveform_time = np.linspace(0, waveform_duration, len(full_waveform))
        
        # Create the axis explicitly
        ax = plt.gca()
        
        # Plot the waveform
        plt.plot(waveform_time, waveform_downsampled, color='blue', alpha=0.7, linewidth=1)
        
        # Resize CAM to match waveform length (use downsampled length for performance)
        cam_time_domain = resize(cam_for_waveform, (len(waveform_downsampled),), anti_aliasing=True)
        
        # Normalize the CAM for waveform overlay
        cam_time_domain = (cam_time_domain - cam_time_domain.min()) / (cam_time_domain.max() - cam_time_domain.min() + 1e-8)
        
        # Scale the CAM to match waveform amplitude range
        waveform_amplitude = np.max(np.abs(waveform_downsampled))
        
        # Create a colormap for the CAM
        import matplotlib.colors as mcolors
        cmap = plt.get_cmap('inferno')
        
        # Create segments for coloring - one segment per point in the downsampled waveform
        for i in range(len(waveform_time) - 1):
            # Color based on CAM activation
            color = cmap(cam_time_domain[i])
            # Alpha based on CAM activation (more important = more opaque)
            alpha = 0.4 * cam_time_domain[i] + 0.1
            
            # Draw a colored vertical span from top to bottom of the plot
            plt.axvspan(
                waveform_time[i], 
                waveform_time[i+1],
                color=color, 
                alpha=alpha,
                zorder=1  # Place behind the waveform
            )
        
        # Redraw the waveform on top for better visibility
        plt.plot(waveform_time, waveform_downsampled, color='blue', linewidth=1, zorder=2)
        
        plt.title(f"Audio Waveform with CAM Overlay - {display_filename}\nPred: {pred_label} ({actual_pred_prob:.4f}), True: {true_label}")
        plt.ylabel('Amplitude')
        plt.xlabel('Time (seconds)')
        plt.grid(alpha=0.3)
        
        # Set y-limits to show the full waveform with some margin
        margin = 0.1 * waveform_amplitude
        plt.ylim(-waveform_amplitude-margin, waveform_amplitude+margin)
        
        # Add colorbar for the CAM overlay - FIX: explicitly pass the axis
        norm = mcolors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap='inferno', norm=norm)
        sm.set_array([])  # Set empty array
        
        # Pass the current axis to the colorbar function
        cbar = plt.colorbar(sm, ax=ax, format='%.1f')
        cbar.set_label('CAM Activation')
        
        # Tight layout for better display
        plt.tight_layout()
        
        # Save waveform/CAM figure if save_path is provided
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.join(save_path, waveform_subdir), exist_ok=True)
            
            # Create file path
            waveform_cam_path = os.path.join(save_path, waveform_subdir, f"{file_id}_waveform_cam.png")
            if total_chunks > 1:
                waveform_cam_path = os.path.join(save_path, waveform_subdir, f"{file_id}_{total_chunks}chunks_waveform_cam.png")
            
            # Save waveform figure
            plt.savefig(waveform_cam_path, dpi=150)
        
        # Close the waveform/CAM figure to free memory
        plt.close()

        # Create interactive HTML visualization
        if save_path:
            # Create directory for HTML files if it doesn't exist
            os.makedirs(os.path.join(save_path, html_subdir), exist_ok=True)
            
            # Create HTML file path
            html_cam_path = os.path.join(save_path, html_subdir, f"{file_id}_interactive.html")
            
            # Create interactive HTML visualization
            create_interactive_cam_html(
                file_id=file_id,
                waveform=full_waveform,
                cam=cam_for_waveform,
                sample_rate=sr,
                output_path=html_cam_path,
                pred_label=pred_label,
                true_label=true_label,
                pred_prob=actual_pred_prob
            )
            
            print(f"Interactive HTML visualization saved to {html_cam_path}")
    
    # Write to log file - ONLY ONCE
    if save_path and audio_paths_dir:
        # Also organize audio paths by epoch if provided
        if epoch is not None:
            audio_paths_dir = os.path.join(audio_paths_dir, f"epoch_{epoch}")
        
        os.makedirs(audio_paths_dir, exist_ok=True)
        paths_filename = "correct_samples.txt" if correct else "incorrect_samples.txt"
        
        # Add the actual file path to the log if available
        # Use display_filename instead of file_id for more readable logs
        path_entry = f"{display_filename}\t{pred_label}\t{true_label}\t{actual_pred_prob:.4f}\t{total_chunks}chunks"
        # Still include the file_id for reference
        path_entry += f"\t{file_id}"
        if file_path:
            path_entry += f"\t{file_path}"
            
        with open(os.path.join(audio_paths_dir, paths_filename), 'a') as f:
            f.write(f"{path_entry}\n")
    
    # Clean up GradCAM
    grad_cam.remove_hooks()
    del grad_cam
    
    # Restore model training state
    if training_state:
        model.train()
    
    return actual_pred_class, target_class, spec_cam_path, waveform_cam_path if show_time_domain else None


def debug_model_gradients(model, input_tensor, target_class=0):
    """Debug function to check which layers capture gradients properly"""
    # Store hooks and layer names
    hooks = []
    layer_gradients = {}
    
    def _gradient_hook(name):
        def hook(module, grad_input, grad_output):
            # Check if gradients exist
            if grad_output and len(grad_output) > 0 and grad_output[0] is not None:
                layer_gradients[name] = {
                    'shape': grad_output[0].shape,
                    'has_grad': grad_output[0].requires_grad,
                    'mean': grad_output[0].abs().mean().item()
                }
            else:
                layer_gradients[name] = {'shape': None, 'has_grad': False, 'mean': 0}
        return hook
    
    # Register hooks for all modules that might have gradients
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            hook = module.register_full_backward_hook(_gradient_hook(name))
            hooks.append(hook)
    
    # Determine model type
    model_type = "cnn14" if hasattr(model, "feature_extractor") else "dualpath"
    
    # Prepare input based on model type
    if model_type == "cnn14":
        if len(input_tensor.shape) == 3:  # [B, 1, T]
            input_tensor = input_tensor.squeeze(1)  # CNN14 expects [B, T]
    else:
        if len(input_tensor.shape) == 2:  # [B, T]
            input_tensor = input_tensor.unsqueeze(1)  # DualPath expects [B, 1, T]
    
    # Forward pass - handle different model types
    model.zero_grad()
    
    try:
        if model_type == "cnn14" and hasattr(model, "classifier"):
            # This is CNN14Classifier
            output = model(input_tensor)
        else:
            # This is PretrainedDualPathAudioClassifier or DualPathAudioClassifier
            # Calculate audio length
            audio_lengths = torch.tensor([input_tensor.shape[-1]], device=input_tensor.device)
            
            # Create dummy prosodic features if needed
            if hasattr(model, "prosodic_encoder"):
                # Determine prosodic feature dimension from model
                if hasattr(model, "prosodic_encoder") and isinstance(model.prosodic_encoder, torch.nn.Sequential):
                    first_layer = next(iter(model.prosodic_encoder.children()))
                    prosodic_dim = first_layer.in_features if hasattr(first_layer, "in_features") else 4
                else:
                    prosodic_dim = 4
                
                # Create a batch of zero prosodic features
                prosodic_features = torch.zeros((1, prosodic_dim), device=input_tensor.device)
                
                # Forward with prosodic features
                output = model(
                    input_tensor, 
                    audio_lengths=audio_lengths,
                    prosodic_features=prosodic_features
                )
            else:
                # Forward without prosodic features
                output = model(input_tensor, audio_lengths=audio_lengths)
                
        # Backward pass
        target = torch.zeros_like(output)
        target[0, target_class] = 1
        output.backward(gradient=target)
        
    except Exception as e:
        print(f"Error during gradient debugging: {str(e)}")
        # Continue with what information we have
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
        
    # Print results
    print("\n=== Gradient Debug Information ===")
    for name, info in layer_gradients.items():
        print(f"{name}:")
        print(f"  Shape: {info['shape']}")
        print(f"  Has gradient: {info['has_grad']}")
        print(f"  Mean absolute value: {info['mean']}")
    
    return layer_gradients


def create_interactive_cam_html(file_id, waveform, cam, sample_rate=16000, output_path=None, 
                           pred_label="Unknown", true_label="Unknown", pred_prob=0.0):
    """
    Create an interactive HTML visualization of the waveform with CAM overlay and audio playback controls.
    
    Args:
        file_id: Identifier for this audio sample
        waveform: Audio waveform as numpy array
        cam: CAM values as numpy array
        sample_rate: Audio sample rate (default: 16000)
        output_path: Path to save the HTML file
        pred_label: Predicted class label
        true_label: True class label
        pred_prob: Prediction probability
        
    Returns:
        Path to the HTML file if saved, or None
    """
    import base64
    import io
    import scipy.io.wavfile
    import numpy as np
    import os
    
    # Clean up the file_id to get a more readable display name
    display_name = file_id
    if isinstance(file_id, str):
        # Check if file_id is a path and extract just the filename
        if "/" in file_id or "\\" in file_id:
            display_name = os.path.basename(file_id)
        # Remove prediction info from the ID if present
        elif "_pred" in file_id:
            display_name = file_id.split("_pred")[0]
    
    # Normalize waveform to avoid clipping in audio playback
    waveform_norm = waveform.copy()
    max_val = np.max(np.abs(waveform_norm))
    if max_val > 0:
        waveform_norm = waveform_norm / max_val * 0.9
    
    # Convert waveform to 16-bit PCM
    waveform_16bit = (waveform_norm * 32767).astype(np.int16)
    
    # Create a WAV file in memory
    wav_buffer = io.BytesIO()
    scipy.io.wavfile.write(wav_buffer, sample_rate, waveform_16bit)
    wav_buffer.seek(0)
    
    # Encode WAV file to base64
    wav_base64 = base64.b64encode(wav_buffer.read()).decode('utf-8')
    
    # Prepare CAM data - resize to match audio duration for visualization
    duration = len(waveform) / sample_rate
    time_points = int(duration * 30)  # 30 points per second (reduced for performance)
    
    # Resize CAM to match time points
    from skimage.transform import resize
    cam_resized = resize(cam, (time_points,), anti_aliasing=True)
    
    # Normalize CAM values for visualization
    cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
    
    # Convert CAM values to JSON-compatible array
    cam_list = cam_norm.tolist()
    
    # Calculate time points for plotting waveform
    downsample_factor = max(1, len(waveform) // (time_points * 10))
    waveform_downsampled = waveform[::downsample_factor]
    
    # Generate timestamps
    time_sec = np.linspace(0, duration, len(waveform_downsampled))
    
    # Create data points for waveform
    waveform_data = []
    for i in range(len(waveform_downsampled)):
        waveform_data.append({"time": float(time_sec[i]), "amplitude": float(waveform_downsampled[i])})
    
    # HTML template with embedded JavaScript for interactive visualization
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interactive WaveformCAM - {display_name}</title>
        <meta charset="UTF-8">
        <script src="https://d3js.org/d3.v6.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f8f9fa;
                color: #333;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                border-radius: 5px;
            }}
            h1 {{
                color: #2c3e50;
                font-size: 24px;
                margin-bottom: 5px;
            }}
            h2 {{
                color: #3498db;
                font-size: 18px;
                margin-bottom: 20px;
            }}
            .info-panel {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                grid-gap: 10px;
                margin-bottom: 20px;
            }}
            .info-item {{
                background-color: #f0f3f6;
                padding: 10px;
                border-radius: 4px;
            }}
            .info-label {{
                font-weight: bold;
                color: #7f8c8d;
            }}
            .controls {{
                margin: 15px 0;
                display: flex;
                align-items: center;
            }}
            .play-btn {{
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                font-size: 18px;
                cursor: pointer;
                margin-right: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .play-btn:hover {{
                background-color: #2980b9;
            }}
            .time-display {{
                margin: 0 10px;
                font-family: monospace;
                min-width: 80px;
            }}
            .progress-container {{
                flex-grow: 1;
                background-color: #ecf0f1;
                height: 10px;
                border-radius: 5px;
                position: relative;
                cursor: pointer;
            }}
            .progress-bar {{
                height: 100%;
                background-color: #3498db;
                border-radius: 5px;
                width: 0%;
            }}
            .progress-marker {{
                position: absolute;
                top: -5px;
                width: 20px;
                height: 20px;
                background-color: #3498db;
                border-radius: 50%;
                transform: translateX(-50%);
                display: none;
            }}
            .tooltip {{
                position: absolute;
                background-color: rgba(44, 62, 80, 0.9);
                color: white;
                padding: 5px;
                border-radius: 3px;
                font-size: 12px;
                pointer-events: none;
                display: none;
            }}
            #waveform-container {{
                position: relative;
                height: 300px;
                margin-top: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
                border: 1px solid #ddd;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Interactive WaveformCAM Visualization</h1>
            <h2>Sample: {display_name}</h2>
            
            <div class="info-panel">
                <div class="info-item">
                    <div class="info-label">Predicted Class:</div>
                    <div>{pred_label} ({pred_prob:.4f})</div>
                </div>
                <div class="info-item">
                    <div class="info-label">True Class:</div>
                    <div>{true_label}</div>
                </div>
            </div>
            
            <div class="controls">
                <button id="play-btn" class="play-btn">▶</button>
                <div class="time-display">0:00 / {duration:.2f}s</div>
                <div class="progress-container" id="progress-container">
                    <div class="progress-bar" id="progress-bar"></div>
                    <div class="progress-marker" id="progress-marker"></div>
                    <div class="tooltip" id="tooltip"></div>
                </div>
            </div>
            
            <div id="waveform-container"></div>
            
            <audio id="audio-element" style="display: none;">
                <source src="data:audio/wav;base64,{wav_base64}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>
        </div>
        
        <script>
            // Audio playback functionality
            const audioElement = document.getElementById('audio-element');
            const playBtn = document.getElementById('play-btn');
            const progressBar = document.getElementById('progress-bar');
            const progressMarker = document.getElementById('progress-marker');
            const progressContainer = document.getElementById('progress-container');
            const timeDisplay = document.querySelector('.time-display');
            const tooltip = document.getElementById('tooltip');
            
            // CAM data
            const camData = {cam_list};
            
            // Waveform data
            const waveformData = {waveform_data};
            
            // Format time as mm:ss
            function formatTime(seconds) {{
                const mins = Math.floor(seconds / 60).toString().padStart(1, '0');
                const secs = Math.floor(seconds % 60).toString().padStart(2, '0');
                return `${{mins}}:${{secs}}`;
            }}
            
            // Toggle play/pause
            playBtn.addEventListener('click', () => {{
                if (audioElement.paused) {{
                    audioElement.play();
                    playBtn.textContent = '⏸';
                }} else {{
                    audioElement.pause();
                    playBtn.textContent = '▶';
                }}
            }});
            
            // Update progress bar during playback
            audioElement.addEventListener('timeupdate', () => {{
                const progress = (audioElement.currentTime / audioElement.duration) * 100;
                progressBar.style.width = `${{progress}}%`;
                progressMarker.style.left = `${{progress}}%`;
                progressMarker.style.display = 'block';
                timeDisplay.textContent = `${{formatTime(audioElement.currentTime)}} / ${{formatTime(audioElement.duration)}}`;
            }});
            
            // Reset UI when playback ends
            audioElement.addEventListener('ended', () => {{
                playBtn.textContent = '▶';
                progressMarker.style.display = 'none';
            }});
            
            // Seek functionality when clicking on progress bar
            progressContainer.addEventListener('click', (e) => {{
                const rect = progressContainer.getBoundingClientRect();
                const pos = (e.clientX - rect.left) / rect.width;
                audioElement.currentTime = pos * audioElement.duration;
            }});
            
            // Show tooltip with CAM value when hovering over progress bar
            progressContainer.addEventListener('mousemove', (e) => {{
                const rect = progressContainer.getBoundingClientRect();
                const pos = (e.clientX - rect.left) / rect.width;
                const timePos = pos * audioElement.duration;
                
                // Find closest CAM value
                const camIndex = Math.floor(pos * camData.length);
                const camValue = camData[Math.min(camIndex, camData.length - 1)];
                
                tooltip.textContent = `Time: ${{timePos.toFixed(2)}}s | CAM: ${{camValue.toFixed(2)}}`;
                tooltip.style.left = `${{e.clientX - rect.left}}px`;
                tooltip.style.top = '-25px';
                tooltip.style.display = 'block';
            }});
            
            progressContainer.addEventListener('mouseout', () => {{
                tooltip.style.display = 'none';
            }});
            
            // Draw the waveform with CAM overlay
            function drawWaveform() {{
                // Set up SVG container
                const margin = {{top: 20, right: 20, bottom: 30, left: 40}};
                const width = document.getElementById('waveform-container').offsetWidth - margin.left - margin.right;
                const height = document.getElementById('waveform-container').offsetHeight - margin.top - margin.bottom;
                
                // Create SVG element
                const svg = d3.select('#waveform-container')
                    .append('svg')
                    .attr('width', width + margin.left + margin.right)
                    .attr('height', height + margin.top + margin.bottom)
                    .append('g')
                    .attr('transform', `translate(${{margin.left}},${{margin.top}})`);
                
                // Set up scales
                const xScale = d3.scaleLinear()
                    .domain([0, d3.max(waveformData, d => d.time)])
                    .range([0, width]);
                
                const yScale = d3.scaleLinear()
                    .domain([-1, 1])
                    .range([height, 0]);
                
                // Create color scale for CAM values
                const colorScale = d3.scaleSequential()
                    .domain([0, 1])
                    .interpolator(d3.interpolateInferno);
                
                // Draw x-axis
                svg.append('g')
                    .attr('transform', `translate(0,${{height/2}}`)
                    .call(d3.axisBottom(xScale).ticks(10).tickFormat(d => d.toFixed(1) + 's'));
                
                // Group data points for visualization
                const bins = Math.min(300, waveformData.length);
                const binSize = waveformData.length / bins;
                
                // Prepare data for area path
                const areaData = [];
                for (let i = 0; i < bins; i++) {{
                    const startIdx = Math.floor(i * binSize);
                    const endIdx = Math.floor((i + 1) * binSize);
                    const binData = waveformData.slice(startIdx, endIdx);
                    
                    if (binData.length > 0) {{
                        const time = binData[0].time;
                        const maxAmp = d3.max(binData, d => d.amplitude);
                        const minAmp = d3.min(binData, d => d.amplitude);
                        
                        // Find CAM value for this time point
                        const camIdx = Math.floor((time / d3.max(waveformData, d => d.time)) * camData.length);
                        const camValue = camData[Math.min(camIdx, camData.length - 1)];
                        
                        areaData.push({{
                            time: time,
                            max: maxAmp,
                            min: minAmp,
                            cam: camValue
                        }});
                    }}
                }}
                
                // Create a clip path for the waveform area
                svg.append("defs").append("clipPath")
                    .attr("id", "clip")
                    .append("rect")
                    .attr("width", width)
                    .attr("height", height);
                
                // Define the line function that was missing
                const line = d3.line()
                    .x(d => xScale(d.time))
                    .y(d => yScale(d.amplitude))
                    .curve(d3.curveLinear);
                
                // Draw waveform with CAM coloring
                // First draw a subtle background waveform
                svg.append('path')
                    .datum(waveformData)
                    .attr('fill', 'none')
                    .attr('stroke', '#cccccc')
                    .attr('stroke-width', 1.5)
                    .attr('d', line)
                    .attr('clip-path', 'url(#clip)');
                
                // Then draw the CAM overlay segments with higher opacity
                for (let i = 0; i < areaData.length - 1; i++) {{
                    const d1 = areaData[i];
                    const d2 = areaData[i + 1];
                    
                    // Use CAM value to determine color
                    const fillColor = colorScale(d1.cam);
                    
                    // Calculate y-positions considering amplitude
                    const y1 = yScale(d1.min);
                    const y2 = yScale(d1.max);
                    const y3 = yScale(d2.max);
                    const y4 = yScale(d2.min);
                    
                    // Create a polygon for this segment - now with higher opacity
                    svg.append('polygon')
                        .attr('points', `
                            ${{xScale(d1.time)}},${{y1}}
                            ${{xScale(d1.time)}},${{y2}}
                            ${{xScale(d2.time)}},${{y3}}
                            ${{xScale(d2.time)}},${{y4}}
                        `)
                        .attr('fill', fillColor)
                        .attr('opacity', 0.5 + d1.cam * 0.5) // Higher opacity based on CAM value
                        .attr('stroke', fillColor)
                        .attr('stroke-opacity', 0.7)
                        .attr('stroke-width', 0.5);
                }}
                
                // Draw amplitude envelope (optional for better visualization)
                const upperEnvelope = d3.line()
                    .x(d => xScale(d.time))
                    .y(d => yScale(d.max))
                    .curve(d3.curveLinear);
                
                const lowerEnvelope = d3.line()
                    .x(d => xScale(d.time))
                    .y(d => yScale(d.min))
                    .curve(d3.curveLinear);
                
                svg.append('path')
                    .datum(areaData)
                    .attr('fill', 'none')
                    .attr('stroke', '#0066cc')
                    .attr('stroke-width', 0.7)
                    .attr('stroke-opacity', 0.6)
                    .attr('d', upperEnvelope);
                
                svg.append('path')
                    .datum(areaData)
                    .attr('fill', 'none')
                    .attr('stroke', '#0066cc')
                    .attr('stroke-width', 0.7)
                    .attr('stroke-opacity', 0.6)
                    .attr('d', lowerEnvelope);
                
                // Create a playback position indicator line
                const playbackLine = svg.append('line')
                    .attr('x1', 0)
                    .attr('x2', 0)
                    .attr('y1', 0)
                    .attr('y2', height)
                    .attr('stroke', 'red')
                    .attr('stroke-width', 2)
                    .attr('display', 'none');
                
                // Update playback line position during audio playback
                audioElement.addEventListener('timeupdate', () => {{
                    const position = xScale(audioElement.currentTime);
                    playbackLine.attr('x1', position)
                        .attr('x2', position)
                        .attr('display', 'block');
                }});
                
                // Hide playback line when audio ends
                audioElement.addEventListener('ended', () => {{
                    playbackLine.attr('display', 'none');
                }});
                
                // Add color legend
                const legendWidth = 200;
                const legendHeight = 15;
                
                const legendX = width - legendWidth - margin.right;
                const legendY = height - legendHeight - 10;
                
                // Create gradient
                const defs = svg.append('defs');
                const gradient = defs.append('linearGradient')
                    .attr('id', 'cam-gradient')
                    .attr('x1', '0%')
                    .attr('x2', '100%')
                    .attr('y1', '0%')
                    .attr('y2', '0%');
                
                // Add color stops
                const stops = 10;
                for (let i = 0; i <= stops; i++) {{
                    gradient.append('stop')
                        .attr('offset', `${{i/stops * 100}}%`)
                        .attr('stop-color', colorScale(i/stops));
                }}
                
                // Draw legend rectangle
                svg.append('rect')
                    .attr('x', legendX)
                    .attr('y', legendY)
                    .attr('width', legendWidth)
                    .attr('height', legendHeight)
                    .style('fill', 'url(#cam-gradient)');
                
                // Add legend title
                svg.append('text')
                    .attr('x', legendX + legendWidth / 2)
                    .attr('y', legendY - 5)
                    .attr('text-anchor', 'middle')
                    .style('font-size', '12px')
                    .text('CAM Activation');
                
                // Add legend ticks
                svg.append('text')
                    .attr('x', legendX)
                    .attr('y', legendY + legendHeight + 15)
                    .attr('text-anchor', 'start')
                    .style('font-size', '10px')
                    .text('Low');
                
                svg.append('text')
                    .attr('x', legendX + legendWidth)
                    .attr('y', legendY + legendHeight + 15)
                    .attr('text-anchor', 'end')
                    .style('font-size', '10px')
                    .text('High');
            }}
            
            // Initialize the visualization when the page loads
            window.addEventListener('load', drawWaveform);
        </script>
    </body>
    </html>
    """
    
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        return output_path
    
    return None