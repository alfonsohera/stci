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
        # For CNN14, we need to access the conv layer inside the conv_block6
        # Conv blocks in PANNs use 'conv1' as the attribute name
        return model.feature_extractor.conv_block6.conv1
    
    elif model_name == "PretrainedDualPathAudioClassifier":
        # For PretrainedDualPathAudioClassifier, access the conv layer in conv_block6
        # Conv blocks in PANNs use 'conv1' as the attribute name
        return model.cnn_extractor.conv_block6.conv1
    
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


def visualize_cam(audio, model, target_class=None, save_path=None, audio_id=None, correct=None, audio_paths_dir=None, epoch=None,                   
                  audio_chunks=None, chunk_outputs=None, show_time_domain=False):
    """
    Visualize CAM for an audio input, with support for chunked processing
    
    Args:
        audio: Audio tensor or a single chunk of audio 
        model: Model to analyze
        target_class: Target class for CAM
        save_path: Directory to save visualizations
        audio_id: ID of the audio sample
        correct: Whether prediction is correct
        audio_paths_dir: Directory to save audio paths
        epoch: Current epoch number
        audio_chunks: Optional list of already chunked audio or audio_id to fetch original audio
        chunk_outputs: Optional list of chunk outputs (logits) from model
        show_time_domain: Whether to display the time-domain representation (waveform) with CAM overlay
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
    
    # Check if audio_chunks is just an ID - if so, we need to fetch the complete original audio
    if isinstance(audio_chunks, str) or isinstance(audio_chunks, int):
        try:
            # Import the necessary function to fetch original audio
            from cnn_rnn_data import get_original_audio_by_id, chunk_audio
            
            print(f"Fetching original audio for ID: {audio_chunks}")
            full_audio = get_original_audio_by_id(audio_chunks)
            
            if full_audio is not None:
                # Convert to tensor if needed
                if not isinstance(full_audio, torch.Tensor):
                    full_audio = torch.tensor(full_audio).float()
                
                # Ensure it has batch dimension [1, T] or [1, 1, T]
                if len(full_audio.shape) == 1:
                    full_audio = full_audio.unsqueeze(0)
                    
                # Move to device
                full_audio = full_audio.to(device)
                
                # Chunk the full audio into 10-second segments
                chunk_size_seconds = 10
                sample_rate = 16000
                
                # Ensure audio is on CPU before chunking (to avoid device mismatch)
                full_audio_cpu = full_audio.cpu()
                
                # Create chunks from full audio
                audio_chunks_list = chunk_audio(full_audio_cpu, chunk_size_seconds=chunk_size_seconds, sample_rate=sample_rate)
                print(f"Created {len(audio_chunks_list)} chunks from original full audio")
                
                # Move chunks back to device
                audio_chunks_list = [chunk.to(device) for chunk in audio_chunks_list]
            else:
                # Fallback to the provided audio if we couldn't get the original
                print(f"Could not fetch original audio for ID: {audio_chunks}, using provided audio chunk")
                audio_chunks_list = [audio_clone]
        except ImportError:
            print("Could not import get_original_audio_by_id, using provided audio chunk")
            audio_chunks_list = [audio_clone]
        except Exception as e:
            print(f"Error fetching original audio: {str(e)}, using provided audio chunk")
            audio_chunks_list = [audio_clone]
    # If audio_chunks is already a dictionary with a list for the audio_id key
    elif isinstance(audio_chunks, dict) and audio_id in audio_chunks and isinstance(audio_chunks[audio_id], list):
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
    true_class_idx = target_class if target_class is not None else None  # Default to provided target class

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
        if target_class is not None:
            file_id += f"_pred{actual_pred_class}_true{target_class}"

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
    print(f"Total chunks: {total_chunks}, Time frames: {time_frames}, Full duration: {total_chunks * chunk_size_seconds}s")
    print(f"Time labels: {time_labels}")

    # CHANGE: Create two separate figures instead of one combined figure
    
    # Figure 1: Spectrogram and CAM visualization
    plt.figure(figsize=(12, 9))
    
    # Plot 1: Spectrogram
    plt.subplot(2, 1, 1)
    plt.imshow(spec_for_plot, origin='lower', aspect='auto', cmap='viridis')
    title = f"Log-Mel Spectrogram\nPred: {pred_label}, True: {true_label}"
    if target_class is not None:
        result = "✓" if actual_pred_class == target_class else "✗"
        title += f" {result}"
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

    title = f"Class Activation Map\nPred: {pred_label}, True: {true_label}" 
    if target_class is not None:
        result = "✓" if actual_pred_class == target_class else "✗"
        title += f" {result}"
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
    
    # Save spectrogram/CAM visualization if save_path is provided
    if save_path:
        # Initialize base subdirectories
        spec_cam_subdir = 'SpectrogramCAMs'
        waveform_subdir = 'WaveformCAMs'
        
        # Add epoch to path if provided
        if epoch is not None:
            spec_cam_subdir = os.path.join(f"epoch_{epoch}", spec_cam_subdir)
            waveform_subdir = os.path.join(f"epoch_{epoch}", waveform_subdir)
        
        # Then add correct/incorrect status
        if correct is not None:
            status = "correct" if correct else "incorrect" 
            spec_cam_subdir = os.path.join(spec_cam_subdir, status)
            waveform_subdir = os.path.join(waveform_subdir, status)

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
        
        # Time axis for waveform (in seconds)
        waveform_duration = len(full_waveform) / sr
        
        # PERFORMANCE OPTIMIZATION: Downsample the waveform and CAM for plotting
        # This prevents hanging with very long audio files
        target_points = 10000  # Maximum number of points to plot
        
        if len(full_waveform) > target_points:
            # Calculate downsample factor
            downsample_factor = int(np.ceil(len(full_waveform) / target_points))
            
            # Downsample waveform
            waveform_downsampled = full_waveform[::downsample_factor]
            
            # Create time axis for downsampled waveform
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
        
        plt.title(f"Audio Waveform with CAM Overlay\nPred: {pred_label}, True: {true_label}")
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
    
    # Write to log file - ONLY ONCE
    if save_path and audio_paths_dir:
        # Also organize audio paths by epoch if provided
        if epoch is not None:
            audio_paths_dir = os.path.join(audio_paths_dir, f"epoch_{epoch}")
        
        os.makedirs(audio_paths_dir, exist_ok=True)
        paths_filename = "correct_samples.txt" if correct else "incorrect_samples.txt"
        with open(os.path.join(audio_paths_dir, paths_filename), 'a') as f:
            f.write(f"{file_id}\t{pred_label}\t{true_label}\t{actual_pred_prob:.4f}\t{total_chunks}chunks\n")
    
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
                if hasattr(model, "prosodic_encoder") and isinstance(self.model.prosodic_encoder, torch.nn.Sequential):
                    first_layer = next(iter(self.model.prosodic_encoder.children()))
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