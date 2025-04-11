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
            
            # Prepare input shape
            if model_type == "cnn14":
                if len(input_tensor.shape) == 3:  # [B, 1, T]
                    input_tensor = input_tensor.squeeze(1)  # CNN14 expects [B, T]
            else:
                if len(input_tensor.shape) == 2:  # [B, T]
                    input_tensor = input_tensor.unsqueeze(1)  # DualPath expects [B, 1, T]
            
            if self.cuda:
                input_tensor = input_tensor.cuda()
            
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


def visualize_cam(audio, model, target_class=None, save_path=None, audio_id=None, correct=None, audio_paths_dir=None, epoch=None):
    """
    Visualize CAM for an audio input
    """
    # Create output directories if needed
    if save_path:
        os.makedirs(os.path.join(save_path, 'LogMelSpecs'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'CAMs'), exist_ok=True)
    
    # Get model's device
    device = next(model.parameters()).device
    
    # Move audio to device
    audio = audio.to(device)
    
    # Make a clone of the input tensor to ensure it's not an inference tensor
    audio_clone = audio.detach().clone()
    
    # Store original training state
    training_state = model.training
    
    # Set model to eval mode temporarily
    model.eval()
    
    # Get target layer for CAM
    target_layer = get_model_target_layers(model)
    
    # Get the actual model prediction first (without forcing a target)
    with torch.no_grad():
        # Calculate proper audio lengths
        audio_lengths = torch.tensor([audio_clone.shape[-1]], device=audio_clone.device)
        
        # Check if model uses prosodic features
        if hasattr(model, "prosodic_encoder"):
            # Find proper prosodic feature size
            if hasattr(model.prosodic_encoder, "0") and hasattr(model.prosodic_encoder[0], "in_features"):
                prosodic_dim = model.prosodic_encoder[0].in_features
            else:
                prosodic_dim = 4  # Default fallback
            
            # Create dummy prosodic features
            prosodic_features = torch.zeros((1, prosodic_dim), device=audio_clone.device)
            
            # Get model's prediction with all required arguments
            actual_logits = model(
                audio_clone, 
                audio_lengths=audio_lengths,
                prosodic_features=prosodic_features
            )
        else:
            # For models that don't use prosodic features
            actual_logits = model(
                audio_clone,
                audio_lengths=audio_lengths
            )
        
        actual_pred_class = actual_logits.argmax(dim=1).item()
        actual_probs = F.softmax(actual_logits, dim=1)
        actual_pred_prob = actual_probs[0, actual_pred_class].item()

    
    # Create subdirectories if needed now that we've verified 'correct'
    if save_path and correct is not None:
        status = "correct" if correct else "incorrect"
        os.makedirs(os.path.join(save_path, 'LogMelSpecs', status), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'CAMs', status), exist_ok=True)

    # Add epoch parameter to function
    if epoch is not None:
        spec_subdir = os.path.join(f"epoch_{epoch}", 'LogMelSpecs')
        cam_subdir = os.path.join(f"epoch_{epoch}", 'CAMs')
    
        os.makedirs(os.path.join(save_path, spec_subdir), exist_ok=True)
        os.makedirs(os.path.join(save_path, cam_subdir), exist_ok=True)

    # Now do the CAM visualization with the desired target class  
    grad_cam = GradCAM(model, target_layer, use_cuda=(device.type == 'cuda'))
    cam, logits, _ = grad_cam(audio_clone, target_class)

    # Map class indices to human-readable labels - DEFINE ONLY ONCE
    class_names = ["Healthy", "MCI", "AD"]
    pred_label = class_names[actual_pred_class]  # Use actual prediction  
    true_label = class_names[target_class] if target_class is not None else "Unknown"

    # Debug print to verify values
    # print(f"CAM Debug: audio_id={audio_id}, pred={actual_pred_class}({pred_label}), true={target_class}({true_label}), correct={correct}")

    # Create filename
    if audio_id is not None and ('pred' in audio_id or 'true' in audio_id):
        file_id = audio_id  
    else:
        file_id = audio_id if audio_id is not None else f"sample_{actual_pred_class}"
        if target_class is not None:
            file_id += f"_pred{actual_pred_class}_true{target_class}"

    # Generate spectrogram - no gradients needed for this
    spectrogram = generate_spectrogram(audio, model)
    
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

    # Create time axis in seconds
    time_frames = spec_for_plot.shape[1]
    time_sec = np.arange(time_frames) * hop_length / sr
    time_ticks = np.linspace(0, time_frames-1, min(10, time_frames), dtype=int)
    time_labels = [f"{time_sec[i]:.1f}" for i in time_ticks]

    # Create figure for visualization
    plt.figure(figsize=(12, 5))
    
    # Plot spectrogram with actual time and frequency units
    plt.subplot(1, 2, 1)
    plt.imshow(spec_for_plot.T, origin='lower', aspect='auto', cmap='viridis')
    title = f"Log-Mel Spectrogram\nPred: {pred_label} ({actual_pred_prob:.2f})"
    if target_class is not None:
        title += f"\nTrue: {true_label}"
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.xticks(time_ticks, time_labels)
    plt.yticks(freq_ticks, freq_labels)
    
    # Plot CAM heatmap with actual time and frequency units
    plt.subplot(1, 2, 2)
    plt.imshow(spec_for_plot.T, origin='lower', aspect='auto', alpha=0.6, cmap='viridis')

    # Resize CAM to match spectrogram dimensions
    cam_resized = resize(cam, (spec_for_plot.shape[1],), anti_aliasing=True)
    cam_2d = np.zeros((spec_for_plot.shape[1], spec_for_plot.shape[0]))
    for i in range(spec_for_plot.shape[0]):
        cam_2d[:, i] = cam_resized

    # Plot the resized CAM
    plt.imshow(cam_2d, origin='lower', aspect='auto', alpha=0.4, cmap='inferno')
    
    title = f"Class Activation Map (Full Audio)\nPred: {pred_label} ({actual_pred_prob:.2f})"
    if target_class is not None:
        result = "✓" if actual_pred_class == target_class else "✗"
        title += f"\nTrue: {true_label} {result}"
        
        # Add chunked prediction info if it differs
        if 'pred' in audio_id:
            chunked_pred = audio_id.split('_')[1].replace('pred', '')
            if int(chunked_pred) != actual_pred_class:
                title += f"\nNote: Chunked pred was {class_names[int(chunked_pred)]}"
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.xticks(time_ticks, time_labels)
    plt.yticks(freq_ticks, freq_labels)
    
    # Save files if path is provided
    spec_path = None
    cam_path = None
    
    if save_path:
        # Initialize base subdirectories
        spec_subdir = 'LogMelSpecs'
        cam_subdir = 'CAMs'
        
        # Add epoch to path if provided
        if epoch is not None:
            spec_subdir = os.path.join(f"epoch_{epoch}", spec_subdir)
            cam_subdir = os.path.join(f"epoch_{epoch}", cam_subdir)
        
        # Then add correct/incorrect status
        if correct is not None:
            status = "correct" if correct else "incorrect" 
            spec_subdir = os.path.join(spec_subdir, status)
            cam_subdir = os.path.join(cam_subdir, status)
        
        # Create directories if they don't exist
        os.makedirs(os.path.join(save_path, spec_subdir), exist_ok=True)
        os.makedirs(os.path.join(save_path, cam_subdir), exist_ok=True)
        
        # Save paths (using file_id that was created above)
        spec_path = os.path.join(save_path, spec_subdir, f"{file_id}_spec.png")
        cam_path = os.path.join(save_path, cam_subdir, f"{file_id}_cam.png")
        
        # Save figure
        plt.tight_layout()
        plt.savefig(cam_path, dpi=150)
        
        # Save spectrogram separately
        plt.figure(figsize=(6, 4))
        plt.imshow(spec_for_plot.T, origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Log-Mel Spectrogram - {file_id}")
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.xticks(time_ticks, time_labels)
        plt.yticks(freq_ticks, freq_labels)
        plt.tight_layout()
        plt.savefig(spec_path, dpi=150)
        
        # Write to log file - ONLY ONCE
        if audio_paths_dir:
            # Also organize audio paths by epoch if provided
            if epoch is not None:
                audio_paths_dir = os.path.join(audio_paths_dir, f"epoch_{epoch}")
            
            os.makedirs(audio_paths_dir, exist_ok=True)
            paths_filename = "correct_samples.txt" if correct else "incorrect_samples.txt"
            with open(os.path.join(audio_paths_dir, paths_filename), 'a') as f:
                f.write(f"{file_id}\t{pred_label}\t{true_label}\t{actual_pred_prob:.4f}\n")
    
    # Close figures to free memory
    plt.close('all')
    
    # Clean up GradCAM
    grad_cam.remove_hooks()
    del grad_cam
    
    # Restore model training state
    if training_state:
        model.train()
    
    return actual_pred_class, target_class, cam_path, spec_path


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