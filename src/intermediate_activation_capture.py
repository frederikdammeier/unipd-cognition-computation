import torch
from typing import Dict
from .vision_vae import VisionVAE

class IntermediateActivationCapture:
    """
    Utility class to capture intermediate activations during VAE inference.
    Uses PyTorch hooks to capture activations at each layer.
    """
    
    def __init__(self, model: VisionVAE):
        self.model = model
        self.activations = {}
        self.hooks = []
    
    def _create_hook(self, name: str):
        """Create a hook function to capture activations."""
        def hook(module, input, output):
            # Detach and store output directly without moving to CPU yet
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach()
            elif isinstance(output, tuple):
                # For modules that return tuples (like encoder)
                self.activations[name] = tuple(o.detach() if isinstance(o, torch.Tensor) else o for o in output)
        return hook
    
    def register_hooks(self):
        """Register hooks on all relevant layers. Must be called after model is on target device."""
        # Clear any previous hooks
        self.remove_hooks()
        
        # Encoder hooks
        self.hooks.append(self.model.encoder.conv1.register_forward_hook(
            self._create_hook('encoder_conv1')))
        self.hooks.append(self.model.encoder.conv2.register_forward_hook(
            self._create_hook('encoder_conv2')))
        self.hooks.append(self.model.encoder.conv3.register_forward_hook(
            self._create_hook('encoder_conv3')))
        self.hooks.append(self.model.encoder.conv4.register_forward_hook(
            self._create_hook('encoder_conv4')))
        self.hooks.append(self.model.encoder.fc_mu.register_forward_hook(
            self._create_hook('encoder_fc_mu')))
        self.hooks.append(self.model.encoder.fc_logvar.register_forward_hook(
            self._create_hook('encoder_fc_logvar')))
        
        # Decoder hooks
        self.hooks.append(self.model.decoder.fc.register_forward_hook(
            self._create_hook('decoder_fc')))
        self.hooks.append(self.model.decoder.deconv1.register_forward_hook(
            self._create_hook('decoder_deconv1')))
        self.hooks.append(self.model.decoder.deconv2.register_forward_hook(
            self._create_hook('decoder_deconv2')))
        self.hooks.append(self.model.decoder.deconv3.register_forward_hook(
            self._create_hook('decoder_deconv3')))
        self.hooks.append(self.model.decoder.deconv4.register_forward_hook(
            self._create_hook('decoder_deconv4')))
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def capture_inference(self, x: torch.Tensor, device: str = 'cpu') -> Dict:
        """
        Run inference and capture all intermediate representations.
        
        Args:
            x: Input batch of shape (batch_size, 3, 64, 64)
            device: Device to run inference on
            
        Returns:
            Dictionary containing:
            - 'input': Original input
            - 'encoder_*': Intermediate encoder activations
            - 'mu': Mean of latent distribution
            - 'logvar': Log variance of latent distribution
            - 'z': Sampled latent vector
            - 'decoder_*': Intermediate decoder activations
            - 'reconstruction': Final reconstructed image
        """
        self.activations.clear()
        
        # Move model to device
        self.model = self.model.to(device)
        # Re-register hooks after moving to device to ensure they work properly
        self.register_hooks()
        
        x = x.to(device)
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            recon_x, mu, logvar, z = self.model(x)
        
        # Move all captured activations to CPU for storage
        cpu_activations = {}
        for name, activation in self.activations.items():
            if isinstance(activation, torch.Tensor):
                cpu_activations[name] = activation.cpu()
            elif isinstance(activation, tuple):
                cpu_activations[name] = tuple(a.cpu() if isinstance(a, torch.Tensor) else a for a in activation)
            else:
                cpu_activations[name] = activation
        
        # Organize results
        results = {
            'input': x.cpu(),
            'reconstruction': recon_x.cpu(),
            'latent': {
                'z': z.cpu(),
                'mu': mu.cpu(),
                'logvar': logvar.cpu(),
                'std': torch.exp(0.5 * logvar).cpu()
            },
            'encoder_activations': {},
            'decoder_activations': {}
        }
        
        # Separate encoder and decoder activations
        for name, activation in cpu_activations.items():
            if 'encoder' in name:
                results['encoder_activations'][name] = activation
            elif 'decoder' in name:
                results['decoder_activations'][name] = activation
        
        return results


# USAGE EXAMPLE:
if __name__ == "__main__":
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VisionVAE(latent_dim=10, beta=1.0)
    model.load_state_dict(torch.load('checkpoints/vae_final.pt', map_location=device))
    
    # Create capture utility
    capturer = IntermediateActivationCapture(model)
    
    # Create dummy batch (replace with real data)
    batch_size = 4
    x_batch = torch.randn(batch_size, 3, 64, 64)
    
    # Run inference and capture representations
    # Hooks are registered automatically inside capture_inference
    results = capturer.capture_inference(x_batch, device=device)
    
    # Clean up hooks when done
    capturer.remove_hooks()
    
    # Access specific representations for analysis
    print(f"Encoder activations: {results['encoder_activations'].keys()}")
    print(f"Decoder activations: {results['decoder_activations'].keys()}")
    print(f"Latent space shape: {results['latent']['z'].shape}")
    
    # For example:
    # encoder_features = results['encoder_activations']['encoder_conv4']
    # latent_codes = results['latent']['z']
    # decoder_features = results['decoder_activations']['decoder_deconv2']