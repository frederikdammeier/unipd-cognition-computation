from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Convolutional encoder for VAE."""
    
    def __init__(self, latent_dim: int = 10):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        
        # Calculate the flattened size after convolutions
        self.flat_size = self._calculate_flat_size()
        
        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
    
    def _calculate_flat_size(self) -> int:
        """Calculate the flattened size after all convolutions."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 64, 64)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            return x.view(1, -1).size(1)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Encode image to latent space.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class Decoder(nn.Module):
    """Convolutional decoder for VAE."""
    
    def __init__(self, latent_dim: int = 10):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Calculate intermediate dimensions
        self.flat_size = self._calculate_flat_size()
        
        # Fully connected layer to expand latent vector
        self.fc = nn.Linear(latent_dim, self.flat_size)
        
        # Transposed convolutional layers
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
    
    def _calculate_flat_size(self) -> int:
        """Calculate the flattened size that corresponds to encoder output."""
        # For 64x64 images: 64 / 2^4 = 4, so 256 * 4 * 4 = 4096
        return 256 * 4 * 4
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to image.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            
        Returns:
            Reconstructed image tensor of shape (batch_size, 3, 64, 64)
        """
        x = self.fc(z)
        x = x.view(-1, 256, 4, 4)
        
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))  # Sigmoid to keep output in [0, 1]
        
        return x


class VisionVAE(nn.Module):
    """
    Vision Beta-VAE with convolutional encoder and decoder.
    
    Designed for 64x64 images.
    """
    
    def __init__(self, latent_dim: int = 10, beta: float = 1.0):
        """
        Initialize Vision VAE.
        
        Args:
            latent_dim: Size of the latent space (default: 10)
            beta: Weight for KL divergence term in loss (default: 1.0)
        """
        super(VisionVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.beta = beta
        
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from latent distribution.
        
        Args:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through VAE.
        
        Args:
            x: Input image tensor of shape (batch_size, 3, height, width)
            
        Returns:
            recon_x: Reconstructed image
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            z: Sampled latent vector
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar, z
    
    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor, 
                      mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Calculate VAE loss (reconstruction + KL divergence).
        
        Args:
            recon_x: Reconstructed image
            x: Original image
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Total loss (reconstruction loss + beta * KL divergence)
        """
        batch_size = x.size(0)

        # Reconstruction loss (binary cross-entropy)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')

        # MSE = F.mse_loss(recon_x, x, reduction='mean')
        # RMSE = torch.sqrt(MSE)
        # recon_loss = RMSE
        
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # Total loss with beta weighting
        loss = recon_loss + self.beta * kl_loss
        
        return loss, recon_loss, kl_loss
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space (deterministic, using mean)."""
        mu, _ = self.encoder(x)
        return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image."""
        return self.decoder(z)

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