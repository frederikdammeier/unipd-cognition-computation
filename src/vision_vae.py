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
