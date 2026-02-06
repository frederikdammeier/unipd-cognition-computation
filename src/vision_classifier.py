import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualClassifier(nn.Module):
    """
    Visual classifier using the same convolutional structure as VAE encoder.
    
    Designed for 64x64 images with output dimension of 36 classes.
    """
    
    def __init__(self, latent_dim: int = 10, num_classes: int = 36):
        """
        Initialize Visual Classifier.
        
        Args:
            latent_dim: Size of the intermediate latent layer (default: 10)
            num_classes: Number of output classes (default: 36)
        """
        super(VisualClassifier, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Convolutional layers (same as VAE encoder)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        
        # Calculate the flattened size after convolutions
        self.flat_size = self._calculate_flat_size()
        
        # Fully connected layer to latent_dim
        self.fc_latent = nn.Linear(self.flat_size, latent_dim)
        
        # Output layer for classification
        self.fc_output = nn.Linear(latent_dim, num_classes)
    
    def _calculate_flat_size(self) -> int:
        """Calculate the flattened size after all convolutions."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 64, 64)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            return x.view(1, -1).size(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classifier.
        
        Args:
            x: Input image tensor of shape (batch_size, 3, 64, 64)
            
        Returns:
            logits: Output logits of shape (batch_size, num_classes)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc_latent(x))
        logits = self.fc_output(x)
        
        return logits


def train_classifier(model: VisualClassifier, 
                    train_loader: torch.utils.data.DataLoader,
                    val_loader: torch.utils.data.DataLoader,
                    num_epochs: int = 10,
                    learning_rate: float = 1e-3,
                    device: torch.device = None) -> dict:
    """
    Train the visual classifier.
    
    Args:
        model: VisualClassifier model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs (default: 10)
        learning_rate: Learning rate for optimizer (default: 1e-3)
        device: Device to train on (default: CPU)
        
    Returns:
        Dictionary containing training and validation metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    metrics = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                logits = model(images)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        
        metrics['train_loss'].append(train_loss)
        metrics['train_accuracy'].append(train_accuracy)
        metrics['val_loss'].append(val_loss)
        metrics['val_accuracy'].append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    return metrics