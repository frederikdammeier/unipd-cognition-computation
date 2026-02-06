import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch

def visualize_activation_maps(activation_tensor: torch.Tensor, 
                              n_maps: int = 8, 
                              figsize: tuple = None,
                              title: str = "Activation Maps",
                              cmap: str = 'viridis') -> plt.Figure:
    """
    Visualize activation maps as a grid of grayscale images.
    
    Args:
        activation_tensor: Tensor of shape [batch_size, depth, width, height]
        n_maps: Number of feature maps to visualize (from depth dimension)
        figsize: Figure size as (width, height). If None, auto-calculated.
        title: Title for the figure
        cmap: Colormap to use ('viridis', 'gray', 'hot', etc.)
        
    Returns:
        matplotlib Figure object
    """
    # Handle different tensor formats and move to CPU/numpy
    if isinstance(activation_tensor, torch.Tensor):
        activation_tensor = activation_tensor.detach().cpu()
    
    # Ensure we have the right shape
    if activation_tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor [batch_size, depth, width, height], got shape {activation_tensor.shape}")
    
    batch_size, depth, width, height = activation_tensor.shape
    n_maps = min(n_maps, depth)  # Don't exceed available depth
    
    # Convert to numpy
    if isinstance(activation_tensor, torch.Tensor):
        activation_np = activation_tensor.numpy()
    else:
        activation_np = activation_tensor
    
    # Create figure
    if figsize is None:
        figsize = (batch_size * 1.5, n_maps * 1.5)
    
    fig, axes = plt.subplots(n_maps, batch_size, figsize=figsize)
    
    # Handle case where there's only one row or one column
    if n_maps == 1 and batch_size == 1:
        axes = np.array([[axes]])
    elif n_maps == 1:
        axes = axes.reshape(1, -1)
    elif batch_size == 1:
        axes = axes.reshape(-1, 1)
    
    # Normalize across all data for consistent visualization
    vmin = activation_np.min()
    vmax = activation_np.max()
    if vmax > vmin:
        activation_normalized = (activation_np - vmin) / (vmax - vmin)
    else:
        activation_normalized = activation_np
    
    # Plot each feature map
    for feature_idx in range(n_maps):
        for batch_idx in range(batch_size):
            ax = axes[feature_idx, batch_idx]
            
            # Extract the feature map for this batch and depth
            feature_map = activation_normalized[batch_idx, feature_idx, :, :]
            
            # Display the feature map
            im = ax.imshow(feature_map, cmap=cmap, aspect='auto')
            
            # Configure axes
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add labels
            if batch_idx == 0:
                ax.set_ylabel(f'Feature {feature_idx}', fontsize=10)
            if feature_idx == 0:
                ax.set_title(f'Sample {batch_idx}', fontsize=10)
    
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    return fig

def plot_confusion_matrix(predictions, targets, label_dict, normalize=True, figsize=(16, 14)):
    """
    Calculate and display confusion matrix for many classes.
    
    Args:
        predictions: predicted class indices
        targets: true class indices
        label_dict: {idx: label_name} dictionary
        normalize: if True, normalize by true labels (shows recall per class)
        figsize: figure size (larger for many classes)
    """
    # Calculate confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    # Normalize if requested (shows recall per class)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create labels in order
    class_labels = [label_dict[i] for i in range(len(label_dict))]
    
    # Plot with matplotlib imshow
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap using imshow
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, fontsize=8)
    ax.set_yticklabels(class_labels, fontsize=8)
    
    # Rotate and align labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Recall' if normalize else 'Count', fontsize=10)
    
    # Add text annotations with adaptive color
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            value = cm[i, j]
            if normalize:
                text = ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                             color='white' if value > 0.5 else 'black', fontsize=6)
            else:
                text = ax.text(j, i, f'{int(value)}', ha='center', va='center',
                             color='white' if value > cm.max()/2 else 'black', fontsize=6)
    
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.set_title(f'Confusion Matrix ({len(label_dict)} classes)', fontsize=12)
    
    plt.tight_layout()
    return fig, cm