import os
from pathlib import Path
from typing import Tuple, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ImageDataset(Dataset):
    """
    PyTorch Dataset for images organized in class-labeled folders.
    
    Expected structure:
    data_root/
    ├── class_1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class_2/
    │   ├── image1.jpg
    │   └── ...
    └── ...
    """
    
    def __init__(
        self,
        data_root: str,
        transform=None,
        image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    ):
        """
        Args:
            data_root: Path to the root directory containing class folders
            transform: Optional torchvision transforms to apply to images
            image_extensions: Tuple of valid image file extensions (case-insensitive)
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self.image_extensions = tuple(ext.lower() for ext in image_extensions)
        
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load image paths and labels from class folders."""
        class_idx = 0
        
        # Iterate through class folders
        for class_folder in sorted(self.data_root.iterdir()):
            if not class_folder.is_dir():
                continue
            
            class_name = class_folder.name
            self.class_to_idx[class_name] = class_idx
            self.idx_to_class[class_idx] = class_name
            
            # Collect images from class folder
            for image_file in sorted(class_folder.iterdir()):
                if image_file.is_file() and image_file.suffix.lower() in self.image_extensions:
                    self.image_paths.append(image_file)
                    self.labels.append(class_idx)
            
            class_idx += 1
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, label) where image is a tensor and label is the class index
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms if provided, ensure output is a tensor if not
        if self.transform:
            image = self.transform(image)
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
        
        return image, label
    
    def get_class_name(self, idx: int) -> str:
        """Get class name from class index."""
        return self.idx_to_class[idx]
    
    def get_num_classes(self) -> int:
        """Get the number of classes."""
        return len(self.class_to_idx)


class ImageDataLoader:
    """Utility class for creating PyTorch DataLoaders for image datasets."""
    
    @staticmethod
    def create_dataloader(
        data_root: str,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        transform=None,
        **kwargs
    ) -> DataLoader:
        """
        Create a DataLoader for an image dataset.
        
        Args:
            data_root: Path to the root directory containing class folders
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the dataset
            num_workers: Number of worker processes for data loading
            transform: Optional torchvision transforms to apply to images
            **kwargs: Additional arguments to pass to DataLoader
            
        Returns:
            PyTorch DataLoader instance
        """
        dataset = ImageDataset(data_root, transform=transform)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )
        
        return dataloader
