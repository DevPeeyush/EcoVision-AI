import os
import random
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

import sys
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from data.transforms import get_train_transforms, get_val_transforms
from data.spatial_split import generate_spatial_split
from configs.config import RANDOM_SEED, BATCH_SIZE, NUM_WORKERS, TRAIN_SPLIT, VAL_SPLIT
from configs.settings import configure_logger

# Reproducibility seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

logger = configure_logger("DatasetLoader")

class SatelliteDatasetLoader:
    def __init__(self, root_dir: Path, split_type: str = 'random'):
        """
        DataLoader manager for Satellite Imagery (EuroSAT or UC Merced).
        
        Args:
            root_dir (Path): Absolute path to the dataset folder.
            split_type (str): 'random' or 'spatial' data splitting strategy.
        """
        self.root_dir = root_dir
        self.split_type = split_type
        
        if not self.root_dir.exists():
            logger.error(f"Dataset directory not found: {self.root_dir}")
            raise FileNotFoundError(f"Dataset root {self.root_dir} does not exist.")
            
        # Support UC Merced structure where categories are inside a nested 'Images' folder
        images_dir = self.root_dir / "Images"
        if images_dir.exists() and images_dir.is_dir():
            subdirs = [d for d in os.listdir(images_dir) if (images_dir / d).is_dir()]
            if len(subdirs) > 0:
                logger.info(f"Auto-resolved dataset loader root directory to nested folder: {images_dir}")
                self.root_dir = images_dir
            
        # Parse datasets
        self.base_dataset = ImageFolder(root=str(self.root_dir))
        self.classes = self.base_dataset.classes
        self.num_classes = len(self.classes)
        
        logger.info(f"Loaded dataset from {self.root_dir} containing {len(self.base_dataset)} images across {self.num_classes} classes.")

    def _get_split_indices(self) -> Tuple[List[int], List[int], List[int]]:
        """Splits data indices based on config ratios and chosen split strategy."""
        num_samples = len(self.base_dataset)
        
        if self.split_type == 'spatial':
            logger.info("Generating Spatial Block split indices...")
            return generate_spatial_split(num_samples, TRAIN_SPLIT, VAL_SPLIT)
        else:
            logger.info("Generating Random split indices...")
            indices = list(range(num_samples))
            random.shuffle(indices)
            
            train_cutoff = int(np.floor(TRAIN_SPLIT * num_samples))
            val_cutoff = int(np.floor((TRAIN_SPLIT + VAL_SPLIT) * num_samples))
            
            train_idx = indices[:train_cutoff]
            val_idx = indices[train_cutoff:val_cutoff]
            test_idx = indices[val_cutoff:]
            
            return train_idx, val_idx, test_idx

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Builds standard training, validation, and test PyTorch DataLoaders.
        Only the training DataLoader applies random augmentations; validation and testing use identity transforms.
        """
        try:
            train_idx, val_idx, test_idx = self._get_split_indices()
            
            # Instantiate datasets with target transforms
            train_dataset = ImageFolder(root=str(self.root_dir), transform=get_train_transforms())
            val_dataset = ImageFolder(root=str(self.root_dir), transform=get_val_transforms())
            test_dataset = ImageFolder(root=str(self.root_dir), transform=get_val_transforms())
            
            # Map subset indices
            train_subset = Subset(train_dataset, train_idx)
            val_subset = Subset(val_dataset, val_idx)
            test_subset = Subset(test_dataset, test_idx)
            
            # Build loaders
            train_loader = DataLoader(
                train_subset, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=NUM_WORKERS, pin_memory=True
            )
            val_loader = DataLoader(
                val_subset, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True
            )
            test_loader = DataLoader(
                test_subset, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True
            )
            
            logger.info(f"Successfully generated DataLoaders. Train size: {len(train_subset)}, Val size: {len(val_subset)}, Test size: {len(test_subset)}")
            return train_loader, val_loader, test_loader
        except Exception as e:
            logger.error(f"Failed to compile PyTorch DataLoaders: {e}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """Calculates image distribution details per category label."""
        targets = self.base_dataset.targets
        unique, counts = np.unique(targets, return_counts=True)
        class_counts = dict(zip([self.classes[i] for i in unique], counts.tolist()))
        
        stats = {
            "total_images": len(targets),
            "num_classes": self.num_classes,
            "classes": self.classes,
            "distribution": class_counts
        }
        
        logger.info(f"Dataset Statistics Summary: Total={stats['total_images']}, Classes={stats['num_classes']}")
        return stats

    def visualize_samples(self, num_samples: int = 5) -> None:
        """Compatibility wrapper for visual plots in notebooks."""
        from data.visualization import visualize_random_samples
        visualize_random_samples(self.root_dir, num_samples=num_samples)

    def plot_class_distribution(self) -> None:
        """Compatibility wrapper for distribution plots in notebooks."""
        from data.visualization import plot_class_distribution
        # Support EuroSAT structure where classes are inside the nested directory
        plot_class_distribution(self.root_dir, title=self.root_dir.name)

    def generate_dataset_statistics(self) -> Dict[str, Any]:
        """Compatibility wrapper for dataset statistics generation in notebooks."""
        return self.get_statistics()

if __name__ == "__main__":
    from configs.paths import EUROSAT_DIR
    try:
        loader = SatelliteDatasetLoader(EUROSAT_DIR, split_type='random')
        stats = loader.get_statistics()
        train_l, val_l, test_l = loader.get_dataloaders()
    except Exception as e:
        logger.error(f"Error executing SatelliteDatasetLoader test run: {e}")
