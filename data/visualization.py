import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
from PIL import Image
from typing import Optional, List

from configs.settings import configure_logger

logger = configure_logger("DataVisualization")

def get_class_distribution(dataset_dir: Path) -> dict:
    """Computes distribution count per class folder."""
    if not dataset_dir.exists():
        logger.error(f"Dataset path does not exist: {dataset_dir}")
        return {}
        
    dist = {}
    for cls in os.listdir(dataset_dir):
        cls_dir = dataset_dir / cls
        if cls_dir.is_dir():
            dist[cls] = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))])
    return dist

def plot_class_distribution(dataset_dir: Path, title: str, save_path: Optional[Path] = None) -> None:
    """Plots and logs dataset class distribution bar graphs."""
    dist = get_class_distribution(dataset_dir)
    if not dist:
        logger.warning(f"No class folders found in {dataset_dir}")
        return

    classes = list(dist.keys())
    counts = list(dist.values())
    
    # Log counts
    logger.info(f"Class distribution for {title}:")
    for cls, cnt in dist.items():
        logger.info(f"  - {cls}: {cnt}")

    plt.figure(figsize=(10, 5))
    sns.barplot(x=counts, y=classes, palette="viridis", hue=classes, legend=False)
    plt.title(f"Class Distribution: {title}")
    plt.xlabel("Image Count")
    plt.ylabel("Category")
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved class distribution plot to: {save_path}")
        plt.close()
    else:
        plt.show()

def visualize_random_samples(dataset_dir: Path, num_samples: int = 5, save_path: Optional[Path] = None) -> None:
    """Selects and plots random images across category directories."""
    if not dataset_dir.exists():
        return
        
    classes = [c for c in os.listdir(dataset_dir) if (dataset_dir / c).is_dir()]
    if not classes:
        return
        
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    if num_samples == 1:
        axes = [axes]
        
    for i in range(num_samples):
        cls = random.choice(classes)
        cls_dir = dataset_dir / cls
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]
        
        if not images:
            continue
            
        img_path = cls_dir / random.choice(images)
        img = Image.open(img_path)
        
        axes[i].imshow(img)
        axes[i].set_title(cls, fontsize=10)
        axes[i].axis('off')
        
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved random samples plot to: {save_path}")
        plt.close()
    else:
        plt.show()
