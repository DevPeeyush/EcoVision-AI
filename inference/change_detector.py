import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import sys

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from models.feature_extractor import FeatureExtractor
from configs.settings import configure_logger

logger = configure_logger("TemporalChangeDetector")

class TemporalChangeDetector:
    def __init__(self, model_path: Path, num_classes: int, device: str = "cpu"):
        """
        Temporal Change Detection Engine utilizing Cosine Similarity between 
        extracted ResNet18 embeddings.
        """
        self.device = device
        self.feature_extractor = FeatureExtractor(model_path=model_path, num_classes=num_classes, device=device)
        self.transform = self.feature_extractor.transform if hasattr(self.feature_extractor, 'transform') else None
        
        # If transform is not mapped, replicate standard ImageNet normalization
        if not self.transform:
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def get_embedding(self, image: Image.Image) -> torch.Tensor:
        """Extracts the 512-dimensional embedding tensor for a single image."""
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return self.feature_extractor(img_tensor)

    def calculate_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """Computes Cosine Similarity between T1 (Before) and T2 (After) image embeddings."""
        emb1 = self.get_embedding(img1)
        emb2 = self.get_embedding(img2)
        
        cos_sim = F.cosine_similarity(emb1, emb2, dim=1)
        sim_val = cos_sim.item()
        logger.info(f"Computed Cosine Similarity: {sim_val:.6f}")
        return sim_val

    def generate_difference_map(self, img1: Image.Image, img2: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates absolute pixel-wise differences and JET heatmap visual representation.
        
        Args:
            img1 (Image.Image): Before image.
            img2 (Image.Image): After image.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Absolute difference image and JET heatmap image.
        """
        # Resize images to match identical pixel coordinates
        i1 = cv2.resize(np.array(img1), (224, 224))
        i2 = cv2.resize(np.array(img2), (224, 224))
        
        # Absolute difference
        diff = cv2.absdiff(i1, i2)
        
        # Convert to grayscale
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        
        # Apply JET colormap mapping
        heatmap = cv2.applyColorMap(gray_diff, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return diff, heatmap
