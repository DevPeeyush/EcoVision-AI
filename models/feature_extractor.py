import torch
import torch.nn as nn
from pathlib import Path
import sys

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from models.resnet18_transfer import ResNet18Transfer
from configs.settings import configure_logger

logger = configure_logger("FeatureExtractor")

class FeatureExtractor(nn.Module):
    def __init__(self, model_path: Path, num_classes: int, device: str = "cpu"):
        """
        Feature Extractor wrapper that strips the classification head of a 
        fine-tuned ResNet18 model to output 512-dimensional image embeddings.
        """
        super(FeatureExtractor, self).__init__()
        self.device = device
        
        # Load full transfer model architecture
        base_model = ResNet18Transfer(num_classes=num_classes, pretrained=False)
        
        # Load weights if available
        if model_path.exists():
            try:
                base_model.load_state_dict(torch.load(model_path, map_location=device))
                logger.info(f"Loaded fine-tuned model checkpoint for feature extraction: {model_path}")
            except Exception as e:
                logger.error(f"Error loading model weights from {model_path}: {e}")
                logger.info("Using random initialized weights for feature extraction.")
        else:
            logger.warning(f"No checkpoint found at {model_path}. Extracting features with random weights.")

        # Strip classification head (final fc layer)
        self.backbone = nn.Sequential(*list(base_model.backbone.children())[:-1])
        self.backbone = self.backbone.to(device)
        self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts raw 512-dimensional embedding features.
        
        Args:
            x (Tensor): Input image tensor of shape (B, 3, 224, 224)
            
        Returns:
            Tensor: Normalized 512-dimensional embedding tensor of shape (B, 512)
        """
        with torch.no_grad():
            features = self.backbone(x)
            features = features.view(features.size(0), -1) # Flatten (B, 512, 1, 1) to (B, 512)
        return features
