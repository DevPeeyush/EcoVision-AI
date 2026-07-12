import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from typing import Tuple
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from data.transforms import get_val_transforms
from configs.settings import configure_logger

logger = configure_logger("GradCAM")

class GradCAMExplainer:
    def __init__(self, model: nn.Module, target_layer: nn.Module, device: str = "cpu"):
        """
        Wrapper to generate GradCAM heatmaps using pytorch-grad-cam library.
        
        Args:
            model (nn.Module): The classification model.
            target_layer (nn.Module): The target convolutional layer (e.g. model.backbone.layer4[-1]).
            device (str): target execution device.
        """
        self.model = model.to(device)
        self.model.eval()
        self.target_layers = [target_layer]
        self.device = device
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)
        self.transform = get_val_transforms()

    def generate_heatmap(self, image: Image.Image, target_category: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Generates GradCAM activation heatmap overlays.
        
        Args:
            image (Image.Image): Input PIL image.
            target_category (int): Target class index.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, int]: Raw heatmap, superimposed overlay, and predicted class index.
        """
        # Preprocess image
        img_resized = image.resize((224, 224))
        rgb_img = np.float32(img_resized) / 255.0
        
        input_tensor = self.transform(img_resized).unsqueeze(0).to(self.device)
        
        # Run forward pass
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_class = predicted_idx.item()
            
        # Target setup (if target_category is None, default to model prediction)
        if target_category is None:
            target_category = predicted_class
            
        targets = [ClassifierOutputTarget(target_category)]
        
        # Generate raw activation maps
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        # Superimpose color mapping
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        logger.info(f"GradCAM Heatmap generated successfully for class target index: {target_category}")
        return grayscale_cam, cam_image, predicted_class
