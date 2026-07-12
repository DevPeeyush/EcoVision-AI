from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from configs.config import IMAGE_SIZE
from data.transforms import get_val_transforms

def load_and_preprocess_image(image_path: str) -> torch.Tensor:
    """Loads an image from disk, converts to RGB, and prepares it as a preprocessed PyTorch tensor."""
    img = Image.open(image_path).convert('RGB')
    transform = get_val_transforms()
    return transform(img).unsqueeze(0)

def convert_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Converts a normalized PyTorch image tensor back to a standard HWC numpy image [0, 255]."""
    # Remove batch dim if present
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    
    # Detach and CPU
    t = tensor.detach().cpu().clone()
    
    # Un-normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    t = t * std + mean
    t = torch.clamp(t, 0.0, 1.0)
    
    # CHW to HWC
    img_np = t.numpy().transpose(1, 2, 0)
    return (img_np * 255.0).astype(np.uint8)
