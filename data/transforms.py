import torchvision.transforms as T
import torch
from configs.config import IMAGE_SIZE

def get_train_transforms() -> T.Compose:
    """
    Returns image transformations for the training split.
    Applies standard ImageNet normalization and data augmentations.
    """
    return T.Compose([
        T.Resize(IMAGE_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms() -> T.Compose:
    """
    Returns image transformations for the validation and testing splits.
    Does NOT include random augmentations, only resizing and normalization.
    """
    return T.Compose([
        T.Resize(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class UnNormalize(object):
    """Utility class to un-normalize PyTorch image tensors back to visual range [0, 1]."""
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Normalized image tensor.
        Returns:
            Tensor: Un-normalized image tensor.
        """
        t = tensor.clone()
        # Un-normalize
        t = t * self.std + self.mean
        # Clamp to [0, 1]
        return torch.clamp(t, 0.0, 1.0)

def get_inverse_transforms() -> UnNormalize:
    """Returns the un-normalization transform object."""
    return UnNormalize()
