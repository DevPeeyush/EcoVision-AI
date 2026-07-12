import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Transfer(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        """
        ResNet18 Model configured for Transfer Learning.
        
        Args:
            num_classes (int): Number of target categories.
            pretrained (bool): If True, initializes weights from ImageNet pre-training.
        """
        super(ResNet18Transfer, self).__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        
        # Replace the fully connected classification head
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
        
    def freeze_backbone(self) -> None:
        """Phase 1: Freezes all layers except the newly initialized classifier head."""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
                
    def unfreeze_last_two_blocks(self) -> None:
        """Phase 2: Unfreezes layer3 and layer4 (last two residual blocks) and fc head for fine-tuning."""
        for name, param in self.backbone.named_parameters():
            if 'layer3' in name or 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
