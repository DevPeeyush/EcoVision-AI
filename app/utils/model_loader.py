import streamlit as st
import torch
import sys
from pathlib import Path
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from configs.paths import OUTPUTS_DIR
from models.resnet_transfer import ResNet18Transfer
from inference.change_detector import TemporalChangeDetector

# Pre-defined classes for EuroSAT
EUROSAT_CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
    'River', 'SeaLake'
]

@st.cache_resource
def load_classification_model():
    """Loads the ResNet18 PyTorch model and caches it into Streamlit's memory."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNet18Transfer(num_classes=len(EUROSAT_CLASSES)).to(device)
    model_path = OUTPUTS_DIR / 'resnet18_best.pth'
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        return None, device

@st.cache_resource
def load_change_detector():
    """Loads the Temporal Change Detector embedding extractor."""
    model_path = OUTPUTS_DIR / 'resnet18_best.pth'
    detector = TemporalChangeDetector(model_path=model_path, num_classes=len(EUROSAT_CLASSES))
    return detector

def get_image_transforms():
    """Returns the standard ImageNet normalization transform."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
