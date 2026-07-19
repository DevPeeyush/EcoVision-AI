import sys
from pathlib import Path
import torch
import logging

# Ensure root directory is in system path
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from configs.paths import EUROSAT_DIR, OUTPUTS_DIR
from configs.settings import DEVICE
from data.dataset_loader import SatelliteDatasetLoader
from models.resnet_transfer import ResNet18Transfer
from training.trainer import Trainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_resnet():
    logging.info("--- Starting ResNet18 Transfer Learning Training Pipeline ---")
    
    # 1. Load Data
    logging.info("Loading EuroSAT dataset...")
    loader = SatelliteDatasetLoader(EUROSAT_DIR, split_type='random')
    train_loader, val_loader, test_loader = loader.get_dataloaders()
    classes = loader.classes
    num_classes = len(classes)
    logging.info(f"Loaded {num_classes} classes: {classes}")
    
    # 2. Initialize Model
    logging.info("Initializing ResNet18 model...")
    model = ResNet18Transfer(num_classes=num_classes)
    save_path = str(OUTPUTS_DIR / 'resnet18_best.pth')
    
    # 3. Phase 1: Freeze backbone & train classifier head
    logging.info("--- Phase 1: Training Classifier Head Only (Backbone Frozen) ---")
    model.freeze_backbone()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model.parameters(), lr=0.001)
    
    trainer = Trainer(model, criterion, optimizer1, device=DEVICE)
    logging.info("Running Phase 1 training for 3 epochs...")
    history_phase1 = trainer.train(train_loader, val_loader, epochs=3, save_path=save_path)
    
    # 4. Phase 2: Unfreeze last blocks & fine-tune
    logging.info("--- Phase 2: Fine-Tuning Model (Last 2 Blocks Unfrozen) ---")
    model.unfreeze_last_two_blocks()
    optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    trainer.optimizer = optimizer2
    
    logging.info("Running Phase 2 training for 5 epochs...")
    history_phase2 = trainer.train(train_loader, val_loader, epochs=5, save_path=save_path)
    
    logging.info("--- Training Pipeline Completed Successfully! ---")
    logging.info(f"Best ResNet18 model weights saved to: {save_path}")

if __name__ == "__main__":
    train_resnet()
