import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Any, Optional

import sys
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from training.evaluator import Evaluator
from configs.settings import configure_logger

logger = configure_logger("Trainer")

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu"
    ):
        """
        Trainer class for satellite classification neural networks.
        
        Args:
            model (nn.Module): The neural network model.
            criterion (nn.Module): Loss function module.
            optimizer (Optimizer): Optimizer module.
            device (str): target device (cpu/cuda).
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.evaluator = Evaluator(self.model, self.criterion, self.device)
        
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_macro_f1": []
        }

    def train_epoch(self, dataloader: DataLoader, limit_batches: Optional[int] = None) -> tuple[float, float]:
        """Executes one training epoch over the dataset."""
        self.model.train()
        running_loss = 0.0
        corrects = 0
        total = 0

        for i, (inputs, labels) in enumerate(tqdm(dataloader, desc="Training Batch", leave=False)):
            if limit_batches is not None and i >= limit_batches:
                break
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data).item()
            total += inputs.size(0)
            
        epoch_loss = running_loss / total
        epoch_acc = corrects / total
        return epoch_loss, epoch_acc

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        limit_batches: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Executes complete multi-epoch training and validation loops.
        Automatically checkpoints the best state dict to local storage.
        """
        # Auto-resolve class names if not provided (for backward compatibility in notebooks)
        if class_names is None:
            if hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'dataset') and hasattr(train_loader.dataset.dataset, 'classes'):
                class_names = train_loader.dataset.dataset.classes
            elif hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'classes'):
                class_names = train_loader.dataset.classes
            else:
                # Fallback to generic names based on model output features
                try:
                    num_classes = self.model.backbone.fc.out_features
                except AttributeError:
                    num_classes = 10
                class_names = [f"Class_{i}" for i in range(num_classes)]
        best_val_acc = 0.0
        
        logger.info(f"Starting model training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Run train epoch
            train_loss, train_acc = self.train_epoch(train_loader, limit_batches=limit_batches)
            
            # Run validation evaluation
            val_loss, val_metrics, _, _ = self.evaluator.evaluate(val_loader, class_names, limit_batches=limit_batches)
            val_acc = val_metrics["accuracy"]
            val_macro_f1 = val_metrics["macro_f1"]
            
            # Store history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_macro_f1"].append(val_macro_f1)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} Summary: "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_macro_f1:.4f}"
            )
            
            # Checkpoint saving if validation accuracy improves
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if save_path:
                    # Ensure directory path exists
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), save_path)
                    logger.info(f"-> Saved new best checkpoint to: {save_path} (Acc: {val_acc:.4f})")
                    
        logger.info("Training complete.")
        return self.history

    def evaluate(self, dataloader: DataLoader) -> tuple[float, float, float, float, float, list[int], list[int]]:
        """Compatibility wrapper for notebook evaluation."""
        val_loss, metrics, all_labels, all_preds = self.evaluator.evaluate(dataloader, class_names=None)
        return (
            val_loss,
            metrics["accuracy"],
            metrics["macro_precision"],
            metrics["macro_recall"],
            metrics["macro_f1"],
            all_labels,
            all_preds
        )
