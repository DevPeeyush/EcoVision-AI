import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import sys
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from training.metrics import calculate_classification_metrics
from configs.paths import CONF_MATRIX_DIR
from configs.settings import configure_logger

logger = configure_logger("Evaluator")

class Evaluator:
    def __init__(self, model: nn.Module, criterion: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device

    def evaluate(self, dataloader: DataLoader, class_names: Optional[List[str]] = None, limit_batches: Optional[int] = None) -> Tuple[float, Dict[str, Any], List[int], List[int]]:
        """
        Evaluates the model on the target data loader.
        
        Args:
            dataloader (DataLoader): PyTorch loader of testing/validation datasets.
            class_names (List[str], optional): List of categories.
            limit_batches (int, optional): Max batches to evaluate (for dry-runs).
            
        Returns:
            Tuple[float, Dict[str, Any], List[int], List[int]]: Loss, metrics dictionary, targets, and predictions.
        """
        # Auto-resolve class names if not provided
        if class_names is None:
            if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'dataset') and hasattr(dataloader.dataset.dataset, 'classes'):
                class_names = dataloader.dataset.dataset.classes
            elif hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'classes'):
                class_names = dataloader.dataset.classes
            else:
                try:
                    num_classes = self.model.backbone.fc.out_features
                except AttributeError:
                    num_classes = 10
                class_names = [f"Class_{i}" for i in range(num_classes)]

        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                if limit_batches is not None and i >= limit_batches:
                    break
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        val_loss = running_loss / len(dataloader.dataset)
        metrics = calculate_classification_metrics(all_labels, all_preds, class_names)
        
        logger.info(f"Evaluation Complete. Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")
        return val_loss, metrics, all_labels, all_preds

    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], class_names: List[str], filename: str) -> Path:
        """Plots and saves confusion matrix heatmap to local storage."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        
        save_path = CONF_MATRIX_DIR / filename
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        logger.info(f"Saved confusion matrix plot to: {save_path}")
        return save_path
