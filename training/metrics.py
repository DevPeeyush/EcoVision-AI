from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from typing import Dict, Any, List, Tuple
import numpy as np

def calculate_classification_metrics(y_true: List[int], y_pred: List[int], class_names: List[str]) -> Dict[str, Any]:
    """
    Computes standard evaluation metrics: Accuracy, Precision, Recall, Macro F1, and Per-class F1.
    
    Args:
        y_true (List[int]): Ground truth labels.
        y_pred (List[int]): Predicted labels.
        class_names (List[str]): List of class category names.
        
    Returns:
        Dict[str, Any]: Structured dictionary containing computed metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(class_names))), zero_division=0
    )
    
    report = classification_report(
        y_true, y_pred, 
        labels=list(range(len(class_names))), 
        target_names=class_names, 
        zero_division=0
    )
    
    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "per_class_f1": {class_names[i]: float(f) for i, f in enumerate(per_class_f1)},
        "per_class_precision": {class_names[i]: float(p) for i, p in enumerate(per_class_precision)},
        "per_class_recall": {class_names[i]: float(r) for i, r in enumerate(per_class_recall)},
        "classification_report": report
    }
    
    return metrics
