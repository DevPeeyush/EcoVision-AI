import torch
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("Callbacks")

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Early stopping helper to terminate training when validation loss stops improving.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logger.info(f"Early Stopping Counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return self.early_stop
