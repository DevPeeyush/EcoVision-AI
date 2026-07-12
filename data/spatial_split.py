import numpy as np
from typing import Tuple, List

def generate_spatial_split(num_samples: int, train_ratio: float, val_ratio: float) -> Tuple[List[int], List[int], List[int]]:
    """
    Generates indices for a simulated spatial block split.
    In spatial datasets (like satellite imagery), adjacent tiles exhibit spatial autocorrelation.
    A simple random split causes 'spatial leakage' because neighboring pixels leak from training to testing,
    artificially inflating evaluation metrics.
    
    To simulate geographic partition splits, we divide the sequential sample indices into contiguous blocks.
    
    Args:
        num_samples (int): Total count of samples in the dataset.
        train_ratio (float): Ratio of data allocated for training.
        val_ratio (float): Ratio of data allocated for validation.
        
    Returns:
        Tuple[List[int], List[int], List[int]]: Tuple containing train, validation, and test indices.
    """
    indices = list(range(num_samples))
    
    # Calculate cutoff indices
    train_cutoff = int(np.floor(train_ratio * num_samples))
    val_cutoff = int(np.floor((train_ratio + val_ratio) * num_samples))
    
    train_idx = indices[:train_cutoff]
    val_idx = indices[train_cutoff:val_cutoff]
    test_idx = indices[val_cutoff:]
    
    return train_idx, val_idx, test_idx
