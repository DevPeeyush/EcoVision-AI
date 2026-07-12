import unittest
import torch
import numpy as np
import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from configs.constants import EUROSAT_CLASSES
from data.spatial_split import generate_spatial_split
from models.baseline_cnn import BaselineCNN
from models.resnet18_transfer import ResNet18Transfer

class TestEcoVisionPipeline(unittest.TestCase):
    def test_spatial_split_boundaries(self):
        """Verifies indices are disjoint and sum to total sample size."""
        num_samples = 100
        train_ratio = 0.70
        val_ratio = 0.15
        
        train, val, test = generate_spatial_split(num_samples, train_ratio, val_ratio)
        
        # Check overlaps
        self.assertTrue(set(train).isdisjoint(set(val)))
        self.assertTrue(set(train).isdisjoint(set(test)))
        self.assertTrue(set(val).isdisjoint(set(test)))
        
        # Check sum
        self.assertEqual(len(train) + len(val) + len(test), num_samples)

    def test_baseline_cnn_forward(self):
        """Checks output dimensions match category shapes."""
        model = BaselineCNN(num_classes=10)
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        
        self.assertEqual(output.shape, (2, 10))

    def test_resnet_transfer_forward(self):
        """Checks ResNet transfer model forward pass dimensions."""
        model = ResNet18Transfer(num_classes=len(EUROSAT_CLASSES), pretrained=False)
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        
        self.assertEqual(output.shape, (1, 10))

if __name__ == "__main__":
    unittest.main()
