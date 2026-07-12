# EcoVision AI - Testing Guide

This document details the test suite and verification scripts created to test the code pipeline, spatial splitting algorithms, and neural network input/output shapes.

---

## 🧪 Unit Tests

We use Python's built-in `unittest` module to write fast, hermetic checks that ensure code changes do not break model interfaces.

### Test Coverage
* **`test_spatial_split_boundaries`**: Asserts that indices returned by the simulated spatial block split are mathematically disjoint (no train-val-test overlap) and sum to the total sample size.
* **`test_baseline_cnn_forward`**: Passes dummy image tensors through the 3-layer CNN and verifies output dimensions match predicted category shapes.
* **`test_resnet_transfer_forward`**: Passes random tensors through the ResNet18 fine-tuned transfer model and verifies shape output correctness.

---

## 🏃 Run the Unit Tests

Ensure your virtual environment is activated and dependencies are synced:

```bash
# Activation
# Windows (PowerShell):
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process; .venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# Execute all tests
python -m unittest tests/test_pipeline.py
```

### Expected Output
```text
...
----------------------------------------------------------------------
Ran 3 tests in 0.208s

OK
```

---

## 🔍 Validation Verification Pipelines

For broader integration testing, you can execute the dry-run pipeline that executes 1-epoch training and evaluation cycles for both models:

```bash
uv run python main.py
```
This script acts as a diagnostic test checking:
1. Dataset paths and downloader loops.
2. OpenCV and Seaborn graph rendering and file output writes.
3. Gradient calculations and optimization steps using Adam/Cross-Entropy.
4. GradCAM gradient hooks and backpropagation targets.
5. High-dimensional feature vector projections using UMAP and t-SNE.
