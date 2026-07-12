# Satellite Image Land-Use Classifier & Temporal Change Detection

EcoVision AI is a production-ready, modular computer vision application designed to perform high-accuracy land-use classification and temporal change detection using satellite imagery. It leverages PyTorch transfer learning with a ResNet18 backbone, custom CNN architectures, model interpretability via GradCAM, high-dimensional embedding visualizations via t-SNE/UMAP, and an interactive Streamlit dashboard.

---

## 🚀 Getting Started

This project uses **[UV](https://github.com/astral-sh/uv)** for fast, reproducible Python packaging and dependency management.

### 1. Prerequisites
Ensure you have Python 3.12+ and `uv` installed. If you don't have `uv` installed, install it via:
* **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
* **macOS/Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### 2. Setup Virtual Environment
Initialize your virtual environment and resolve dependencies:
```bash
# Create a virtual environment using Python 3.12
uv venv --python 3.12

# Activate the virtual environment
# Windows (PowerShell):
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process; .venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
```

### 3. Install & Sync Dependencies
Installs all libraries specified in `pyproject.toml` and locks versions using `uv.lock`:
```bash
uv sync
```

---

## 🛠️ Usage

### Running System Ingestion & Workflows
To download datasets, verify integrity, run splits, or execute custom training pipelines:
```bash
uv run python main.py
```

### Launching the Dashboard
Start the modern, nature-inspired Streamlit user interface:
```bash
uv run streamlit run app/streamlit_app.py
```

---

## 🧪 Testing

To run the pipeline's automated unit tests (verifying splits, input/output tensors, and architectures):
```bash
uv run python -m unittest tests/test_pipeline.py
```
For detailed coverage specifications, see the **[Testing Guide](file:///D:/Main%20Project/report/TESTING_GUIDE.md)**.

---

## 🌐 Deployment

For step-by-step instructions on containerizing with Docker or deploying to AWS, GCP, or Streamlit Cloud, see the **[Deployment Guide](file:///D:/Main%20Project/report/DEPLOYMENT_GUIDE.md)**.

---

## 📁 Folder Structure

* `app/`: Contains Streamlit pages, styles, utils, and assets.
* `configs/`: Centralized settings, path definitions, and constant variables.
* `data/`: Ingestion, spatial splitting, dataset downloaders, loaders, and preprocessing.
* `models/`: Neural network definitions (Baseline CNN, ResNet18, GradCAM, Embedding handlers).
* `training/`: Training and validation scripts, metric calculations, and callbacks.
* `inference/`: Target predictor and temporal change detection engines.
* `notebooks/`: Step-by-step EDA, prototyping, and evaluation notebooks.
* `outputs/`: Holds training checkpoints, logs, confusion matrices, and change maps.
* `report/`: Central folder containing reports, guides, and documentations.

---

## 📋 GitHub Submission Checklist

Before submitting the project, verify that all eight core requirements are fulfilled:
1. **Clean Repository:** Code contains organized folders, `README.md`, and clean dependency specifications.
2. **Data Ingestion:** Automates downloads using KaggleHub and mitigates spatial leakage using contiguous index splits.
3. **Baseline Model:** A 3-layer scratch CNN model is included with training pipelines and accuracy checks.
4. **Fine-Tuned Model:** A fine-tuned ResNet18 model is saved at `outputs/checkpoints/resnet18_best.pth`.
5. **Change Detection:** Stripped backbone model extracts 512-D embeddings and computes similarity maps.
6. **Geo-Dashboard:** Interactive Streamlit app runs locally on port 8501.
7. **Unit Tests:** Passes automated verification (`tests/test_pipeline.py`).
8. **Final Report & Video:** Formal PDF report compiled from `report/REPORT.pdf`
