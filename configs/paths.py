import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "datasets"

# Specific dataset directories
EUROSAT_DOWNLOAD_DIR = DATA_DIR / "eurosat"
EUROSAT_DIR = EUROSAT_DOWNLOAD_DIR / "EuroSAT"
UCMERCED_DIR = DATA_DIR / "ucmerced"

# Outputs
OUTPUTS_DIR = BASE_DIR / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
CONF_MATRIX_DIR = OUTPUTS_DIR / "confusion_matrix"
HEATMAPS_DIR = OUTPUTS_DIR / "heatmaps"
ROC_DIR = OUTPUTS_DIR / "roc"
GRADCAM_DIR = OUTPUTS_DIR / "gradcam"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
LOGS_DIR = OUTPUTS_DIR / "logs"
REPORTS_DIR = OUTPUTS_DIR / "reports"

MODELS_DIR = BASE_DIR / "models"
REPORTS_ROOT_DIR = BASE_DIR / "report"

# Ensure essential directories exist
DIRS = [
    DATA_DIR, OUTPUTS_DIR, CHECKPOINTS_DIR, CONF_MATRIX_DIR,
    HEATMAPS_DIR, ROC_DIR, GRADCAM_DIR, EMBEDDINGS_DIR,
    PREDICTIONS_DIR, LOGS_DIR, REPORTS_DIR, MODELS_DIR, REPORTS_ROOT_DIR
]

for _dir in DIRS:
    _dir.mkdir(parents=True, exist_ok=True)
