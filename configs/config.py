import sys
from pathlib import Path

# Hyperparameters and Dataset Configurations
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_WORKERS = 4
RANDOM_SEED = 42

# Training Splits
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Dataset Kaggle Handles
EUROSAT_KAGGLE_PATH = "apollo2506/eurosat-dataset"
UCMERCED_KAGGLE_PATH = "abdulhasibuddin/uc-merced-land-use-dataset"
