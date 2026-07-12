import os
import shutil
import kagglehub
from pathlib import Path
import sys

# Support relative path resolution
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from configs.paths import EUROSAT_DOWNLOAD_DIR, EUROSAT_DIR, UCMERCED_DIR
from configs.config import EUROSAT_KAGGLE_PATH, UCMERCED_KAGGLE_PATH
from configs.settings import configure_logger

logger = configure_logger("DatasetDownloader")

class DatasetDownloader:
    def __init__(self):
        self.valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')

    def move_files(self, src: Path, dest: Path) -> None:
        """Recursively copies files from kagglehub temp downloads directory to our local datasets folder."""
        try:
            os.makedirs(dest, exist_ok=True)
            # Check for single nested root directory
            contents = os.listdir(src)
            if len(contents) == 1 and (src / contents[0]).is_dir():
                src = src / contents[0]
                
            for item in os.listdir(src):
                s = src / item
                d = dest / item
                if s.is_dir():
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            logger.info(f"Successfully moved dataset files from cache to target path: {dest}")
        except Exception as e:
            logger.error(f"Failed to move files from {src} to {dest}: {e}")
            raise

    def check_integrity(self, path: Path) -> bool:
        """Verifies integrity of download by ensuring valid images are present."""
        if not path.exists():
            logger.error(f"Path does not exist for integrity check: {path}")
            return False
            
        img_count = 0
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(self.valid_extensions):
                    img_count += 1
                    
        if img_count > 0:
            logger.info(f"Integrity check PASSED for {path}. Found {img_count} images.")
            return True
        else:
            logger.error(f"Integrity check FAILED for {path}. Zero valid images found.")
            return False

    def download_eurosat(self) -> Path:
        """Downloads EuroSAT if not present."""
        if EUROSAT_DIR.exists() and len(os.listdir(EUROSAT_DIR)) > 0:
            logger.info(f"EuroSAT already configured at {EUROSAT_DIR}. Skipping download.")
            return EUROSAT_DIR

        logger.info("Initializing EuroSAT download via KaggleHub...")
        try:
            cache_path = kagglehub.dataset_download(EUROSAT_KAGGLE_PATH)
            logger.info(f"EuroSAT downloaded to KaggleHub cache at {cache_path}")
            self.move_files(Path(cache_path), EUROSAT_DOWNLOAD_DIR)
            
            # Resolve potential nested folder structure
            if not EUROSAT_DIR.exists():
                nested_dir = EUROSAT_DOWNLOAD_DIR / "EuroSAT"
                if nested_dir.exists():
                    logger.info("EuroSAT folder is nested correctly.")
            return EUROSAT_DIR
        except Exception as e:
            logger.error(f"Error downloading EuroSAT: {e}")
            raise

    def download_ucmerced(self) -> Path:
        """Downloads UC Merced dataset if not present."""
        if UCMERCED_DIR.exists() and len(os.listdir(UCMERCED_DIR)) > 0:
            logger.info(f"UC Merced already configured at {UCMERCED_DIR}. Skipping download.")
            return UCMERCED_DIR

        logger.info("Initializing UC Merced download via KaggleHub...")
        try:
            cache_path = kagglehub.dataset_download(UCMERCED_KAGGLE_PATH)
            logger.info(f"UC Merced downloaded to KaggleHub cache at {cache_path}")
            self.move_files(Path(cache_path), UCMERCED_DIR)
            return UCMERCED_DIR
        except Exception as e:
            logger.error(f"Error downloading UC Merced: {e}")
            raise

    def run(self) -> None:
        """Runs the complete ingestion and integrity verification pipeline."""
        logger.info("Starting Dataset Downloader Execution...")
        self.download_eurosat()
        self.check_integrity(EUROSAT_DIR)
        
        self.download_ucmerced()
        self.check_integrity(UCMERCED_DIR)
        logger.info("Dataset Downloader Ingestion Complete.")

def download_eurosat() -> Path:
    """Compatibility wrapper for notebook imports."""
    return DatasetDownloader().download_eurosat()

def download_ucmerced() -> Path:
    """Compatibility wrapper for notebook imports."""
    return DatasetDownloader().download_ucmerced()

if __name__ == "__main__":
    downloader = DatasetDownloader()
    downloader.run()
