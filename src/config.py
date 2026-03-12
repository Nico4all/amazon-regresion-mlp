from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
ARDUINO_DIR = BASE_DIR / "arduino"

DATASET_FILENAME = "Amazon Sale Report.xlsx"
DATASET_PATH = RAW_DATA_DIR / DATASET_FILENAME

TARGET_COLUMN_CANDIDATES = ["Amount", "Qty"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2