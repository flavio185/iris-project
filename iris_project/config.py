from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
BRONZE_DATA_DIR = DATA_DIR / "bronze"
SILVER_DATA_DIR = DATA_DIR / "silver"
GOLD_DATA_DIR = DATA_DIR / "gold"
VALIDATION_REPORTS_DIR = DATA_DIR / "validation_reports"

MODELS_DIR = PROJ_ROOT / "models"
LOGS_DIR = PROJ_ROOT / "logs"
REPORTS_DIR = PROJ_ROOT / "reports"

S3_BUCKET = "datamasters2025"
S3_BRONZE_PREFIX = "iris/bronze"
S3_SILVER_PREFIX = "iris/silver"
S3_GOLD_PREFIX = "iris/gold"

IRIS_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
IRIS_TARGET = "species"
IRIS_CLASSES = ["setosa", "versicolor", "virginica"]
