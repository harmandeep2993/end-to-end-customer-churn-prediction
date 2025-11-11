import joblib
import pandas as pd
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

def save_pickle(obj, path: str):
    """Save Python object as pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    logger.info(f"Saved pickle file: {path}")

def load_pickle(path: str):
    """Load pickle object."""
    if not os.path.exists(path):
        logger.error(f"Pickle file not found: {path}")
        raise FileNotFoundError(f"{path} not found")
    logger.info(f"Loaded pickle file: {path}")
    return joblib.load(path)

def save_csv(df: pd.DataFrame, path: str):
    """Save DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved CSV file: {path}")

def load_csv(path: str):
    """Load CSV as DataFrame."""
    if not os.path.exists(path):
        logger.error(f"CSV file not found: {path}")
        raise FileNotFoundError(f"{path} not found")
    logger.info(f"Loaded CSV file: {path}")
    return pd.read_csv(path)