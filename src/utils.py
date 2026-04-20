import time
import logging
from pathlib import Path
from functools import wraps

import numpy as np
import pandas as pd



def get_logger(name: str = "livestock_anom") -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger(name)



def timed(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        print(f"[timer] {fn.__name__} completed in {time.perf_counter()-t0:.2f}s")
        return result
    return wrapper



def load_or_generate(csv_path: Path) -> pd.DataFrame:
    """Load CSV if it exists, otherwise regenerate dataset."""
    if csv_path.exists():
        print(f"[utils] Loading existing dataset ← {csv_path}")
        return pd.read_csv(csv_path)
    from src.data_generator import generate_dataset
    return generate_dataset()


def validate_dataframe(df: pd.DataFrame, required_cols: list) -> None:
    """Raise ValueError if required columns are missing."""
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Upload missing required columns: {missing}")


def build_results_df(df: pd.DataFrame,
                     scores: np.ndarray,
                     categories: np.ndarray,
                     top_features: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Merge original dataframe with model outputs for display.
    """
    out = df.copy().reset_index(drop=True)
    out["anomaly_score"] = np.round(scores, 4)
    out["risk_category"] = categories

    if top_features is not None and len(top_features):
        feat_cols = [c for c in top_features.columns if c.startswith("feature_")]
        top_features = top_features.set_index("record_idx")
        for col in feat_cols:
            out[col] = top_features[col].reindex(out.index).fillna("—")

    return out.sort_values("anomaly_score", ascending=False).reset_index(drop=True)


REQUIRED_RAW_COLS = [
    "farm_id", "animal_type", "temperature", "humidity",
    "feed_intake", "water_intake", "movement_level", "milk_output",
    "heart_rate", "body_temperature", "vaccination_gap_days",
    "waste_output", "cough_frequency", "nearby_outbreak_score",
    "region", "timestamp",
]
