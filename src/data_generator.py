"""
data_generator.py
Generates a realistic synthetic livestock health dataset (20,000 rows).
Anomalous patterns are embedded but NOT labeled — the unsupervised models
must discover them. Pattern types: disease_outbreak, feed_contamination,
dehydration, stress, environmental_danger.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

SEED = 42
np.random.seed(SEED)

# ── Output path ──────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = DATA_DIR / "livestock_data.csv"


# ── Normal distribution parameters per animal type ───────────────────────────
NORMAL_PARAMS = {
    "cow": dict(
        temperature=(22, 3), humidity=(60, 8),
        feed_intake=(18, 2), water_intake=(45, 5),
        movement_level=(55, 10), milk_output=(18, 4),
        heart_rate=(65, 8), body_temperature=(38.5, 0.4),
        waste_output=(28, 4), cough_frequency=(0.5, 0.5),
    ),
    "buffalo": dict(
        temperature=(23, 3), humidity=(62, 8),
        feed_intake=(20, 2.5), water_intake=(50, 6),
        movement_level=(50, 10), milk_output=(10, 3),
        heart_rate=(60, 7), body_temperature=(38.3, 0.4),
        waste_output=(32, 5), cough_frequency=(0.4, 0.4),
    ),
    "goat": dict(
        temperature=(22, 3), humidity=(55, 7),
        feed_intake=(3, 0.5), water_intake=(8, 1.5),
        movement_level=(70, 12), milk_output=(2, 0.8),
        heart_rate=(80, 10), body_temperature=(39.2, 0.5),
        waste_output=(5, 1), cough_frequency=(0.3, 0.4),
    ),
    "poultry": dict(
        temperature=(24, 3), humidity=(65, 8),
        feed_intake=(0.12, 0.02), water_intake=(0.3, 0.05),
        movement_level=(65, 12), milk_output=(0, 0),
        heart_rate=(275, 25), body_temperature=(41.0, 0.5),
        waste_output=(0.18, 0.04), cough_frequency=(0.2, 0.3),
    ),
}

ANIMAL_TYPES = list(NORMAL_PARAMS.keys())
REGIONS = ["Punjab", "Haryana", "UP", "Rajasthan", "Gujarat", "MP"]
FARM_IDS = [f"FARM-{str(i).zfill(3)}" for i in range(1, 71)]


def _sample_normal(animal: str, n: int) -> dict:
    """Sample feature values from normal (healthy) distribution."""
    p = NORMAL_PARAMS[animal]
    row = {}
    for feat, (mu, sigma) in p.items():
        row[feat] = np.random.normal(mu, sigma, n)
    return row


def _inject_disease_outbreak(df: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    """
    Disease outbreak: elevated body temp, high heart rate, reduced feed/water,
    increased cough, high nearby_outbreak_score.
    """
    df.loc[idx, "body_temperature"] += np.random.uniform(1.5, 3.5, len(idx))
    df.loc[idx, "heart_rate"] *= np.random.uniform(1.15, 1.35, len(idx))
    df.loc[idx, "feed_intake"] *= np.random.uniform(0.4, 0.65, len(idx))
    df.loc[idx, "water_intake"] *= np.random.uniform(0.5, 0.7, len(idx))
    df.loc[idx, "cough_frequency"] += np.random.uniform(3, 8, len(idx))
    df.loc[idx, "movement_level"] *= np.random.uniform(0.5, 0.75, len(idx))
    df.loc[idx, "nearby_outbreak_score"] = np.random.uniform(0.75, 1.0, len(idx))
    return df


def _inject_feed_contamination(df: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    """
    Feed contamination: sharply reduced feed intake, abnormal waste output,
    mild fever, reduced milk.
    """
    df.loc[idx, "feed_intake"] *= np.random.uniform(0.2, 0.5, len(idx))
    df.loc[idx, "waste_output"] *= np.random.uniform(1.5, 2.5, len(idx))
    df.loc[idx, "body_temperature"] += np.random.uniform(0.8, 2.0, len(idx))
    df.loc[idx, "milk_output"] *= np.random.uniform(0.4, 0.7, len(idx))
    df.loc[idx, "movement_level"] *= np.random.uniform(0.6, 0.8, len(idx))
    return df


def _inject_dehydration(df: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    """
    Dehydration: very low water intake, elevated body temp, reduced milk,
    low movement.
    """
    df.loc[idx, "water_intake"] *= np.random.uniform(0.1, 0.3, len(idx))
    df.loc[idx, "body_temperature"] += np.random.uniform(0.5, 1.5, len(idx))
    df.loc[idx, "milk_output"] *= np.random.uniform(0.5, 0.75, len(idx))
    df.loc[idx, "movement_level"] *= np.random.uniform(0.4, 0.65, len(idx))
    df.loc[idx, "heart_rate"] *= np.random.uniform(1.1, 1.25, len(idx))
    return df


def _inject_stress(df: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    """
    Stress: high movement variability, elevated heart rate, reduced milk,
    slightly elevated body temp.
    """
    df.loc[idx, "movement_level"] *= np.random.uniform(1.3, 1.8, len(idx))
    df.loc[idx, "heart_rate"] *= np.random.uniform(1.2, 1.4, len(idx))
    df.loc[idx, "milk_output"] *= np.random.uniform(0.6, 0.85, len(idx))
    df.loc[idx, "body_temperature"] += np.random.uniform(0.3, 1.0, len(idx))
    df.loc[idx, "feed_intake"] *= np.random.uniform(0.7, 0.9, len(idx))
    return df


def _inject_environmental_danger(df: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    """
    Environmental danger (heat stress / cold snap): extreme ambient temperature,
    high humidity, low movement, elevated heart rate.
    """
    # either heat or cold spike
    direction = np.random.choice([1, -1], len(idx))
    df.loc[idx, "temperature"] += direction * np.random.uniform(10, 18, len(idx))
    df.loc[idx, "humidity"] += np.random.uniform(15, 25, len(idx))
    df.loc[idx, "movement_level"] *= np.random.uniform(0.4, 0.7, len(idx))
    df.loc[idx, "heart_rate"] *= np.random.uniform(1.1, 1.3, len(idx))
    df.loc[idx, "feed_intake"] *= np.random.uniform(0.6, 0.85, len(idx))
    return df


def generate_dataset(n: int = 20_000) -> pd.DataFrame:
    """
    Build the full synthetic dataset with normal + anomalous records.

    Anomaly injection strategy
    --------------------------
    ~5% of each cluster will be injected with one of the five anomaly types.
    Injection is done AFTER building the base dataframe so patterns are
    embedded without any explicit label column.
    """
    print(f"[data_generator] Generating {n:,} records …")

    # ── 1. Base records ───────────────────────────────────────────────────────
    records = []
    base_ts = datetime(2023, 1, 1)

    for i in range(n):
        animal = np.random.choice(ANIMAL_TYPES, p=[0.40, 0.25, 0.20, 0.15])
        p = NORMAL_PARAMS[animal]
        row = {"farm_id": np.random.choice(FARM_IDS),
               "animal_type": animal,
               "region": np.random.choice(REGIONS),
               "timestamp": base_ts + timedelta(hours=int(np.random.uniform(0, 8760)))}
        for feat, (mu, sigma) in p.items():
            row[feat] = float(np.random.normal(mu, sigma))
        # vaccination gap: mostly 0-90 days, heavier tail for risk
        row["vaccination_gap_days"] = float(np.random.exponential(scale=30))
        row["nearby_outbreak_score"] = float(np.random.beta(2, 8))
        records.append(row)

    df = pd.DataFrame(records)

    # ── 2. Anomaly injection (no labels) ──────────────────────────────────────
    anomaly_frac = 0.05  # 5% total
    anomaly_n = int(n * anomaly_frac)
    anom_idx_all = np.random.choice(df.index, size=anomaly_n, replace=False)

    # Split across 5 anomaly types roughly equally
    splits = np.array_split(anom_idx_all, 5)
    injectors = [
        _inject_disease_outbreak,
        _inject_feed_contamination,
        _inject_dehydration,
        _inject_stress,
        _inject_environmental_danger,
    ]
    for inj, idx in zip(injectors, splits):
        df = inj(df, idx)

    # ── 3. Introduce realistic nulls (~2%) ────────────────────────────────────
    null_features = ["milk_output", "movement_level", "waste_output",
                     "cough_frequency", "water_intake"]
    for feat in null_features:
        null_mask = np.random.rand(n) < 0.02
        df.loc[null_mask, feat] = np.nan

    # ── 4. Clip to physically plausible ranges ─────────────────────────────────
    df["humidity"] = df["humidity"].clip(10, 100)
    df["temperature"] = df["temperature"].clip(-5, 50)
    df["body_temperature"] = df["body_temperature"].clip(35, 45)
    df["heart_rate"] = df["heart_rate"].clip(30, 400)
    df["movement_level"] = df["movement_level"].clip(0, 100)
    df["feed_intake"] = df["feed_intake"].clip(0, None)
    df["water_intake"] = df["water_intake"].clip(0, None)
    df["cough_frequency"] = df["cough_frequency"].clip(0, None)
    df["vaccination_gap_days"] = df["vaccination_gap_days"].clip(0, 365)
    df["nearby_outbreak_score"] = df["nearby_outbreak_score"].clip(0, 1)

    df = df.reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[data_generator] Saved {len(df):,} rows → {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    generate_dataset()
