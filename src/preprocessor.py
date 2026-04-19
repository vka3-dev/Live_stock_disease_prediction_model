"""
preprocessor.py
Full preprocessing pipeline:
  - Missing value imputation
  - Categorical encoding
  - Time-based feature extraction
  - Feature engineering (rolling stats, deviation from mean, severity flags)
  - Scaling (StandardScaler)

Returns a preprocessed numpy array + feature name list suitable for
unsupervised model training.
"""

import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class LivestockPreprocessor:
    """
    Stateful preprocessor — call fit_transform() during training,
    transform() at inference time.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: dict = {}
        self.feature_names_: list = []
        self._fitted = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit on training data and return transformed array."""
        processed = self._engineer(df.copy())
        X = processed[self.feature_names_].values.astype(np.float32)
        X = self.scaler.fit_transform(X)
        self._fitted = True
        return X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using already-fitted scaler."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform().")
        processed = self._engineer(df.copy())
        # Align columns — fill missing engineered cols with 0
        for col in self.feature_names_:
            if col not in processed.columns:
                processed[col] = 0.0
        X = processed[self.feature_names_].values.astype(np.float32)
        return self.scaler.transform(X)

    def save(self, path: Path = MODELS_DIR / "preprocessor.joblib"):
        joblib.dump(self, path)
        print(f"[preprocessor] Saved → {path}")

    @staticmethod
    def load(path: Path = MODELS_DIR / "preprocessor.joblib") -> "LivestockPreprocessor":
        obj = joblib.load(path)
        print(f"[preprocessor] Loaded ← {path}")
        return obj

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values:
          - Numeric: median imputation within animal_type group
          - Categorical: mode fill
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if df[col].isna().any():
                # Group-wise median is more realistic than global median
                df[col] = df.groupby("animal_type")[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # Global median fallback if a group had all-NaN
                df[col] = df[col].fillna(df[col].median())
        for col in ["animal_type", "region", "farm_id"]:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode()[0])
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label-encode farm_id; one-hot encode animal_type and region.
        Encoder state is stored so inference uses the same mapping.
        """
        # farm_id → label encode (high cardinality, one-hot would be huge)
        if "farm_id" not in self.label_encoders:
            le = LabelEncoder()
            df["farm_id_enc"] = le.fit_transform(df["farm_id"].astype(str))
            self.label_encoders["farm_id"] = le
        else:
            le = self.label_encoders["farm_id"]
            known = set(le.classes_)
            df["farm_id"] = df["farm_id"].apply(
                lambda x: x if x in known else le.classes_[0]
            )
            df["farm_id_enc"] = le.transform(df["farm_id"].astype(str))

        # One-hot encode animal_type and region
        for col in ["animal_type", "region"]:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df = pd.concat([df, dummies], axis=1)

        return df

    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract hour, day_of_week, month from timestamp.
        Cyclical encoding (sin/cos) avoids ordinal discontinuities.
        """
        if "timestamp" not in df.columns:
            return df
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["month"] = ts.dt.month
        # Cyclical encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        return df

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Additional engineered features:
          1. Deviation from animal-type mean for key vitals
             (captures within-species anomalies better than raw values)
          2. Vaccination gap severity flag
             (>90 days = 1, >180 days = 2, else 0)
          3. Hydration ratio: water_intake / feed_intake
          4. Thermal stress index: body_temp × ambient_temp interaction
        """
        vitals = ["body_temperature", "heart_rate", "feed_intake", "water_intake"]
        for v in vitals:
            if v in df.columns:
                group_mean = df.groupby("animal_type")[v].transform("mean")
                group_std = df.groupby("animal_type")[v].transform("std").replace(0, 1)
                df[f"{v}_dev"] = (df[v] - group_mean) / group_std

        if "vaccination_gap_days" in df.columns:
            df["vacc_severity"] = (
                (df["vaccination_gap_days"] > 90).astype(int)
                + (df["vaccination_gap_days"] > 180).astype(int)
            )

        if "water_intake" in df.columns and "feed_intake" in df.columns:
            safe_feed = df["feed_intake"].replace(0, 1e-6)
            df["hydration_ratio"] = df["water_intake"] / safe_feed

        if "body_temperature" in df.columns and "temperature" in df.columns:
            df["thermal_stress"] = df["body_temperature"] * df["temperature"]

        return df

    def _select_model_features(self, df: pd.DataFrame) -> list:
        """
        Build the final list of numeric columns to pass to models.
        Drops raw categorical and timestamp columns.
        """
        drop_cols = {
            "farm_id", "animal_type", "region", "timestamp",
            "hour", "day_of_week", "month",
        }
        feature_cols = [
            c for c in df.columns
            if c not in drop_cols and df[c].dtype in [np.float32, np.float64, np.int64, np.int32, bool]
        ]
        return feature_cols

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full pipeline and set self.feature_names_ on first call."""
        df = self._impute(df)
        df = self._encode_categoricals(df)
        df = self._extract_time_features(df)
        df = self._feature_engineering(df)
        # Drop any NaN columns that slipped through
        df = df.fillna(0)

        if not self._fitted:
            self.feature_names_ = self._select_model_features(df)

        return df


# ── Standalone run ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_generator import generate_dataset, OUTPUT_FILE

    df = pd.read_csv(OUTPUT_FILE) if OUTPUT_FILE.exists() else generate_dataset()
    prep = LivestockPreprocessor()
    X = prep.fit_transform(df)
    prep.save()
    print(f"[preprocessor] Feature matrix shape: {X.shape}")
    print(f"[preprocessor] Features: {prep.feature_names_[:10]} … ({len(prep.feature_names_)} total)")
