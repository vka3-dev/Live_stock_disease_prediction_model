import os
import numpy as np
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _risk_category(scores: np.ndarray,
                   low_thresh: float = 0.33,
                   high_thresh: float = 0.66) -> np.ndarray:
  
    cats = np.where(scores >= high_thresh, "high",
           np.where(scores >= low_thresh, "medium", "low"))
    return cats


def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


class IsolationForestModel:
    

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200):
        from sklearn.ensemble import IsolationForest
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, X: np.ndarray) -> "IsolationForestModel":
        print("[IsolationForest] Training …")
        self.model.fit(X)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        anomaly_score : float array in [0, 1] — higher = more anomalous
        risk_category : str array — 'low' / 'medium' / 'high'
        """
        # decision_function returns higher values for inliers → negate
        raw = -self.model.decision_function(X)
        scores = _minmax_normalize(raw)
        return scores, _risk_category(scores)

    def save(self, path: Path = MODELS_DIR / "isolation_forest.joblib"):
        joblib.dump(self.model, path)
        print(f"[IsolationForest] Saved → {path}")

    @classmethod
    def load(cls, path: Path = MODELS_DIR / "isolation_forest.joblib") -> "IsolationForestModel":
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        obj._fitted = True
        print(f"[IsolationForest] Loaded ← {path}")
        return obj


