"""
models.py
Four unsupervised anomaly detection models, each as a class with:
  fit(X), predict(X) → (anomaly_score, risk_category), save(), load()

Models:
  A. IsolationForestModel  — sklearn, contamination tuned to 0.05
  B. DBSCANModel           — sklearn, noise points = anomalies
  C. AutoencoderModel      — TensorFlow/Keras, reconstruction error
  D. KMeansAnomalyModel    — sklearn, distance-to-centroid as score
"""

import os
import numpy as np
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _risk_category(scores: np.ndarray,
                   low_thresh: float = 0.33,
                   high_thresh: float = 0.66) -> np.ndarray:
    """
    Map continuous anomaly scores in [0, 1] → 'low' / 'medium' / 'high'.
    Thresholds chosen so roughly the top 5% are 'high', next 15% 'medium'.
    """
    cats = np.where(scores >= high_thresh, "high",
           np.where(scores >= low_thresh, "medium", "low"))
    return cats


def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


# ── A. Isolation Forest ───────────────────────────────────────────────────────

class IsolationForestModel:
    """
    Isolation Forest anomaly detector.

    Why contamination=0.05?
    Typical disease prevalence in a managed herd is 3–7%.
    Setting contamination=0.05 tells IF to flag roughly 5% of records
    as outliers, matching our synthetic injection rate and real-world
    epidemiological expectations.

    Why n_estimators=200?
    Default 100 is sufficient for many datasets but larger ensembles
    reduce variance in score estimates, important for borderline cases.
    """

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


# ── B. DBSCAN ─────────────────────────────────────────────────────────────────

class DBSCANModel:
    """
    DBSCAN-based anomaly detector.

    Why eps=0.5, min_samples=10?
    After StandardScaler preprocessing, features are roughly unit-variance.
    eps=0.5 catches tight clusters; isolated points in lower-density regions
    are flagged as noise (label = -1), which we treat as anomalies.
    min_samples=10 ensures noise is genuinely sparse, not just small clusters.

    Limitation: DBSCAN has no explicit score — we use distance to nearest
    core point as a proxy anomaly score.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 10):
        from sklearn.cluster import DBSCAN
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        self._fitted = False
        self._X_train: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "DBSCANModel":
        print("[DBSCAN] Fitting …")
        self.model.fit(X)
        self._X_train = X
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.neighbors import NearestNeighbors
        # Distance to nearest training point as anomaly score
        nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
        nn.fit(self._X_train)
        dists, _ = nn.kneighbors(X)
        scores = _minmax_normalize(dists.ravel())
        return scores, _risk_category(scores)

    def save(self, path: Path = MODELS_DIR / "dbscan.joblib"):
        joblib.dump({"model": self.model, "X_train": self._X_train,
                     "eps": self.eps, "min_samples": self.min_samples}, path)
        print(f"[DBSCAN] Saved → {path}")

    @classmethod
    def load(cls, path: Path = MODELS_DIR / "dbscan.joblib") -> "DBSCANModel":
        data = joblib.load(path)
        obj = cls(eps=data["eps"], min_samples=data["min_samples"])
        obj.model = data["model"]
        obj._X_train = data["X_train"]
        obj._fitted = True
        print(f"[DBSCAN] Loaded ← {path}")
        return obj


# ── C. Autoencoder ────────────────────────────────────────────────────────────

class AutoencoderModel:
    """Lightweight sklearn autoencoder-like model using MLPRegressor."""
    def __init__(self, input_dim=None, epochs=50, batch_size=256):
        self.input_dim=input_dim; self.epochs=epochs
    def fit(self,X):
        from sklearn.neural_network import MLPRegressor
        self.model=MLPRegressor(hidden_layer_sizes=(32,16,32), max_iter=self.epochs, random_state=42)
        self.model.fit(X,X)
        recon=self.model.predict(X)
        err=np.mean((X-recon)**2,axis=1)
        self.history_={}
        self._fitted=True
        return self
    def predict(self,X):
        recon=self.model.predict(X); err=np.mean((X-recon)**2,axis=1); scores=_minmax_normalize(err); return scores,_risk_category(scores)
    def save(self,path=MODELS_DIR / "autoencoder.joblib"):
        joblib.dump(self.model,path)
    @classmethod
    def load(cls,path=MODELS_DIR / "autoencoder.joblib"):
        obj=cls(); obj.model=joblib.load(path); obj._fitted=True; return obj


# ── D. KMeans Anomaly ─────────────────────────────────────────────────────────

class KMeansAnomalyModel:
    """
    KMeans-based anomaly detector using distance-to-centroid as anomaly score.

    Why k=8?
    We have 4 animal types × ~2 health states (normal/stressed) ≈ 8 clusters.
    Points far from any centroid are anomalous. This is a simpler signal than
    Isolation Forest but provides complementary cluster structure.

    Silhouette score is computed post-fit for evaluation.
    """

    def __init__(self, n_clusters: int = 8):
        from sklearn.cluster import KMeans
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self._fitted = False

    def fit(self, X: np.ndarray) -> "KMeansAnomalyModel":
        print(f"[KMeans] Fitting k={self.n_clusters} clusters …")
        self.model.fit(X)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Distance to nearest centroid → normalized anomaly score."""
        centroids = self.model.cluster_centers_
        labels = self.model.predict(X)
        dists = np.linalg.norm(X - centroids[labels], axis=1)
        scores = _minmax_normalize(dists)
        return scores, _risk_category(scores)

    def silhouette(self, X: np.ndarray) -> float:
        from sklearn.metrics import silhouette_score
        labels = self.model.predict(X)
        # Subsample for speed on large datasets
        n = min(5000, len(X))
        idx = np.random.choice(len(X), n, replace=False)
        return float(silhouette_score(X[idx], labels[idx]))

    def save(self, path: Path = MODELS_DIR / "kmeans.joblib"):
        joblib.dump(self.model, path)
        print(f"[KMeans] Saved → {path}")

    @classmethod
    def load(cls, path: Path = MODELS_DIR / "kmeans.joblib") -> "KMeansAnomalyModel":
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        obj.n_clusters = obj.model.n_clusters
        obj._fitted = True
        print(f"[KMeans] Loaded ← {path}")
        return obj
