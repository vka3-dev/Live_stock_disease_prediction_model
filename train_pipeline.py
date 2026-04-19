"""
train_pipeline.py
End-to-end pipeline:
  1. Generate synthetic dataset (if not already present)
  2. Preprocess and engineer features
  3. Train all four unsupervised models
  4. Run SHAP explainability
  5. Evaluate and generate charts

Run with:  python train_pipeline.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OUTPUT_FILE = ROOT / "data" / "livestock_data.csv"
from src.preprocessor import LivestockPreprocessor
from src.models import (
    IsolationForestModel, DBSCANModel, AutoencoderModel, KMeansAnomalyModel
)
from src.explainability import AnomalyExplainer
from src.evaluator import ModelEvaluator
from src.utils import timed

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


@timed
def step_data() -> pd.DataFrame:
    print(f"[pipeline] Loading dataset ← {OUTPUT_FILE}")
    return pd.read_csv(OUTPUT_FILE)


@timed
def step_preprocess(df: pd.DataFrame):
    prep = LivestockPreprocessor()
    X = prep.fit_transform(df)
    prep.save(MODELS_DIR / "preprocessor.joblib")
    print(f"[pipeline] Feature matrix: {X.shape}")
    return prep, X


@timed
def step_train_all(X: np.ndarray):
    models = {}

    # A. Isolation Forest
    iforest = IsolationForestModel(contamination=0.05, n_estimators=200)
    iforest.fit(X)
    iforest.save()
    models["IsolationForest"] = iforest

    # B. DBSCAN
    dbscan = DBSCANModel(eps=0.5, min_samples=10)
    dbscan.fit(X)
    dbscan.save()
    models["DBSCAN"] = dbscan

    # C. Autoencoder
    ae = AutoencoderModel(input_dim=X.shape[1], epochs=50, batch_size=256)
    ae.fit(X)
    ae.save()
    models["Autoencoder"] = ae

    # D. KMeans
    km = KMeansAnomalyModel(n_clusters=8)
    km.fit(X)
    km.save()
    sil = km.silhouette(X)
    print(f"[pipeline] KMeans silhouette score: {sil:.4f}")
    models["KMeans"] = km

    return models


@timed
def step_explain(iforest: IsolationForestModel,
                 X: np.ndarray,
                 feature_names: list,
                 scores: np.ndarray):
    try:
        explainer = AnomalyExplainer(iforest, feature_names)
        explain_df = explainer.explain_batch(X, scores, top_k=5)
        print(f"[pipeline] Explained {len(explain_df)} anomalous records")
        explainer.plot_summary(X)
        return explain_df
    except ImportError:
        print("[pipeline] SHAP not installed — skipping explainability step")
        return pd.DataFrame()


@timed
def step_evaluate(models: dict, X: np.ndarray, feature_names: list, ae_history: dict):
    evaluator = ModelEvaluator(feature_names)
    for name, model in models.items():
        scores, cats = model.predict(X)
        evaluator.add_model(name, scores, cats)
    evaluator.run_all(X, autoencoder_history=ae_history)
    return evaluator


if __name__ == "__main__":
    print("=" * 60)
    print("  Smart Livestock Health — Training Pipeline")
    print("=" * 60)

    df = step_data()
    prep, X = step_preprocess(df)

    models = step_train_all(X)

    if_scores, if_cats = models["IsolationForest"].predict(X)
    step_explain(models["IsolationForest"], X, prep.feature_names_, if_scores)

    ae_history = getattr(models["Autoencoder"], "history_", {})
    evaluator = step_evaluate(models, X, prep.feature_names_, ae_history)

    print("\n[pipeline] ✅ All steps complete. Launch dashboard with:")
    print("           streamlit run app.py")
