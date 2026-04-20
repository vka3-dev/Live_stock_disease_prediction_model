import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CHARTS_DIR = Path(__file__).resolve().parent.parent / "data" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


class AnomalyExplainer:
    """
    Wraps SHAP TreeExplainer for Isolation Forest.

    Why SHAP?
    SHAP values provide theoretically grounded (Shapley) attribution:
    each feature's contribution is the average marginal impact across
    all possible feature subsets.  For anomaly detection this tells us
    WHICH features pushed a record away from normality.
    """

    def __init__(self, model, feature_names: list):
        """
        Parameters
        ----------
        model        : fitted IsolationForestModel instance
        feature_names: list of feature column names matching training data
        """
        self.if_model = model.model  # raw sklearn IsolationForest
        self.feature_names = feature_names
        self._explainer = None

    def _get_explainer(self, X_background: np.ndarray):
        """
        Build SHAP TreeExplainer lazily (expensive first call).
        Uses a shrunken background sample for speed.
        """
        import shap
        if self._explainer is None:
            n_bg = min(500, len(X_background))
            bg = shap.sample(X_background, n_bg, random_state=42)
            self._explainer = shap.TreeExplainer(self.if_model, data=bg)
        return self._explainer

    def explain_batch(self,
                      X: np.ndarray,
                      anomaly_scores: np.ndarray,
                      top_k: int = 5,
                      max_records: int = 200) -> pd.DataFrame:
        """
        Compute SHAP values for the most anomalous records.

        Returns a DataFrame with columns:
          record_idx, top_feature_1..top_k, shap_value_1..top_k
        """
        import shap

        # Only explain truly anomalous records for efficiency
        anom_mask = anomaly_scores > np.percentile(anomaly_scores, 90)
        X_anom = X[anom_mask][:max_records]
        anom_indices = np.where(anom_mask)[0][:max_records]

        explainer = self._get_explainer(X)
        shap_values = explainer.shap_values(X_anom)

        rows = []
        for i, (shap_row, orig_idx) in enumerate(zip(shap_values, anom_indices)):
            abs_shap = np.abs(shap_row)
            top_idx = abs_shap.argsort()[::-1][:top_k]
            row = {"record_idx": int(orig_idx),
                   "anomaly_score": float(anomaly_scores[orig_idx])}
            for rank, fi in enumerate(top_idx, 1):
                row[f"feature_{rank}"] = self.feature_names[fi]
                row[f"shap_{rank}"] = float(shap_row[fi])
            rows.append(row)

        return pd.DataFrame(rows)

    def plot_summary(self,
                     X: np.ndarray,
                     max_display: int = 15,
                     save_path: Path = CHARTS_DIR / "shap_summary.png"):
        """
        Generate SHAP summary (mean |SHAP|) bar plot and save to disk.
        """
        import shap
        print("[Explainer] Computing SHAP values for summary plot …")
        n = min(1000, len(X))
        idx = np.random.choice(len(X), n, replace=False)
        X_sample = X[idx]

        explainer = self._get_explainer(X)
        shap_values = explainer.shap_values(X_sample)

        fig, ax = plt.subplots(figsize=(8, 6))
        mean_abs = np.abs(shap_values).mean(axis=0)
        sorted_idx = mean_abs.argsort()[::-1][:max_display]

        feat_labels = [self.feature_names[i] for i in sorted_idx]
        values = mean_abs[sorted_idx]

        ax.barh(feat_labels[::-1], values[::-1], color="#D85A30", height=0.6)
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title("Feature Importance — Isolation Forest (SHAP)", fontsize=12)
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Explainer] SHAP summary plot saved → {save_path}")
        return save_path
