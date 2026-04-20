import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / Streamlit Cloud
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

CHARTS_DIR = Path(__file__).resolve().parent.parent / "data" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "IsolationForest": "#D85A30",
    "DBSCAN":          "#378ADD",
    "Autoencoder":     "#1D9E75",
    "KMeans":          "#BA7517",
}


class ModelEvaluator:
    """Collects predictions from all models and generates evaluation charts."""

    def __init__(self, feature_names: list):
        self.feature_names = feature_names
        self.results: dict = {}  # model_name → {"scores": ..., "categories": ...}

    def add_model(self, name: str, scores: np.ndarray, categories: np.ndarray):
        self.results[name] = {"scores": scores, "categories": categories}

    # ── Summary statistics ────────────────────────────────────────────────────

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, data in self.results.items():
            cats = data["categories"]
            rows.append({
                "model": name,
                "total": len(cats),
                "high_risk": int((cats == "high").sum()),
                "medium_risk": int((cats == "medium").sum()),
                "low_risk": int((cats == "low").sum()),
                "anomaly_rate_pct": round(
                    100 * (cats != "low").mean(), 2
                ),
                "mean_score": round(float(data["scores"].mean()), 4),
                "p95_score": round(float(np.percentile(data["scores"], 95)), 4),
            })
        return pd.DataFrame(rows)

    # ── Chart 1: Score distribution histograms ────────────────────────────────

    def plot_score_distributions(self,
                                  save_path: Path = CHARTS_DIR / "score_distributions.png"):
        models = list(self.results.keys())
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.ravel()

        for i, name in enumerate(models):
            scores = self.results[name]["scores"]
            ax = axes[i]
            color = COLORS.get(name, "#888")
            ax.hist(scores, bins=60, color=color, alpha=0.85, edgecolor="none")
            ax.axvline(np.percentile(scores, 95), color="black",
                       linestyle="--", linewidth=1.2, label="p95 threshold")
            ax.set_title(name, fontsize=11, fontweight="bold")
            ax.set_xlabel("Anomaly score", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.legend(fontsize=8)
            ax.tick_params(labelsize=8)

        fig.suptitle("Anomaly Score Distributions — All Models", fontsize=13)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Evaluator] Score distributions saved → {save_path}")

    # ── Chart 2: Anomaly count comparison bar chart ───────────────────────────

    def plot_model_comparison(self,
                               save_path: Path = CHARTS_DIR / "model_comparison.png"):
        df = self.summary()
        fig, ax = plt.subplots(figsize=(8, 5))

        x = np.arange(len(df))
        w = 0.25
        ax.bar(x - w, df["high_risk"],   width=w, label="High",   color="#E24B4A")
        ax.bar(x,     df["medium_risk"], width=w, label="Medium", color="#EF9F27")
        ax.bar(x + w, df["low_risk"],    width=w, label="Low",    color="#97C459")

        ax.set_xticks(x)
        ax.set_xticklabels(df["model"], fontsize=10)
        ax.set_ylabel("Record count", fontsize=10)
        ax.set_title("Anomaly Count by Risk Category — Model Comparison", fontsize=12)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Evaluator] Model comparison saved → {save_path}")

    # ── Chart 3: Autoencoder reconstruction loss ──────────────────────────────

    def plot_reconstruction_loss(self,
                                  history: dict,
                                  save_path: Path = CHARTS_DIR / "autoencoder_loss.png"):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history.get("loss", []), label="Train loss", color="#1D9E75", linewidth=2)
        ax.plot(history.get("val_loss", []), label="Val loss",
                color="#1D9E75", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("MSE Loss", fontsize=10)
        ax.set_title("Autoencoder Reconstruction Loss", fontsize=12)
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Evaluator] Reconstruction loss saved → {save_path}")

    # ── Chart 4: PCA cluster visualization ───────────────────────────────────

    def plot_pca_clusters(self,
                           X: np.ndarray,
                           scores: np.ndarray,
                           categories: np.ndarray,
                           model_name: str = "IsolationForest",
                           save_path: Path = CHARTS_DIR / "pca_clusters.png"):
        from sklearn.decomposition import PCA

        print("[Evaluator] PCA projection for cluster plot …")
        pca = PCA(n_components=2, random_state=42)
        n = min(5000, len(X))
        idx = np.random.choice(len(X), n, replace=False)
        X2 = pca.fit_transform(X[idx])
        cats = categories[idx]

        fig, ax = plt.subplots(figsize=(9, 6))
        palette = {"low": "#97C459", "medium": "#EF9F27", "high": "#E24B4A"}
        for cat in ["low", "medium", "high"]:
            mask = cats == cat
            ax.scatter(X2[mask, 0], X2[mask, 1],
                       c=palette[cat], label=cat.capitalize(),
                       alpha=0.5, s=8, rasterized=True)

        ax.set_xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
        ax.set_ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
        ax.set_title(f"PCA Cluster View — {model_name}", fontsize=12)
        ax.legend(markerscale=3, fontsize=9)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Evaluator] PCA cluster plot saved → {save_path}")

    def run_all(self, X: np.ndarray, autoencoder_history: dict | None = None):
        """Generate all evaluation charts in one call."""
        self.plot_score_distributions()
        self.plot_model_comparison()
        if autoencoder_history:
            self.plot_reconstruction_loss(autoencoder_history)
        if "IsolationForest" in self.results:
            self.plot_pca_clusters(
                X,
                self.results["IsolationForest"]["scores"],
                self.results["IsolationForest"]["categories"],
            )
        print("\n[Evaluator] ── Summary ──────────────────────────────────────")
        print(self.summary().to_string(index=False))
