import os
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure src/ is importable when run from project root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils import REQUIRED_RAW_COLS, build_results_df, validate_dataframe
from src.preprocessor import LivestockPreprocessor
from src.models import (
    IsolationForestModel, DBSCANModel, AutoencoderModel, KMeansAnomalyModel
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = ROOT / "data"
MODELS_DIR = ROOT / "models"
CSV_PATH   = DATA_DIR / "livestock_data.csv"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Livestock Anomaly Detection",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset …")
def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


@st.cache_resource(show_spinner="Loading preprocessor …")
def load_preprocessor() -> tuple[LivestockPreprocessor, np.ndarray]:
    prep_path = MODELS_DIR / "preprocessor.joblib"
    if prep_path.exists():
        prep = LivestockPreprocessor.load(prep_path)
        df = load_raw_data()
        X = prep.transform(df)
    else:
        df = load_raw_data()
        prep = LivestockPreprocessor()
        X = prep.fit_transform(df)
        prep.save(prep_path)
    return prep, X


@st.cache_resource(show_spinner="Loading Isolation Forest …")
def load_isolation_forest(X: np.ndarray) -> IsolationForestModel:
    p = MODELS_DIR / "isolation_forest.joblib"
    if p.exists():
        return IsolationForestModel.load(p)
    m = IsolationForestModel()
    m.fit(X)
    m.save(p)
    return m


@st.cache_resource(show_spinner="Loading DBSCAN …")
def load_dbscan(X: np.ndarray) -> DBSCANModel:
    p = MODELS_DIR / "dbscan.joblib"
    if p.exists():
        return DBSCANModel.load(p)
    m = DBSCANModel()
    m.fit(X)
    m.save(p)
    return m


@st.cache_resource(show_spinner="Loading Autoencoder …")
def load_autoencoder(X: np.ndarray) -> AutoencoderModel:
    p = MODELS_DIR / "autoencoder"
    if Path(p).exists():
        return AutoencoderModel.load(Path(p))
    m = AutoencoderModel(input_dim=X.shape[1])
    m.fit(X)
    m.save(Path(p))
    return m


@st.cache_resource(show_spinner="Loading KMeans …")
def load_kmeans(X: np.ndarray) -> KMeansAnomalyModel:
    p = MODELS_DIR / "kmeans.joblib"
    if p.exists():
        return KMeansAnomalyModel.load(p)
    m = KMeansAnomalyModel()
    m.fit(X)
    m.save(p)
    return m


def get_model(name: str, X: np.ndarray):
    if name == "Isolation Forest":
        return load_isolation_forest(X)
    if name == "DBSCAN":
        return load_dbscan(X)
    if name == "Autoencoder":
        return load_autoencoder(X)
    if name == "KMeans":
        return load_kmeans(X)
    raise ValueError(f"Unknown model: {name}")


# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.image("https://img.icons8.com/color/96/cow.png", width=60)
st.sidebar.title("Livestock Anomaly Detection")
st.sidebar.markdown("---")

model_choice = st.sidebar.selectbox(
    "Active model",
    ["Isolation Forest", "DBSCAN", "Autoencoder", "KMeans"],
    index=0,
    help="Choose the unsupervised model to run anomaly detection.",
)

uploaded_file = st.sidebar.file_uploader(
    "Upload new farm CSV",
    type=["csv"],
    help=f"CSV must contain columns: {', '.join(REQUIRED_RAW_COLS[:6])} …",
)

st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ for AI Engineer Portfolio")

# ── Load base data ─────────────────────────────────────────────────────────────

raw_df = load_raw_data()
prep, X_train = load_preprocessor()

# Determine source dataframe
if uploaded_file is not None:
    try:
        user_df = pd.read_csv(uploaded_file)
        validate_dataframe(user_df, REQUIRED_RAW_COLS)
        X_infer = prep.transform(user_df)
        source_df = user_df
        st.sidebar.success(f"Uploaded {len(user_df):,} records")
    except Exception as e:
        st.sidebar.error(f"Upload error: {e}")
        source_df = raw_df
        X_infer = X_train
else:
    source_df = raw_df
    X_infer = X_train

# ── Run model ──────────────────────────────────────────────────────────────────

model = get_model(model_choice, X_train)
scores, categories = model.predict(X_infer)
results_df = build_results_df(source_df, scores, categories)

# ── KPIs ───────────────────────────────────────────────────────────────────────

st.title("🐄 Smart Livestock Health — Anomaly Detection")
st.caption(f"Model: **{model_choice}** · Records: **{len(source_df):,}**")

high  = int((categories == "high").sum())
med   = int((categories == "medium").sum())
anom_rate = round(100 * (categories != "low").mean(), 1)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total records",      f"{len(source_df):,}")
c2.metric("High-risk records",  f"{high:,}",  delta=f"{round(100*high/len(source_df),1)}%", delta_color="inverse")
c3.metric("Medium-risk records", f"{med:,}")
c4.metric("Anomaly rate",       f"{anom_rate}%")

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Results table",
    "🌡️ Risk heatmap",
    "📈 Charts",
    "🔵 Cluster view",
    "🏆 Farm ranking",
])

# ── Tab 1: Results table ───────────────────────────────────────────────────────
with tab1:
    st.subheader("Anomaly Detection Results")

    risk_filter = st.multiselect(
        "Filter by risk category",
        ["high", "medium", "low"],
        default=["high", "medium"],
    )
    show_df = results_df[results_df["risk_category"].isin(risk_filter)].head(500)

    def color_risk(val):
        colors = {"high": "background-color:#FCEBEB;color:#791F1F",
                  "medium": "background-color:#FAEEDA;color:#633806",
                  "low": "background-color:#EAF3DE;color:#27500A"}
        return colors.get(val, "")

    display_cols = ["farm_id", "animal_type", "region",
                    "body_temperature", "heart_rate", "anomaly_score", "risk_category"]
    display_cols = [c for c in display_cols if c in show_df.columns]

    styled_df = show_df[display_cols].style.map(
        color_risk,
        subset=["risk_category"]
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400
    )

    csv_out = results_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download full results CSV", csv_out,
                       "anomaly_results.csv", "text/csv")

# ── Tab 2: Risk heatmap ────────────────────────────────────────────────────────
with tab2:
    st.subheader("Risk Heatmap — Farms × Features")

    heatmap_features = ["body_temperature", "heart_rate", "feed_intake",
                        "water_intake", "movement_level", "cough_frequency",
                        "vaccination_gap_days", "nearby_outbreak_score"]
    heatmap_features = [f for f in heatmap_features if f in results_df.columns]

    top_farms = (results_df.groupby("farm_id")["anomaly_score"]
                  .mean().nlargest(20).index.tolist())
    hm_df = results_df[results_df["farm_id"].isin(top_farms)]
    hm_pivot = (hm_df.groupby("farm_id")[heatmap_features]
                 .mean().reindex(top_farms))

    fig_hm = px.imshow(
        hm_pivot,
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        title="Average Feature Value — Top 20 Highest-Risk Farms",
        labels={"color": "Feature value"},
    )
    fig_hm.update_layout(height=500)
    st.plotly_chart(fig_hm, use_container_width=True)

# ── Tab 3: Charts ──────────────────────────────────────────────────────────────
with tab3:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Score distribution")
        fig_hist = px.histogram(
            results_df, x="anomaly_score", nbins=80,
            color="risk_category",
            color_discrete_map={"high": "#E24B4A", "medium": "#EF9F27", "low": "#97C459"},
            title="Anomaly Score Distribution",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        st.subheader("Anomalies by animal type")
        count_df = (results_df[results_df["risk_category"] != "low"]
                    .groupby(["animal_type", "risk_category"])
                    .size().reset_index(name="count"))
        fig_bar = px.bar(
            count_df, x="animal_type", y="count", color="risk_category",
            color_discrete_map={"high": "#E24B4A", "medium": "#EF9F27"},
            barmode="stack", title="Anomalies by Animal Type",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Anomalies by region")
    reg_df = (results_df[results_df["risk_category"] != "low"]
              .groupby(["region", "risk_category"])
              .size().reset_index(name="count"))
    fig_reg = px.bar(
        reg_df, x="region", y="count", color="risk_category",
        color_discrete_map={"high": "#E24B4A", "medium": "#EF9F27"},
        barmode="group", title="Risk Distribution by Region",
    )
    st.plotly_chart(fig_reg, use_container_width=True)

# ── Tab 4: Cluster view ────────────────────────────────────────────────────────
with tab4:
    st.subheader("2D PCA Cluster Visualization")
    from sklearn.decomposition import PCA

    n_pca = min(4000, len(X_infer))
    pca_idx = np.random.choice(len(X_infer), n_pca, replace=False)
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_infer[pca_idx])

    pca_df = pd.DataFrame({
        "PC1": X2[:, 0], "PC2": X2[:, 1],
        "risk": categories[pca_idx],
        "score": np.round(scores[pca_idx], 3),
        "farm_id": source_df["farm_id"].values[pca_idx],
        "animal_type": source_df["animal_type"].values[pca_idx],
    })

    fig_pca = px.scatter(
        pca_df, x="PC1", y="PC2", color="risk",
        color_discrete_map={"high": "#E24B4A", "medium": "#EF9F27", "low": "#97C459"},
        hover_data=["farm_id", "animal_type", "score"],
        opacity=0.6,
        title=f"PCA Clusters — {model_choice} (n={n_pca:,})",
        labels={"PC1": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                "PC2": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"},
    )
    fig_pca.update_traces(marker_size=4)
    fig_pca.update_layout(height=550)
    st.plotly_chart(fig_pca, use_container_width=True)

# ── Tab 5: Farm ranking ────────────────────────────────────────────────────────
with tab5:
    st.subheader("Farm Risk Ranking")

    farm_rank = (results_df.groupby("farm_id")
                  .agg(
                      total_animals=("animal_type", "count"),
                      avg_score=("anomaly_score", "mean"),
                      high_risk=("risk_category", lambda x: (x == "high").sum()),
                      medium_risk=("risk_category", lambda x: (x == "medium").sum()),
                      dominant_type=("animal_type", lambda x: x.mode()[0]),
                      region=("region", "first"),
                  )
                  .reset_index()
                  .sort_values("avg_score", ascending=False)
                  .reset_index(drop=True))
    farm_rank.index += 1
    farm_rank.columns = ["Farm ID", "Animals", "Avg Score",
                         "High Risk", "Medium Risk",
                         "Primary Animal", "Region"]
    farm_rank["Avg Score"] = farm_rank["Avg Score"].round(4)

    col_sort = st.selectbox("Sort by", ["Avg Score", "High Risk", "Animals"])
    farm_rank = farm_rank.sort_values(col_sort, ascending=False).reset_index(drop=True)
    farm_rank.index += 1

    st.dataframe(farm_rank, use_container_width=True, height=450)
