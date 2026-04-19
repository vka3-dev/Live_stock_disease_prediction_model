# 🐄 Smart Livestock Health Anomaly Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-blue?logo=scikit-learn)
![SHAP](https://img.shields.io/badge/SHAP-0.42%2B-green)

**Unsupervised machine learning system that detects abnormal behaviour patterns in farm data to predict possible livestock disease outbreaks — before any labels are available.**

---

## 📌 Motivation

In large-scale livestock farming, disease outbreaks cost billions annually. Traditional rule-based alert systems miss subtle, multi-feature patterns. This project builds an **unsupervised anomaly detection pipeline** that:

- Ingests sensor/farm data (temperature, heart rate, feed intake, etc.)
- Learns what *normal* looks like from the data itself — no labels needed
- Flags individual animals or farms that deviate from normality
- Explains *why* a record is anomalous (SHAP)
- Visualises risk across farms in a Streamlit dashboard

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                        │
│   data_generator.py  →  data/livestock_data.csv (20,000 rows)  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                   PREPROCESSING LAYER                           │
│   preprocessor.py — imputation · encoding · feature engineering │
│                    · StandardScaler · save/load                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
         ┌─────────────────────┼──────────────────────┐
         ▼                     ▼                       ▼           ▼
┌────────────────┐   ┌──────────────┐   ┌──────────────────┐ ┌──────────┐
│ Isolation      │   │   DBSCAN     │   │  Autoencoder     │ │  KMeans  │
│ Forest         │   │              │   │  (TF/Keras)      │ │          │
│ anomaly_score  │   │ dist-to-core │   │ reconstruction   │ │ dist-to- │
│                │   │              │   │ error            │ │ centroid │
└───────┬────────┘   └──────┬───────┘   └────────┬─────────┘ └────┬─────┘
        │                   │                    │                 │
        └───────────────────┴────────────────────┴─────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────┐
│                  EXPLAINABILITY LAYER                           │
│   explainability.py — SHAP TreeExplainer (Isolation Forest)    │
│   Top contributing features per anomalous record               │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│               EVALUATION & VISUALISATION LAYER                  │
│   evaluator.py  —  score distributions · model comparison      │
│   PCA cluster plot · reconstruction loss curve                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                     DASHBOARD (app.py)                          │
│   Streamlit — upload CSV · live detection · heatmap            │
│   farm ranking · cluster scatter · downloadable results        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Algorithms

| Model | Approach | Why chosen |
|---|---|---|
| **Isolation Forest** | Random partitioning trees; anomalies isolated in fewer splits | Fast, scalable, no distance metric needed; contamination can be tuned |
| **DBSCAN** | Density-based clustering; noise points = anomalies | Finds arbitrarily shaped clusters; no need to specify k |
| **Autoencoder** | Neural encoder-decoder; high reconstruction error → anomaly | Learns non-linear manifold; captures complex feature interactions |
| **KMeans** | Distance to cluster centroid as anomaly score | Interpretable clusters; fast; silhouette score for quality assessment |

**Primary model**: Isolation Forest (fastest, most robust, native SHAP support).

---

## 🚀 Setup & Run

### 1. Clone and install

```bash
git clone https://github.com/yourname/livestock-anomaly-detection.git
cd livestock-anomaly-detection
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate data

```bash
python -m src.data_generator
# → data/livestock_data.csv  (20,000 rows)
```

### 3. Train all models

```bash
python train_pipeline.py
# Runs: preprocess → train IF / DBSCAN / Autoencoder / KMeans → evaluate → save
```

Or run each step individually:

```python
from src.data_generator import generate_dataset
from src.preprocessor import LivestockPreprocessor
from src.models import IsolationForestModel

df = generate_dataset()
prep = LivestockPreprocessor()
X = prep.fit_transform(df)
prep.save()

model = IsolationForestModel()
model.fit(X)
model.save()

scores, categories = model.predict(X)
```

### 4. Launch dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## ☁️ Streamlit Cloud Deployment

1. Push repo to GitHub (make sure `requirements.txt` is at root).
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app.
3. Select your repo, branch `main`, main file `app.py`.
4. Click **Deploy**. Streamlit Cloud will install requirements automatically.

> **Note**: First run will generate data and train models on the cloud instance.
> Pre-train locally and commit the `models/` folder to avoid cold-start delay.

---

## 📸 Screenshots

![Dashboard Overview](screenshots/dashboard.png)
![Risk Heatmap](screenshots/heatmap.png)
![PCA Clusters](screenshots/clusters.png)
![SHAP Explainability](screenshots/shap.png)

---

## 🗂️ Folder Structure

```
livestock-anomaly-detection/
├── data/
│   ├── livestock_data.csv          ← generated dataset
│   └── charts/                     ← evaluation plots
├── models/                         ← trained model artifacts
├── notebooks/                      ← exploration notebooks
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── preprocessor.py
│   ├── models.py
│   ├── explainability.py
│   ├── evaluator.py
│   └── utils.py
├── app.py                          ← Streamlit dashboard
├── train_pipeline.py               ← end-to-end training script
├── requirements.txt
└── README.md
```

---

## 🔮 Future Improvements

- **Real sensor integration**: MQTT / IoT feed ingestion for live streaming detection
- **Semi-supervised learning**: Use confirmed disease records to fine-tune thresholds
- **Federated learning**: Train per-farm models without centralising sensitive data
- **Time-series models**: LSTM autoencoders to detect temporal anomaly patterns
- **Mobile alerts**: Push notifications when a farm crosses high-risk threshold
- **Ensemble scoring**: Combine all 4 model scores with a meta-learner

---

## 🧰 Tech Stack

Python · NumPy · Pandas · Scikit-Learn · TensorFlow/Keras · SHAP · Streamlit · Plotly · Matplotlib · Joblib
