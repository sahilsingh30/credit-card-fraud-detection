# 🛡️ Credit Card Fraud Detection System

An end-to-end machine learning pipeline for detecting fraudulent credit card transactions, built on the Kaggle Credit Card Fraud dataset. Features a Random Forest classifier with class imbalance handling, full evaluation suite, and an interactive Streamlit dashboard.

---

## 📊 Results

| Metric | Score |
|---|---|
| AUC-ROC | **0.9946** |
| Average Precision | **0.9769** |
| Fraud Recall | 0.71 |
| Fraud Precision | 1.00 |

> Dataset: 284,807 transactions · 492 fraud cases · **0.17% fraud rate**

---

## 🚀 Features

- **Severe class imbalance handling** via `class_weight='balanced'` in Random Forest
- **Stratified train/test split** to preserve fraud ratio across sets
- **5-fold Stratified Cross-Validation** with AUC-ROC and Average Precision scoring
- **Full evaluation suite** — ROC curve, Precision-Recall curve, Confusion Matrix, Classification Report
- **Feature importance ranking** across all 30 features
- **Model serialisation** via `joblib` for production deployment
- **Interactive Streamlit dashboard** with live single-transaction predictor and batch prediction

---

## 🗂️ Project Structure

```
credit-card-fraud-detection/
│
├── creditcard.csv          # Dataset (download from Kaggle — see below)
├── fraud_detection.py      # Core ML pipeline (terminal script)
├── app.py                  # Streamlit web dashboard
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/sahilsingh30/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download `creditcard.csv` from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root.

---

## ▶️ Usage

### Option A — Run the terminal pipeline

```bash
python fraud_detection.py
```

Outputs:
- Printed metrics (AUC-ROC, Average Precision, Confusion Matrix)
- `fraud_rf_model.joblib` — serialised model
- `fraud_scaler.joblib` — serialised scaler
- `fraud_detection_results.png` — evaluation plots

### Option B — Launch the Streamlit dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 📱 Dashboard Preview

The Streamlit dashboard includes 4 tabs:

| Tab | Contents |
|---|---|
| 📊 Overview | Dataset stats, class distribution, pipeline architecture |
| 📈 Model Performance | ROC curve, PR curve, confusion matrix, feature importances |
| 🔍 Live Prediction | Single transaction predictor with adjustable threshold |
| 📋 Dataset Explorer | Filter, preview, and inspect the dataset |

---

## 🧠 Model Details

```
Algorithm   : Random Forest Classifier
Estimators  : 100 trees
Max Depth   : 12
Class Weight: balanced  ← key for handling 0.17% fraud rate
CV Strategy : StratifiedKFold (k=5)
Test Size   : 20% (stratified)
Random State: 42
```

### Why `class_weight='balanced'`?

With only 0.17% fraud rate, a naive model predicts "Legit" for everything and achieves 99.8% accuracy — completely useless. `class_weight='balanced'` automatically penalises misclassifying the minority (fraud) class by a factor proportional to its rarity, forcing the model to actually learn fraud patterns.

---

## 📦 Requirements

```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
streamlit>=1.28.0
joblib>=1.3.0
```

Install via:
```bash
pip install -r requirements.txt
```

---

## 📁 Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
**Credits:** ULB Machine Learning Group

| Column | Description |
|---|---|
| `Time` | Seconds elapsed since first transaction |
| `V1–V28` | PCA-anonymised features (original features confidential) |
| `Amount` | Transaction amount in USD |
| `Class` | Target — `0` = Legit, `1` = Fraud |

> The dataset is not included in this repository due to its size (~150MB). Download it directly from Kaggle.

---

## 👤 Author

**Sahil Singh**  
GitHub: [@sahilsingh30](https://github.com/sahilsingh30)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
