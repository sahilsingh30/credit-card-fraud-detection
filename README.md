# 🛡️ Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red?style=flat-square&logo=streamlit)
![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-green?style=flat-square)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.9946-brightgreen?style=flat-square)

An end-to-end machine learning pipeline for detecting fraudulent credit card transactions. Built on the Kaggle Credit Card Fraud dataset with a Random Forest classifier, class imbalance handling, full evaluation suite, and an interactive Streamlit dashboard.

---

## 📊 Results

| Metric | Score |
|---|---|
| AUC-ROC | **0.9946** |
| Average Precision | **0.9769** |
| Fraud Precision | 1.00 |
| Fraud Recall | 0.71 |

> Dataset: 284,807 transactions · 492 fraud cases · **0.17% fraud rate**

---

## 🚀 Features

- **Severe class imbalance handling** via `class_weight='balanced'` in Random Forest
- **Stratified train/test split** to preserve fraud ratio across sets
- **5-fold Stratified Cross-Validation** with AUC-ROC and Average Precision scoring
- **Full evaluation suite** — ROC curve, Precision-Recall curve, Confusion Matrix, Classification Report
- **Feature importance ranking** across all 30 features
- **Interactive Streamlit dashboard** with live single-transaction predictor and batch prediction

---

## 🗂️ Project Structure

```
Credit-Card-Fraud-Detection/
│
├── app.py              ← Streamlit dashboard (main file)
├── requirements.txt    ← Python dependencies
├── README.md
└── creditcard.csv      ← Dataset (download from Kaggle — see below)
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/sahilsingh30/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
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

Download `creditcard.csv` from [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root folder.

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Opens automatically at `http://localhost:8501`

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
Algorithm    : Random Forest Classifier
Estimators   : 100 trees
Max Depth    : 12
Class Weight : balanced  ← key for handling 0.17% fraud rate
CV Strategy  : StratifiedKFold (k=5)
Test Size    : 20% (stratified)
Random State : 42
```

### Why `class_weight='balanced'`?

With only 0.17% fraud rate, a naive model predicts "Legit" for everything and still achieves 99.8% accuracy — completely useless. `class_weight='balanced'` automatically penalises misclassifying the minority (fraud) class proportional to its rarity, forcing the model to actually learn fraud patterns.

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

> `creditcard.csv` is not included in this repository due to its size (~150MB). Download it from Kaggle and place it in the project root before running.

---

## 👤 Author

**Sahil Singh**  
GitHub: [@sahilsingh30](https://github.com/sahilsingh30)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
