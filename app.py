import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import os
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    RocCurveDisplay, PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
)

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0d1117; }
    .block-container { padding: 2rem 2.5rem; max-width: 1400px; }

    /* Header */
    .hero-header {
        background: linear-gradient(135deg, #0f2027 0%, #1a1a2e 50%, #16213e 100%);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(0,212,255,0.06) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #e6edf3;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #7d8590;
        margin-top: 0.4rem;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(0,212,255,0.1);
        border: 1px solid rgba(0,212,255,0.25);
        color: #00d4ff;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 0.8rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* Metric cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #00d4ff44; }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4ff;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #7d8590;
        margin-top: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 500;
    }
    .metric-delta {
        font-size: 0.8rem;
        color: #3fb950;
        margin-top: 0.2rem;
        font-weight: 500;
    }

    /* Section headers */
    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #7d8590;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.8rem;
        border-left: 3px solid #00d4ff;
        padding-left: 0.6rem;
    }

    /* Prediction result */
    .pred-fraud {
        background: linear-gradient(135deg, #2d1116, #1e0a0e);
        border: 2px solid #f85149;
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
    }
    .pred-legit {
        background: linear-gradient(135deg, #0d2117, #091a11);
        border: 2px solid #3fb950;
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
    }
    .pred-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
    }
    .pred-prob {
        font-size: 3rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.1;
        margin: 0.5rem 0;
    }
    .pred-desc { font-size: 0.85rem; color: #7d8590; margin-top: 0.4rem; }

    /* Sidebar */
    .stSidebar { background-color: #0d1117; border-right: 1px solid #21262d; }
    .stSidebar .stSlider > div > div > div { background: #00d4ff; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #161b22;
        border-radius: 10px;
        padding: 4px;
        border: 1px solid #21262d;
    }
    .stTabs [data-baseweb="tab"] {
        color: #7d8590;
        font-weight: 500;
        font-size: 0.87rem;
    }
    .stTabs [aria-selected="true"] {
        background: #1f6feb !important;
        color: #ffffff !important;
        border-radius: 8px;
    }

    /* Status pills */
    .pill-loaded {
        display: inline-block;
        background: rgba(63,185,80,0.12);
        border: 1px solid rgba(63,185,80,0.3);
        color: #3fb950;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .pill-warning {
        display: inline-block;
        background: rgba(210,153,34,0.12);
        border: 1px solid rgba(210,153,34,0.3);
        color: #d2992a;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.6rem 1.4rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

# ── Helper: load/train model ─────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_or_train(csv_path):
    df = pd.read_csv(csv_path)

    scaler_amt = StandardScaler()
    scaler_time = StandardScaler()
    df["scaled_amount"] = scaler_amt.fit_transform(df[["Amount"]])
    df["scaled_time"]   = scaler_time.fit_transform(df[["Time"]])

    feature_cols = ["scaled_time", "scaled_amount"] + [f"V{i}" for i in range(1, 29)]
    X = df[feature_cols].values
    y = df["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=12,
        class_weight="balanced", n_jobs=-1, random_state=42
    )
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    metrics = {
        "auc_roc"   : roc_auc_score(y_test, y_prob),
        "avg_prec"  : average_precision_score(y_test, y_prob),
        "cm"        : confusion_matrix(y_test, y_pred),
        "report"    : classification_report(y_test, y_pred,
                          target_names=["Legit", "Fraud"], output_dict=True),
        "importances": clf.feature_importances_,
        "feature_cols": feature_cols,
        "fraud_rate": df["Class"].mean() * 100,
        "n_total"   : len(df),
        "n_fraud"   : int(df["Class"].sum()),
        "y_test"    : y_test,
        "y_prob"    : y_prob,
    }
    return clf, scaler_amt, scaler_time, metrics, df

# ── Hero Header ──────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">🛡️ ML Security System</div>
    <h1 class="hero-title">Credit Card Fraud Detection</h1>
    <p class="hero-subtitle">Random Forest · Scikit-learn · AUC-ROC Evaluation · Real-time Prediction</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    uploaded = st.file_uploader("Upload `creditcard.csv`", type=["csv"])
    use_demo = st.checkbox("Use built-in demo dataset", value=True)

    st.markdown("---")
    st.markdown("### 🎛️ Model Settings")
    n_est  = st.slider("n_estimators", 50, 200, 100, 10)
    max_d  = st.selectbox("max_depth", [8, 10, 12, 15, "None"], index=2)
    thresh = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05,
                       help="Adjust fraud detection sensitivity")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#7d8590; line-height:1.6'>
    <b style='color:#e6edf3'>Tech Stack</b><br>
    Python · Random Forest<br>
    Scikit-learn · Pandas<br>
    Streamlit · Matplotlib
    </div>
    """, unsafe_allow_html=True)

# ── Load Dataset ─────────────────────────────────────────────
csv_path = None
if uploaded:
    with open("temp_upload.csv", "wb") as f:
        f.write(uploaded.read())
    csv_path = "temp_upload.csv"
    st.markdown('<span class="pill-loaded">✓ Custom dataset loaded</span>', unsafe_allow_html=True)
elif use_demo and os.path.exists("creditcard.csv"):
    csv_path = "creditcard.csv"
    st.markdown('<span class="pill-loaded">✓ Demo dataset ready</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="pill-warning">⚠ Upload creditcard.csv to begin</span>', unsafe_allow_html=True)
    st.info("Please upload the `creditcard.csv` dataset using the sidebar, or check 'Use built-in demo dataset' if you have it in the same folder.")
    st.stop()

# ── Train ────────────────────────────────────────────────────
with st.spinner("🔄 Training Random Forest model..."):
    clf, scaler_amt, scaler_time, m, df = load_or_train(csv_path)

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Overview",
    "📈  Model Performance",
    "🔍  Live Prediction",
    "📋  Dataset Explorer",
])

# ════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-label">Dataset Statistics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, f"{m['n_total']:,}", "Total Transactions", "284,807 records"),
        (c2, f"{m['n_fraud']:,}", "Fraud Cases", "Confirmed fraud"),
        (c3, f"{m['fraud_rate']:.4f}%", "Fraud Rate", "Severe imbalance"),
        (c4, f"{m['auc_roc']:.4f}", "AUC-ROC Score", "Model performance"),
    ]
    for col, val, label, delta in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-delta">{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown('<div class="section-label">Class Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor('#161b22')
        ax.set_facecolor('#161b22')
        counts = [m['n_total'] - m['n_fraud'], m['n_fraud']]
        colors = ['#1f6feb', '#f85149']
        bars = ax.bar(['Legit', 'Fraud'], counts, color=colors, width=0.4, edgecolor='none')
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                    f'{cnt:,}', ha='center', va='bottom', color='#e6edf3', fontsize=10, fontweight='600')
        ax.set_ylabel('Count', color='#7d8590', fontsize=9)
        ax.tick_params(colors='#7d8590')
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.grid(axis='y', color='#21262d', linewidth=0.8)
        ax.set_ylim(0, max(counts) * 1.15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.markdown('<div class="section-label">Amount Distribution by Class</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor('#161b22')
        ax.set_facecolor('#161b22')
        legit_amt  = df[df['Class'] == 0]['Amount'].clip(0, 500)
        fraud_amt  = df[df['Class'] == 1]['Amount'].clip(0, 500)
        ax.hist(legit_amt, bins=50, color='#1f6feb', alpha=0.6, label='Legit', density=True)
        ax.hist(fraud_amt, bins=50, color='#f85149', alpha=0.8, label='Fraud', density=True)
        ax.set_xlabel('Amount (USD, clipped at $500)', color='#7d8590', fontsize=9)
        ax.set_ylabel('Density', color='#7d8590', fontsize=9)
        ax.tick_params(colors='#7d8590')
        ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3', fontsize=9)
        for spine in ax.spines.values(): spine.set_color('#21262d')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-label">Pipeline Architecture</div>', unsafe_allow_html=True)
    steps = [
        ("01", "Load CSV", "284,807 transactions"),
        ("02", "Scale Features", "StandardScaler on Amount & Time"),
        ("03", "Stratified Split", "80% train / 20% test"),
        ("04", "Random Forest", "class_weight='balanced'"),
        ("05", "Cross-Validate", "5-Fold StratifiedKFold"),
        ("06", "Evaluate", "AUC-ROC · Avg Precision · CM"),
    ]
    cols = st.columns(6)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style='background:#161b22; border:1px solid #21262d; border-radius:10px;
                        padding:1rem 0.8rem; text-align:center; height:100px;'>
                <div style='font-size:0.65rem; color:#00d4ff; font-weight:700;
                            letter-spacing:1px; margin-bottom:0.3rem;'>{num}</div>
                <div style='font-size:0.85rem; font-weight:600; color:#e6edf3;
                            margin-bottom:0.2rem;'>{title}</div>
                <div style='font-size:0.72rem; color:#7d8590; line-height:1.3'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ════════════════════════════════════════════════════════════
with tab2:
    r2c1, r2c2, r2c3 = st.columns(3)
    report = m['report']
    perf_cards = [
        (r2c1, f"{m['auc_roc']:.4f}", "AUC-ROC"),
        (r2c2, f"{m['avg_prec']:.4f}", "Average Precision"),
        (r2c3, f"{report['Fraud']['f1-score']:.4f}", "Fraud F1-Score"),
    ]
    for col, val, label in perf_cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-label">ROC Curve</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#161b22'); ax.set_facecolor('#161b22')
        RocCurveDisplay.from_predictions(m['y_test'], m['y_prob'], ax=ax, color='#00d4ff',
                                          name=f"RF (AUC={m['auc_roc']:.4f})")
        ax.plot([0,1],[0,1],'--', color='#444d56', lw=1)
        ax.set_xlabel('False Positive Rate', color='#7d8590'); ax.set_ylabel('True Positive Rate', color='#7d8590')
        ax.tick_params(colors='#7d8590')
        ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3')
        for spine in ax.spines.values(): spine.set_color('#21262d')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_b:
        st.markdown('<div class="section-label">Precision-Recall Curve</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#161b22'); ax.set_facecolor('#161b22')
        PrecisionRecallDisplay.from_predictions(m['y_test'], m['y_prob'], ax=ax, color='#f78166',
                                                  name=f"RF (AP={m['avg_prec']:.4f})")
        ax.set_xlabel('Recall', color='#7d8590'); ax.set_ylabel('Precision', color='#7d8590')
        ax.tick_params(colors='#7d8590')
        ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3')
        for spine in ax.spines.values(): spine.set_color('#21262d')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-label">Confusion Matrix</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        fig.patch.set_facecolor('#161b22'); ax.set_facecolor('#161b22')
        cm = m['cm']
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Legit','Fraud'], color='#e6edf3')
        ax.set_yticklabels(['Legit','Fraud'], color='#e6edf3')
        ax.set_xlabel('Predicted', color='#7d8590'); ax.set_ylabel('Actual', color='#7d8590')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                        color='white' if cm[i,j] > cm.max()/2 else '#0d1117', fontsize=12, fontweight='700')
        plt.colorbar(im, ax=ax)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_d:
        st.markdown('<div class="section-label">Top-15 Feature Importances</div>', unsafe_allow_html=True)
        feat_imp = pd.Series(m['importances'], index=m['feature_cols']).sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        fig.patch.set_facecolor('#161b22'); ax.set_facecolor('#161b22')
        colors_imp = ['#00d4ff' if i == 0 else '#1f6feb' if i < 5 else '#21262d' for i in range(15)]
        ax.barh(feat_imp.index[::-1], feat_imp.values[::-1], color=colors_imp[::-1], edgecolor='none')
        ax.set_xlabel('Importance', color='#7d8590'); ax.tick_params(colors='#7d8590')
        for spine in ax.spines.values(): spine.set_color('#21262d')
        ax.grid(axis='x', color='#21262d', lw=0.8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-label">Full Classification Report</div>', unsafe_allow_html=True)
    report_df = pd.DataFrame(report).T.iloc[:2].round(4)
    st.dataframe(report_df.style.format("{:.4f}").background_gradient(cmap='Blues', axis=1), use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — Live Prediction
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">Single Transaction Predictor</div>', unsafe_allow_html=True)
    st.markdown("Adjust the sliders to simulate a transaction, then click **Predict**.")

    col_pred, col_result = st.columns([1.2, 1])

    with col_pred:
        amount = st.number_input("Transaction Amount (USD)", 0.01, 25000.0, 150.0, step=10.0)
        time_s = st.number_input("Time (seconds elapsed)", 0, 172792, 50000, step=1000)

        st.markdown("**PCA Features (V1–V10)** — adjust for anomaly simulation")
        v_cols = st.columns(2)
        v_vals = {}
        for i in range(1, 29):
            col = v_cols[(i-1) % 2]
            default = float(np.random.normal(0, 1)) if i > 10 else 0.0
            if i <= 10:
                v_vals[f"V{i}"] = col.slider(f"V{i}", -10.0, 10.0, default, 0.1, key=f"v{i}")
            else:
                v_vals[f"V{i}"] = default  # auto-fill the rest with normal values

        predict_btn = st.button("🔍 Predict Transaction")

    with col_result:
        if predict_btn:
            scaled_amt  = (amount - 88.35) / 250.12
            scaled_time = (time_s - 94813) / 47488

            row = [scaled_time, scaled_amt] + [v_vals[f"V{i}"] for i in range(1, 29)]
            X_input = np.array(row).reshape(1, -1)

            prob = clf.predict_proba(X_input)[0][1]
            pred = 1 if prob >= thresh else 0

            if pred == 1:
                st.markdown(f"""
                <div class="pred-fraud">
                    <div class="pred-title" style="color:#f85149">⚠️ FRAUD DETECTED</div>
                    <div class="pred-prob" style="color:#f85149">{prob*100:.1f}%</div>
                    <div class="pred-desc">Fraud probability · threshold {thresh:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-legit">
                    <div class="pred-title" style="color:#3fb950">✅ LEGITIMATE</div>
                    <div class="pred-prob" style="color:#3fb950">{(1-prob)*100:.1f}%</div>
                    <div class="pred-desc">Legit probability · threshold {thresh:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Probability Gauge</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 1.2))
            fig.patch.set_facecolor('#161b22'); ax.set_facecolor('#161b22')
            ax.barh([0], [1], color='#21262d', height=0.5, edgecolor='none')
            bar_color = '#f85149' if prob > thresh else '#3fb950'
            ax.barh([0], [prob], color=bar_color, height=0.5, edgecolor='none')
            ax.axvline(thresh, color='#ffa657', lw=2, linestyle='--', label=f'Threshold {thresh}')
            ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.5)
            ax.axis('off')
            ax.text(prob, 0.28, f'{prob:.3f}', ha='center', color='#e6edf3', fontsize=10, fontweight='700')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        else:
            st.markdown("""
            <div style='background:#161b22; border:1px dashed #30363d; border-radius:12px;
                        padding:3rem; text-align:center; color:#7d8590;'>
                <div style='font-size:2.5rem; margin-bottom:0.5rem'>🔍</div>
                <div style='font-weight:600; color:#e6edf3; margin-bottom:0.3rem'>Ready to Predict</div>
                <div style='font-size:0.85rem'>Set transaction parameters and click Predict</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Batch Prediction (Random Samples)</div>', unsafe_allow_html=True)
        if st.button("🎲 Run 10 Random Samples"):
            sample = df.sample(10, random_state=np.random.randint(0, 1000))
            feat_cols = ["scaled_time", "scaled_amount"] + [f"V{i}" for i in range(1, 29)]
            X_sample = sample[feat_cols].values
            probs = clf.predict_proba(X_sample)[:, 1]
            preds = (probs >= thresh).astype(int)
            result_df = pd.DataFrame({
                "Amount ($)": sample["Amount"].values.round(2),
                "True Label": ["🚨 Fraud" if c else "✅ Legit" for c in sample["Class"].values],
                "Predicted" : ["🚨 Fraud" if p else "✅ Legit" for p in preds],
                "Fraud Prob": [f"{p:.4f}" for p in probs],
                "Correct?"  : ["✓" if t == p else "✗" for t, p in zip(sample["Class"].values, preds)],
            })
            st.dataframe(result_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════
# TAB 4 — Dataset Explorer
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-label">Dataset Preview</div>', unsafe_allow_html=True)
    filter_class = st.radio("Filter by class", ["All", "Legit Only", "Fraud Only"], horizontal=True)
    view_df = df.copy()
    if filter_class == "Legit Only":  view_df = view_df[view_df["Class"] == 0]
    elif filter_class == "Fraud Only": view_df = view_df[view_df["Class"] == 1]

    display_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 11)] + ["Class"]
    st.dataframe(
        view_df[display_cols].head(100).style.applymap(
            lambda v: "color: #f85149; font-weight:600" if v == 1 else "",
            subset=["Class"]
        ),
        use_container_width=True, hide_index=True
    )
    st.caption(f"Showing first 100 of {len(view_df):,} rows · Columns V11–V28 hidden for clarity")

    st.markdown('<div class="section-label">Summary Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df[["Time", "Amount"] + [f"V{i}" for i in range(1,8)]].describe().round(3),
                 use_container_width=True)