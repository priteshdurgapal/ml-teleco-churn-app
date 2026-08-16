"""
Telco Customer Churn — Multi-Model Metrics
Models are pre-trained and saved to disk (see train_and_export.py).

On load: shows ONLY the pre-computed metrics table from the original training run.
On upload: additionally shows a model-selection dropdown, and for the selected
model — metrics, confusion matrix, classification report — plus an all-models
metrics table for the uploaded data. The pre-computed table stays at the bottom.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                              recall_score, f1_score, matthews_corrcoef,
                              confusion_matrix, ConfusionMatrixDisplay,
                              classification_report)

st.set_page_config(page_title="Telco Churn — Model Metrics", layout="wide")

# ---------------------------------------------------------------
# Load trained model artifacts (cached — loads once per session)
# ---------------------------------------------------------------
MODEL_FILES = {
    'Logistic Regression': 'model/logistic_regression.joblib',
    'Decision Tree': 'model/decision_tree.joblib',
    'K-Nearest Neighbors': 'model/knn.joblib',
    'Naive Bayes (Gaussian)': 'model/naive_bayes.joblib',
    'Random Forest': 'model/random_forest.joblib',
}

@st.cache_resource
def load_artifacts():
    models = {name: joblib.load(path) for name, path in MODEL_FILES.items()}
    scaler = joblib.load('model/scaler.joblib')
    feature_columns = joblib.load('model/feature_columns.joblib')
    column_types = joblib.load('model/column_types.joblib')
    precomputed_metrics = joblib.load('model/metrics.joblib')
    return models, scaler, feature_columns, column_types, precomputed_metrics

models, scaler, feature_columns, column_types, precomputed_metrics = load_artifacts()
numeric_cols = column_types['numeric_cols']
categorical_cols = column_types['categorical_cols']
REQUIRED_RAW_COLS = set(numeric_cols) | set(categorical_cols) | {'Churn'}


def preprocess(df_raw_subset: pd.DataFrame) -> np.ndarray:
    """Apply the exact same encoding + scaling pipeline used at training time."""
    df = df_raw_subset.drop(columns=['customerID', 'Churn'], errors='ignore').copy()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
    return scaler.transform(df_encoded)


@st.cache_data(show_spinner=False)
def compute_all_metrics(df_raw: pd.DataFrame, _models_key: str):
    X_scaled = preprocess(df_raw)
    y_true = df_raw['Churn'].map({'Yes': 1, 'No': 0})
    out = {}
    for name, model in models.items():
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]
        out[name] = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'AUC': roc_auc_score(y_true, y_proba),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'MCC': matthews_corrcoef(y_true, y_pred),
        }
    return out


def to_display_df(metrics_dict):
    df = pd.DataFrame(metrics_dict).T.round(4)
    df.insert(0, 'ML Model Name', df.index)
    df = df.reset_index(drop=True)
    return df


def get_col(df: pd.DataFrame, keyword: str):
    return next((c for c in df.columns if keyword in c), None)


def compute_overall_winner(df: pd.DataFrame):
    """Rank models by AUC + F1 + MCC combined (most reliable trio under class imbalance).
    Returns (winner_name, winner_idx, recall_leader_name_or_None)."""
    auc_col, f1_col, mcc_col = get_col(df, 'AUC'), get_col(df, 'F1'), get_col(df, 'MCC')
    if not all([auc_col, f1_col, mcc_col]):
        return None, None, None

    ranks = df[[auc_col, f1_col, mcc_col]].rank(ascending=False)
    avg_rank = ranks.mean(axis=1)
    winner_idx = avg_rank.idxmin()
    winner_name = df.loc[winner_idx, 'ML Model Name']

    recall_col = get_col(df, 'Recall')
    recall_leader_name = None
    if recall_col is not None:
        recall_idx = df[recall_col].idxmax()
        if recall_idx != winner_idx:
            recall_leader_name = df.loc[recall_idx, 'ML Model Name']

    return winner_name, winner_idx, recall_leader_name


def show_metrics_table_with_best(df: pd.DataFrame):
    """Render a metrics table with the best model (by F1) highlighted, computed live from df."""
    f1_col = next((c for c in df.columns if 'F1' in c), None)
    if f1_col is None:
        st.dataframe(df, use_container_width=True, hide_index=True)
        return

    best_idx = df[f1_col].idxmax()
    best_name = df.loc[best_idx, 'ML Model Name']
    best_score = df.loc[best_idx, f1_col]

    def highlight_best(row):
        color = 'background-color: #c6f6c6' if row.name == best_idx else ''
        return [color] * len(row)

    st.dataframe(
        df.style.apply(highlight_best, axis=1),
        use_container_width=True,
        hide_index=True,
    )


def show_overall_winner(df: pd.DataFrame):
    """Dynamic 'Overall Winner' narrative — same logic as the README, computed live from df."""
    winner_name, winner_idx, recall_leader_name = compute_overall_winner(df)
    if winner_name is None:
        return
    st.markdown("**Overall Winner for this dataset**")
    st.success(
        f"**{winner_name}** — leads across AUC, F1, and MCC combined, the metrics most "
        f"reliable under class imbalance (plain Accuracy can be misleading here)."
    )



# ---------------------------------------------------------------
# Sidebar — download bundled test data, upload your own
# ---------------------------------------------------------------
st.sidebar.subheader("Download Test Data")

with open('test_data.csv', 'rb') as f:
    st.sidebar.download_button(
        label="Download test_data.csv",
        data=f,
        file_name="test_data.csv",
        mime="text/csv",
    )

st.sidebar.subheader("Upload Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload test data (CSV)",
    type="csv",
    help="Upload a CSV with the same columns as the original dataset (including the "
         "true 'Churn' column) to see metrics computed on your own data. Only test "
         "data should be uploaded, not the full training set.",
)

test_raw = None
if uploaded_file is not None:
    try:
        user_df = pd.read_csv(uploaded_file)
        missing = REQUIRED_RAW_COLS - set(user_df.columns)
        if missing:
            st.sidebar.error(
                f"Uploaded CSV is missing required column(s): {', '.join(sorted(missing))}."
            )
        else:
            if 'customerID' not in user_df.columns:
                user_df.insert(0, 'customerID', [f"row_{i}" for i in range(len(user_df))])
            test_raw = user_df
            st.sidebar.success(f"Loaded {len(test_raw)} rows.")
    except Exception as e:
        st.sidebar.error(f"Couldn't read that CSV ({e}).")

# Model dropdown only appears once data is uploaded
selected_model_name = None
if test_raw is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model")
    selected_model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))

# ---------------------------------------------------------------
# Main page
# ---------------------------------------------------------------
st.title("Telco Customer Churn — Classification Dashboard")
st.markdown("This project builds and compares multiple binary classification models to predict whether a customer will churn (Churn: Yes/No), based on their account, service, and billing information.")
if test_raw is not None:
    # --- Data preview — only shown once a file has been uploaded ---
    st.subheader("Data Preview")
    st.caption(f"First rows of uploaded file ({len(test_raw)} rows total)")
    st.dataframe(test_raw.head(9), use_container_width=True)
    st.markdown("---")

    selected_model = models[selected_model_name]
    X_scaled = preprocess(test_raw)
    y_true = test_raw['Churn'].map({'Yes': 1, 'No': 0})
    y_pred = selected_model.predict(X_scaled)
    y_proba = selected_model.predict_proba(X_scaled)[:, 1]

    # Compute all-models metrics now so we can flag the overall winner in the header below
    uploaded_result = compute_all_metrics(test_raw, _models_key=",".join(models.keys()))
    uploaded_df = to_display_df(uploaded_result)
    uploaded_winner_name, _, _ = compute_overall_winner(uploaded_df)

    # --- Selected model: metrics, confusion matrix, classification report ---
    st.subheader(f"Performance: {selected_model_name}")
    st.caption(f"Computed on uploaded file ({len(test_raw)} rows)")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2f}")
    m2.metric("AUC Score", f"{roc_auc_score(y_true, y_proba):.2f}")
    m3.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.2f}")
    m4.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.2f}")
    m5.metric("F1 Score", f"{f1_score(y_true, y_pred, zero_division=0):.2f}")
    m6.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.2f}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Confusion Matrix**")
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4.5))
        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot(ax=ax_cm, cmap='Blues', colorbar=False)
        st.pyplot(fig_cm)

    with c2:
        st.markdown("**Classification Report**")
        report_dict = classification_report(
            y_true, y_pred, target_names=['No Churn', 'Churn'],
            output_dict=True, zero_division=0,
        )
        report_df = pd.DataFrame(report_dict).T.round(4)
        st.dataframe(report_df, use_container_width=True)

    st.markdown("---")

    # --- All-models metrics table for the uploaded data ---
    st.subheader("Metrics on Uploaded Test Data")
    show_metrics_table_with_best(uploaded_df)
    show_overall_winner(uploaded_df)

    st.markdown("---")
else:
    st.info("Download the sample CSV from the sidebar, then upload it back to see the model predictions.")    

# --- Pre-computed metrics: full section when no upload, collapsible once data is uploaded ---
precomputed_df = to_display_df(precomputed_metrics)

if test_raw is None:
    st.subheader("Pre-computed Metrics")
    show_metrics_table_with_best(precomputed_df)
    show_overall_winner(precomputed_df)
else:
    st.subheader("Pre-computed Metrics")
    with st.expander("(Expand)", expanded=False):
        show_metrics_table_with_best(precomputed_df)
        show_overall_winner(precomputed_df)