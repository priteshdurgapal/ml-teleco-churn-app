"""
Trains all 5 classification models on the Telco Churn dataset and exports:
  - model/*.joblib          -> each trained model
  - model/scaler.joblib     -> fitted StandardScaler
  - model/feature_columns.joblib -> exact column order after one-hot encoding (needed at inference)
  - model/metrics.joblib    -> dict of evaluation metrics per model (for the Streamlit app to display)
  - test_data.csv           -> raw-format (unencoded) held-out test rows + true Churn label, for demoing
                               predictions in the Streamlit app
"""
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                              recall_score, f1_score, matthews_corrcoef,
                              confusion_matrix, roc_curve)

RANDOM_STATE = 42
os.makedirs('model', exist_ok=True)

# ---------------------------------------------------------------
# Load & clean (mirrors the notebook)
# ---------------------------------------------------------------
df_raw = pd.read_csv('telco_data.csv')
df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')
df_raw['TotalCharges'] = df_raw['TotalCharges'].fillna(0)

df = df_raw.drop(columns=['customerID']).copy()
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

target = 'Churn'
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_cols = [c for c in df.columns if c not in numeric_cols + [target]]

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
feature_columns = [c for c in df_encoded.columns if c != target]

X = df_encoded[feature_columns]
y = df_encoded[target]

# Keep index alignment so we can pull the matching RAW rows for test_data.csv
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# Train all 5 models
# ---------------------------------------------------------------
models = {
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'decision_tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'knn': KNeighborsClassifier(n_neighbors=5),
    'naive_bayes': GaussianNB(),
    'random_forest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

display_names = {
    'logistic_regression': 'Logistic Regression',
    'decision_tree': 'Decision Tree',
    'knn': 'K-Nearest Neighbors',
    'naive_bayes': 'Naive Bayes (Gaussian)',
    'random_forest': 'Random Forest',
}

metrics_all = {}

for key, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics_all[display_names[key]] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC Score': roc_auc_score(y_test, y_proba),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'MCC Score': matthews_corrcoef(y_test, y_pred),
    }

    joblib.dump(model, f'model/{key}.joblib')
    print(f"Saved model/{key}.joblib")

joblib.dump(scaler, 'model/scaler.joblib')
joblib.dump(feature_columns, 'model/feature_columns.joblib')
joblib.dump({'numeric_cols': numeric_cols, 'categorical_cols': categorical_cols}, 'model/column_types.joblib')
joblib.dump(metrics_all, 'model/metrics.joblib')

print("\nSaved scaler, feature_columns, column_types, metrics.")

# ---------------------------------------------------------------
# Export raw-format test_data.csv (original columns, for the app to demo on)
# ---------------------------------------------------------------
test_raw = df_raw.loc[X_test.index].copy()  # original, unencoded columns incl. customerID
test_raw.to_csv('test_data.csv', index=False)
print(f"\nSaved test_data.csv with {len(test_raw)} rows (raw/unencoded format).")

print("\n=== Metrics summary ===")
print(pd.DataFrame(metrics_all).T.round(4))
