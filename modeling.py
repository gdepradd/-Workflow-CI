import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import mlflow
import mlflow.sklearn
import dagshub

# Import fungsi tuning dari file terpisah
from modelling_tuning import tune_model

# Inisialisasi koneksi ke DagsHub
dagshub.init(repo_owner='gdepradd', repo_name='mlflow-diabetic-project', mlflow=True)

# Set MLflow tracking URI ke DagsHub
mlflow.set_tracking_uri("https://dagshub.com/gdepradd/mlflow-diabetic-project.mlflow")

# Set nama eksperimen MLflow
mlflow.set_experiment("mlflow-diabetic-project")

# Memuat dataset
data_path = "../preprocessing/preprocessing/dataset_smote.csv"
df = pd.read_csv(data_path)

# Memisahkan fitur dan target
X = df.drop("Diabetes_binary", axis=1)
y = df["Diabetes_binary"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Nonaktifkan autolog agar kita pakai manual logging
mlflow.sklearn.autolog(disable=True)

# ================================
# 1. Basic Model Training
# ================================
print("\n=== Training Basic RandomForestClassifier Model ===")
with mlflow.start_run(run_name="basic_rf_model"):
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC: {roc_auc}")

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 150)
    mlflow.log_param("max_depth", 10)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.log_param("data_path", data_path)
    mlflow.log_param("train_size", X_train.shape[0])
    mlflow.log_param("test_size", X_test.shape[0])
    mlflow.log_param("features_count", X_train.shape[1])

    mlflow.sklearn.log_model(model, "model")

# ================================
# 2. Tuning Model
# ================================
print("\n=== Training Tuned RandomForestClassifier Model ===")
tuned_model = tune_model(X_train, y_train, X_test, y_test)

# ================================
# Simpan URL DagsHub
# ================================
with open("DagsHub.txt", "w") as f:
    f.write("https://dagshub.com/gdepradd/mlflow-diabetic-project")

print("\n✅ Pelatihan selesai! Data telah dicatat di:")
print("https://dagshub.com/gdepradd/mlflow-diabetic-project")
