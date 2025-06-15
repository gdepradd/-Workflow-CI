import argparse
import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import mlflow

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="preprocessing")
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# MLflow setup
if os.environ.get("DAGSHUB_USERNAME") and os.environ.get("DAGSHUB_TOKEN"):
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USERNAME"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]
    mlflow.set_tracking_uri("https://dagshub.com/gdepradd/mlflow-diabetic-project.mlflow")  # Ganti sesuai proyekmu
    mlflow.set_experiment("mlflow-diabetic-project")
    use_remote = True
else:
    os.makedirs("mlruns", exist_ok=True)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("mlflow-diabetic-project")
    use_remote = False

# Load and preprocess dataset
with mlflow.start_run(run_name="Preprocessing_Diabetes_SMOTE"):
    df = pd.read_csv(args.input_file)

    selected_columns = ['Age', 'Sex', 'GenHlth', 'MentHlth', 'PhysHlth',
                        'BMI', 'HvyAlcoholConsump', 'HighChol', 'PhysActivity', 'Diabetes_binary']
    df = df[selected_columns]

    X = df.drop("Diabetes_binary", axis=1)
    y = df["Diabetes_binary"]

    smote = SMOTE(random_state=args.random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled["Diabetes_binary"] = y_resampled

    os.makedirs(args.output_dir, exist_ok=True)

    cleaned_path = os.path.join(args.output_dir, "dataset_smote.csv")
    train_path = os.path.join(args.output_dir, "train_data.csv")
    test_path = os.path.join(args.output_dir, "test_data.csv")

    df_resampled.to_csv(cleaned_path, index=False)

    train_df, test_df = train_test_split(
        df_resampled, test_size=args.test_size, random_state=args.random_state, stratify=y_resampled
    )
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    mlflow.log_param("input_file", args.input_file)
    mlflow.log_param("output_dir", args.output_dir)
    mlflow.log_param("test_size", args.test_size)
    mlflow.log_param("random_state", args.random_state)

    mlflow.log_metric("rows_train", train_df.shape[0])
    mlflow.log_metric("rows_test", test_df.shape[0])
    mlflow.log_metric("total_rows", df_resampled.shape[0])

    mlflow.log_artifact(cleaned_path)
    mlflow.log_artifact(train_path)
    mlflow.log_artifact(test_path)

    print("✅ Preprocessing selesai. File disimpan di:", args.output_dir)

    if use_remote:
        print("📍 Tracking MLflow di DagsHub aktif.")
    else:
        print("📍 Tracking MLflow tersimpan secara lokal di ./mlruns")

    #Tes
