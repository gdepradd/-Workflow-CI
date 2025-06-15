import pandas as pd
import os
import mlflow
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Setup
random_state = 42
test_size = 0.2
output_dir = "preprocessing"
input_path = "dataset_smote.csv"

# Tracking URI
if os.environ.get("DAGSHUB_USERNAME") and os.environ.get("DAGSHUB_TOKEN"):
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USERNAME"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]
    mlflow.set_tracking_uri("https://dagshub.com/gdepradd/mlflow-diabetic-project.mlflow")
    use_remote = True
else:
    os.makedirs("mlruns", exist_ok=True)
    mlflow.set_tracking_uri("file:./mlruns")
    use_remote = False

mlflow.set_experiment("mlflow-diabetic-project")

def preprocess_and_log():
    with mlflow.start_run(run_name="Preprocessing_Diabetes_SMOTE", nested=True):

        df = pd.read_csv(input_path)

        selected_columns = ['Age', 'Sex', 'GenHlth', 'MentHlth', 'PhysHlth',
                            'BMI', 'HvyAlcoholConsump', 'HighChol', 'PhysActivity', 'Diabetes_binary']
        df = df[selected_columns]

        X = df.drop("Diabetes_binary", axis=1)
        y = df["Diabetes_binary"]

        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled["Diabetes_binary"] = y_resampled

        os.makedirs(output_dir, exist_ok=True)

        cleaned_path = os.path.join(output_dir, "dataset_smote.csv")
        train_path = os.path.join(output_dir, "train_data.csv")
        test_path = os.path.join(output_dir, "test_data.csv")

        df_resampled.to_csv(cleaned_path, index=False)

        train_df, test_df = train_test_split(
            df_resampled, test_size=test_size, random_state=random_state, stratify=y_resampled
        )
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        # Log parameters
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("output_dir", output_dir)

        # Log metrics
        mlflow.log_metric("rows_train", train_df.shape[0])
        mlflow.log_metric("rows_test", test_df.shape[0])
        mlflow.log_metric("total_rows", df_resampled.shape[0])

        # Log artifacts
        mlflow.log_artifact(cleaned_path)
        mlflow.log_artifact(train_path)
        mlflow.log_artifact(test_path)

        print(f"✅ Preprocessing selesai. File disimpan di: {output_dir}")
        if use_remote:
            print("📍 Tracking MLflow di DagsHub aktif.")
        else:
            print("📍 Tracking MLflow tersimpan secara lokal di ./mlruns")

if __name__ == "__main__":
    preprocess_and_log()
