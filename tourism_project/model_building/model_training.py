
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from huggingface_hub import HfApi, hf_hub_download, create_repo
from huggingface_hub.utils import RepositoryNotFoundError # Corrected import
import os
from google.colab import userdata
import mlflow
import mlflow.sklearn
import skops.io as sio # Import skops.io

# --- Configuration ---
# Hugging Face Token (from Colab secrets)
hf_token = userdata.get("HF_TOKEN")

# Hugging Face Dataset Repository details
dataset_repo_id = "revathisrihari/tourism-dataset" # This is where your processed data is
dataset_repo_type = "dataset"
train_file = "tourism_train.csv"
test_file = "tourism_test.csv"

# Hugging Face Model Repository details
model_repo_id = "revathisrihari/tourism-model" # New repository for the model
model_repo_type = "model"

# MLflow setup
mlflow_tracking_uri = "mlruns" # Local MLflow tracking server
mlflow_experiment_name = "Tourism_Package_Prediction"

# Allow MLflow filesystem backend
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# --- Step 1: Load the train and test data from the Hugging Face data space ---
print(f"Downloading {train_file} from Hugging Face Hub (repo: {dataset_repo_id})...")
local_train_path = hf_hub_download(
    repo_id=dataset_repo_id,
    repo_type=dataset_repo_type,
    filename=train_file,
    token=hf_token
)
print(f"Training data downloaded to: {local_train_path}")

print(f"Downloading {test_file} from Hugging Face Hub (repo: {dataset_repo_id})...")
local_test_path = hf_hub_download(
    repo_id=dataset_repo_id,
    repo_type=dataset_repo_type,
    filename=test_file,
    token=hf_token
)
print(f"Test data downloaded to: {local_test_path}")

train_df = pd.read_csv(local_train_path)
test_df = pd.read_csv(local_test_path)

print("Train DataFrame head:")
print(train_df.head())
print("Test DataFrame head:")
print(test_df.head())

# Separate features (X) and target (y)
X_train = train_df.drop('ProdTaken', axis=1)
y_train = train_df['ProdTaken']
X_test = test_df.drop('ProdTaken', axis=1)
y_test = test_df['ProdTaken']

# --- Step 2: Preprocessing for Model Training (One-Hot Encoding) ---
categorical_features = X_train.select_dtypes(include=['object']).columns
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns

# Create a column transformer for one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep numerical features as they are
)

# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert to DataFrame with feature names (optional, for better readability)
# Get feature names after one-hot encoding
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = list(ohe_feature_names) + list(numerical_features)

X_train_processed = pd.DataFrame(X_train_processed, columns=all_feature_names)
X_test_processed = pd.DataFrame(X_test_processed, columns=all_feature_names)

print("Processed X_train head:")
print(X_train_processed.head())

# --- Step 3: Define a model and parameters & Tune the model with the defined parameters ---
# For demonstration, we'll use DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)

# Define parameter grid for GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7, 9],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# --- Step 4: Log all the tuned parameters using MLflow ---
# Set MLflow tracking URI
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment(mlflow_experiment_name)

with mlflow.start_run():
    # Log model type
    mlflow.log_param("model_type", "DecisionTreeClassifier")

    # Log the entire parameter grid as a single string to avoid conflicts
    mlflow.log_param("grid_search_params_space", str(param_grid))

    # Perform GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_processed, y_train)

    # Get the best estimator
    best_model = grid_search.best_estimator_

    # Log best parameters and best score directly from grid_search
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_score_roc_auc", grid_search.best_score_)

    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation ROC AUC score: ", grid_search.best_score_)

    # --- Step 5: Evaluate the model performance ---
    y_pred = best_model.predict(X_test_processed)
    y_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print(f"Test ROC AUC: {roc_auc:.4f}")

    # Log evaluation metrics
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1_score", f1)
    mlflow.log_metric("test_roc_auc", roc_auc)

    # --- Step 6: Register the best model in the Hugging Face model hub ---
    # Create directory for model if it doesn't exist
    os.makedirs("tourism_project/model_building/model_output", exist_ok=True)
    model_path = "tourism_project/model_building/model_output/best_decision_tree_model.skops" # Change to .skops
    sio.dump(best_model, model_path) # Save using skops.io.dump
    print(f"Model saved locally to {model_path}")

    # Log the model to MLflow, specifying skops serialization format
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="decision_tree_model",
        registered_model_name="DecisionTreeClassifierTourism",
        signature=mlflow.models.infer_signature(X_train_processed, best_model.predict(X_train_processed)),
        input_example=X_train_processed.head(1).to_dict(orient="records"),
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS # Specify skops serialization
    )

    # Create a new Hugging Face model repo if it doesn't exist
    api = HfApi(token=hf_token)
    try:
        api.repo_info(repo_id=model_repo_id, repo_type=model_repo_type)
        print(f"Model space '{model_repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model space '{model_repo_id}' not found. Creating new space...")
        create_repo(repo_id=model_repo_id, repo_type=model_repo_type, private=False, token=hf_token)
        print(f"Model space '{model_repo_id}' created.")

    # Upload the saved model to Hugging Face Model Hub
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_decision_tree_model.skops", # Change to .skops
        repo_id=model_repo_id,
        repo_type=model_repo_type,
        commit_message="Upload best Decision Tree model (skops format)"
    )
    print(f"Best model uploaded to Hugging Face Model Hub: {model_repo_id}/best_decision_tree_model.skops")

print("Model training, evaluation, and registration complete!")
