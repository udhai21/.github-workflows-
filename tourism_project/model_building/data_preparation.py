
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download
import os
from google.colab import userdata

# Get Hugging Face token from Colab secrets
hf_token = userdata.get("HF_TOKEN")

# Define Hugging Face repository details
repo_id = "revathisrihari/tourism-dataset" # Ensure this matches the uploaded repo_id
repo_type = "dataset"
file_path = "tourism.csv"

# Step 1: Load the dataset directly from the Hugging Face data space.
print(f"Downloading {file_path} from Hugging Face Hub (repo: {repo_id})...")
local_file_path = hf_hub_download(
    repo_id=repo_id,
    repo_type=repo_type,
    filename=file_path,
    token=hf_token
)
print(f"Dataset downloaded to: {local_file_path}")

df = pd.read_csv(local_file_path)
print("Original DataFrame head:")
df.head()

# Step 2: Perform data cleaning and remove any unnecessary columns.
# Remove 'Unnamed: 0' and 'CustomerID' as they are not needed for modeling
df_cleaned = df.drop(columns=['Unnamed: 0', 'CustomerID'], errors='ignore')

# Handle missing values: Fill numerical NaNs with median, categorical NaNs with mode
for col in df_cleaned.columns:
    if df_cleaned[col].dtype == 'object': # Categorical columns
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
    else: # Numerical columns
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

print("Cleaned DataFrame head:")
df_cleaned.head()
print("Missing values after cleaning:")
print(df_cleaned.isnull().sum())

# Step 3: Split the cleaned dataset into training and testing sets, and save them locally.
X = df_cleaned.drop('ProdTaken', axis=1)
y = df_cleaned['ProdTaken']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Recombine for saving, ensuring 'ProdTaken' is present
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Create a directory to save processed data if it doesn't exist
output_dir = "tourism_project/data"
os.makedirs(output_dir, exist_ok=True)

train_path = os.path.join(output_dir, "tourism_train.csv")
test_path = os.path.join(output_dir, "tourism_test.csv")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"Training data saved to {train_path}")
print(f"Test data saved to {test_path}")

# Step 4: Upload the resulting train and test datasets back to the Hugging Face data space.
print("Uploading processed data to Hugging Face Hub...")
api = HfApi(token=hf_token)
api.upload_file(
    path_or_fileobj=train_path,
    path_in_repo="tourism_train.csv",
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message="Upload processed training data"
)
api.upload_file(
    path_or_fileobj=test_path,
    path_in_repo="tourism_test.csv",
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message="Upload processed test data"
)
print("Processed train and test data uploaded to Hugging Face Hub.")
