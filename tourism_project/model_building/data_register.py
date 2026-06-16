
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os
from google.colab import userdata

# IMPORTANT: Replace <YOUR_HF_USERNAME> with your Hugging Face username
repo_id = "revathisrihari/tourism-dataset"
repo_type = "dataset"

# Get Hugging Face token from Colab secrets
hf_token = userdata.get("HF_TOKEN")

# Initialize API client
api = HfApi(token=hf_token)

# Step 1: Check if the space exists and create it if not
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=hf_token)
    print(f"Space '{repo_id}' created.") # Corrected line: Removed the newline from the f-string

# Step 2: Upload the data folder to the Hugging Face Hub
print(f"Uploading data from 'tourism_project/data' to '{repo_id}'...")
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
    token=hf_token,
    commit_message="Upload tourism.csv"
)
print("Data upload complete!")
