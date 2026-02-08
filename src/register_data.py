# src/register_data.py
import pandas as pd
from huggingface_hub import HfApi
import os

# --- CONFIGURATION ---
# 1. Get Username (Default to 'iStillWaters' if not set in env)
HF_USERNAME = os.getenv("HF_USERNAME", "iStillWaters")

# 2. Get Repo Name (Default to 'auto_predictive_maintenance_data')
DATASET_REPO_NAME = os.getenv("DATASET_REPO_NAME", "auto_predictive_maintenance_data")

# 3. CONSTRUCT THE REPO ID
REPO_ID = f"{HF_USERNAME}/{DATASET_REPO_NAME}"

# 4. Get Token
HF_TOKEN = os.getenv("HF_TOKEN")

DATA_FILE = "data/engine_data.csv"

def register_data():
    print("Starting Data Registration...")
    
    # Check if Token exists before proceeding
    if not HF_TOKEN:
        raise ValueError("❌ HF_TOKEN is missing! Please set the environment variable.")

    # 1. Load Local Data
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"❌ File not found: {DATA_FILE}")
    
    df = pd.read_csv(DATA_FILE)
    print(f"✅ Loaded {len(df)} rows from {DATA_FILE}")

    # 2. Create/Connect to Repo
    # We pass the token explicitly here
    api = HfApi(token=HF_TOKEN)
    
    try:
        # Now REPO_ID is correctly defined
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
        print(f"✅ Connected to HF Repo: {REPO_ID}")
    except Exception as e:
        print(f"⚠️ Repo creation warning (might already exist): {e}")

    # 3. Upload Raw Data
    print(f"Uploading raw data to {REPO_ID}...")
    api.upload_file(
        path_or_fileobj=DATA_FILE,
        path_in_repo="raw/engine_data.csv",
        repo_id=REPO_ID,  # Uses the constructed ID
        repo_type="dataset"
    )
    print("🎉 Raw data registered successfully!")

if __name__ == "__main__":
    register_data()