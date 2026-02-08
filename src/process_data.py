# src/process_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
from huggingface_hub import HfApi
import joblib
import os

# --- CONFIGURATION ---
HF_USERNAME = os.getenv("HF_USERNAME", "iStillWaters")
DATASET_REPO_NAME = os.getenv("DATASET_REPO_NAME", "auto_predictive_maintenance_data")
HF_TOKEN = os.getenv("HF_TOKEN")

# 1. CONSTRUCT REPO_ID (The Missing Piece)
REPO_ID = f"{HF_USERNAME}/{DATASET_REPO_NAME}"

def process_data():
    print("Starting Data Processing...")
    
    # Check Token
    if not HF_TOKEN:
        raise ValueError("❌ HF_TOKEN is missing!")

    # 1. Load Raw Data from HF
    # IMPORTANT: We added `token=HF_TOKEN` here. 
    # Without this, the script cannot read the data you just uploaded.
    print(f"Downloading from {REPO_ID}...")
    dataset = load_dataset(REPO_ID, data_files="raw/engine_data.csv", split="train", token=HF_TOKEN)
    
    df = dataset.to_pandas()
    print(f"✅ Loaded {len(df)} rows from Hugging Face.")

    # 2. Data Cleaning
    initial_count = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_count - len(df)} duplicates.")
    
    # 3. Feature Selection & Splitting
    target = 'Engine Condition'
    if target not in df.columns:
         raise ValueError(f"❌ Target column '{target}' not found in dataset!")

    X = df.drop(target, axis=1)
    y = df[target]
    
    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Scaling (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # 5. Save Artifacts Locally
    os.makedirs("processed_data", exist_ok=True)
    
    # Reset indices to avoid concat issues
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    pd.concat([X_train_scaled, y_train], axis=1).to_csv("processed_data/train.csv", index=False)
    pd.concat([X_test_scaled, y_test], axis=1).to_csv("processed_data/test.csv", index=False)
    joblib.dump(scaler, "processed_data/scaler.joblib")
    
    print("✅ Processed files saved locally.")

    # 6. Upload to Hugging Face
    api = HfApi(token=HF_TOKEN)
    print("Uploading processed artifacts...")
    
    artifacts = {
        "processed_data/train.csv": "processed/train.csv",
        "processed_data/test.csv": "processed/test.csv",
        "processed_data/scaler.joblib": "artifacts/scaler.joblib"
    }
    
    for local_path, repo_path in artifacts.items():
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="dataset"
        )
        
    print("Data processing complete & uploaded!")

if __name__ == "__main__":
    process_data()