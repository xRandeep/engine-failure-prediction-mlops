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

# Construct Repo ID
REPO_ID = f"{HF_USERNAME}/{DATASET_REPO_NAME}"

# --- FEATURE DEFINITION ---
# We explicitly list physical sensors to avoid accidental ID/Timestamp leakage.
EXPECTED_FEATURES = [
    'Engine rpm', 
    'Lub oil pressure', 
    'Fuel pressure', 
    'Coolant pressure', 
    'lub oil temp', 
    'Coolant temp'
]
TARGET = 'Engine Condition'

def process_data():
    print("Starting Data Processing...")
    
    if not HF_TOKEN:
        raise ValueError("❌ HF_TOKEN is missing!")

    # 1. Load Raw Data (Replaces your Try/Except block)
    # In MLOps, we want to fail if the Cloud Source is missing, not fallback to local.
    print(f"Downloading raw data from {REPO_ID}...")
    try:
        dataset = load_dataset(
            "csv", 
            data_files=f"hf://datasets/{REPO_ID}/raw/engine_data.csv", 
            split="train", 
            token=HF_TOKEN
        )
        df = dataset.to_pandas()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise e

    # 2. Data Cleaning
    initial_count = len(df)
    df = df.drop_duplicates()
    print(f"Duplicates removed: {initial_count - len(df)}")
    
    # 3. Feature Selection & Splitting
    # Check if target exists
    if TARGET not in df.columns:
        raise ValueError(f"❌ Target column '{TARGET}' not found!")

    # Select Features (X) and Target (y)
    # NOTE: This is safer than your notebook's ".drop()" method
    try:
        X = df[EXPECTED_FEATURES]
    except KeyError as e:
        raise ValueError(f"❌ Missing expected columns in raw CSV: {e}")
        
    y = df[TARGET]
    
    # Stratified Split (Matches your Notebook logic)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Scaling (StandardScaler)
    # Matches your Notebook: Fit on Train, Transform Test
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled_array = scaler.fit_transform(X_train)
    X_test_scaled_array = scaler.transform(X_test)
    
    # Convert back to DataFrame (Matches your Notebook)
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=EXPECTED_FEATURES)
    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=EXPECTED_FEATURES)

    # 5. Reassemble & Save Locally
    os.makedirs("processed_data", exist_ok=True)
    
    # Reset indices (Matches your Notebook - Critical!)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    pd.concat([X_train_scaled, y_train], axis=1).to_csv("processed_data/train.csv", index=False)
    pd.concat([X_test_scaled, y_test], axis=1).to_csv("processed_data/test.csv", index=False)
    
    # Save Scaler (Matches your Notebook)
    joblib.dump(scaler, "processed_data/scaler.joblib")
    print("✅ Processed files saved locally.")

    # 6. Upload to Hugging Face
    api = HfApi(token=HF_TOKEN)
    print(f"Uploading artifacts to {REPO_ID}...")
    
    artifacts = {
        "processed_data/train.csv": "processed/train.csv",
        "processed_data/test.csv": "processed/test.csv",
        "processed_data/scaler.joblib": "artifacts/scaler.joblib"
    }
    
    for local_path, repo_path in artifacts.items():
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="dataset"
            )
            print(f"   uploaded: {repo_path}")
        except Exception as e:
            print(f"   ❌ Failed to upload {repo_path}: {e}")
        
    print("Data processing complete & uploaded!")

if __name__ == "__main__":
    process_data()