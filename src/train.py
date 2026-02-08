# src/train.py
import pandas as pd
import joblib
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import fbeta_score, recall_score, accuracy_score
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
import mlflow

# --- CONFIGURATION ---
HF_USERNAME = os.getenv("HF_USERNAME", "iStillWaters")
DATASET_REPO_NAME = os.getenv("DATASET_REPO_NAME", "auto_predictive_maintenance_data")
MODEL_REPO_NAME = os.getenv("MODEL_REPO_NAME", "auto_predictive_maintenance_model")
HF_TOKEN = os.getenv("HF_TOKEN")

# --- CONSTRUCT REPO IDs ---
# Source: Where we get the data from
DATA_REPO_ID = f"{HF_USERNAME}/{DATASET_REPO_NAME}"

# Destination: Where we save the model to
MODEL_REPO_ID = f"{HF_USERNAME}/{MODEL_REPO_NAME}"

def train_model():
    print("Starting Model Training...")
    
    # Check Token
    if not HF_TOKEN:
        raise ValueError("❌ HF_TOKEN is missing!")

    # 1. Load Processed Data
    print(f"Downloading processed data from {DATA_REPO_ID}...")
    
    # ADDED token=HF_TOKEN here to ensure access to private/new repos
    train_ds = load_dataset(DATA_REPO_ID, data_files="processed/train.csv", split="train", token=HF_TOKEN)
    test_ds = load_dataset(DATA_REPO_ID, data_files="processed/test.csv", split="train", token=HF_TOKEN)
    
    train_df = train_ds.to_pandas()
    test_df = test_ds.to_pandas()
    
    target = 'Engine Condition'
    X_train = train_df.drop(target, axis=1)
    y_train = train_df[target]
    X_test = test_df.drop(target, axis=1)
    y_test = test_df[target]

    # 2. Train Champion Model (AdaBoost)
    print("Training AdaBoost Classifier...")
    model = AdaBoostClassifier(
        n_estimators=50, 
        learning_rate=1.0, 
        algorithm='SAMME', 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 3. Evaluation
    y_pred = model.predict(X_test)
    
    f2 = fbeta_score(y_test, y_pred, beta=2)
    recall = recall_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n🏆 Model Performance:")
    print(f"   F2-Score: {f2:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   Accuracy: {acc:.4f}")

    # 4. Save Model Locally
    joblib.dump(model, "best_engine_model.pkl")
    print("✅ Model saved locally.")

    # 5. Fetch Scaler (We need to re-upload it to the Model Repo for the App)
    print("Fetching scaler for bundling...")
    # FIXED: Uses DATA_REPO_ID correctly now
    scaler_path = hf_hub_download(
        repo_id=DATA_REPO_ID, 
        filename="artifacts/scaler.joblib", 
        repo_type="dataset",
        token=HF_TOKEN
    )
    
    # 6. Register to HF Model Hub
    api = HfApi(token=HF_TOKEN)
    
    # Create Model Repo if needed
    try:
        api.create_repo(repo_id=MODEL_REPO_ID, repo_type="model", exist_ok=True)
        print(f"✅ Connected to Model Repo: {MODEL_REPO_ID}")
    except Exception as e:
        print(f"⚠️ Model Repo creation warning: {e}")
    
    print(f"Uploading artifacts to {MODEL_REPO_ID}...")
    
    # Upload Model
    api.upload_file(
        path_or_fileobj="best_engine_model.pkl",
        path_in_repo="best_engine_model.pkl",
        repo_id=MODEL_REPO_ID,
        repo_type="model"
    )
    
    # Upload Scaler
    api.upload_file(
        path_or_fileobj=scaler_path,
        path_in_repo="scaler.joblib",
        repo_id=MODEL_REPO_ID,
        repo_type="model"
    )
    
    # Log metrics (Model Card)
    card_text = f"""
---
tags:
- predictive-maintenance
- sklearn
metrics:
- f2_score: {f2}
- recall: {recall}
- accuracy: {acc}
---
# Engine Failure Prediction Model
Champion model trained via Automated MLOps Pipeline.

## Performance
- **Recall (Safety):** {recall:.2%}
- **F2-Score:** {f2:.4f}
- **Accuracy:** {acc:.2%}
"""
    api.upload_file(
        path_or_fileobj=card_text.encode(),
        path_in_repo="README.md",
        repo_id=MODEL_REPO_ID,
        repo_type="model"
    )
    
    print("Model Training & Registration Complete!")

if __name__ == "__main__":
    train_model()