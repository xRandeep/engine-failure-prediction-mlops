# src/train.py
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import warnings
import logging
from huggingface_hub import HfApi, hf_hub_download
from datasets import load_dataset

# Modeling Imports
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, fbeta_score, make_scorer

# MLflow
import mlflow
import mlflow.sklearn

# --- CONFIGURATION ---
HF_USERNAME = os.getenv("HF_USERNAME", "iStillWaters")
DATASET_REPO_NAME = os.getenv("DATASET_REPO_NAME", "auto_predictive_maintenance_data")
MODEL_REPO_NAME = os.getenv("MODEL_REPO_NAME", "auto_predictive_maintenance_model")
HF_TOKEN = os.getenv("HF_TOKEN")

# Construct IDs
DATA_REPO_ID = f"{HF_USERNAME}/{DATASET_REPO_NAME}"
MODEL_REPO_ID = f"{HF_USERNAME}/{MODEL_REPO_NAME}"

# Silence Warnings
logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

def train_and_register():
    print("Starting AutoML Training Pipeline...")
    
    if not HF_TOKEN:
        raise ValueError("❌ HF_TOKEN is missing!")

    # ==========================================
    # 1. LOAD DATA 
    # ==========================================
    print(f"Downloading processed data from {DATA_REPO_ID}...")
    try:
        dataset = load_dataset(
            "csv",
            data_files={
                "train": f"hf://datasets/{DATA_REPO_ID}/processed/train.csv",
                "test":  f"hf://datasets/{DATA_REPO_ID}/processed/test.csv"
            },
            token=HF_TOKEN
        )
    except Exception as e:
        print(f"❌ Failed to load data. Error: {e}")
        raise e
    
    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()
    
    TARGET = 'Engine Condition'
    
    # Split Features/Target
    # Note: Data is ALREADY scaled from process_data.py
    X_train = train_df.drop(TARGET, axis=1)
    y_train = train_df[TARGET]
    X_test = test_df.drop(TARGET, axis=1)
    y_test = test_df[TARGET]
    
    print(f"✅ Data Loaded. Train Shape: {X_train.shape}")

    # ==========================================
    # 2. DEFINE MODELS & PARAMS
    # ==========================================
    model_params = {
        "Decision_Tree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {"max_depth": [5, 10, None], "min_samples_leaf": [1, 5]}
        },
        "Random_Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {"n_estimators": [50, 100], "max_depth": [10, 20]}
        },
        "Bagging": {
            "model": BaggingClassifier(random_state=42),
            "params": {"n_estimators": [10, 50]}
        },
        "AdaBoost": {
            "model": AdaBoostClassifier(random_state=42),
            "params": {"n_estimators": [50, 100], "learning_rate": [0.1, 1.0]}
        },
        "Gradient_Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}
        },
        "XGBoost": {
            "model": XGBClassifier(eval_metric='logloss', random_state=42),
            "params": {"n_estimators": [50, 100], "max_depth": [3, 6]}
        }
    }

    # ==========================================
    # 3. TRAINING LOOP (GridSearch + MLflow)
    # ==========================================
    # Setup MLflow
    mlflow.set_experiment("Engine_Failure_Prediction_Pipeline")
    
    all_results = []
    best_overall_f2 = 0
    best_model_name = ""
    best_overall_model = None
    
    # Custom Scorer: F2 Score (Recall Weighted)
    ftwo_scorer = make_scorer(fbeta_score, greater_is_better=True, beta=2)

    print(f"{'Model':<20} | {'F2-Score':<10} | {'Recall':<10}")
    print("-" * 45)

    for name, mp in model_params.items():
        with mlflow.start_run(run_name=name):
            # Grid Search
            clf = GridSearchCV(mp['model'], mp['params'], cv=3, scoring=ftwo_scorer, refit=True, n_jobs=-1)
            clf.fit(X_train, y_train)

            best_model = clf.best_estimator_
            best_params = clf.best_params_

            # Evaluate
            y_pred = best_model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f2 = fbeta_score(y_test, y_pred, beta=2)

            # Log to MLflow
            mlflow.log_params(best_params)
            mlflow.log_metrics({"accuracy": acc, "f1": f1, "recall": rec, "f2_score": f2})
            mlflow.sklearn.log_model(best_model, name)

            # Store results
            all_results.append({
                "Model": name, "F2_Score": f2, "Recall": rec, 
                "Accuracy": acc, "F1_Score": f1
            })

            print(f"{name:<20} | {f2:.4f}     | {rec:.4f}")

            # Track Winner
            if f2 > best_overall_f2:
                best_overall_f2 = f2
                best_overall_model = best_model
                best_model_name = name

    print("-" * 45)
    print(f"CHAMPION MODEL: {best_model_name} (F2={best_overall_f2:.4f})")

    # ==========================================
    # 4. GENERATE COMPARISON PLOT
    # ==========================================
    results_df = pd.DataFrame(all_results).sort_values(by="F2_Score", ascending=False)
    
    # Save plot to file (headless environment)
    plt.figure(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.2
    
    plt.bar(x - 1.5*width, results_df['F2_Score'], width, label='F2', color='#1f77b4')
    plt.bar(x - 0.5*width, results_df['Recall'], width, label='Recall', color='#2ca02c')
    plt.bar(x + 0.5*width, results_df['F1_Score'], width, label='F1', color='#ff7f0e')
    plt.bar(x + 1.5*width, results_df['Accuracy'], width, label='Acc', color='#d62728')
    
    plt.xticks(x, results_df['Model'], rotation=15)
    plt.title('Model Leaderboard')
    plt.legend()
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    print("Comparison plot saved.")

    # ==========================================
    # 5. REGISTER ARTIFACTS TO HUGGING FACE
    # ==========================================
    
    # A. Save Champion Model Locally
    joblib.dump(best_overall_model, "best_engine_model.pkl")
    
    # B. Fetch Scaler (Crucial for Inference)
    print("Fetching scaler for bundling...")
    scaler_path = hf_hub_download(
        repo_id=DATA_REPO_ID, 
        filename="artifacts/scaler.joblib", 
        repo_type="dataset",
        token=HF_TOKEN
    )

    # C. Upload Everything to Model Hub
    api = HfApi(token=HF_TOKEN)
    
    try:
        api.create_repo(repo_id=MODEL_REPO_ID, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"⚠️ Model Repo note: {e}")

    print(f"Uploading artifacts to {MODEL_REPO_ID}...")
    
    # Upload List
    upload_files = {
        "best_engine_model.pkl": "best_engine_model.pkl",
        scaler_path: "scaler.joblib",      # The downloaded scaler
        "model_comparison.png": "model_comparison.png" # The plot
    }

    for local, remote in upload_files.items():
        if local:
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=remote,
                repo_id=MODEL_REPO_ID,
                repo_type="model"
            )

    # D. Create & Upload Model Card
    card_content = f"""
---
tags:
- predictive-maintenance
- sklearn
- {best_model_name}
metrics:
- f2_score: {best_overall_f2}
---
# 🚛 Engine Failure Prediction Model

## 🏆 Champion Model: **{best_model_name}**

This model was automatically selected from a pool of candidates based on the **F2-Score**, which prioritizes Recall (minimizing false negatives) to ensure safety.

### 📊 Performance Metrics
| Metric | Score | Explanation |
| :--- | :--- | :--- |
| **F2-Score** | **{best_overall_f2:.4f}** | Primary selection metric (Recall-weighted). |
| **Recall** | **{results_df.iloc[0]['Recall']:.4f}** | Ability to catch failures. |
| **Accuracy** | **{results_df.iloc[0]['Accuracy']:.4f}** | Overall correctness. |

### 🔍 Model Leaderboard
![Model Comparison](model_comparison.png)

### ⚙️ Pipeline Details
- **Training Date:** {pd.Timestamp.now()}
- **Scaler:** Standard Scaler (Bundled)
- **Deployment:** [Live App Link](https://huggingface.co/spaces/{HF_USERNAME}/Engine-Reliability-Dashboard)
"""
    
    api.upload_file(
        path_or_fileobj=card_content.encode(),
        path_in_repo="README.md",
        repo_id=MODEL_REPO_ID,
        repo_type="model"
    )

    print("Pipeline Complete! Champion Model Registered.")

if __name__ == "__main__":
    train_and_register()