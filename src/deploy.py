# src/deploy.py
import os
from huggingface_hub import HfApi

# --- CONFIGURATION ---
# 1. User Configuration
HF_USERNAME = os.getenv("HF_USERNAME", "iStillWaters")
SPACE_NAME = os.getenv("SPACE_NAME", "Engine-Reliability-Dashboard") 
HF_TOKEN = os.getenv("HF_TOKEN")

# 2. Construct The Space ID
REPO_ID = f"{HF_USERNAME}/{SPACE_NAME}"

def deploy_app():
    print(f"Starting App Deployment to Space: {REPO_ID}...")

    # Safety Check
    if not HF_TOKEN:
        raise ValueError("❌ HF_TOKEN is missing! Check your GitHub Secrets.")

    api = HfApi(token=HF_TOKEN)

    # 3. Create the Space (Must be Docker SDK)
    try:
        print(f"   Connecting to Hugging Face Space: {REPO_ID}")
        api.create_repo(
            repo_id=REPO_ID, 
            repo_type="space", 
            space_sdk="docker", # Crucial for Dockerfile apps
            exist_ok=True
        )
        print("   ✅ Space connection established.")
    except Exception as e:
        print(f"   ⚠️ Note: {e}")

    # 4. Upload Deployment Files
    # These files must exist in the root of your repo when running this script
    files_to_deploy = [
        "app.py", 
        "requirements.txt", 
        "Dockerfile"
    ]
    
    print("Uploading application files...")
    for file in files_to_deploy:
        if os.path.exists(file):
            print(f"   Uploading {file}...")
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=REPO_ID,
                repo_type="space"
            )
        else:
            print(f"   ❌ Error: {file} not found locally! Deployment will be incomplete.")

    print(f"\nDeployment Success! Live App: https://huggingface.co/spaces/{REPO_ID}")

if __name__ == "__main__":
    deploy_app()