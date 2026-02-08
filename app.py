import streamlit as st
import pandas as pd
import joblib
import os
import sklearn  # Explicit import prevents some joblib errors
from huggingface_hub import hf_hub_download

# --- 1. PAGE CONFIG (MUST BE FIRST) ---
st.set_page_config(page_title="Engine Failure Prediction", page_icon="🚛")

# --- CONFIGURATION ---
HF_USERNAME = os.getenv("HF_USERNAME", "iStillWaters")
MODEL_REPO_NAME = os.getenv("MODEL_REPO_NAME", "auto_predictive_maintenance_model")
MODEL_REPO_ID = f"{HF_USERNAME}/{MODEL_REPO_NAME}"

MODEL_FILENAME = "best_engine_model.pkl"
SCALER_FILENAME = "scaler.joblib"

# CRITICAL: Must match the order in process_data.py exactly!
EXPECTED_FEATURES = [
    'Engine rpm', 
    'Lub oil pressure', 
    'Fuel pressure', 
    'Coolant pressure', 
    'lub oil temp', 
    'Coolant temp'
]

# --- LOAD MODEL & SCALER ---
@st.cache_resource
def load_artifacts():
    print(f"Loading artifacts from {MODEL_REPO_ID}...")
    try:
        # Download Model
        model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
        model = joblib.load(model_path)
        
        # Download Scaler
        scaler_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=SCALER_FILENAME)
        scaler = joblib.load(scaler_path)
        
        return model, scaler
    except Exception as e:
        # We cannot use st.error here easily if it's cached, so we print to logs
        print(f"❌ Error loading artifacts: {e}")
        return None, None

# Load them now
model, scaler = load_artifacts()

# --- UI LAYOUT ---
st.title("🚛 Engine Failure Prediction System")
st.markdown(f"**Model Source:** `{MODEL_REPO_ID}`")
st.markdown("Enter real-time sensor data to predict engine health status.")

# --- INPUT FORM ---
with st.form("prediction_form"):
    st.subheader("Sensor Telemetry")
    col1, col2 = st.columns(2)
    
    with col1:
        rpm = st.number_input("Engine RPM", min_value=0, max_value=10000, value=2000)
        lub_oil_p = st.number_input("Lub Oil Pressure", min_value=0.0, max_value=10.0, value=4.5)
        fuel_p = st.number_input("Fuel Pressure", min_value=0.0, max_value=20.0, value=7.0)
    
    with col2:
        coolant_p = st.number_input("Coolant Pressure", min_value=0.0, max_value=10.0, value=3.0)
        lub_oil_t = st.number_input("Lub Oil Temp (°C)", min_value=0.0, max_value=150.0, value=75.0)
        coolant_t = st.number_input("Coolant Temp (°C)", min_value=0.0, max_value=150.0, value=80.0)
    
    submit_button = st.form_submit_button("Predict Engine Status")

# --- PREDICTION LOGIC ---
if submit_button:
    if model is None or scaler is None:
        st.error("Cannot predict: Model or Scaler not loaded. Check HF Space Logs.")
    else:
        # 1. Create Dataframe
        input_data = pd.DataFrame({
            'Engine rpm': [rpm],
            'Lub oil pressure': [lub_oil_p],
            'Fuel pressure': [fuel_p],
            'Coolant pressure': [coolant_p],
            'lub oil temp': [lub_oil_t],
            'Coolant temp': [coolant_t]
        })
        
        # 2. Reorder Columns
        input_data = input_data[EXPECTED_FEATURES]
        
        try:
            # 3. Scale
            scaled_data = scaler.transform(input_data)
            
            # 4. Predict
            prediction = model.predict(scaled_data)[0]
            
            try:
                probability = model.predict_proba(scaled_data)[0][1]
            except:
                probability = 0.0
            
            # 5. Display
            st.divider()
            if prediction == 1:
                st.error(f"🚨 CRITICAL WARNING: Engine Failure Predicted!")
                st.write(f"**Confidence Level:** {probability:.2%}")
                st.warning("Recommendation: Stop vehicle immediately and inspect cooling system.")
            else:
                st.success(f"✅ SYSTEM NORMAL: Engine is Healthy.")
                st.write(f"**Failure Probability:** {probability:.2%}")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")