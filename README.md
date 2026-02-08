# 🚛 Engine Failure Prediction System (MLOps Capstone)

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20App-yellow)](https://huggingface.co/spaces/iStillWaters/YOUR_SPACE_NAME)
[![ML Pipeline](https://github.com/xRandeep/engine-failure-prediction-mlops/actions/workflows/pipeline.yml/badge.svg)](https://github.com/xRandeep/engine-failure-prediction-mlops/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
This project delivers an end-to-end **Predictive Maintenance System** designed to identify potential engine failures before they occur. By analyzing sensor telemetry (RPM, Pressure, Temperature), the system predicts failure probability and recommends maintenance actions.

The solution is engineered as a full **MLOps Pipeline**, featuring automated training, model versioning, and cloud deployment.

### 🎯 Key Objectives
* **Predict:** Detect engine failure signatures with >95% Recall.
* **Automate:** Trigger model retraining automatically via GitHub Actions.
* **Deploy:** Serve predictions via a user-friendly Streamlit Dashboard.

---

## 🏗️ System Architecture
The project follows a modern MLOps workflow:

1.  **Data Ingestion:** Raw sensor data is stored and versioned on **Hugging Face Datasets**.
2.  **Experimentation:** Models trained & tuned using **Scikit-Learn** & **MLflow**.
3.  **CI/CD Pipeline:** **GitHub Actions** automates the training and deployment process.
4.  **Deployment:** The final model is hosted on **Hugging Face Spaces** as a Dockerized Streamlit App.

---

## 📊 Model Performance
After experimenting with Random Forest, XGBoost, and AdaBoost, the final champion model was selected based on the **F2-Score** (prioritizing Recall to minimize missed failures).

| Metric | Score | Insight |
| :--- | :--- | :--- |
| **Recall** | **97.5%** |Catches nearly all critical failures (Safety First). |
| **F2-Score** | **0.88** | Balanced metric prioritizing failure detection. |
| **Accuracy** | **65.1%** | Accepts higher false positives to ensure safety. |

**Champion Model:** `AdaBoostClassifier`

---

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone [https://github.com/xRandeep/engine-failure-prediction-mlops.git](https://github.com/xRandeep/engine-failure-prediction-mlops.git)
cd engine-failure-prediction-mlops
```

**2. Install Dependencies**
```Bash
pip install -r requirements.txt
```

**3. Run the App**
```Bash
streamlit run app.py
```

## 🔗 Deployment

The application is live and can be accessed here:
**[Link to your Hugging Face Space]**

## 🤝 Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
