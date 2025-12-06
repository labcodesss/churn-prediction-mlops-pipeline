

# churn-prediction-mlops-pipeline 
# 🔥 Churn Prediction — End-to-End MLOps Pipeline  
### FastAPI • scikit-learn • CI/CD • Model Registry • Monitoring • Local Deployment (no Docker)

This repository contains a **production-ready churn prediction system** built with real MLOps practices:
- Automated training  
- Reproducible preprocessing & feature engineering  
- FastAPI inference service with hot-reload  
- Simple web UI for business users  
- Evaluation pipeline  
- CI/CD with GitHub Actions  
- Prometheus metrics endpoint for monitoring  

The project demonstrates how to take a machine-learning model **from data → model → API → UI → CI/CD**.

---

## 🚀 Key Features

### **✔ End-to-End Machine Learning Pipeline**
- Data preprocessing  
- Feature engineering  
- Random Forest training  
- Model persistence (`joblib`)  
- Evaluation (Accuracy, Precision, Recall, ROC-AUC)

### **✔ Production-style API (FastAPI)**
- `/predict` – returns churn probability  
- `/health` – health check  
- `/reload-model` – hot swap model without restarting server  
- `/metrics` – Prometheus-ready monitoring  

### **✔ Simple Web UI**
A clean HTML page calls the `/predict` API and displays the results.

### **✔ CI Pipeline (GitHub Actions)**
- Install dependencies  
- Run pytest  
- Train new model  
- Save trained model + metrics as artifacts  

### **✔ Monitoring**
- Exposes Prometheus metrics via `/metrics`  
- Tracks inference count per endpoint & status  

---

---

## 🧠 Model Performance (Evaluation Output)

From `model/metrics.json`:

| Metric        | Score  |
|---------------|--------|
| **Accuracy**  | 0.865  |
| **Precision** | 0.839  |
| **Recall**    | 0.937  |
| **ROC-AUC**   | 0.935  |

➡ **High recall** is valuable for churn detection because it identifies most at-risk customers.

---

## 🧪 Run the Project Locally

### 1️⃣ Create virtual environment
```bash
python -m venv .venv

pip install -r requirements.txt

2️⃣ Train the model
python -m src.train --data data/churn_sample.csv --model-out model/model.joblib

3️⃣ Start the FastAPI server
.\.venv\Scripts\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8080 --reload

API docs available at:
http://127.0.0.1:8080/docs

4️⃣ Start the Web UI
.\.venv\Scripts\python.exe -m http.server 5500 --bind 127.0.0.1 -d web

Open UI in browser:
http://127.0.0.1:5500/index.html


📡 API Endpoints
Method	 Endpoint	     Description
GET   	/health	         Service heartbeat
POST	/predict	     Returns churn probability
POST	/reload-model	 Reload latest model  without restart
GET	    /metrics	     Prometheus monitoring metrics

Example request:

{
  "features": [39.99, 12, 479.88]
}


Example response:

{
  "churn_proba": 0.29
}

🔄 CI/CD (GitHub Actions)

Workflow: .github/workflows/mlops-ci.yml
Pipeline steps:
Setup Python
Install dependencies
Run pytest
Train the model
Upload model + metrics as artifacts
Every push to main triggers automated testing & training.

📜 License

MIT License — free to use and modify.

👤 Author

Developed by labcodesss
For questions or improvements, feel free to open an issue or pull request.
