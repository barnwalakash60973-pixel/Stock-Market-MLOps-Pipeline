# 📈 Stock Market Prediction MLOps Pipeline

> End-to-end MLOps pipeline for multi-class stock movement prediction
> using **CatBoost**, **FastAPI**, **Streamlit**, **DVC**, **MLflow**,
> **Docker**, **GitHub Actions**, and **Render**.

------------------------------------------------------------------------

## 🛠 Tech Stack

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/CatBoost-ML-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge" />
  <img src="https://img.shields.io/badge/DVC-Versioning-13ADC7?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Docker-Container-2496ED?style=for-the-badge&logo=docker" />
  <img src="https://img.shields.io/badge/GitHub_Actions-CI-2088FF?style=for-the-badge&logo=githubactions" />
  <img src="https://img.shields.io/badge/Render-Deployment-46E3B7?style=for-the-badge&logo=render" />
</p>

------------------------------------------------------------------------

## 📑 Table of Contents

-   Features
-   Architecture
-   Project Structure
-   Installation
-   Training Pipeline
-   MLflow
-   DVC
-   Docker
-   FastAPI
-   Streamlit
-   CI/CD
-   Model
-   Results
-   Future Improvements
-   Author

------------------------------------------------------------------------

## 🚀 Features

-   End-to-end machine learning pipeline
-   Automated data ingestion and validation
-   Time-series feature engineering
-   Walk-forward validation
-   CatBoost multi-class classification
-   Final model retraining using validated iterations
-   MLflow experiment tracking
-   DVC data & model versioning
-   FastAPI REST API
-   Streamlit dashboard
-   Dockerized backend and frontend
-   GitHub Actions CI
-   Automatic deployment with Render

------------------------------------------------------------------------

## 🏗️ Architecture

``` text
Dataset
   │
   ▼
Data Ingestion
   │
   ▼
Data Validation
   │
   ▼
Feature Engineering
   │
   ▼
Model Training
   ├── Walk-Forward Validation
   ├── Early Stopping
   ├── Final Retraining
   ├── Final Evaluation
   ├── MLflow Tracking
   └── DVC Versioning
   │
   ▼
FastAPI Backend
   │
   ▼
Streamlit Dashboard
```

------------------------------------------------------------------------

## 📂 Project Structure

``` text
.
├── .github/
│   └── workflows/
├── app/
├── config/
├── data/
├── frontend_streamlit/
├── models/
├── reports/
├── requirements/
├── src/
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── training_pipeline.py
└── README.md
```

------------------------------------------------------------------------

## ⚙️ Installation

``` bash
git clone <repository-url>
cd Stock-Market-MLOps-Pipeline
python -m venv venu
```

Activate environment:

``` bash
# Windows
venu\Scripts\activate

# Linux/macOS
source venu/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements/backend.txt
pip install -r requirements/frontend.txt
```

------------------------------------------------------------------------

## 🏃 Training Pipeline

``` bash
python training_pipeline.py
```

Pipeline stages:

1.  Data Ingestion
2.  Data Validation
3.  Feature Engineering
4.  Walk-forward Validation
5.  Final Model Training
6.  Final Model Evaluation
7.  Save Pipeline
8.  MLflow Tracking
9.  DVC Model Versioning

------------------------------------------------------------------------

## 📊 MLflow

Start MLflow UI:

``` bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open:

``` text
http://127.0.0.1:5000
```

Tracked: - Parameters - Metrics - Artifacts - Logged Model

------------------------------------------------------------------------

## 📦 DVC

``` bash
dvc add models
git add models.dvc
git commit -m "Update trained model"
dvc push
```

------------------------------------------------------------------------

## 🐳 Docker

``` bash
docker compose up --build
```

Services

-   Backend → Port 8000
-   Frontend → Port 8501

------------------------------------------------------------------------

## 🔌 FastAPI

``` bash
uvicorn app.main:app --reload
```

Swagger UI:

``` text
http://127.0.0.1:8000/docs
```

------------------------------------------------------------------------

## 💻 Streamlit

``` bash
streamlit run frontend_streamlit/streamlit_app.py
```

------------------------------------------------------------------------

## 🔄 CI/CD

### Continuous Integration

-   Black
-   Flake8
-   MyPy
-   Pytest

### Continuous Delivery

``` text
Git Push
   │
   ▼
GitHub
   │
   ▼
GitHub Actions
   │
   ├── CI Checks
   └── Render Deployment
```

------------------------------------------------------------------------

## 📈 Model

-   Algorithm: CatBoostClassifier
-   Time-Series Walk-forward Validation
-   Early Stopping
-   Final Retraining
-   Multi-class Prediction

------------------------------------------------------------------------

## 📌 Results

The pipeline logs every experiment using MLflow and versions trained
models with DVC.

Tracked information includes:

-   Hyperparameters
-   Final Iterations
-   Accuracy
-   Precision
-   Recall
-   F1 Score
-   Model Artifacts

------------------------------------------------------------------------

## 📸 Screenshots

-   Streamlit Dashboard *(Add Screenshot)*
-   MLflow Experiment Tracking *(Add Screenshot)*
-   FastAPI Swagger UI *(Add Screenshot)*

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Integrate real-time financial news sentiment using FinBERT.
-   Register production-ready models using MLflow Model Registry.
-   Add model monitoring dashboards.
-   Implement data drift and concept drift detection.
-   Automate scheduled model retraining.
-   Integrate live market data.
-   Hyperparameter optimization using Optuna.
-   Deploy on Kubernetes for scalable inference.

------------------------------------------------------------------------

## 👨‍💻 Author

**Akash Kumar Barnwal**

-   🎓 M.Sc. Artificial Intelligence & Machine Learning, IIIT Lucknow
-   💻 GitHub: https://github.com/barnwalakash60973-pixel
-   🔗 LinkedIn:
    https://www.linkedin.com/in/akash-kumar-barnwal-31968a380/

------------------------------------------------------------------------

## 📄 License

This project is licensed under the MIT License.
