# 🚀 MLOps Churn Prediction System

End-to-end MLOps system for customer churn prediction with automated training, drift detection, and production-ready deployment.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![Tests](https://img.shields.io/badge/Tests-13%20passing-success)
![Coverage](https://img.shields.io/badge/Coverage-36%25-yellow)

## 📊 Project Overview

This project demonstrates production-grade MLOps practices for predicting customer churn. It features automated model training, real-time drift detection, webhook-driven retraining, and a complete monitoring stack.

**Key Metrics:**
- 🎯 Model Accuracy: 81%
- 📈 ROC-AUC: 0.85
- ⚡ API Response Time: <100ms
- 🔧 46 Engineered Features

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────┐
│                   User Interfaces                        │
├──────────────────┬──────────────────┬───────────────────┤
│  Streamlit UI    │   FastAPI Docs   │   MLflow UI       │
│  Port 8501       │   Port 8000      │   Port 5000       │
└────────┬─────────┴────────┬─────────┴────────┬──────────┘
         │                  │                  │
         ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────┐
│              Docker Container Layer                      │
├──────────────────┬──────────────────┬───────────────────┤
│   Dashboard      │      API         │     MLflow        │
│   Container      │   Container      │   Container       │
└────────┬─────────┴────────┬─────────┴────────┬──────────┘
         │                  │                  │
         └──────────────────┴──────────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │  Feature Store  │
                  │  Model Registry │
                  │  Data Pipeline  │
                  └─────────────────┘
```

## ✨ Features

### 🤖 Machine Learning
- **Multi-Model Training**: XGBoost, LightGBM, CatBoost
- **Automated Model Selection**: Best model based on ROC-AUC
- **Feature Engineering**: 46 features from raw data
- **Hyperparameter Optimization**: Grid search with cross-validation

### 🔧 MLOps Infrastructure
- **Feature Store**: Versioned feature management with Parquet
- **Experiment Tracking**: MLflow for metrics, parameters, and artifacts
- **Model Registry**: Versioned model storage and lifecycle management
- **Drift Detection**: Evidently AI for data and prediction drift

### 🚀 Production Ready
- **REST API**: FastAPI with 15+ endpoints
- **Real-time Predictions**: Sub-100ms response time
- **Batch Processing**: Async batch prediction endpoint
- **Health Monitoring**: Prometheus-compatible metrics
- **Webhook Automation**: Triggered retraining on drift detection

### 🐳 Deployment
- **Docker Containers**: Multi-service orchestration
- **Health Checks**: Automated service health verification
- **Auto-restart**: Resilient container management
- **Volume Mounting**: Persistent data and models

## 📂 Project Structure
```
mlops-churn-prediction/
├── src/
│   ├── data/              # Data loading and validation
│   ├── features/          # Feature engineering & store
│   ├── training/          # Model training pipeline
│   ├── serving/           # FastAPI application
│   ├── monitoring/        # Drift detection & metrics
│   └── frontend/          # Streamlit dashboard
├── tests/
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.dashboard
│   ├── Dockerfile.mlflow
│   └── docker-compose.yml
├── models/                # Trained models
├── data/                  # Raw and processed data
├── setup.bat              # Initial setup script
├── demo-start.bat         # Start demo environment
├── demo-stop.bat          # Stop demo environment
└── quick-check.bat        # System health check
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Docker Desktop
- 8GB RAM minimum

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/EstebanM-M/mlops-churn-prediction.git
cd mlops-churn-prediction
```

**2. Run automated setup:**
```cmd
setup.bat
```

This will:
- Create virtual environment
- Install dependencies
- Train initial model
- Build Docker images

**3. Start the demo environment:**
```cmd
demo-start.bat
```

**4. Access the services:**
- 📊 Dashboard: http://localhost:8501
- 🔧 API Docs: http://localhost:8000/docs
- 📈 MLflow UI: http://localhost:5000

## 📖 Usage

### Making Predictions

**Via Dashboard:**
1. Navigate to http://localhost:8501
2. Go to "Single Prediction" tab
3. Fill in customer details
4. Click "Predict Churn"

**Via API:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "customerID": "TEST-001",
    "tenure": 12,
    "MonthlyCharges": 70.35,
    "Contract": "Month-to-month",
    ...
  }'
```

### Training Models
```cmd
# Set MLflow tracking
set MLFLOW_TRACKING_URI=http://localhost:5000

# Run training
python src/training/train.py
```

### Running Tests
```cmd
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html
```

## 🔬 Model Performance

| Model      | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------------|----------|-----------|--------|----------|---------|
| CatBoost   | 0.811    | 0.703     | 0.498  | 0.583    | **0.849** |
| LightGBM   | 0.807    | 0.686     | 0.505  | 0.582    | 0.844   |
| XGBoost    | 0.806    | 0.683     | 0.498  | 0.576    | 0.840   |

**Best Model:** CatBoost (ROC-AUC: 0.849)

## 🧪 Testing

- **13 Tests** covering:
  - API endpoints
  - Feature Store operations
  - Drift detection
  - Model predictions
- **36% Code Coverage**
- **CI/CD Ready** (GitHub Actions compatible)

## 🛠️ Tech Stack

**ML & Data:**
- Scikit-learn
- XGBoost, LightGBM, CatBoost
- Pandas, NumPy
- MLflow

**Backend:**
- FastAPI
- Uvicorn
- SQLAlchemy
- Pydantic

**Frontend:**
- Streamlit
- Plotly

**MLOps:**
- Docker & Docker Compose
- Evidently AI
- Prometheus

**Testing:**
- Pytest
- Coverage.py

## 📊 API Endpoints

### Prediction
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions

### Monitoring
- `GET /health` - Service health check
- `GET /monitoring/health` - Monitoring system status
- `GET /monitoring/metrics` - Prometheus metrics

### Webhooks
- `POST /webhook/retrain` - Trigger model retraining
- `POST /webhook/drift-alert` - Drift detection alert

## 🔍 Monitoring & Drift Detection

The system automatically monitors:
- **Data Drift**: Distribution changes in input features
- **Prediction Drift**: Changes in model outputs
- **Model Performance**: Accuracy degradation over time

**Alerts trigger when:**
- Data drift score > 0.5
- Prediction drift detected
- Model accuracy drops > 5%

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 👤 Author

**Esteban Morales**
- GitHub: [EstebanM-M](https://github.com/EstebanM-M)
- LinkedIn: [Esteban_Morales_Mahecha] (https://www.linkedin.com/in/esteban-morales-mahecha/)

## 🙏 Acknowledgments

- Telco Customer Churn Dataset
- MLflow Community
- FastAPI Documentation

## 📧 Contact

For questions or opportunities, reach out via:
- Email: estebanmmahecha@outlook.com 
- LinkedIn: [Esteban_Morales_Mahecha](https://www.linkedin.com/in/esteban-morales-mahecha/)

---

**⭐ If you find this project useful, please consider giving it a star!**