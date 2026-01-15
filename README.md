# 🚀 MLOps Churn Prediction Pipeline

End-to-end ML pipeline for customer churn prediction with automated training, monitoring, and deployment.


---

## ⚡ Quick Start
```bash
# 1. Clone repository
git clone https://github.com/EstebanM-M/mlops-churn-prediction
cd mlops-churn-prediction

# 2. Create virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -e .

# 4. Run the complete pipeline
# Download data
python -m data.data_loader

# Validate data quality
python -m data.data_validator

# Split data into train/val/test
python -m data.data_splitter

# Create features
python -m features.feature_store

# Train models
python -m training.train

# 5. View MLflow results
mlflow ui
# Open browser at http://localhost:5000

# 6. Start API server
uvicorn serving.api:app --reload
# Open browser at http://localhost:8000/docs

```

---

## 🎯 Features

### ✅ Implemented

- **Data Pipeline**
  - Automatic dataset download from IBM repository
  - Custom data validation (6 quality checks)
  - Stratified train/val/test split (64%/16%/20%)
  
- **Feature Store** 
  - 46 engineered features from 21 raw columns
  - Feature caching with Parquet format
  - Feature validation and statistics
  - Single source of truth for training and serving

- **Training Pipeline**
  - 3 Gradient Boosting models: XGBoost, LightGBM, CatBoost
  - MLflow experiment tracking
  - Automatic model comparison and selection
  - Best model: **CatBoost (ROC-AUC: 0.8485)**

- **API Serving** 
  - FastAPI REST API with Swagger documentation
  - `/predict` endpoint for single predictions
  - `/predict/batch` endpoint for batch predictions
  - `/health` endpoint for monitoring
  - Pydantic validation for request/response
  - Automatic model loading on startup

  - **Monitoring & Drift Detection** 
  - Statistical drift detection (Kolmogorov-Smirnov test)
  - Automated drift alerts and reporting
  - `/monitoring/check-drift` endpoint
  - `/monitoring/summary` endpoint
  - Integration with webhook retraining system
```

### 🔜 Coming Soon

- Streamlit dashboard for predictions
- CI/CD pipeline with GitHub Actions
- Docker multi-service deployment
- Automated retraining triggers

---

## 🏗️ System Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                              │
│  Raw CSV → Validation → Train/Val/Test Split → Feature Store   │
│  (7,043 samples → 46 engineered features → Parquet cache)      │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                       TRAINING LAYER                            │
│  XGBoost + LightGBM + CatBoost → MLflow Tracking               │
│  Automatic model comparison → Champion selection               │
│  Best: CatBoost (ROC-AUC: 0.8485)                              │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                       SERVING LAYER                             │
│  FastAPI REST API + Pydantic Validation + Swagger Docs         │
│  Endpoints: /predict, /health, /batch                          │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    AUTOMATION LAYER                             │
│  Webhooks: /webhook/retrain, /webhook/new-data                 │
│  Background job system with status tracking                     │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                     MONITORING LAYER                            │
│  Drift Detection (KS test) + Automated Alerts                  │
│  Health checks + Performance tracking                           │
│  Auto-trigger retraining when drift exceeds threshold          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

**Machine Learning:**
- scikit-learn 1.3+
- XGBoost 2.0+
- LightGBM 4.1+
- CatBoost 1.2+

**MLOps:**
- MLflow 2.9+ (Experiment tracking & model registry)
- Evidently AI 0.4+ (Drift detection - WIP)
- DVC 3.30+ (Data versioning - configured)

**API & Web:**
- FastAPI 0.104+ (API serving - WIP)
- Streamlit 1.29+ (Dashboard - WIP)
- Pydantic 2.5+ (Data validation)

**Database:**
- SQLAlchemy 2.0+ (ORM)
- PostgreSQL (via psycopg2-binary)

**Visualization:**
- Matplotlib 3.8+
- Seaborn 0.13+
- Plotly 5.18+

**Development:**
- pytest 7.4+ (Testing)
- black 23.11+ (Code formatting)
- isort 5.12+ (Import sorting)
- pre-commit 3.5+ (Git hooks)

---

## 📦 Project Structure
```
mlops-churn-prediction/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py       # Download & load data
│   │   ├── data_validator.py    # Data quality checks
│   │   └── data_splitter.py     # Train/val/test split
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_store.py     # Feature engineering
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py             # Model training with MLflow
│   ├── serving/                 # (WIP)
│   │   ├── api.py
│   │   └── webhooks.py
│   ├── monitoring/              # (WIP)
│   │   └── drift_detector.py
│   └── frontend/                # (WIP)
│       └── app.py
├── data/
│   ├── raw/                     # Raw CSV data
│   ├── processed/               # Train/val/test splits
│   └── features/                # Cached features (parquet)
├── models/                      # Saved models (.joblib)
├── mlruns/                      # MLflow experiments
├── tests/                       # Unit & integration tests (WIP)
├── docker/                      # Docker configs (WIP)
├── .github/workflows/           # CI/CD pipelines (WIP)
├── setup.py                     # Package configuration
├── requirements.txt             # Dependencies (-e .)
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

---

## 📊 Current Results

### Dataset
- **Source:** IBM Telco Customer Churn
- **Total Samples:** 7,043 customers
- **Features:** 21 raw → 46 engineered
- **Target:** Churn (Yes/No)
- **Class Distribution:** 73.5% No Churn, 26.5% Churn

### Model Performance (Validation Set)

| Model     | ROC-AUC | Accuracy | F1 Score | Precision | Recall |
|-----------|---------|----------|----------|-----------|--------|
| 🏆 CatBoost | **0.8485** | 81.10% | 0.5832 | 70.28% | 49.83% |
| LightGBM  | 0.8441  | 80.75%   | 0.5819   | 68.64%    | 50.50% |
| XGBoost   | 0.8403  | 80.57%   | 0.5764   | 68.35%    | 49.83% |

**Champion Model:** CatBoost v1  
**Saved at:** `models/champion_catboost_v1.joblib`

---

## 🚧 Development Status

- [x] Project setup & configuration
- [x] Data pipeline (loader, validator, splitter)
- [x] Feature Store with 46 engineered features
- [x] Training pipeline with 3 models + MLflow
- [x] Model comparison and selection
- [x] FastAPI serving layer
- [x] Webhook-driven automation
- [x] Drift detection & monitoring
- [ ] Streamlit dashboard
- [ ] CI/CD with GitHub Actions
- [ ] Docker deployment
- [ ] Comprehensive testing
- [ ] Documentation (MkDocs)


---

## 🔍 Key Features Explained

### Feature Store 
Centralized feature engineering ensuring consistency between training and serving:
- Prevents training-serving skew (same features in training and production)
- Caches computed features in Parquet format for performance
- Single source of truth for all 46 engineered features
- Built-in feature validation and statistical summaries
- Version control for feature definitions

### MLflow Integration
Complete experiment tracking and model management:
- All training runs logged automatically with parameters and metrics
- Model versioning and registry for reproducibility
- Easy model comparison across experiments
- Champion model selection and promotion
- Experiment visualization and analysis

### Automated Model Selection
Intelligent champion selection based on ROC-AUC:
- Trains 3 models in parallel (XGBoost, LightGBM, CatBoost)
- Compares performance metrics across validation set
- Automatically selects and saves best model
- Champion model ready for immediate serving

### Webhook Automation System
Event-driven automation for MLOps workflows:
- `/webhook/retrain`: Trigger model retraining on-demand
- `/webhook/new-data`: Notify system of new data arrivals
- Background job system with status tracking
- Automatic retraining when data threshold is met
- Job queue with monitoring and cancellation support

### Drift Detection & Monitoring
Production model health monitoring:
- Statistical drift detection using Kolmogorov-Smirnov test
- Monitors 28 numerical features for distribution changes
- Automated alerts when drift exceeds configurable thresholds
- Integration with webhook system for auto-retraining
- Detailed drift reports with per-feature analysis
- `/monitoring/check-drift`: On-demand drift analysis
- `/monitoring/summary`: Comprehensive system health status

### API Serving
Production-ready REST API with FastAPI:
- `/predict`: Single customer churn prediction
- `/predict/batch`: Batch predictions for multiple customers
- `/health`: Service health check endpoint
- Pydantic validation for request/response data
- Interactive Swagger documentation at `/docs`
- Automatic feature engineering via Feature Store
- Risk level classification (Low/Medium/High)

---

## 🧪 Testing
```bash
# Run all tests (when implemented)
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/unit/test_feature_store.py
```

---

## 📝 Example Usage

### Training a New Model
```python
from features.feature_store import FeatureStore
from training.train import ModelTrainer

# Load features
feature_store = FeatureStore()
train_features = feature_store.load_features("train")
val_features = feature_store.load_features("val")

# Train models
trainer = ModelTrainer()
results = trainer.train_all_models(train_features, val_features)

# Compare and save best
comparison = trainer.compare_models(results)
print(comparison)

best_name, best_model, best_metrics = trainer.select_best_model(results)
trainer.save_model(best_model, f"champion_{best_name}")
```

### Creating Features
```python
from features.feature_store import FeatureStore
import pandas as pd

# Create Feature Store
fs = FeatureStore()

# Load raw data
df = pd.read_csv("data/raw/telco_churn.csv")

# Create features
features_df = fs.create_features(df)

# Validate features
validation = fs.validate_features(features_df)
print(validation)

# Save features
fs.save_features(features_df, "train", version="v1")
```

### Making Predictions via API
```python
import requests

# Customer data
customer = {
    "customerID": "TEST-001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.20
}

# Make prediction
response = requests.post("http://localhost:8000/predict", json=customer)
prediction = response.json()

print(f"Churn Probability: {prediction['churn_probability']:.2%}")
print(f"Prediction: {prediction['churn_prediction']}")
print(f"Risk Level: {prediction['risk_level']}")
```

---

## 👤 Author

**Esteban**  
Electronic Engineer (Escuela Colombiana de Ingeniería Julio Garavito, 2024)  
Transitioning to ML/AI Engineering

**Skills Demonstrated:**
- End-to-end ML pipeline design
- MLOps best practices
- Feature engineering
- Model training and evaluation
- Experiment tracking with MLflow
- Clean code architecture

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- Dataset: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d)
- Inspired by production ML systems at Uber, Airbnb, and Netflix

---

## 📞 Contact

For questions or collaboration:
- GitHub: https://github.com/EstebanM-M?tab=repositories
- LinkedIn: https://www.linkedin.com/in/esteban-morales-mahecha/
- Email: estebanmoralesm@outlook.com