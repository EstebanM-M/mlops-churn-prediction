# 🚀 MLOps Churn Prediction Pipeline

End-to-end ML pipeline for customer churn prediction with automated training, monitoring, and deployment.

**Project Status:** 🟢 60% Complete (Data Pipeline + Feature Store + Training Pipeline)

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
```

---

## 🎯 Features

### ✅ Implemented

- **Data Pipeline**
  - Automatic dataset download from IBM repository
  - Custom data validation (6 quality checks)
  - Stratified train/val/test split (64%/16%/20%)
  
- **Feature Store** ⭐
  - 46 engineered features from 21 raw columns
  - Feature caching with Parquet format
  - Feature validation and statistics
  - Single source of truth for training and serving

- **Training Pipeline**
  - 3 Gradient Boosting models: XGBoost, LightGBM, CatBoost
  - MLflow experiment tracking
  - Automatic model comparison and selection
  - Best model: **CatBoost (ROC-AUC: 0.8485)**

### 🔜 Coming Soon

- FastAPI serving with webhook triggers
- Evidently AI drift detection and monitoring
- Streamlit dashboard for predictions
- CI/CD pipeline with GitHub Actions
- Docker multi-service deployment
- Automated retraining triggers

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                        │
│  Raw Data → Validation → Train/Val/Test Split          │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                   FEATURE STORE                         │
│  Raw Features → Engineered Features → Cached (.parquet)│
│  • 46 features created                                  │
│  • Validation checks                                    │
│  • Version control                                      │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                  TRAINING PIPELINE                      │
│  MLflow Tracking → 3 Models → Best Model Selection     │
│  • XGBoost    (ROC-AUC: 0.8403)                        │
│  • LightGBM   (ROC-AUC: 0.8441)                        │
│  • CatBoost   (ROC-AUC: 0.8485) 🏆                     │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│             SERVING + MONITORING (WIP)                  │
│  FastAPI → A/B Testing → Drift Detection → Alerts      │
└─────────────────────────────────────────────────────────┘
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
- [ ] FastAPI serving layer
- [ ] Webhook-driven automation
- [ ] Drift detection & monitoring
- [ ] Streamlit dashboard
- [ ] CI/CD with GitHub Actions
- [ ] Docker deployment
- [ ] Comprehensive testing
- [ ] Documentation (MkDocs)

**Progress:** 🟢🟢🟢🟢🟢🟢⚪⚪⚪⚪ 60%

---

## 🔍 Key Features Explained

### Feature Store
Centralized feature engineering ensuring consistency between training and serving:
- Prevents training-serving skew
- Caches computed features for performance
- Single source of truth for all features
- Feature validation and statistics

### MLflow Integration
Complete experiment tracking:
- All training runs logged automatically
- Model parameters and metrics stored
- Model versioning and registry
- Easy model comparison

### Model Selection
Automated champion selection based on ROC-AUC:
- Trains 3 models in parallel
- Compares performance metrics
- Selects best model automatically
- Saves champion for serving

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