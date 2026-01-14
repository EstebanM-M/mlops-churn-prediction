# 🚀 MLOps Churn Prediction Pipeline

End-to-end ML pipeline for customer churn prediction with automated training, monitoring, and deployment.

## ⚡ Quick Start
```bash
# 1. Clone repository
git clone <your-repo-url>
cd mlops-churn-prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -e .

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# 5. Download data
python src/data/data_loader.py

# 6. Train model
python src/training/train.py

# 7. Start API
uvicorn serving.api:app --reload

# 8. Start frontend
streamlit run src/frontend/app.py
```

## 🏗️ Architecture
```
Data Pipeline → Feature Store → Training Pipeline → Model Registry
                                                          ↓
                                               CI/CD Pipeline
                                                          ↓
                                            Serving + Monitoring
```

## 🛠️ Tech Stack

- **ML**: XGBoost, LightGBM, CatBoost
- **MLOps**: MLflow, DVC, Great Expectations
- **API**: FastAPI, Streamlit
- **Database**: PostgreSQL
- **Monitoring**: Evidently AI
- **CI/CD**: GitHub Actions

## 📦 Project Structure
```
mlops-churn-prediction/
├── src/
│   ├── data/          # Data pipeline
│   ├── features/      # Feature store
│   ├── training/      # Model training
│   ├── serving/       # API & serving
│   └── monitoring/    # Drift detection
├── tests/             # Unit & integration tests
├── docker/            # Docker configuration
└── docs/              # Documentation
```

## 🚧 Development Status

- [x] Project setup
- [ ] Data pipeline
- [ ] Feature store
- [ ] Training pipeline
- [ ] API serving
- [ ] Monitoring
- [ ] CI/CD
- [ ] Documentation

## 👤 Author

**Esteban** - Electronic Engineer transitioning to ML/AI

## 📄 License

MIT License
```

---

## **Resumen de Cambios Necesarios** ✅
```
Archivos que NECESITAN actualización:
├── setup.py                 ✅ Ya lo actualizamos
├── .gitignore              ⏭️ Actualizar (nuevo código arriba)
├── .env.example            ⏭️ Crear (nuevo archivo)
└── README.md               ⏭️ Crear (opcional pero recomendado)

Archivos que NO necesitan cambios:
├── requirements.txt        ✅ Está bien (solo "-e .")
├── pyproject.toml         ✅ Está bien
└── .pre-commit-config.yaml ✅ Está bien