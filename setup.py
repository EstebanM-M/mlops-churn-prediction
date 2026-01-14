from setuptools import setup, find_packages

setup(
    name="mlops-churn-prediction",
    version="1.0.0",
    author="Esteban",
    author_email="tu_email@example.com",
    description="End-to-end MLOps pipeline for customer churn prediction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        # Core ML
        "numpy>=1.24.0,<2.0.0",
        "pandas>=2.0.0,<3.0.0",
        "scikit-learn>=1.3.0,<2.0.0",
        "scipy>=1.11.0,<2.0.0",
        
        # Gradient Boosting Models (3 modelos)
        "xgboost>=2.0.0,<3.0.0",
        "lightgbm>=4.1.0,<5.0.0",
        "catboost>=1.2.0,<2.0.0",
        
        # MLOps Tools ⭐
        "mlflow>=2.9.0,<3.0.0",              # Experiment tracking & registry
        "dvc>=3.30.0,<4.0.0",                # Data versioning
        "great-expectations>=0.18.0,<1.0.0", # Data validation
        
        # API & Web
        "fastapi>=0.104.0,<1.0.0",
        "uvicorn[standard]>=0.24.0,<1.0.0",
        "streamlit>=1.29.0,<2.0.0",
        "pydantic>=2.5.0,<3.0.0",
        "pydantic-settings>=2.1.0,<3.0.0",
        "python-multipart>=0.0.6,<1.0.0",    # Para file uploads en FastAPI
        
        # Database
        "sqlalchemy>=2.0.0,<3.0.0",
        "psycopg2-binary>=2.9.0,<3.0.0",     # PostgreSQL driver
        "alembic>=1.12.0,<2.0.0",            # Database migrations
        
        # Monitoring ⭐
        "evidently>=0.4.11,<1.0.0",          # Drift detection
        "prometheus-client>=0.19.0,<1.0.0",  # Metrics
        
        # Utilities
        "python-dotenv>=1.0.0,<2.0.0",
        "pyyaml>=6.0.0,<7.0.0",              # Config files
        "joblib>=1.3.0,<2.0.0",              # Model serialization
        "requests>=2.31.0,<3.0.0",
        "tqdm>=4.66.0,<5.0.0",
        
        # Visualization
        "matplotlib>=3.8.0,<4.0.0",
        "seaborn>=0.13.0,<1.0.0",
        "plotly>=5.18.0,<6.0.0",
        
        # Feature Engineering
        "category-encoders>=2.6.0,<3.0.0",   # Encoders avanzados
        "imbalanced-learn>=0.11.0,<1.0.0",   # SMOTE si necesitamos
    ],
    extras_require={
        "dev": [
            # Testing
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "httpx>=0.25.0",                 # Para testing de FastAPI
            
            # Code Quality
            "black>=23.11.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "pre-commit>=3.5.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.5.0",
            "mkdocstrings[python]>=0.24.0",
        ],
    },
)