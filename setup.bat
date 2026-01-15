@echo off
echo ========================================
echo MLOps Churn Prediction - Initial Setup
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

REM Create .env file if doesn't exist
if not exist ".env" (
    echo Creating .env file...
    echo # MLflow Configuration > .env
    echo MLFLOW_TRACKING_URI=http://localhost:5000 >> .env
    echo MODEL_DIR=models >> .env
    echo DATA_DIR=data >> .env
)

REM Set permanent environment variable
echo Setting permanent environment variable...
setx MLFLOW_TRACKING_URI http://localhost:5000

REM Create necessary directories
if not exist "models\production\" mkdir models\production
if not exist "data\features\" mkdir data\features

REM Train initial model
echo.
echo ========================================
echo Training initial model (5-10 minutes)...
echo ========================================
python src\training\train.py

REM Build Docker images
echo.
echo ========================================
echo Building Docker images...
echo ========================================
docker-compose -f docker\docker-compose.yml build

echo.
echo ========================================
echo ✅ Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. CLOSE this terminal
echo 2. OPEN a NEW terminal
echo 3. Run: demo-start.bat
echo.
pause