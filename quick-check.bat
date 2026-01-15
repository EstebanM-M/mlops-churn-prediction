@echo off
echo ========================================
echo System Health Check
echo ========================================
echo.

echo [1/5] Checking Docker...
docker --version
if %errorlevel% neq 0 (
    echo ❌ Docker not installed
    goto :end
) else (
    echo ✅ Docker OK
)

echo.
echo [2/5] Checking Python...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python not installed
    goto :end
) else (
    echo ✅ Python OK
)

echo.
echo [3/5] Checking Model...
if exist "models\champion_catboost_v1.joblib" (
    echo ✅ Model exists
) else (
    echo ❌ Model not found - run setup.bat
)

echo.
echo [4/5] Checking Docker Containers...
docker ps | findstr churn
if %errorlevel% neq 0 (
    echo ⚠️ Containers not running
    echo Run: demo-start.bat
) else (
    echo ✅ Containers running
)

echo.
echo [5/5] Checking Environment Variable...
if "%MLFLOW_TRACKING_URI%"=="http://localhost:5000" (
    echo ✅ MLFLOW_TRACKING_URI configured
) else (
    echo ⚠️ MLFLOW_TRACKING_URI not set
    echo Run: setx MLFLOW_TRACKING_URI http://localhost:5000
)

:end
echo.
echo ========================================
echo Health Check Complete
echo ========================================
pause