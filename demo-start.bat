@echo off
echo ========================================
echo MLOps Churn Prediction - Starting Demo
echo ========================================
echo.

REM Start Docker containers
echo Starting Docker containers...
docker-compose -f docker\docker-compose.yml up -d

REM Wait for services to be ready
echo.
echo Waiting for services to start (30 seconds)...
timeout /t 30 /nobreak

REM Check container status
echo.
echo ========================================
echo Container Status:
echo ========================================
docker ps

REM Open services in browser
echo.
echo ========================================
echo Opening services in browser...
echo ========================================
timeout /t 5 /nobreak

start http://localhost:8501
timeout /t 2 /nobreak
start http://localhost:8000/docs
timeout /t 2 /nobreak
start http://localhost:5000

echo.
echo ========================================
echo ✅ Demo Environment Ready!
echo ========================================
echo.
echo Services:
echo - Dashboard:  http://localhost:8501
echo - API Docs:   http://localhost:8000/docs
echo - MLflow UI:  http://localhost:5000
echo.
echo Press any key to view logs...
pause
docker-compose -f docker\docker-compose.yml logs --tail=50