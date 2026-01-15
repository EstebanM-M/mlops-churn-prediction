@echo off
echo ========================================
echo MLOps Churn Prediction - Stopping Demo
echo ========================================
echo.

REM Stop Docker containers
echo Stopping Docker containers...
docker-compose -f docker\docker-compose.yml down

echo.
echo ========================================
echo ✅ Demo Environment Stopped
echo ========================================
echo.
echo To start again, run: demo-start.bat
echo.
pause