@echo off
TITLE PulseIQ Healthcare Pipeline & Dashboard Runner
color 0A

echo ===============================================================================
echo                PULSEIQ - HEALTHCARE READMISSION PIPELINE & DASHBOARD
echo                    Project Folder: %~dp0
echo ===============================================================================
echo.

cd /d "%~dp0"

echo [1/4] Verifying Python and dependencies...
python -m pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b %errorlevel%
)
echo       Dependencies verified.
echo.

echo [2/4] Executing ETL Data Pipeline...
python run_pipeline.py
if %errorlevel% neq 0 (
    echo [ERROR] Pipeline execution failed.
    pause
    exit /b %errorlevel%
)
echo       Data pipeline completed successfully!
echo.

echo [3/4] Launching FastAPI MLOps Prediction Microservice on port 8000...
start /B python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
echo       FastAPI service started in background (http://127.0.0.1:8000).
echo.

echo [4/4] Launching 4K 120fps Streamlit Analytics Dashboard...
echo       Dashboard will open in your default browser at http://localhost:8501
echo       (Press Ctrl+C in this terminal to stop the servers)
echo.
python -m streamlit run app/streamlit_app.py

pause
