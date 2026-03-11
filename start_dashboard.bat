@echo off
REM ============================================================
REM AI-Based Cybersecurity Threat Prediction System
REM Live Detection Dashboard Launcher
REM ============================================================

echo.
echo ================================================================================
echo  🛡️  CYBER THREAT DETECTION SYSTEM - WEB DASHBOARD
echo ================================================================================
echo.
echo Starting Live Detection System with Web Dashboard...
echo.
echo Browser will open at: http://localhost:5000
echo.
echo Press CTRL+C to stop monitoring
echo.
pause

REM Change to project directory
cd /d A:\Final\Final

REM Run the live detection system with dashboard
.\.venv\Scripts\python.exe live_detection.py --dashboard

pause
