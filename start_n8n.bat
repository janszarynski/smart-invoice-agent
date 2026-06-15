@echo off
echo =======================================================
echo     Smart Invoice Agent - Control Panel (n8n)
echo =======================================================
echo.
echo [1] Starting local AI engine (Ollama) in the background...
start /b ollama serve
timeout /t 3 >nul

echo [2] Starting n8n server...
echo.
echo NOTE: If the system asks "Need to install the following packages: n8n", type 'y' and press Enter.
echo Once started, open your browser and navigate to: http://localhost:5678
echo.
echo To stop the system, simply close this window.
echo =======================================================
set NODES_EXCLUDE=[]
npx n8n
pause
