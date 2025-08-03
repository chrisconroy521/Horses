@echo off
echo 🐎 Ragozin Sheets Parser - Server Launcher
echo ============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Check if virtual environment exists and activate it
if exist "venv\Scripts\activate.bat" (
    echo 🔄 Activating virtual environment...
    call venv\Scripts\activate.bat
    echo ✅ Virtual environment activated
) else (
    echo ⚠️  Virtual environment not found, using system Python
)

echo.

REM Check if API key is set
python -c "import os; exit(0 if os.getenv('OPENAI_API_KEY') else 1)" 2>nul
if errorlevel 1 (
    echo ⚠️  OpenAI API key not set
    echo    The GPT parser will not be available
    echo    Run 'python set_api_key.py' to set your API key
    echo.
)

echo 🚀 Starting both servers...
echo.

REM Start backend server in background
echo Starting FastAPI backend server...
start "Backend Server" cmd /k "python api.py"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend server in background
echo Starting Streamlit frontend server...
start "Frontend Server" cmd /k "streamlit run streamlit_app.py --server.port 8501"

REM Wait a moment for frontend to start
timeout /t 5 /nobreak >nul

echo.
echo ============================================
echo 🎯 Both servers are starting!
echo.
echo 📱 Access your application:
echo    Frontend: http://localhost:8501
echo    Backend API: http://localhost:8000
echo    API Docs: http://localhost:8000/docs
echo.
echo 🛑 Close the server windows to stop the servers
echo ============================================
echo.

REM Open the frontend in default browser
echo 🌐 Opening web interface in browser...
start http://localhost:8501

echo.
echo ✅ Servers launched successfully!
echo    Both server windows should now be open
echo.
pause 