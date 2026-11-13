@echo off
title PolicyLens - Hospital Policy Intelligence Assistant
cd /d "%~dp0"

echo ========================================================
echo   Starting PolicyLens Assistant...
echo ========================================================
echo.

:: Detect virtual environment
set "PYTHON_EXE="
if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else if exist "myenv\Scripts\python.exe" (
    set "PYTHON_EXE=myenv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        set "PYTHON_EXE=python"
    )
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python could not be located.
    echo Please make sure Python or the venv virtual environment is installed.
    echo.
    pause
    exit /b 1
)

echo [OK] Using Python: %PYTHON_EXE%
echo [OK] Opening browser at http://localhost:8000 ...
echo.

:: Launch browser in background after 5 seconds to allow RAG model to initialize
start "" cmd /c "timeout /t 5 /nobreak >nul && start http://localhost:8000"

:: Start PolicyLens API and Web Server
"%PYTHON_EXE%" api.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] PolicyLens stopped unexpectedly.
    pause
)
