@echo off
echo Installing EDI Claims Processing System...

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

REM Check Python version is 3.8+
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>nul
if errorlevel 1 (
    echo Error: Python 3.8 or higher is required
    python --version
    pause
    exit /b 1
)

echo Python version check passed.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

REM Create directories
echo Creating directories...
mkdir logs 2>nul
mkdir models 2>nul
mkdir config 2>nul

REM Copy example configuration
if not exist "config\config.yaml" (
    echo Copying example configuration...
    if exist "config\config.example.yaml" (
        copy "config\config.example.yaml" "config\config.yaml"
        echo Configuration template copied to config\config.yaml
        echo Please edit config\config.yaml with your database settings before running.
    ) else (
        echo Warning: config.example.yaml not found
        echo Please create config\config.yaml manually using the provided template
    )
) else (
    echo Configuration file already exists: config\config.yaml
)

REM Create __init__.py files for Python packages
echo Creating package initialization files...
echo. > src\__init__.py
echo. > src\config\__init__.py
echo. > src\database\__init__.py
echo. > src\monitoring\__init__.py
echo. > src\utils\__init__.py

echo.
echo Installation completed successfully!
echo.
echo Next steps:
echo 1. Edit config\config.yaml with your database settings
echo 2. Run database setup scripts in sql\ directory:
echo    - PostgreSQL: sql\postgresql_create_edi_databases.sql
echo    - SQL Server: sql\sqlserver_create_results_database.sql
echo 3. Generate encryption key: 
echo    python -c "from cryptography.fernet import Fernet; print('Encryption key:', Fernet.generate_key().decode())"
echo 4. Load sample data (optional): python load.py --operation sample --count 1000
echo 5. Train ML model (optional): python src\train_filter_predictor.py
echo 6. Test the system: python run_edi.py --config config\config.yaml
echo.
echo For monitoring setup, see docs\prometheus_setup_windows.md
echo.
pause