@echo off

REM Check if .venv directory exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    echo Virtual environment created.
    echo Activating virtual environment...
    call .\.venv\Scripts\activate
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    echo Virtual environment already exists.
    echo Activating virtual environment...
    call .\.venv\Scripts\activate
)

REM Run your Python script
echo Running main.py...
python main.py