@echo off
:: Activate the virtual environment
call .\.venv\Scripts\activate

:: Run the Python script
python main.py

:: Keep the command window open after execution
pause