@echo off
title Cogni Subtitles Launcher
cd /d "%~dp0"

echo ⚡ Activating Python 3.11 Environment Matrix...
call venv311\Scripts\activate

echo 🧠 Launching Cogni Subtitles Live Translator Engine...
python translator.py

echo 👋 Virtual Environment Deactivated.
pause
