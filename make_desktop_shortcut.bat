@echo off
set "APPDIR=%~dp0"
powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\Live Translator.lnk'); $s.TargetPath='%APPDIR%venv311\Scripts\pythonw.exe'; $s.Arguments='translator.py'; $s.WorkingDirectory='%APPDIR%'; $s.Description='Live Translator'; $s.Save()"
echo.
echo Shortcut "Live Translator" created on your Desktop.
pause
