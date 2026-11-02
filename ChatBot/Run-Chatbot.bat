@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Run-Chatbot.ps1"
if errorlevel 1 pause
