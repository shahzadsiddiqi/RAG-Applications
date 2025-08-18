@echo off
REM Simple Git sync script for Windows

set BRANCH=main
set REMOTE=origin

REM Add all changes
git add .

REM Commit (use default if no message provided)
set MSG=%1
if "%MSG%"=="" set MSG=Sync update
git commit -m "%MSG%"

REM Push to GitHub
git push %REMOTE% %BRANCH%

pause