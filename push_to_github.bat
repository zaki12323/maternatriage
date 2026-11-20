@echo off
echo ========================================================
echo   Pushing MaternaTriage to GitHub
echo ========================================================
echo.
cd /d "%~dp0"

echo Current Status:
git status
echo.

echo Pushing to GitHub (origin main)...
echo If a window pops up, click "Sign in with your browser".
echo.
git push -u origin main

echo.
if %ERRORLEVEL% EQU 0 (
    echo ========================================================
    echo   SUCCESS! Your project is now live on GitHub!
    echo   Repository: https://github.com/zaki12323/maternatriage
    echo ========================================================
) else (
    echo ========================================================
    echo   If it failed: Make sure you created a repository named
    echo   'maternatriage' on https://github.com/new first!
    echo ========================================================
)
echo.
pause
