@echo off
rem Color Grader: connect the MCP server to Claude.
rem Registers it with Claude Code when the claude command exists, then prints the Claude Desktop lines.
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
set "PY=%~dp0.venv\Scripts\python.exe"
set "SCRIPT=%~dp0colorgrader\mcp.py"

if not exist "%PY%" (
    echo Color Grader is not installed yet. Run install.bat first.
    pause
    exit /b 1
)

where claude >nul 2>nul
if errorlevel 1 (
    echo Claude Code not found. Skipping the Claude Code setup.
) else (
    echo Registering color-grader with Claude Code...
    call claude mcp remove color-grader -s user >nul 2>nul
    call claude mcp add color-grader -s user -- "%PY%" "%SCRIPT%"
    if errorlevel 1 (
        echo Claude Code registration failed. See the message above.
    ) else (
        echo Done. Claude Code now has the color-grader tools in every folder.
    )
)

echo.
"%PY%" -m colorgrader.mcp --config
echo.
pause
