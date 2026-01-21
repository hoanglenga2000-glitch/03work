@echo off
chcp 65001 >nul
echo ========================================
echo WPS 表格格式化工具
echo ========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：未找到 Python
    echo 请先安装 Python 3.x
    pause
    exit /b 1
)

REM 检查是否安装了 pywin32
python -c "import win32com.client" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  检测到未安装 pywin32，正在安装...
    pip install pywin32
    if errorlevel 1 (
        echo ❌ 安装失败，请手动运行: pip install pywin32
        pause
        exit /b 1
    )
)

echo ✅ 环境检查通过
echo.
echo 请确保：
echo   1. WPS 已打开
echo   2. 要处理的文档已在 WPS 中打开
echo.
pause

REM 运行 Python 脚本
python "%~dp0wps_table_formatter.py"

echo.
pause
