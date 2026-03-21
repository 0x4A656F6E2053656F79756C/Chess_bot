@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

title Chess AI Simulator
echo === Chess AI Setup ^& Run ===

:: 가상환경이 없으면 새로 생성하고 패키지 설치 [cite: 1]
if not exist venv (
    echo [1/3] 파이썬 가상환경을 생성합니다...
    python -m venv venv
    
    echo [2/3] 가상환경을 활성화합니다...
    call venv\Scripts\activate
    
    echo [3/3] 필수 라이브러리를 설치합니다... (시간이 다소 소요될 수 있습니다)
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    :: GPU 가속을 위한 PyTorch CUDA 버전 설치 [cite: 2]
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    echo 세팅이 완료되었습니다!
) else (
    call venv\Scripts\activate
)

echo.
echo =========================================
echo 1. 로컬 AI 대국 시뮬레이터 (play.py)
echo 2. 체스닷컴 인터넷 봇 (internet.py)
echo 3. 체스 AI 관전(중계) 모드 (spectator.py)
echo =========================================
set /p choice="원하는 모드의 번호를 입력하세요 (1, 2, 3): "

if "%choice%"=="1" python play.py
if "%choice%"=="2" python internet.py
if "%choice%"=="3" python spectator.py

pause