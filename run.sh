#!/bin/bash

# 한글 깨짐 방지 및 환경 설정
export LANG=ko_KR.UTF-8

echo "=== Chess AI Setup & Run (macOS) ==="

# 1. 가상환경이 없으면 생성 및 패키지 설치
if [ ! -d "venv" ]; then
    echo "[1/3] 파이썬 가상환경을 생성합니다..."
    python3 -m venv venv
    
    echo "[2/3] 가상환경을 활성화합니다..."
    source venv/bin/activate
    
    echo "[3/3] 라이브러리 설치 및 Mac 최적화 작업을 진행합니다..."
    python3 -m pip install --upgrade pip
    
    # 먼저 기본 요구사항 설치
    pip install -r requirements.txt
    
    # 🌟 Mac 전용 PyTorch 재설치 로직 (기존 torch 삭제 후 재설치)
    echo "⚙️ 기존 Torch 제거 후 Mac M1/M2 가속 버전을 설치합니다..."
    pip uninstall -y torch torchvision torchaudio
    pip install torch torchvision torchaudio
    
    echo "✅ 세팅이 완료되었습니다!"
else
    source venv/bin/activate
fi

echo ""
echo "========================================="
echo "1. 로컬 AI 대국 시뮬레이터 (play.py)"
echo "2. 체스닷컴 인터넷 봇 (internet.py)"
echo "3. 체스 AI 관전(중계) 모드 (spectator.py)"
echo "========================================="
read -p "원하는 모드의 번호를 입력하세요 (1, 2, 3): " choice

# 2. 선택에 따른 실행
if [ "$choice" == "1" ]; then
    python3 play.py
elif [ "$choice" == "2" ]; then
    python3 internet.py
elif [ "$choice" == "3" ]; then
    python3 spectator.py
else
    echo "잘못된 입력입니다."
fi

read -p "엔터를 누르면 종료됩니다..."