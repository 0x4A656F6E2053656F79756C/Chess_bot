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
    
    echo "[3/3] 필수 라이브러리를 설치합니다..."
    python3 -m pip install --upgrade pip
    pip install -r requirements.txt
    
    # macOS용 PyTorch 설치 (MPS 가속 활용 가능)
    # 기존 cu121 대신 macOS용 기본 torch 설치를 권장합니다.
    pip install torch
    echo "세팅이 완료되었습니다!"
else
    # 이미 가상환경이 있다면 활성화만 진행
    source venv/bin/activate
fi

echo ""
echo "========================================="
echo "1. 로컬 AI 대국 시뮬레이터 (play.py)"
echo "2. 체스닷컴 인터넷 봇 (internet.py)"
echo "========================================="
read -p "원하는 모드의 번호를 입력하세요 (1 또는 2): " choice

# 2. 선택에 따른 실행
if [ "$choice" == "1" ]; then
    python3 play.py
elif [ "$choice" == "2" ]; then
    python3 internet.py
else
    echo "잘못된 입력입니다."
fi

# 종료 전 대기 (선택 사항)
read -p "엔터를 누르면 종료됩니다..."