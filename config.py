import torch

def get_device():
    # 1. Mac M1/M2/M3 (MPS) 확인
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    # 2. NVIDIA GPU (CUDA) 확인
    elif torch.cuda.is_available():
        return torch.device("cuda")
    # 3. 그 외에는 CPU
    else:
        return torch.device("cpu")

# 전역 변수로 선언
device = get_device()

if __name__ == "__main__":
    print(f"📡 현재 시스템에서 감지된 최적 장치: {device}")