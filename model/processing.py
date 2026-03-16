import torch

# 1. 보따리(체크포인트) 파일 불러오기 (CPU로 안전하게 로드)
checkpoint_path = "chess_two_head_epoch_5_FINAL.pth"  # 저장된 모델 경로에 맞게 수정하세요!
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# 2. 순수 가중치만 추출하기
pure_weights = checkpoint['model_state_dict']

# 3. 새로운 이름으로 깔끔하게 저장하기
save_path = "model_v5.pth"
torch.save(pure_weights, save_path)

print(f"✅ 가중치 분리 완료! 이제 {save_path} 파일을 봇에 바로 적용할 수 있습니다.")