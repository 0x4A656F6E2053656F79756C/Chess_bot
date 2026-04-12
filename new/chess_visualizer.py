import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt
import seaborn as sns

# 💡 글로벌 상수를 제거하고 동적으로 구조를 세팅할 수 있도록 수정
def parse_fen_to_76_chars(fen):
    try:
        board, color, castling, ep, half, full = fen.split(' ')
    except ValueError:
        return None

    for i in range(1, 9):
        board = board.replace(str(i), '.' * i)
    board = board.replace('/', '')
    if len(board) != 64: return None

    castling = castling.ljust(4, '.')[:4]
    ep = ep.ljust(2, '.')[:2]
    half = half.zfill(2)[:2]
    full = full.zfill(3)[:3]

    return board + color + castling + ep + half + full

class ChessTransformer(nn.Module):
    # 💡 init 시 모델의 크기 파라미터를 입력받도록 변경
    def __init__(self, d_model, n_heads, n_layers, dim_feedforward, vocab_size=129, num_classes=1968):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 77, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, 
            batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, src):
        x = self.embedding(src) + self.pos_encoder
        attn_map = None
        
        for i, layer in enumerate(self.transformer_encoder.layers):
            if i == len(self.transformer_encoder.layers) - 1:
                # 마지막 레이어의 Self-Attention 가중치 추출
                _, attn_map = layer.self_attn(layer.norm1(x), layer.norm1(x), layer.norm1(x), need_weights=True, average_attn_weights=True)
            x = layer(x)
            
        logits = self.fc_out(x[:, 0, :])
        return logits, attn_map

def visualize(fen, model_size="9M"):
    """
    model_size: "9M" 또는 "136M"
    """
    # 💡 선택한 모델 크기에 따라 하이퍼파라미터 및 파일명 세팅
    if model_size == "9M":
        d_model, n_heads, n_layers, dim_feedforward = 256, 8, 8, 1024
        model_path = "chess_transformer_best_9M.pth"
    elif model_size == "136M":
        d_model, n_heads, n_layers, dim_feedforward = 1024, 8, 8, 4096
        model_path = "chess_transformer_best_136M.pth"
    else:
        print("❌ 지원하지 않는 모델 크기입니다. '9M' 또는 '136M'을 입력하세요.")
        return

    # 1. 환경 준비
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        with open("move_vocab.json", "r") as f:
            id_to_move = json.load(f)
    except FileNotFoundError:
        print("❌ move_vocab.json 파일이 없습니다.")
        return
    
    # 2. 동적 모델 초기화 및 가중치 로드
    model = ChessTransformer(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, 
        dim_feedforward=dim_feedforward, num_classes=len(id_to_move)
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ [{model_size}] 모델 가중치 로드 완료: {model_path}")
    except FileNotFoundError:
        print(f"❌ {model_path} 파일이 없습니다. 학습이 완료되었는지 확인하세요.")
        return
        
    model.eval()

    # 3. 전처리
    parsed = parse_fen_to_76_chars(fen)
    if parsed is None:
        print("❌ 잘못된 FEN 형식입니다.")
        return
        
    input_ids = torch.tensor([[128] + [ord(c) for c in parsed]], dtype=torch.long).to(device)

    # 4. 추론
    with torch.no_grad():
        logits, attn = model(input_ids)
        move_idx = logits.argmax(dim=1).item()
        predicted_move = id_to_move[str(move_idx)]
    
    # 5. 시각화 (CLS 토큰의 보드 64칸 어텐션)
    board_attn = attn[0, 0, 1:65].cpu().numpy().reshape(8, 8)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(board_attn, annot=False, cmap="viridis", 
                xticklabels=['a','b','c','d','e','f','g','h'], 
                yticklabels=['8','7','6','5','4','3','2','1'])
    plt.title(f"[{model_size}] Predicted Move: {predicted_move}\nTarget FEN: {fen}")
    plt.show()

# ==========================================
# 실행 예시
# ==========================================
if __name__ == "__main__":
    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    # 9M 모델 결과 확인
    visualize(test_fen, model_size="9M")
    
    # 136M 모델 결과 확인 (학습 완료 후 주석 해제)
    visualize(test_fen, model_size="136M")