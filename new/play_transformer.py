import torch
import torch.nn as nn
import chess
import json
import time

# 💡 업로드해주신 GUI.py에서 게임 엔진과 플레이어 클래스를 그대로 가져옵니다.
from GUI import ChessGame, HumanPlayer, Player

# ==========================================
# 1. 트랜스포머 모델 및 파서 정의
# ==========================================
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
        x = self.transformer_encoder(x)
        return self.fc_out(x[:, 0, :]), None  # (logits, attn_map) 반환

# ==========================================
# 2. 트랜스포머 AI 플레이어 클래스
# ==========================================
class TransformerPlayer(Player):
    def __init__(self, model_size="9M", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        
        # Vocab 로드
        try:
            with open("move_vocab.json", "r") as f:
                self.id_to_move = json.load(f)
        except FileNotFoundError:
            raise Exception("❌ move_vocab.json 파일이 없습니다.")
            
        # 모델 구조 세팅
        if model_size == "9M":
            d_model, n_heads, n_layers, dim_feedforward = 256, 8, 8, 1024
            # 이전에 저장했던 9M 모델 파일명 (실제 파일명에 맞게 수정하세요)
            model_path = "chess_transformer_best_9M.pth" 
        elif model_size == "136M":
            d_model, n_heads, n_layers, dim_feedforward = 1024, 8, 8, 4096
            model_path = "chess_transformer_best_136M.pth"
        else:
            raise ValueError("지원하지 않는 모델 크기입니다.")

        # 모델 초기화 및 가중치 로드
        self.model = ChessTransformer(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, 
            dim_feedforward=dim_feedforward, num_classes=len(self.id_to_move)
        ).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ {model_size} 트랜스포머 장착 완료! ({model_path})")
        except FileNotFoundError:
            raise Exception(f"❌ {model_path} 가중치 파일을 찾을 수 없습니다.")
            
        self.model.eval()

    def is_human(self):
        return False

    def get_move(self, board):
        # AI가 너무 빨리 두면 사람이 당황하므로 약간의 딜레이 추가
        time.sleep(0.5) 
        
        fen = board.fen()
        parsed = parse_fen_to_76_chars(fen)
        
        if parsed is None:
            print("⚠️ FEN 파싱 실패! 랜덤 수를 둡니다.")
            return list(board.legal_moves)[0]

        input_ids = torch.tensor([[128] + [ord(c) for c in parsed]], dtype=torch.long).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast('cuda' if 'cuda' in str(self.device) else 'cpu'):
                logits, _ = self.model(input_ids)
        
        # 💡 핵심 로직: 확률(Logit)이 높은 순서대로 정렬하여, 합법적인 수인지 검사
        sorted_indices = torch.argsort(logits[0], descending=True)
        
        for idx in sorted_indices:
            move_str = self.id_to_move[str(idx.item())]
            try:
                move = chess.Move.from_uci(move_str)
                # 이 수가 현재 체스 규칙에 어긋나지 않는지 확인
                if move in board.legal_moves:
                    return move
            except ValueError:
                continue
                
        # 모든 수를 다 뒤졌는데도 둘 게 없다면 (거의 불가능) 첫 번째 합법적 수 반환
        print("⚠️ 합법적인 수를 찾지 못했습니다. 랜덤 수를 둡니다.")
        return list(board.legal_moves)[0]

# ==========================================
# 3. 게임 실행
# ==========================================
if __name__ == "__main__":
    print("="*40)
    print("🤖 Transformer Chess Simulator")
    print("="*40)
    
    # 설정 선택
    size_input = input("상대할 모델 크기를 선택하세요 (1: 9M, 2: 136M) [기본 1]: ")
    model_size = "136M" if size_input.strip() == "2" else "9M"
    
    color_input = input("당신의 진영을 선택하세요 (1: 백, 2: 흑) [기본 1]: ")
    is_human_white = False if color_input.strip() == "2" else True
    
    print("\n게임 로딩 중...")
    
    try:
        transformer_ai = TransformerPlayer(model_size=model_size)
        human = HumanPlayer()
        
        white_player = human if is_human_white else transformer_ai
        black_player = transformer_ai if is_human_white else human
        
        # GUI.py의 ChessGame 인스턴스 생성 
        # (model_path=None으로 주면 기존 CNN 평가바는 꺼집니다)
        game = ChessGame(white_player, black_player, model_path=None, spectator_mode=False)
        
        # 만약 플레이어가 흑이면 보드를 뒤집어 줌
        if not is_human_white:
            game.flip_board = True
            
        print("\n게임을 시작합니다! GUI 창을 확인하세요.")
        game.run()
        
    except Exception as e:
        print(f"\n❌ 실행 중 오류 발생: {e}")