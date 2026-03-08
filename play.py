import chess.pgn
import datetime
import os

from GUI import ChessGame, HumanPlayer
from AI import CNNPlayer, MCTSPlayer

if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)
    os.makedirs("notes", exist_ok=True)

    print("=== 체스 AI 로컬 매치 시뮬레이터 ===")
    
    model_name = input("사용할 모델 파일 이름 (model/ 폴더 내, 기본: model_v2.pth): ").strip()
    if not model_name:
        model_name = "model_v2.pth"
    
    MODEL_PATH = os.path.join("model", model_name)
    
    print("1. 사람 vs MCTS 봇 (가장 추천)")
    print("2. 직관(CNN) 봇 vs 수읽기(MCTS) 봇")
    print("3. MCTS vs MCTS")
    print("4. 사람 vs 사람")
    
    mode = input("모드 선택 (1~4): ").strip()
    
    if mode == '2':
        player1 = CNNPlayer(model_path=MODEL_PATH)
        player2 = MCTSPlayer(model_path=MODEL_PATH, simulations=100)
    elif mode == '3':
        player1 = MCTSPlayer(model_path=MODEL_PATH, simulations=100, explore_moves=20, add_noise=True)
        player2 = MCTSPlayer(model_path=MODEL_PATH, simulations=100, explore_moves=20, add_noise=True)
    elif mode == '4':
        player1 = HumanPlayer()
        player2 = HumanPlayer()
    else: 
        player1 = HumanPlayer()
        player2 = MCTSPlayer(model_path=MODEL_PATH, simulations=100, explore_moves=6, add_noise=False)

    # 수정됨: MODEL_PATH를 ChessGame에 전달하여 GUI에서 평가치를 로드할 수 있게 함
    game = ChessGame(white_player=player1, black_player=player2, model_path=MODEL_PATH)
    game.run()

    os.makedirs("notes", exist_ok=True)
    pgn_game = chess.pgn.Game.from_board(game.board)
    pgn_game.headers["Event"] = "Local AI Test Match"
    pgn_game.headers["White"] = player1.__class__.__name__ 
    pgn_game.headers["Black"] = player2.__class__.__name__
    
    if hasattr(game, 'explicit_result') and game.explicit_result:
        pgn_game.headers["Result"] = game.explicit_result
    else:
        pgn_game.headers["Result"] = game.board.result()
        
    filename = datetime.datetime.now().strftime("match_%Y%m%d_%H%M%S.pgn")
    with open("notes/" + filename, "w", encoding="utf-8") as f:
        f.write(str(pgn_game))
        
    print(f"\n✅ 기보 저장 완료: notes/{filename}")