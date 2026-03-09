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
    
    print("\n[플레이 모드]")
    print("1. 사람 vs MCTS 봇 (가장 추천)")
    print("2. 직관(CNN) 봇 vs 수읽기(MCTS) 봇")
    print("3. MCTS vs MCTS")
    print("4. 사람 vs 사람")
    print("5. MCTS 봇 vs 사람")
    print("6. 사람 vs CNN 봇")
    print("7. CNN 봇 vs CNN 봇")
    
    mode = input("모드 선택 (1~7): ").strip()
    
    print("\n[핸디캡 설정]")
    print("0. 핸디캡 없음 (기본)")
    print("1. 백 퀸(Queen) 제거")
    print("2. 백 룩(Rook, a1) 제거")
    print("3. 백 나이트(Knight, b1) 제거")
    print("4. 흑 퀸(Queen) 제거")
    print("5. 흑 룩(Rook, a8) 제거")
    handicap = input("핸디캡 선택 (0~5) [기본: 0]: ").strip()
    if not handicap:
        handicap = '0'
    
    if mode == '2':
        player1 = CNNPlayer(model_path=MODEL_PATH)
        player2 = MCTSPlayer(model_path=MODEL_PATH, simulations=100, explore_moves=10, add_noise=False)
    elif mode == '3':
        player1 = MCTSPlayer(model_path=MODEL_PATH, simulations=100, explore_moves=20, add_noise=True)
        player2 = MCTSPlayer(model_path=MODEL_PATH, simulations=100, explore_moves=20, add_noise=True)
    elif mode == '4':
        player1 = HumanPlayer()
        player2 = HumanPlayer()
    elif mode == '5': 
        player1 = MCTSPlayer(model_path=MODEL_PATH, simulations=100, explore_moves=6, add_noise=False)
        player2 = HumanPlayer()
    elif mode == '6':
        player1 = HumanPlayer()
        player2 = CNNPlayer(model_path=MODEL_PATH)
    elif mode == '7':
        player1 = CNNPlayer(model_path=MODEL_PATH)
        player2 = CNNPlayer(model_path=MODEL_PATH)
    else: 
        player1 = HumanPlayer()
        player2 = MCTSPlayer(model_path=MODEL_PATH, simulations=100, explore_moves=6, add_noise=False)

    game = ChessGame(white_player=player1, black_player=player2, model_path=MODEL_PATH)

    # --- 핸디캡 적용 로직 ---
    if handicap == '1':
        game.board.remove_piece_at(chess.D1) # 백 퀸 제거
    elif handicap == '2':
        game.board.remove_piece_at(chess.A1) # 백 퀸사이드 룩 제거
    elif handicap == '3':
        game.board.remove_piece_at(chess.B1) # 백 퀸사이드 나이트 제거
    elif handicap == '4':
        game.board.remove_piece_at(chess.D8) # 흑 퀸 제거
    elif handicap == '5':
        game.board.remove_piece_at(chess.A8) # 흑 퀸사이드 룩 제거

    if handicap in ['1', '2', '3', '4', '5']:
        # 기물이 사라졌으므로, 룩이나 킹이 없어졌을 때를 대비해 캐슬링 권한을 정리합니다.
        game.board.clean_castling_rights()
        print(f"\n✅ 핸디캡 모드({handicap}번)가 적용되었습니다!")

    game.run()

    os.makedirs("notes", exist_ok=True)
    pgn_game = chess.pgn.Game.from_board(game.board)
    pgn_game.headers["Event"] = "Local AI Test Match (Handicap)" if handicap != '0' else "Local AI Test Match"
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