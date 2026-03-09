import chess
import chess.pgn
import datetime
import os
import pygame
import threading
import time

from AI import MCTSPlayer
from GUI import ChessGame  # GUI.py에서 ChessGame 클래스 임포트

GAMES = 6
MODEL_PATH = "model/model_v2.pth"

def auto_close_monitor(game):
    """게임이 종료되면 3초 대기 후 자동으로 GUI 창을 닫는 데몬 스레드"""
    while not getattr(game, 'game_over', False):
        time.sleep(0.5)
    
    # 결과를 확인할 수 있게 3초 대기
    time.sleep(3)
    
    # Pygame 종료 이벤트를 발생시켜 game.run() 루프를 탈출하게 만듦
    pygame.event.post(pygame.event.Event(pygame.QUIT))

def play_single_game_with_gui(white_player, black_player):
    # [추가] 창을 화면 중앙에 띄우기 위한 설정 (반드시 set_mode 호출 전에 실행)
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    
    # GUI 인스턴스 생성
    game = ChessGame(white_player, black_player, model_path=MODEL_PATH)
    
    # 평가 막대 켜기
    game.show_eval = True
    
    # [수정] 전체 화면 대신 적절한 고정 크기 설정 (예: 1200x800)
    # 모니터 해상도보다 작게 설정해야 '창'으로 보입니다.
    window_width = 1200
    window_height = 800
    
    game.width, game.height = window_width, window_height
    # pygame.FULLSCREEN 제거
    game.screen = pygame.display.set_mode((game.width, game.height)) 
    
    # GUI 내부 크기 업데이트
    game.update_dimensions(game.width, game.height)

    # AI 메서드 안전 장치
    white_player.is_human = lambda: False
    black_player.is_human = lambda: False

    # 자동 닫기 스레드 시작
    threading.Thread(target=auto_close_monitor, args=(game,), daemon=True).start()

    # 게임 진행
    game.run()

    return game.board

if __name__ == "__main__":

    os.makedirs("notes", exist_ok=True)

    print("=== Player 1 vs Player 2 GUI 테스트 ===")

    results = {
        "player1_win": 0,
        "player2_win": 0,
        "draw": 0
    }

    # 초반 동안 다양한 수 탐색
    EXPLORE_MOVES = 10
    
    # 각 플레이어의 시뮬레이션 수치
    P1_SIMULATIONS = 50
    P2_SIMULATIONS = 100

    for game_idx in range(GAMES):

        print(f"\n===== Game {game_idx+1}/{GAMES} =====")

        # 색 번갈아 할당
        if game_idx % 2 == 0:
            white_player = MCTSPlayer(model_path=MODEL_PATH, simulations=P1_SIMULATIONS, explore_moves=EXPLORE_MOVES, add_noise=True)
            black_player = MCTSPlayer(model_path=MODEL_PATH, simulations=P2_SIMULATIONS, explore_moves=EXPLORE_MOVES, add_noise=True)

            white_tag = "Player 1"
            black_tag = "Player 2"

        else:
            white_player = MCTSPlayer(model_path=MODEL_PATH, simulations=P2_SIMULATIONS, explore_moves=EXPLORE_MOVES, add_noise=True)
            black_player = MCTSPlayer(model_path=MODEL_PATH, simulations=P1_SIMULATIONS, explore_moves=EXPLORE_MOVES, add_noise=True)

            white_tag = "Player 2"
            black_tag = "Player 1"

        # GUI와 함께 게임 실행
        board = play_single_game_with_gui(white_player, black_player)

        result = board.result()
        print("Result:", result)

        # 승패 기록
        if result == "1-0":
            if white_tag == "Player 1":
                results["player1_win"] += 1
            else:
                results["player2_win"] += 1

        elif result == "0-1":
            if black_tag == "Player 1":
                results["player1_win"] += 1
            else:
                results["player2_win"] += 1

        else:
            results["draw"] += 1

        # PGN 저장
        game_pgn = chess.pgn.Game.from_board(board)

        game_pgn.headers["Event"] = "Player 1 vs Player 2 Test"
        game_pgn.headers["White"] = white_tag
        game_pgn.headers["Black"] = black_tag
        game_pgn.headers["Result"] = result

        filename = datetime.datetime.now().strftime(f"notes/mcts_test_%Y%m%d_%H%M%S_{game_idx}.pgn")

        with open(filename, "w", encoding="utf-8") as f:
            f.write(str(game_pgn))

        print("PGN saved:", filename)

    print("\n===== 최종 결과 =====")

    total = GAMES
    print(f"Player 1 승: {results['player1_win']}")
    print(f"Player 2 승: {results['player2_win']}")
    print(f"무승부: {results['draw']}")

    print("\n승률")
    print(f"Player 1: {results['player1_win']/total*100:.1f}%")
    print(f"Player 2: {results['player2_win']/total*100:.1f}%")