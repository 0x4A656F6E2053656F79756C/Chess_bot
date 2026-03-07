import chess
import chess.pgn
import datetime
import os

from AI import MCTSPlayer

GAMES = 2
MODEL_PATH = "model/model_v2.pth"

def play_single_game(white_player, black_player):
    board = chess.Board()

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = white_player.get_move(board)
        else:
            move = black_player.get_move(board)

        board.push(move)

    return board


if __name__ == "__main__":

    os.makedirs("notes", exist_ok=True)

    print("=== MCTS vs MCTS 테스트 ===")

    results = {
        "100_win": 0,
        "200_win": 0,
        "draw": 0
    }

    for game_idx in range(GAMES):

        print(f"\n===== Game {game_idx+1}/{GAMES} =====")

    # 색 번갈아
        if game_idx % 2 == 0:
            white_player = MCTSPlayer(model_path=MODEL_PATH, simulations=100, explore_moves=20, add_noise=True)
            black_player = MCTSPlayer(model_path=MODEL_PATH, simulations=200, explore_moves=20, add_noise=True)

            white_tag = "MCTS100"
            black_tag = "MCTS200" # (원본 코드의 200 태그 유지)

        else:
            white_player = MCTSPlayer(model_path=MODEL_PATH, simulations=200, explore_moves=20, add_noise=True)
            black_player = MCTSPlayer(model_path=MODEL_PATH, simulations=100, explore_moves=20, add_noise=True)

            white_tag = "MCTS200"
            black_tag = "MCTS100"

        board = play_single_game(white_player, black_player)

        result = board.result()
        print("Result:", result)

        # 승패 기록
        if result == "1-0":
            if white_tag == "MCTS100":
                results["100_win"] += 1
            else:
                results["200_win"] += 1

        elif result == "0-1":
            if black_tag == "MCTS100":
                results["100_win"] += 1
            else:
                results["200_win"] += 1

        else:
            results["draw"] += 1

        # PGN 저장
        game = chess.pgn.Game.from_board(board)

        game.headers["Event"] = "MCTS 100 vs 200 Test"
        game.headers["White"] = white_tag
        game.headers["Black"] = black_tag
        game.headers["Result"] = result

        filename = datetime.datetime.now().strftime(f"notes/mcts_test_%Y%m%d_%H%M%S_{game_idx}.pgn")

        with open(filename, "w", encoding="utf-8") as f:
            f.write(str(game))

        print("PGN saved:", filename)

    print("\n===== 최종 결과 =====")

    total = GAMES
    print(f"MCTS100 승: {results['100_win']}")
    print(f"MCTS200 승: {results['200_win']}")
    print(f"무승부: {results['draw']}")

    print("\n승률")
    print(f"MCTS100: {results['100_win']/total*100:.1f}%")
    print(f"MCTS200: {results['200_win']/total*100:.1f}%")