import chess
import torch
import os
import glob
from AI import TwoHeadChessCNN, MCTSPlayer
from config import device

def play_game(white_player, black_player):
    board = chess.Board()
    # 🌟 공정한 대결을 위해 턴 수를 제한하거나 무승부 조건을 체크합니다.
    move_count = 0
    while not board.is_game_over() and move_count < 150:
        current_player = white_player if board.turn == chess.WHITE else black_player
        
        # 🌡️ 실력 측정을 위한 온도 전략: 
        # 처음 8수(4무브)까지만 탐험(1.0)을 허용하고 이후엔 최선의 수(0.0)만 둡니다.
        temp = 1.0 if move_count < 8 else 0.0
        
        # MCTSPlayer의 get_move를 활용하되, 내부적으로 temp가 적용되도록 합니다.
        # (필요 시 MCTSPlayer 클래스의 get_move에 temp 인자를 추가하거나 
        # 아래처럼 직접 mcts.search를 호출할 수 있습니다.)
        move = current_player.mcts.search(board, add_noise=False, temperature=temp)
        
        board.push(move)
        current_player.mcts.update_with_move(move)
        # 상대방 플레이어의 트리도 동기화해줍니다.
        other_player = black_player if board.turn == chess.WHITE else white_player
        other_player.mcts.update_with_move(move)
        
        move_count += 1
    
    return board.result()

def run_arena(model_path_a, model_path_b, total_games=10):
    print(f"⚔️  대결 시작: {os.path.basename(model_path_a)} vs {os.path.basename(model_path_b)}")
    print(f"Using device: {device}\n")

    # 플레이어 초기화 (수읽기 깊이는 공평하게 100으로 설정)
    p_a = MCTSPlayer(model_path_a, simulations=100)
    p_b = MCTSPlayer(model_path_b, simulations=100)

    results = {"A_win": 0, "B_win": 0, "Draw": 0}

    for i in range(total_games):
        # 진영 교대 (공정성을 위해 짝수 판은 진영을 바꿉니다)
        if i % 2 == 0:
            white, black = p_a, p_b
            side_a = chess.WHITE
        else:
            white, black = p_b, p_a
            side_a = chess.BLACK
            
        print(f"🎮 [{i+1}/{total_games}] 경기 진행 중...", end="\r")
        res_str = play_game(white, black)
        
        # 결과 집계
        if res_str == "1-0":
            if side_a == chess.WHITE: results["A_win"] += 1
            else: results["B_win"] += 1
        elif res_str == "0-1":
            if side_a == chess.BLACK: results["A_win"] += 1
            else: results["B_win"] += 1
        else:
            results["Draw"] += 1
            
        # 매 게임 후 트리 초기화 (다음 게임에 영향 주지 않기 위해)
        from AI import ChessMCTS
        p_a.mcts = ChessMCTS(p_a.model, device, num_simulations=100)
        p_b.mcts = ChessMCTS(p_b.model, device, num_simulations=100)

    print(f"\n\n🏆 최종 결과:")
    print(f" - {os.path.basename(model_path_a)}: {results['A_win']}승")
    print(f" - {os.path.basename(model_path_b)}: {results['B_win']}승")
    print(f" - 무승부: {results['Draw']}회")
    
    win_rate = (results['A_win'] + 0.5 * results['Draw']) / total_games * 100
    print(f"\n📊 {os.path.basename(model_path_a)}의 기대 승률: {win_rate:.1f}%")

if __name__ == "__main__":
    # 🔍 비교할 모델 경로 설정
    OLD_MODEL = "model/model_v2.pth"
    NEW_MODEL = "model/model_v5.pth"

    run_arena(NEW_MODEL, OLD_MODEL, total_games=20) # 20판 정도 대결