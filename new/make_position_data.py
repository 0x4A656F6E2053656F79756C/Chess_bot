import chess
import chess.engine
import chess.pgn
import csv
import multiprocessing
import os
from tqdm import tqdm

# ==========================================
# --- ⚙️ 업데이트된 설정 값 ---
# ==========================================
STOCKFISH_PATH = "/usr/games/stockfish" 
PGN_FILE = "./2013-01.pgn"

# 💡 파일명 변경 (기존 데이터와 섞임 방지)
CSV_OUTPUT = "./chess_training_data_2000.csv" 

# 💡 100만 개로 넉넉하게 잡아 파일 끝까지 파싱
MAX_GAMES_TO_READ = 1000000       

TIME_LIMIT_PER_MOVE = 0.1       
MULTI_PV = 3                    
# ==========================================

def get_already_processed_fens(csv_path):
    processed = set()
    if os.path.exists(csv_path):
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row["FEN"])
    return processed

def extract_fens_from_pgn(pgn_path, max_games):
    fens = set()
    games_processed = 0
    
    with open(pgn_path, "r", encoding="utf-8") as pgn:
        while games_processed < max_games:
            game = chess.pgn.read_game(pgn)
            if game is None: 
                break # 파일 끝에 도달하면 안전하게 종료
            
            white_elo_str = game.headers.get("WhiteElo", "0")
            black_elo_str = game.headers.get("BlackElo", "0")
            
            if not white_elo_str.isdigit() or not black_elo_str.isdigit():
                continue
                
            # 💡 [핵심] 양측 레이팅 2000 이상으로 고급 데이터만 추출
            if int(white_elo_str) >= 2000 and int(black_elo_str) >= 2000:
                games_processed += 1
                board = game.board()
                for i, move in enumerate(game.mainline_moves()):
                    board.push(move)
                    # 💡 [핵심] 10수(5턴)만 스킵하여 미들게임 초입부터 학습
                    if i > 10: 
                        fens.add(board.fen())
                        
    return fens 

def evaluate_fen_chunk(fens_chunk):
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    results = []
    
    for fen in fens_chunk:
        board = chess.Board(fen)
        if board.is_game_over():
            continue
            
        try:
            info = engine.analyse(board, chess.engine.Limit(time=TIME_LIMIT_PER_MOVE), multipv=MULTI_PV)
            row_data = {"FEN": fen}
            
            for i, line in enumerate(info):
                rank = i + 1
                move_str = line["pv"][0].uci() if "pv" in line else ""
                score = line["score"].pov(chess.WHITE).score(mate_score=10000) 
                
                row_data[f"Move{rank}"] = move_str
                row_data[f"Eval{rank}"] = score
            
            for i in range(len(info) + 1, MULTI_PV + 1):
                row_data[f"Move{i}"] = ""
                row_data[f"Eval{i}"] = ""
                
            results.append(row_data)
        except Exception:
            pass
            
    engine.quit()
    return results

def main():
    print("1. 기존 저장된 데이터 확인 중...")
    processed_fens = get_already_processed_fens(CSV_OUTPUT)
    print(f"-> 이미 완료된 FEN: {len(processed_fens)}개")

    print(f"2. PGN 파일에서 FEN 추출 시작 (Elo 2000+, 10수 스킵)...")
    all_fens = extract_fens_from_pgn(PGN_FILE, MAX_GAMES_TO_READ)
    
    fens_to_process = list(all_fens - processed_fens)
    print(f"-> 총 {len(all_fens)}개 중 남은 작업 대상: {len(fens_to_process)}개\n")

    if len(fens_to_process) == 0:
        print("모든 FEN 평가가 이미 완료되었습니다!")
        return

    # OS 및 모델 학습(GPU Dataloader)을 위해 코어 2개 비워두기
    total_cores = multiprocessing.cpu_count()
    num_cores = max(1, total_cores - 2) 
    
    num_chunks = num_cores * 10
    chunk_size = max(1, len(fens_to_process) // num_chunks)
    fen_chunks = [fens_to_process[i:i + chunk_size] for i in range(0, len(fens_to_process), chunk_size)]
    
    print(f"3. {total_cores}개의 전체 코어 중 {num_cores}개를 사용하여 엔진 평가 시작 (남은 작업 {len(fen_chunks)} 단위)...")
    
    fieldnames = ["FEN", "Move1", "Eval1", "Move2", "Eval2", "Move3", "Eval3"]
    file_exists = os.path.exists(CSV_OUTPUT)
    
    with open(CSV_OUTPUT, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        with multiprocessing.Pool(processes=num_cores) as pool:
            for result_chunk in tqdm(pool.imap_unordered(evaluate_fen_chunk, fen_chunks), total=len(fen_chunks), desc="평가 진행률"):
                for row in result_chunk:
                    writer.writerow(row)
                csv_file.flush() 
                
    print(f"완료! 데이터가 {CSV_OUTPUT}에 안전하게 저장되었습니다.")

if __name__ == "__main__":
    main()