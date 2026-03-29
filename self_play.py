import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
import os
import time
import glob
import csv
import urllib.request
import re
import logging

from AI import TwoHeadChessCNN, ChessMCTS, board_to_tensor, MOVE_TO_ID
from config import device

# =========================================================================
# 0. 로깅 설정 (파일과 콘솔 동시 출력)
# =========================================================================
logger = logging.getLogger("ChessPipeline")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

file_handler = logging.FileHandler("pipeline_log.txt", encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def print(*args, **kwargs):
    msg = " ".join(map(str, args))
    logger.info(msg)

# =========================================================================
# 🔔 ntfy 알림 전송 도우미 함수
# =========================================================================
def send_ntfy_notification(message, topic="raspberry_4b"):
    try:
        url = f"https://ntfy.sh/{topic}"
        req = urllib.request.Request(url, data=message.encode('utf-8'), method="POST")
        req.add_header("Title", "Chess AI Pipeline")
        req.add_header("Tags", "chess_pawn,robot")
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"⚠️ ntfy 알림 전송 실패: {e}")

# =========================================================================
# 1. 자가 대국 전용 데이터셋
# =========================================================================
class SelfPlayDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.X = torch.from_numpy(data['X'])
        self.Y = torch.from_numpy(data['Y']).float()
        self.Z = torch.from_numpy(data['Z']).float().unsqueeze(1) 

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.Z[idx]

# =========================================================================
# 2. 자가 대국 워커 (데이터 생성)
# =========================================================================
def self_play_worker(worker_id, model_path, num_games, output_dir, simulations, lock, shared_chunk_counter):
    torch.set_num_threads(1)
    model = TwoHeadChessCNN().to(device)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
        else: model.load_state_dict(checkpoint)
    model.eval()

    X_chunk, Y_chunk, Z_chunk = [], [], []
    NUM_CLASSES = 4224

    for game_idx in range(num_games):
        board = chess.Board()
        mcts = ChessMCTS(model, device, num_simulations=simulations)
        
        states, policies, players = [], [], []
        move_count = 0
        auto_resigned = False
        global_z = 0.0

        while not board.is_game_over() and not board.can_claim_draw() and move_count < 150:
            temp = 1.0 if move_count < 30 else 0.0
            best_move = mcts.search(board, add_noise=True, temperature=temp)

            action_probs = np.zeros(NUM_CLASSES, dtype=np.float32)
            actions = list(mcts.root.children.keys())
            visits = np.array([mcts.root.children[a].visits for a in actions])
            
            if temp > 0.1: probs = visits / np.sum(visits)
            else:
                probs = np.zeros_like(visits, dtype=np.float32)
                probs[np.argmax(visits)] = 1.0

            for action, prob in zip(actions, probs):
                idx = MOVE_TO_ID.get(action.uci())
                if idx is not None: action_probs[idx] = prob

            states.append(board_to_tensor(board))
            policies.append(action_probs)
            players.append(board.turn)

            board.push(best_move)
            mcts.update_with_move(best_move)
            move_count += 1

            if mcts.root.q_value() < -0.90 and np.random.rand() > 0.1:
                auto_resigned = True
                global_z = -1.0 if board.turn == chess.WHITE else 1.0
                print(f"[Worker-{worker_id}] 🏳️ 자동 기권 발생! (턴 수: {move_count})")
                break

        if not auto_resigned:
            result = board.result()
            if result == '1-0': global_z = 1.0
            elif result == '0-1': global_z = -1.0
            else: global_z = 0.0

        for i in range(len(states)):
            turn = players[i]
            z_val = global_z if turn == chess.WHITE else -global_z
            X_chunk.append(states[i])
            Y_chunk.append(policies[i]) 
            Z_chunk.append(z_val)

        print(f"[Worker-{worker_id}] 게임 {game_idx+1}/{num_games} 완료 | 턴 수: {move_count}")

        if len(X_chunk) >= 1024:
            with lock:
                chunk_idx = shared_chunk_counter.value
                shared_chunk_counter.value += 1
                
            out_path = os.path.join(output_dir, f"selfplay_chunk_{chunk_idx:04d}.npz")
            np.savez_compressed(out_path, X=np.array(X_chunk, dtype=np.float32), Y=np.array(Y_chunk, dtype=np.float32), Z=np.array(Z_chunk, dtype=np.float32))
            X_chunk, Y_chunk, Z_chunk = [], [], []

    if len(X_chunk) > 0:
        with lock:
            chunk_idx = shared_chunk_counter.value
            shared_chunk_counter.value += 1
        out_path = os.path.join(output_dir, f"selfplay_chunk_{chunk_idx:04d}.npz")
        np.savez_compressed(out_path, X=np.array(X_chunk, dtype=np.float32), Y=np.array(Y_chunk, dtype=np.float32), Z=np.array(Z_chunk, dtype=np.float32))

# =========================================================================
# 3. 자가 대국 전용 학습 로직
# =========================================================================
def train_selfplay_model(data_dir="dataset_selfplay", batch_size=4096, epochs=1, lr=0.0001, pretrained_path=None, iteration=1):
    if device.type == 'cuda': torch.backends.cudnn.benchmark = True

    model = TwoHeadChessCNN(hidden_channels=256, num_res_blocks=20).to(device)
    criterion_policy = nn.CrossEntropyLoss() 
    criterion_value = nn.MSELoss() 
    
    # 학습률 0.0001 적용
    optimizer = optim.Adam(model.parameters(), lr=lr)

    log_file = "selfplay_training_log.csv"
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Iteration', 'Epoch', 'Chunk', 'Policy_Loss', 'Value_Loss', 'Accuracy', 'LR'])

    if pretrained_path and os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
        else: model.load_state_dict(checkpoint)

    for epoch in range(epochs):
        npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not npz_files: return

        model.train()
        current_lr = optimizer.param_groups[0]['lr']

        for file_idx, npz_file in enumerate(npz_files):
            dataset = SelfPlayDataset(npz_file)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
            
            total_p_loss, total_v_loss, correct_preds, total_samples = 0, 0, 0, 0
            
            for batch_X, batch_Y, batch_Z in dataloader:
                batch_X, batch_Y, batch_Z = batch_X.to(device, non_blocking=True), batch_Y.to(device, non_blocking=True), batch_Z.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    pred_policy, pred_value = model(batch_X)
                    loss_p = criterion_policy(pred_policy, batch_Y)
                    loss_v = criterion_value(pred_value, batch_Z)
                    loss = loss_p + (1.5 * loss_v) 
                
                loss.backward()
                optimizer.step()
                
                total_p_loss += loss_p.item() * batch_X.size(0)
                total_v_loss += loss_v.item() * batch_X.size(0)
                _, predicted = torch.max(pred_policy.data, 1)
                _, target_indices = torch.max(batch_Y, 1)
                total_samples += batch_Y.size(0)
                correct_preds += (predicted == target_indices).sum().item()

            chunk_p_loss = total_p_loss / total_samples
            chunk_v_loss = total_v_loss / total_samples
            chunk_acc = 100 * correct_preds / total_samples

            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([iteration, epoch+1, file_idx+1, f"{chunk_p_loss:.4f}", f"{chunk_v_loss:.4f}", f"{chunk_acc:.2f}", f"{current_lr:.6f}"])

    os.makedirs("model", exist_ok=True)
    save_path = f"model/chess_selfplay_iter_{iteration}_FINAL.pth"
    torch.save({'iteration': iteration, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, save_path)

# =========================================================================
# 4. 파이프라인 무한 루프 
# =========================================================================
def run_alphazero_pipeline(initial_model="model/model_v5.pth", output_dir="dataset_selfplay", games_per_iteration=100, simulations=100, epochs_per_train=1):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("model", exist_ok=True)
    
    # 🌟 버퍼 크기 설정 (약 100만 개의 포지션 데이터 유지)
    MAX_BUFFER_CHUNKS = 1000 
    
    latest_models = glob.glob(f"model/chess_selfplay_iter_*_FINAL.pth")
    
    if latest_models:
        current_model = max(latest_models, key=os.path.getctime)
        match = re.search(r'iter_(\d+)', os.path.basename(current_model))
        iteration = int(match.group(1)) + 1 if match else 1
        print(f"🔄 기존 기록 발견! Iteration {iteration}부터 재개합니다. (기존 모델: {current_model})")
    else:
        current_model = initial_model
        iteration = 1
        
    # 🌟 프로그램 시작/재시작 시점의 청크 개수를 파악하여 다음 1000개 단위 알림 기준점 설정
    initial_chunks = len(glob.glob(os.path.join(output_dir, "selfplay_chunk_*.npz")))
    next_notify_chunk = ((initial_chunks // 1000) + 1) * 1000
    
    while True:
        print(f"\n==================================================")
        print(f"🚀 [Iteration {iteration}] 자가 대국(Self-Play) 데이터 생성 시작")
        print(f"==================================================")
        
        num_workers = max(1, mp.cpu_count() - 1)
        games_per_worker = games_per_iteration // num_workers
        
        lock = mp.Lock()
        existing_chunks = glob.glob(os.path.join(output_dir, "selfplay_chunk_*.npz"))
        shared_chunk_counter = mp.Value('i', len(existing_chunks))
        
        workers = []
        for i in range(num_workers):
            p = mp.Process(target=self_play_worker, args=(i, current_model, games_per_worker, output_dir, simulations, lock, shared_chunk_counter))
            p.start()
            workers.append(p)

        for p in workers:
            p.join()

        # 현재 수집된 청크 개수 확인
        npz_files = sorted(glob.glob(os.path.join(output_dir, "selfplay_chunk_*.npz")))
        current_chunks = len(npz_files)

        # 슬라이딩 윈도우 로직 (오래된 데이터 삭제)
        if current_chunks > MAX_BUFFER_CHUNKS:
            files_to_delete = npz_files[:-MAX_BUFFER_CHUNKS]
            print(f"🧹 슬라이딩 윈도우 적용: 버퍼 초과. 오래된 데이터 {len(files_to_delete)}개를 삭제합니다.")
            for f in files_to_delete:
                try: os.remove(f)
                except OSError as e: print(f"⚠️ 파일 삭제 실패: {f} ({e})")
            current_chunks = MAX_BUFFER_CHUNKS # 삭제 후 개수 보정

        # 🌟 버퍼 웜업(충전) 대기 로직
        if current_chunks < MAX_BUFFER_CHUNKS:
            print(f"⏳ 데이터 수집 중... (현재 버퍼: {current_chunks}/{MAX_BUFFER_CHUNKS} 청크)")
            
            # 버퍼가 차는 동안에는 1000개 단위 돌파 시에만 알림 전송
            if current_chunks >= next_notify_chunk:
                send_ntfy_notification(f"⏳ [버퍼 충전 중] 현재 데이터가 {current_chunks}개에 도달했습니다! ({current_chunks}/{MAX_BUFFER_CHUNKS})")
                next_notify_chunk = ((current_chunks // 1000) + 1) * 1000 # 다음 알림 기준 업데이트
                
            iteration += 1
            continue # 버퍼가 꽉 찰 때까지 모델 학습을 건너뜀

        # 🌟 버퍼가 꽉 찬 이후(학습 단계): 매 이터레이션 시작 시 알림
        send_ntfy_notification(f"🚀 [Iteration {iteration}] 자가 대국 완료 (버퍼 100%). 모델 학습을 시작합니다!")

        print(f"\n✅ [Iteration {iteration}] 버퍼 준비 완료. 학습을 시작합니다.")

        # 모델 학습 진행
        train_selfplay_model(
            data_dir=output_dir, 
            batch_size=1024, 
            epochs=epochs_per_train, 
            lr=0.0001, 
            pretrained_path=current_model,
            iteration=iteration
        )
        
        # 🌟 버퍼가 꽉 찬 이후(학습 단계): 매 이터레이션 종료 시 알림
        send_ntfy_notification(f"✅ [Iteration {iteration}] 모델 학습 및 저장 완료! 다음 사이클을 준비합니다.")

        latest_models = glob.glob(f"model/chess_selfplay_iter_*_FINAL.pth")
        if latest_models:
            current_model = max(latest_models, key=os.path.getctime)
            print(f"🔄 다음 이터레이션에 사용할 모델로 교체: {current_model}")
            
        iteration += 1

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    run_alphazero_pipeline(
        initial_model="model/model_v5.pth", 
        output_dir="dataset_selfplay",      
        games_per_iteration=200,            
        simulations=100,                    
        epochs_per_train=1                  
    )
