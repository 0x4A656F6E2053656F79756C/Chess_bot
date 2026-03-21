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
import csvr
import urllib.request # 🔔 ntfy 알림 전송용 내장 라이브러리 추가
import re

from AI import TwoHeadChessCNN, ChessMCTS, board_to_tensor, MOVE_TO_ID
from config import device

# =========================================================================
# 🔔 ntfy 알림 전송 도우미 함수
# =========================================================================
def send_ntfy_notification(message, topic="raspberry_4b"):
    try:
        url = f"https://ntfy.sh/{topic}"
        req = urllib.request.Request(url, data=message.encode('utf-8'), method="POST")
        # 한글 깨짐 방지를 위해 헤더 추가
        req.add_header("Title", "Chess AI Pipeline")
        req.add_header("Tags", "chess_pawn,robot")
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"⚠️ ntfy 알림 전송 실패: {e}")

# =========================================================================
# 1. 자가 대국 전용 데이터셋 (확률 분포 지원)
# =========================================================================
class SelfPlayDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.X = torch.from_numpy(data['X'])
        self.Y = torch.from_numpy(data['Y']).float() # 4224차원 확률 분포
        self.Z = torch.from_numpy(data['Z']).float().unsqueeze(1) 

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.Z[idx]

# =========================================================================
# 2. 자가 대국 워커 (데이터 생성)
# =========================================================================
def self_play_worker(worker_id, model_path, num_games, output_dir, simulations, lock, shared_chunk_counter):
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
def train_selfplay_model(data_dir="dataset_selfplay", batch_size=4096, epochs=1, lr=0.0005, pretrained_path=None, iteration=1):
    if device.type == 'cuda': torch.backends.cudnn.benchmark = True

    model = TwoHeadChessCNN(hidden_channels=256, num_res_blocks=20).to(device)
    criterion_policy = nn.CrossEntropyLoss() 
    criterion_value = nn.MSELoss() 
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
    
    for npz_file in npz_files:
        try: os.remove(npz_file)
        except OSError: pass

# =========================================================================
# 4. 파이프라인 무한 루프 
# =========================================================================
def run_alphazero_pipeline(initial_model="model/model_v5.pth", output_dir="dataset_selfplay", games_per_iteration=100, simulations=100, epochs_per_train=1):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("model", exist_ok=True)
    
    # 🌟 [수정] 기존에 학습된 가장 최신 이터레이션 모델 찾기
    latest_models = glob.glob(f"model/chess_selfplay_iter_*_FINAL.pth")
    
    if latest_models:
        current_model = max(latest_models, key=os.path.getctime)
        # 파일명에서 숫자 추출 (예: iter_5 -> 5)
        match = re.search(r'iter_(\d+)', os.path.basename(current_model))
        iteration = int(match.group(1)) + 1 if match else 1
        print(f"🔄 기존 기록 발견! Iteration {iteration}부터 재개합니다. (기존 모델: {current_model})")
    else:
        current_model = initial_model
        iteration = 1
    
    while True:
        print(f"\n==================================================")
        print(f"🚀 [Iteration {iteration}] 자가 대국(Self-Play) 데이터 생성 시작")
        print(f"==================================================")
        
        num_workers = max(1, mp.cpu_count() - 2)
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

        # 🔔 조건: 1회차 또는 10의 배수 회차일 때 (10, 20, 30...) 데이터 생성 완료 및 학습 시작 알림 전송
        should_notify = (iteration == 1 or iteration % 10 == 0)

        if should_notify:
            send_ntfy_notification(f"🚀 [Iteration {iteration}] 자가 대국 데이터 생성 완료. 모델 학습을 시작합니다!")

        print(f"\n✅ [Iteration {iteration}] 자가 대국 완료. 학습을 시작합니다.")

        # 모델 학습 진행
        train_selfplay_model(
            data_dir=output_dir, 
            batch_size=1024, 
            epochs=epochs_per_train, 
            lr=0.0005, 
            pretrained_path=current_model,
            iteration=iteration
        )
        
        # 🔔 학습이 끝나면 완료 알림 전송
        if should_notify:
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