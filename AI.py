import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import time

# ==========================================
# 1. 보드 & 이동 매핑 유틸리티
# ==========================================
def create_move_mapping():
    moves = []
    files = "abcdefgh"
    ranks = "12345678"
    squares = [f+r for f in files for r in ranks]

    for sq1 in squares:
        for sq2 in squares:
            if sq1 != sq2:
                moves.append(sq1 + sq2)

    promotions = ['q', 'r', 'b', 'n']
    for f1_idx, f1 in enumerate(files):
        for f2_idx in [f1_idx - 1, f1_idx, f1_idx + 1]: 
            if 0 <= f2_idx <= 7:
                f2 = files[f2_idx]
                for p in promotions:
                    moves.append(f"{f1}7{f2}8{p}")
                    moves.append(f"{f1}2{f2}1{p}")

    move_to_id = {move: idx for idx, move in enumerate(moves)}
    id_to_move = {idx: move for move, idx in move_to_id.items()}
    return move_to_id, id_to_move

MOVE_TO_ID, ID_TO_MOVE = create_move_mapping()
NUM_CLASSES = len(MOVE_TO_ID)

def board_to_tensor(board):
    tensor = np.zeros((14, 8, 8), dtype=np.float32)
    piece_to_channel = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            channel = piece_to_channel[piece.symbol()]
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            tensor[channel][7 - rank][file] = 1.0

    if board.turn == chess.WHITE:
        tensor[12].fill(1.0)
    else:
        tensor[12].fill(-1.0)

    if board.has_kingside_castling_rights(chess.WHITE): tensor[13][7][7] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE): tensor[13][7][0] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK): tensor[13][0][7] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK): tensor[13][0][0] = 1.0

    if board.ep_square is not None:
        ep_rank = chess.square_rank(board.ep_square)
        ep_file = chess.square_file(board.ep_square)
        tensor[13][7 - ep_rank][ep_file] = 1.0

    return tensor

# ==========================================
# 2. 투 헤드 신경망 (Policy & Value)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class TwoHeadChessCNN(nn.Module):
    def __init__(self, num_classes=4224, hidden_channels=256, num_res_blocks=20):
        super().__init__()
        self.conv_initial = nn.Conv2d(14, hidden_channels, kernel_size=3, padding=1)
        self.bn_initial = nn.BatchNorm2d(hidden_channels)
        
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_channels) for _ in range(num_res_blocks)
        ])

        # Policy Head
        self.conv_policy = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2 * 8 * 8, num_classes)

        # Value Head
        self.conv_value = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(1 * 8 * 8, 256)
        self.fc_value2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn_initial(self.conv_initial(x)))
        for block in self.res_blocks:
            x = block(x)

        p = F.relu(self.bn_policy(self.conv_policy(x)))
        p = p.view(p.size(0), -1)
        p = self.fc_policy(p)

        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.fc_value1(v))
        v = torch.tanh(self.fc_value2(v)) 
        return p, v

# ==========================================
# 3. 몬테카를로 트리 탐색 (MCTS)
# ==========================================
class MCTSNode:
    def __init__(self, parent=None, action=None, prior_prob=0.0):
        self.parent = parent
        self.action = action  
        self.children = {}  
        self.visits = 0       
        self.value_sum = 0.0  
        self.prior_prob = prior_prob  

    def q_value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def ucb_score(self, parent_visits, c_puct=2.0):
        q = -self.q_value()
        u = c_puct * self.prior_prob * math.sqrt(parent_visits) / (1 + self.visits)
        return q + u

    def is_expanded(self):
        return len(self.children) > 0

# ==========================================
# 3. 몬테카를로 트리 탐색 (MCTS)
# ==========================================
class MCTSNode:
    def __init__(self, parent=None, action=None, prior_prob=0.0):
        self.parent = parent
        self.action = action  
        self.children = {}  
        self.visits = 0       
        self.value_sum = 0.0  
        self.prior_prob = prior_prob  

    def q_value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def ucb_score(self, parent_visits, c_puct=2.0):
        q = -self.q_value()
        u = c_puct * self.prior_prob * math.sqrt(parent_visits) / (1 + self.visits)
        return q + u

    def is_expanded(self):
        return len(self.children) > 0

class ChessMCTS:
    def __init__(self, model, device, num_simulations=800): 
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.root = MCTSNode() # 🌟 개선점 3: 트리 재사용을 위해 루트 노드 유지

    def update_with_move(self, move):
        """실제 게임에서 상대방이나 내가 수를 두었을 때 트리를 해당 노드로 전진시킵니다."""
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None # 메모리 확보를 위해 부모 노드와의 연결 해제
        else:
            self.root = MCTSNode() # 예상치 못한 수일 경우 트리를 초기화

    def search(self, initial_board, add_noise=False, temperature=0.0):
        board = initial_board.copy()
        
        # 🌟 매번 새로 루트를 만드는 대신 유지된 self.root를 확장
        if not self.root.is_expanded():
            self.expand_node(self.root, board)
        
        # 탐색 전 루트 노드에 디리클레 노이즈 추가 (다양성 확보)
        if add_noise:
            self.add_dirichlet_noise(self.root)

        for _ in range(self.num_simulations):
            node = self.root
            search_path = [node]
            
            while node.is_expanded():
                best_action, best_child = max(
                    node.children.items(),
                    key=lambda item: item[1].ucb_score(node.visits)
                )
                node = best_child
                board.push(best_action) 
                search_path.append(node)

            value = self.evaluate_node(node, board)
            self.backpropagate(search_path, value, board)

        if not self.root.children:
            return random.choice(list(initial_board.legal_moves))
            
        # --- 온도(Temperature) 기반 확률적 수 선택 ---
        if temperature > 0:
            actions = list(self.root.children.keys())
            visits = np.array([self.root.children[a].visits for a in actions])
            
            visits = visits ** (1.0 / temperature)
            probs = visits / np.sum(visits)
            
            chosen_idx = np.random.choice(len(actions), p=probs)
            return actions[chosen_idx]
        else:
            return max(self.root.children.items(), key=lambda item: item[1].visits)[0]

    def add_dirichlet_noise(self, node, epsilon=0.25, alpha=0.3):
        actions = list(node.children.keys())
        if not actions: return
        noise = np.random.dirichlet([alpha] * len(actions))
        for i, action in enumerate(actions):
            child = node.children[action]
            child.prior_prob = (1 - epsilon) * child.prior_prob + epsilon * noise[i]

    def expand_node(self, node, board):
        policy_probs, _ = self.get_model_output(board)
        for move, prob in policy_probs.items():
            node.children[move] = MCTSNode(parent=node, action=move, prior_prob=prob)

    def evaluate_node(self, node, board):
        if board.is_game_over() or board.can_claim_draw():
            if board.is_checkmate(): return -1.0
            return 0.0
        _, value = self.get_model_output(board)
        self.expand_node(node, board)
        return value

    def get_model_output(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}, 0.0

        with torch.no_grad():
            tensor = board_to_tensor(board)
            tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
            logits, value_tensor = self.model(tensor)
            
            logits = logits[0].cpu().numpy()
            value = value_tensor.item()
            
            move_probs = {}
            legal_indices = []
            valid_moves = []
            
            for move in legal_moves:
                idx = MOVE_TO_ID.get(move.uci())
                if idx is not None:
                    legal_indices.append(idx)
                    valid_moves.append(move)
                    
            if legal_indices:
                # 🌟 개선점 1: 수치적으로 안정적인 Softmax 연산 적용 (오버플로우 방지)
                legal_logits = logits[legal_indices]
                max_logit = np.max(legal_logits)
                exp_logits = np.exp(legal_logits - max_logit)
                probs = exp_logits / np.sum(exp_logits)
                
                for move, prob in zip(valid_moves, probs):
                    move_probs[move] = prob
            else:
                # 예외 처리: 매핑되지 않은 수가 있을 경우 균등 분포
                prob = 1.0 / len(legal_moves)
                for move in legal_moves:
                    move_probs[move] = prob
                
        return move_probs, value
    
    def backpropagate(self, search_path, value, board):
        for node in reversed(search_path):
            node.visits += 1
            node.value_sum += value
            value = -value 
            if node.parent is not None:
                board.pop() 

# ==========================================
# 4. AI 봇 인터페이스 (직관 봇 & MCTS 봇)
# ==========================================
class AIPlayer:
    def is_human(self):
        return False

class CNNPlayer(AIPlayer):
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CNN 직관 봇] {model_path} 로딩 중... (장치: {self.device})")
        self.model = TwoHeadChessCNN().to(self.device) 
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        except FileNotFoundError:
            print(f"❌ 경고: {model_path} 없음. 무작위 수 반환.")
            self.model = None
        if self.model: self.model.eval()

    def get_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves: return None
        if self.model is None:
            time.sleep(0.5); return random.choice(legal_moves)

        with torch.no_grad():
            state_tensor = board_to_tensor(board)
            state_tensor = torch.from_numpy(state_tensor).unsqueeze(0).to(self.device)
            logits, _ = self.model(state_tensor) 
            sorted_indices = torch.argsort(logits[0], descending=True).cpu().numpy()
            
            for idx in sorted_indices:
                uci_move_str = ID_TO_MOVE[idx]
                move = chess.Move.from_uci(uci_move_str)
                if move in legal_moves:
                    time.sleep(0.1) 
                    return move
        return random.choice(legal_moves)

class MCTSPlayer(AIPlayer):
    def __init__(self, model_path, simulations=100, explore_moves=20, add_noise=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MCTS 수읽기 봇] {model_path} 로딩 중... (장치: {self.device})")
        self.model = TwoHeadChessCNN().to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        except FileNotFoundError:
            print(f"❌ 경고: {model_path} 없음. 무작위 수 반환.")
            self.model = None
            
        self.simulations = simulations
        self.explore_moves = explore_moves
        self.add_noise = add_noise
        self.last_move_count = 0 # 🌟 보드 기록 동기화를 위해 추가
        
        if self.model:
            self.model.eval() 
            self.mcts = ChessMCTS(model=self.model, device=self.device, num_simulations=self.simulations)

    def get_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves: return None
        if self.model is None:
            time.sleep(0.5); return random.choice(legal_moves)

        # 🌟 현재 보드 상태에 맞춰 밀린 수(상대방의 수 등)를 트리에 반영하여 동기화
        while self.last_move_count < len(board.move_stack):
            move = board.move_stack[self.last_move_count]
            self.mcts.update_with_move(move)
            self.last_move_count += 1

        print(f"\n🤔 [MCTS] {self.simulations}개의 가상 미래 탐색 중...")
        start_time = time.time()
        
        current_temp = 0.7 if len(board.move_stack) < self.explore_moves else 0.0
        
        best_move = self.mcts.search(board, add_noise=self.add_noise, temperature=current_temp)
        
        print(f"💡 [MCTS] 선택: {best_move} (탐색 시간: {time.time() - start_time:.2f}초)")
        return best_move