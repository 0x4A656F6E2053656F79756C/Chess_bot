import chess
import pygame
import threading
import time
import torch

# AI 관련 모듈 임포트
from AI import TwoHeadChessCNN, board_to_tensor
from config import device

class Player:
    def is_human(self): return True
    def get_move(self, board): pass

class HumanPlayer(Player):
    def is_human(self): return True

class ChessGame:
    # 파라미터에 spectator_mode=False 추가
    def __init__(self, white_player, black_player, model_path=None, spectator_mode=False):
        pygame.init()
        self.width, self.height = 900, 700
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        
        # 창 제목 변경
        pygame.display.set_caption("Chess Simulator" if not spectator_mode else "Chess Spectator Mode")
        self.clock = pygame.time.Clock()
        
        self.board = chess.Board()
        self.players = {chess.WHITE: white_player, chess.BLACK: black_player}
        
        # 관전 모드 여부와 FPS 저장 (관전 모드면 2 FPS로 낮춰서 연산량 대폭 감소)
        self.spectator_mode = spectator_mode 
        self.fps = 2 if spectator_mode else 60
        
        self.selected_square = None
        self.ai_thinking = False
        self.promotion_pending = None
        self.game_over = False
        self.game_result_text = ""
        self.dragging = False
        self.drag_pos = (0, 0)
        self.clicked_already_selected = False
        self.promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        
        # UI 버튼들
        self.resign_button_rect = pygame.Rect(0, 0, 0, 0)
        self.eval_button_rect = pygame.Rect(0, 0, 0, 0)
        self.flip_button_rect = pygame.Rect(0, 0, 0, 0)
        self.undo_button_rect = pygame.Rect(0, 0, 0, 0)
        self.heatmap_button_rect = pygame.Rect(0, 0, 0, 0) # 🌟 히트맵 버튼 추가
        self.explicit_result = None
        
        self.UI_WIDTH = 250 
        
        self.show_eval = False
        self.show_heatmap = False # 🌟 히트맵 토글 상태
        self.current_eval = 0.0
        self.last_eval_fen = ""
        self.flip_board = False 
        
        self.device = device
        self.model = None
        if model_path:
            try:
                self.model = TwoHeadChessCNN().to(self.device)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
                self.model.eval()
            except Exception as e:
                print(f"평가치용 모델 로드 실패: {e}")
                self.model = None
                
        self.update_dimensions(self.width, self.height)

    def update_dimensions(self, width, height):
        board_area_width = max(width - self.UI_WIDTH, 400) 
        min_dim = min(board_area_width, height)
        
        self.SQUARE_SIZE = int(min_dim * 0.8) // 8
        self.board_x = (board_area_width - self.SQUARE_SIZE * 8) // 2
        self.board_y = (height - self.SQUARE_SIZE * 8) // 2
        self.piece_images = self.load_images()
        
        btn_width, btn_height = 120, 45
        btn_x = width - self.UI_WIDTH + (self.UI_WIDTH - btn_width) // 2
        
        # 버튼들을 아래에서 위로 차례대로 쌓습니다.
        self.resign_button_rect = pygame.Rect(btn_x, height - btn_height - 30, btn_width, btn_height)
        self.eval_button_rect = pygame.Rect(btn_x, self.resign_button_rect.y - btn_height - 15, btn_width, btn_height)
        self.flip_button_rect = pygame.Rect(btn_x, self.eval_button_rect.y - btn_height - 15, btn_width, btn_height)
        self.undo_button_rect = pygame.Rect(btn_x, self.flip_button_rect.y - btn_height - 15, btn_width, btn_height)
        self.heatmap_button_rect = pygame.Rect(btn_x, self.undo_button_rect.y - btn_height - 15, btn_width, btn_height) # 히트맵 버튼 위치

    def get_visual_pos(self, file, rank):
        v_file = 7 - file if self.flip_board else file
        v_rank = rank if self.flip_board else 7 - rank
        return self.board_x + v_file * self.SQUARE_SIZE, self.board_y + v_rank * self.SQUARE_SIZE

    def load_images(self):
        symbols = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
        images = {}
        pygame.font.init()
        font = pygame.font.SysFont('Arial', max(10, self.SQUARE_SIZE * 2 // 3), bold=True)
        for symbol in symbols:
            color_prefix = 'w' if symbol.isupper() else 'b'
            filename = f"images/{color_prefix}{symbol.upper()}.png"
            try:
                img = pygame.image.load(filename)
                images[symbol] = pygame.transform.scale(img, (self.SQUARE_SIZE, self.SQUARE_SIZE))
            except:
                surf = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE), pygame.SRCALPHA)
                text_color = (255, 255, 255) if symbol.isupper() else (0, 0, 0)
                bg_color = (150, 150, 150) if symbol.isupper() else (100, 100, 100)
                pygame.draw.circle(surf, bg_color, (self.SQUARE_SIZE//2, self.SQUARE_SIZE//2), self.SQUARE_SIZE*3//8)
                text = font.render(symbol, True, text_color)
                text_rect = text.get_rect(center=(self.SQUARE_SIZE//2, self.SQUARE_SIZE//2))
                surf.blit(text, text_rect)
                images[symbol] = surf
        return images

    def _update_evaluation(self):
        if not self.model: return
        with torch.no_grad():
            tensor = board_to_tensor(self.board)
            tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
            _, value_tensor = self.model(tensor)
            val = value_tensor.item()
            if self.board.turn == chess.BLACK:
                val = -val
            self.current_eval = val

    def draw_board(self):
        self.screen.fill((40, 44, 52))
        
        ui_rect = pygame.Rect(self.width - self.UI_WIDTH, 0, self.UI_WIDTH, self.height)
        pygame.draw.rect(self.screen, (30, 33, 39), ui_rect)
        pygame.draw.line(self.screen, (60, 64, 72), (self.width - self.UI_WIDTH, 0), (self.width - self.UI_WIDTH, self.height), 2)
        
        colors = [pygame.Color(240, 217, 181), pygame.Color(181, 136, 99)]
        coord_font = pygame.font.SysFont('Arial', max(12, self.SQUARE_SIZE // 4), bold=True)
        
        for i in range(8):
            _, vy = self.get_visual_pos(0, i)
            rank_text = coord_font.render(str(i + 1), True, (200, 200, 200))
            rank_rect = rank_text.get_rect(center=(self.board_x - 15, vy + self.SQUARE_SIZE // 2))
            self.screen.blit(rank_text, rank_rect)
            
            vx, _ = self.get_visual_pos(i, 0)
            file_text = coord_font.render(chr(ord('a') + i), True, (200, 200, 200))
            file_rect = file_text.get_rect(center=(vx + self.SQUARE_SIZE // 2, self.board_y + 8 * self.SQUARE_SIZE + 15))
            self.screen.blit(file_text, file_rect)

        for rank in range(8):
            for file in range(8):
                color = colors[int((rank + file) % 2 == 0)]
                vx, vy = self.get_visual_pos(file, rank)
                rect = pygame.Rect(vx, vy, self.SQUARE_SIZE, self.SQUARE_SIZE)
                pygame.draw.rect(self.screen, color, rect)

        if self.board.move_stack:
            last_move = self.board.peek()
            for sq in (last_move.from_square, last_move.to_square):
                file, rank = chess.square_file(sq), chess.square_rank(sq)
                vx, vy = self.get_visual_pos(file, rank)
                rect = pygame.Rect(vx, vy, self.SQUARE_SIZE, self.SQUARE_SIZE)
                highlight = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE))
                highlight.set_alpha(80)
                highlight.fill((50, 150, 200))
                self.screen.blit(highlight, rect.topleft)

        if self.selected_square is not None:
            file, rank = chess.square_file(self.selected_square), chess.square_rank(self.selected_square)
            vx, vy = self.get_visual_pos(file, rank)
            rect = pygame.Rect(vx, vy, self.SQUARE_SIZE, self.SQUARE_SIZE)
            highlight = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE))
            highlight.set_alpha(100)
            highlight.fill((255, 255, 50))
            self.screen.blit(highlight, rect.topleft)

        # 🌟 히트맵 데이터 미리 계산
        top_moves_data = []
        if self.show_heatmap:
            # 🐛 버그 수정: 항상 '현재 턴'인 플레이어의 MCTS 트리를 참조하도록 변경
            current_player = self.players[self.board.turn]
            mcts_player = current_player if hasattr(current_player, 'mcts') and current_player.mcts is not None else None
            
            if mcts_player and hasattr(mcts_player.mcts, 'root'):
                try:
                    children = list(mcts_player.mcts.root.children.items())
                    valid_children = [(move, child) for move, child in children if child.visits > 0]
                    valid_children.sort(key=lambda x: x[1].visits, reverse=True)
                    
                    # 🌟 표시 개수를 5개에서 최상위 3개로 축소
                    top_children = valid_children[:3]
                    
                    if top_children:
                        max_v = top_children[0][1].visits
                        for idx, (move, child) in enumerate(top_children):
                            win_rate = (-child.q_value() + 1.0) / 2.0 * 100.0
                            top_moves_data.append({
                                'move': move,
                                'visits': child.visits,
                                'win_rate': win_rate,
                                'is_best': (idx == 0),
                                'intensity': child.visits / max_v if max_v > 0 else 0
                            })
                except RuntimeError:
                    pass

        # 🌟 1단계: 히트맵 바닥 색칠 및 빨간색 원 (기물 밑에 깔림)
        if self.show_heatmap and top_moves_data:
            drawn_sqs = set()
            for data in top_moves_data:
                to_sq = data['move'].to_square
                if to_sq not in drawn_sqs:
                    drawn_sqs.add(to_sq) 
                    vx, vy = self.get_visual_pos(chess.square_file(to_sq), chess.square_rank(to_sq))
                    overlay = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE), pygame.SRCALPHA)
                    
                    if data['is_best']:
                        overlay.fill((0, 200, 255, 140))
                    else:
                        r = int(255 * (1 - data['intensity']))
                        g = int(255 * data['intensity'])
                        overlay.fill((r, g, 0, 120))
                    
                    self.screen.blit(overlay, (vx, vy))

                    # 최선의 수인 경우: 사각형 테두리 대신 칸 중앙을 기준으로 빨간색 두꺼운 원 그리기
                    if data['is_best']:
                        center = (vx + self.SQUARE_SIZE // 2, vy + self.SQUARE_SIZE // 2)
                        radius = self.SQUARE_SIZE // 2 - 2
                        pygame.draw.circle(self.screen, (255, 0, 0), center, radius, 4)

        # 🌟 2단계: 기물 렌더링
        for rank in range(8):
            for file in range(8):
                sq = chess.square(file, rank)
                piece = self.board.piece_at(sq)
                if piece:
                    if self.dragging and sq == self.selected_square: continue
                    piece_image = self.piece_images.get(piece.symbol())
                    if piece_image:
                        vx, vy = self.get_visual_pos(file, rank)
                        rect = pygame.Rect(vx, vy, self.SQUARE_SIZE, self.SQUARE_SIZE)
                        self.screen.blit(piece_image, rect.topleft)

        if self.selected_square is not None:
            for move in self.board.legal_moves:
                if move.from_square == self.selected_square:
                    target_file, target_rank = chess.square_file(move.to_square), chess.square_rank(move.to_square)
                    vx, vy = self.get_visual_pos(target_file, target_rank)
                    circle_surface = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE), pygame.SRCALPHA)
                    if self.board.piece_at(move.to_square):
                        pygame.draw.circle(circle_surface, (0, 0, 0, 60), (self.SQUARE_SIZE // 2, self.SQUARE_SIZE // 2), self.SQUARE_SIZE // 2 - 1, max(3, self.SQUARE_SIZE//10))
                    else:
                        pygame.draw.circle(circle_surface, (0, 0, 0, 60), (self.SQUARE_SIZE // 2, self.SQUARE_SIZE // 2), self.SQUARE_SIZE // 6)
                    self.screen.blit(circle_surface, (vx, vy))

        if self.dragging and self.selected_square is not None:
            piece = self.board.piece_at(self.selected_square)
            if piece and self.piece_images.get(piece.symbol()):
                self.screen.blit(self.piece_images.get(piece.symbol()), self.piece_images.get(piece.symbol()).get_rect(center=self.drag_pos))

        # 🌟 3단계: 화살표 그리기 (더 두껍고 뚜렷하게)
        if self.show_heatmap and top_moves_data:
            import math
            for data in reversed(top_moves_data): 
                from_sq = data['move'].from_square
                to_sq = data['move'].to_square
                vx, vy = self.get_visual_pos(chess.square_file(to_sq), chess.square_rank(to_sq))
                fx, fy = self.get_visual_pos(chess.square_file(from_sq), chess.square_rank(from_sq))

                start_pos = (fx + self.SQUARE_SIZE // 2, fy + self.SQUARE_SIZE // 2)
                end_pos = (vx + self.SQUARE_SIZE // 2, vy + self.SQUARE_SIZE // 2)

                distance = math.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
                if distance > self.SQUARE_SIZE * 0.4:
                    angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
                    
                    adjusted_end = (
                        end_pos[0] - math.cos(angle) * (self.SQUARE_SIZE * 0.4),
                        end_pos[1] - math.sin(angle) * (self.SQUARE_SIZE * 0.4)
                    )

                    arrow_color = (255, 50, 50) if data['is_best'] else (180, 180, 180)
                    
                    # 선 두께 증가 (최선의 수: 6, 일반 수: 3)
                    line_thickness = 6 if data['is_best'] else 3

                    pygame.draw.line(self.screen, arrow_color, start_pos, adjusted_end, line_thickness)

                    # 화살표 머리(삼각형) 크기 증가 (12 -> 16으로 더 큼지막하게)
                    head_size = 16 
                    p1 = (adjusted_end[0] - head_size * math.cos(angle - math.pi/6), adjusted_end[1] - head_size * math.sin(angle - math.pi/6))
                    p2 = (adjusted_end[0] - head_size * math.cos(angle + math.pi/6), adjusted_end[1] - head_size * math.sin(angle + math.pi/6))
                    pygame.draw.polygon(self.screen, arrow_color, [adjusted_end, p1, p2])

        # 🌟 4단계: 텍스트 렌더링 (화살표, 기물 위에 항상 고정)
        if self.show_heatmap and top_moves_data:
            font_small = pygame.font.SysFont('Arial', max(11, self.SQUARE_SIZE // 5), bold=True)
            drawn_text_sqs = set()
            
            for data in top_moves_data:
                to_sq = data['move'].to_square
                
                if to_sq not in drawn_text_sqs:
                    drawn_text_sqs.add(to_sq)
                    
                    vx, vy = self.get_visual_pos(chess.square_file(to_sq), chess.square_rank(to_sq))
                    
                    mid_x = vx + self.SQUARE_SIZE // 2
                    mid_y = vy + self.SQUARE_SIZE // 2

                    text_v = font_small.render(f"N:{data['visits']}", True, (255, 255, 255))
                    text_q = font_small.render(f"{data['win_rate']:.1f}%", True, (255, 255, 255))

                    v_rect = text_v.get_rect(center=(mid_x, mid_y - 10))
                    q_rect = text_q.get_rect(center=(mid_x, mid_y + 10))

                    bg_rect = v_rect.union(q_rect).inflate(10, 6)
                    bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
                    bg_surface.fill((0, 0, 0, 180)) 
                    self.screen.blit(bg_surface, bg_rect.topleft)

                    self.screen.blit(text_v, v_rect)
                    self.screen.blit(text_q, q_rect)

        # --- 우측 UI 요소 렌더링 ---
        btn_font = pygame.font.SysFont('Arial', 18, bold=True)
        current_player = self.players[self.board.turn]
        
        # 1. 기권 버튼
        if not self.spectator_mode and current_player.is_human() and not self.game_over:
            pygame.draw.rect(self.screen, (200, 60, 60), self.resign_button_rect, border_radius=5)
            btn_text = btn_font.render("Resign", True, (255, 255, 255))
            self.screen.blit(btn_text, btn_text.get_rect(center=self.resign_button_rect.center))

        # 2. 평가치 토글 버튼
        if self.model:
            eval_color = (60, 160, 200) if self.show_eval else (80, 80, 80)
            pygame.draw.rect(self.screen, eval_color, self.eval_button_rect, border_radius=5)
            eval_btn_text = btn_font.render(f"Eval: {'ON' if self.show_eval else 'OFF'}", True, (255, 255, 255))
            self.screen.blit(eval_btn_text, eval_btn_text.get_rect(center=self.eval_button_rect.center))

        # 3. 보드 회전 버튼
        pygame.draw.rect(self.screen, (100, 150, 100), self.flip_button_rect, border_radius=5)
        flip_btn_text = btn_font.render("Flip Board", True, (255, 255, 255))
        self.screen.blit(flip_btn_text, flip_btn_text.get_rect(center=self.flip_button_rect.center))

        # 4. 무르기 (Undo) 버튼
        if not self.spectator_mode:
            undo_color = (200, 150, 50) if not self.ai_thinking and len(self.board.move_stack) >= 2 else (80, 80, 80)
            pygame.draw.rect(self.screen, undo_color, self.undo_button_rect, border_radius=5)
            undo_btn_text = btn_font.render("Undo (2 Moves)", True, (255, 255, 255))
            self.screen.blit(undo_btn_text, undo_btn_text.get_rect(center=self.undo_button_rect.center))

        # 5. 히트맵 버튼
        heatmap_color = (150, 80, 200) if self.show_heatmap else (80, 80, 80)
        pygame.draw.rect(self.screen, heatmap_color, self.heatmap_button_rect, border_radius=5)
        hm_btn_text = btn_font.render(f"Heatmap: {'ON' if self.show_heatmap else 'OFF'}", True, (255, 255, 255))
        self.screen.blit(hm_btn_text, hm_btn_text.get_rect(center=self.heatmap_button_rect.center))

        # 평가치 바 시각화
        if self.show_eval and self.model:
            current_fen = self.board.fen()
            if current_fen != self.last_eval_fen:
                self._update_evaluation()
                self.last_eval_fen = current_fen

            bar_width = 30
            bar_height = 250
            eval_x = self.width - self.UI_WIDTH + (self.UI_WIDTH - bar_width) // 2
            eval_y = (self.height - bar_height) // 2 - 130 

            white_ratio = (self.current_eval + 1.0) / 2.0
            white_ratio = max(0.0, min(1.0, white_ratio)) 
            
            white_height = int(bar_height * white_ratio)
            black_height = bar_height - white_height

            pygame.draw.rect(self.screen, (20, 20, 20), (eval_x-2, eval_y-2, bar_width+4, bar_height+4))
            pygame.draw.rect(self.screen, (50, 50, 50), (eval_x, eval_y, bar_width, black_height))
            pygame.draw.rect(self.screen, (230, 230, 230), (eval_x, eval_y + black_height, bar_width, white_height))

            eval_val_font = pygame.font.SysFont('Arial', 22, bold=True)
            eval_str = f"{self.current_eval:+.2f}"
            val_text = eval_val_font.render(eval_str, True, (255, 255, 255))
            self.screen.blit(val_text, val_text.get_rect(center=(eval_x + bar_width // 2, eval_y - 20)))

        if self.promotion_pending: self.draw_promotion_menu()
        if self.game_over: self.draw_game_over_screen()
        pygame.display.flip()

    def draw_promotion_menu(self):
        menu_width, menu_height = self.SQUARE_SIZE * 4, self.SQUARE_SIZE
        start_x = self.board_x + (self.SQUARE_SIZE * 8 - menu_width) // 2
        start_y = self.board_y + (self.SQUARE_SIZE * 8 - menu_height) // 2

        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))

        menu_rect = pygame.Rect(start_x, start_y, menu_width, menu_height)
        pygame.draw.rect(self.screen, (200, 200, 200), menu_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), menu_rect, 2)

        for i, piece_type in enumerate(self.promotion_pieces):
            piece_symbol = chess.piece_symbol(piece_type).upper() if self.board.turn == chess.WHITE else chess.piece_symbol(piece_type)
            if img := self.piece_images.get(piece_symbol):
                self.screen.blit(img, img.get_rect(center=(start_x + self.SQUARE_SIZE * i + self.SQUARE_SIZE // 2, start_y + self.SQUARE_SIZE // 2)))

    def draw_game_over_screen(self):
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        font = pygame.font.SysFont('Arial', max(24, self.SQUARE_SIZE // 2), bold=True)
        text_surf = font.render(self.game_result_text, True, (255, 255, 255))
        self.screen.blit(text_surf, text_surf.get_rect(center=(self.width // 2, self.height // 2)))

    def handle_ai_move(self):
        current_player = self.players[self.board.turn]
        def ai_worker():
            move = current_player.get_move(self.board)
            if move and move in self.board.legal_moves: self.board.push(move)
            self.ai_thinking = False
        self.ai_thinking = True
        threading.Thread(target=ai_worker, daemon=True).start()

    def _try_make_move(self, from_sq, to_sq):
        move = chess.Move(from_sq, to_sq)
        moving_piece = self.board.piece_at(from_sq)
        
        if moving_piece and moving_piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in [0, 7]:
            if chess.Move(from_sq, to_sq, promotion=chess.QUEEN) in self.board.legal_moves:
                self.promotion_pending = (from_sq, to_sq)
                return True 
        elif move in self.board.legal_moves:
            self.board.push(move)
            return True
        return False

    def run(self):
        running = True
        while running:
            if not self.game_over:
                if self.board.is_game_over():
                    self.game_over = True
                    outcome = self.board.outcome()
                    self.game_result_text = f"Game Over: {self.board.result()} ({outcome.termination.name if outcome else 'Unknown'})"
                elif self.board.can_claim_draw():
                    self.game_over, self.game_result_text = True, "Game Over: 1/2-1/2 (Draw Claimable)"

            current_player = self.players[self.board.turn]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.width, self.height = event.w, event.h
                    self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                    self.update_dimensions(self.width, self.height)
                elif self.promotion_pending and event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    menu_width = self.SQUARE_SIZE * 4
                    start_x, start_y = self.board_x + (self.SQUARE_SIZE * 8 - menu_width) // 2, self.board_y + (self.SQUARE_SIZE * 8 - self.SQUARE_SIZE) // 2
                    if start_y <= y <= start_y + self.SQUARE_SIZE and start_x <= x <= start_x + menu_width:
                        if 0 <= (index := (x - start_x) // self.SQUARE_SIZE) < 4:
                            self.board.push(chess.Move(self.promotion_pending[0], self.promotion_pending[1], promotion=self.promotion_pieces[index]))
                            self.promotion_pending = None
                
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.eval_button_rect.collidepoint(event.pos) and self.model:
                        self.show_eval = not self.show_eval
                        continue
                        
                    if self.flip_button_rect.collidepoint(event.pos):
                        self.flip_board = not self.flip_board
                        continue
                        
                    if self.heatmap_button_rect.collidepoint(event.pos):
                        self.show_heatmap = not self.show_heatmap
                        continue
                    
                    # 🌟 관전 모드에서는 기권, 무르기 버튼 기능 차단
                    if not self.spectator_mode:
                        if self.undo_button_rect.collidepoint(event.pos):
                            if not self.ai_thinking and len(self.board.move_stack) >= 2:
                                self.board.pop(); self.board.pop() 
                                self.selected_square = None; self.dragging = False; self.game_over = False
                                self.explicit_result = None; self.promotion_pending = None
                            continue
                            
                        if self.resign_button_rect.collidepoint(event.pos):
                            if current_player.is_human() and not self.game_over:
                                self.game_over = True
                                self.explicit_result = "0-1" if self.board.turn == chess.WHITE else "1-0"
                                winner = "Black" if self.board.turn == chess.WHITE else "White"
                                loser = "White" if self.board.turn == chess.WHITE else "Black"
                                self.game_result_text = f"Game Over: {winner} wins ({loser} Resigned)"
                            continue

                    # 🌟 관전 모드에서는 마우스 드래그 조작 전면 차단
                    if self.spectator_mode:
                        continue

                    if not self.ai_thinking and current_player.is_human() and not self.promotion_pending and not self.game_over:
                        bx, by = event.pos[0] - self.board_x, event.pos[1] - self.board_y
                        if 0 <= bx < self.SQUARE_SIZE * 8 and 0 <= by < self.SQUARE_SIZE * 8:
                            v_file, v_rank = bx // self.SQUARE_SIZE, by // self.SQUARE_SIZE
                            file = 7 - v_file if self.flip_board else v_file
                            rank = v_rank if self.flip_board else 7 - v_rank
                            clicked_square = chess.square(file, rank)
                            
                            if self.selected_square is None:
                                piece = self.board.piece_at(clicked_square)
                                if piece and piece.color == self.board.turn:
                                    self.selected_square = clicked_square; self.dragging = True; self.drag_pos = event.pos
                            else:
                                if clicked_square == self.selected_square:
                                    self.dragging = True; self.drag_pos = event.pos; self.clicked_already_selected = True 
                                else:
                                    piece = self.board.piece_at(clicked_square)
                                    if piece and piece.color == self.board.turn:
                                        self.selected_square = clicked_square; self.dragging = True; self.drag_pos = event.pos; self.clicked_already_selected = False
                                    else:
                                        if self._try_make_move(self.selected_square, clicked_square): self.selected_square = None
                                        self.clicked_already_selected = False
                                        
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                    if not self.spectator_mode and not self.ai_thinking and current_player.is_human() and not self.game_over:
                        self.selected_square, self.dragging, self.clicked_already_selected = None, False, False
                        
                elif event.type == pygame.MOUSEMOTION:
                    if not self.spectator_mode and self.dragging: self.drag_pos = event.pos
                    
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if not self.spectator_mode and self.dragging:
                        self.dragging = False
                        if self.selected_square is not None:
                            bx, by = event.pos[0] - self.board_x, event.pos[1] - self.board_y
                            if 0 <= bx < self.SQUARE_SIZE * 8 and 0 <= by < self.SQUARE_SIZE * 8:
                                v_file, v_rank = bx // self.SQUARE_SIZE, by // self.SQUARE_SIZE
                                file, rank = 7 - v_file if self.flip_board else v_file, v_rank if self.flip_board else 7 - v_rank
                                released_square = chess.square(file, rank)
                                if released_square != self.selected_square:
                                    if self._try_make_move(self.selected_square, released_square): self.selected_square = None
                                elif getattr(self, 'clicked_already_selected', False): self.selected_square = None
                            else: self.selected_square = None
                            self.clicked_already_selected = False

            if not self.spectator_mode and not self.ai_thinking and not current_player.is_human() and not self.promotion_pending and not self.game_over:
                self.handle_ai_move()

            self.draw_board()
            
            # 🌟 관전 모드면 2 FPS로 떨어뜨려 엄청난 최적화 달성
            if self.ai_thinking:
                self.clock.tick(1) 
            else:
                self.clock.tick(self.fps) 
        pygame.quit()