import chess
import pygame
import threading
import time

class Player:
    def is_human(self): return True
    def get_move(self, board): pass

class HumanPlayer(Player):
    def is_human(self): return True

class ChessGame:
    def __init__(self, white_player, black_player):
        pygame.init()
        self.width, self.height = 900, 700  # 우측 패널을 위해 기본 너비를 늘렸습니다.
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("Chess Simulator")
        self.clock = pygame.time.Clock()
        
        self.board = chess.Board()
        self.players = {chess.WHITE: white_player, chess.BLACK: black_player}
        
        self.selected_square = None
        self.ai_thinking = False
        self.promotion_pending = None
        self.game_over = False
        self.game_result_text = ""
        self.dragging = False
        self.drag_pos = (0, 0)
        self.clicked_already_selected = False
        self.promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        
        self.resign_button_rect = pygame.Rect(0, 0, 0, 0)
        self.explicit_result = None
        
        # 우측 UI 패널 너비 지정
        self.UI_WIDTH = 250 
        
        self.update_dimensions(self.width, self.height)

    def update_dimensions(self, width, height):
        # 체스판이 그려질 왼쪽 영역 계산
        board_area_width = max(width - self.UI_WIDTH, 400) 
        min_dim = min(board_area_width, height)
        
        self.SQUARE_SIZE = int(min_dim * 0.8) // 8
        self.board_x = (board_area_width - self.SQUARE_SIZE * 8) // 2
        self.board_y = (height - self.SQUARE_SIZE * 8) // 2
        self.piece_images = self.load_images()
        
        # 기권 버튼 위치 설정 (우측 UI 패널 하단 중앙)
        btn_width, btn_height = 120, 45
        self.resign_button_rect = pygame.Rect(
            width - self.UI_WIDTH + (self.UI_WIDTH - btn_width) // 2,
            height - btn_height - 40, # 바닥에서 40px 위
            btn_width, btn_height
        )

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

    def draw_board(self):
        self.screen.fill((40, 44, 52)) # 전체 배경색
        
        # --- 우측 UI 패널 배경 ---
        ui_rect = pygame.Rect(self.width - self.UI_WIDTH, 0, self.UI_WIDTH, self.height)
        pygame.draw.rect(self.screen, (30, 33, 39), ui_rect) # 약간 더 어두운 색상
        pygame.draw.line(self.screen, (60, 64, 72), (self.width - self.UI_WIDTH, 0), (self.width - self.UI_WIDTH, self.height), 2) # 구분선
        
        colors = [pygame.Color(240, 217, 181), pygame.Color(181, 136, 99)]
        coord_font = pygame.font.SysFont('Arial', max(12, self.SQUARE_SIZE // 4), bold=True)
        
        # 좌표(a~h, 1~8) 그리기
        for i in range(8):
            rank_text = coord_font.render(str(i + 1), True, (200, 200, 200))
            rank_rect = rank_text.get_rect(center=(self.board_x - 15, self.board_y + (7 - i) * self.SQUARE_SIZE + self.SQUARE_SIZE // 2))
            self.screen.blit(rank_text, rank_rect)
            file_text = coord_font.render(chr(ord('a') + i), True, (200, 200, 200))
            file_rect = file_text.get_rect(center=(self.board_x + i * self.SQUARE_SIZE + self.SQUARE_SIZE // 2, self.board_y + 8 * self.SQUARE_SIZE + 15))
            self.screen.blit(file_text, file_rect)

        # 체스판 타일 그리기
        for rank in range(8):
            for file in range(8):
                color = colors[int((rank + file) % 2 == 0)]
                rect = pygame.Rect(self.board_x + file * self.SQUARE_SIZE, self.board_y + (7 - rank) * self.SQUARE_SIZE, self.SQUARE_SIZE, self.SQUARE_SIZE)
                pygame.draw.rect(self.screen, color, rect)

        if self.board.move_stack:
            last_move = self.board.peek()
            for sq in (last_move.from_square, last_move.to_square):
                file, rank = chess.square_file(sq), chess.square_rank(sq)
                rect = pygame.Rect(self.board_x + file * self.SQUARE_SIZE, self.board_y + (7 - rank) * self.SQUARE_SIZE, self.SQUARE_SIZE, self.SQUARE_SIZE)
                highlight = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE))
                highlight.set_alpha(80)
                highlight.fill((50, 150, 200))
                self.screen.blit(highlight, rect.topleft)

        if self.selected_square is not None:
            file, rank = chess.square_file(self.selected_square), chess.square_rank(self.selected_square)
            rect = pygame.Rect(self.board_x + file * self.SQUARE_SIZE, self.board_y + (7 - rank) * self.SQUARE_SIZE, self.SQUARE_SIZE, self.SQUARE_SIZE)
            highlight = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE))
            highlight.set_alpha(100)
            highlight.fill((255, 255, 50))
            self.screen.blit(highlight, rect.topleft)

        for rank in range(8):
            for file in range(8):
                sq = chess.square(file, rank)
                piece = self.board.piece_at(sq)
                if piece:
                    if self.dragging and sq == self.selected_square: continue
                    piece_image = self.piece_images.get(piece.symbol())
                    if piece_image:
                        rect = pygame.Rect(self.board_x + file * self.SQUARE_SIZE, self.board_y + (7 - rank) * self.SQUARE_SIZE, self.SQUARE_SIZE, self.SQUARE_SIZE)
                        self.screen.blit(piece_image, rect.topleft)

        if self.selected_square is not None:
            for move in self.board.legal_moves:
                if move.from_square == self.selected_square:
                    target_file, target_rank = chess.square_file(move.to_square), chess.square_rank(move.to_square)
                    circle_surface = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE), pygame.SRCALPHA)
                    if self.board.piece_at(move.to_square):
                        pygame.draw.circle(circle_surface, (0, 0, 0, 60), (self.SQUARE_SIZE // 2, self.SQUARE_SIZE // 2), self.SQUARE_SIZE // 2 - 1, max(3, self.SQUARE_SIZE//10))
                    else:
                        pygame.draw.circle(circle_surface, (0, 0, 0, 60), (self.SQUARE_SIZE // 2, self.SQUARE_SIZE // 2), self.SQUARE_SIZE // 6)
                    self.screen.blit(circle_surface, (self.board_x + target_file * self.SQUARE_SIZE, self.board_y + (7 - target_rank) * self.SQUARE_SIZE))

        if self.dragging and self.selected_square is not None:
            piece = self.board.piece_at(self.selected_square)
            if piece and self.piece_images.get(piece.symbol()):
                self.screen.blit(self.piece_images.get(piece.symbol()), self.piece_images.get(piece.symbol()).get_rect(center=self.drag_pos))

        # --- 기권 버튼 그리기 ---
        current_player = self.players[self.board.turn]
        if current_player.is_human() and not self.game_over:
            pygame.draw.rect(self.screen, (200, 60, 60), self.resign_button_rect, border_radius=5)
            btn_font = pygame.font.SysFont('Arial', 18, bold=True)
            btn_text = btn_font.render("Resign", True, (255, 255, 255))
            self.screen.blit(btn_text, btn_text.get_rect(center=self.resign_button_rect.center))
            
            # 추후 타이머 위치를 표시하기 위한 텍스트 (임시)
            timer_font = pygame.font.SysFont('Arial', 24, bold=True)
            timer_text = timer_font.render("Clock Space", True, (150, 150, 150))
            self.screen.blit(timer_text, timer_text.get_rect(center=(self.width - self.UI_WIDTH//2, 100)))

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
                elif not self.ai_thinking and current_player.is_human() and not self.promotion_pending and not self.game_over:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1: 
                            
                            # --- 기권 버튼 클릭 확인 ---
                            if self.resign_button_rect.collidepoint(event.pos):
                                self.game_over = True
                                # 👇 백이 기권하면 0-1, 흑이 기권하면 1-0 저장
                                self.explicit_result = "0-1" if self.board.turn == chess.WHITE else "1-0"
                                
                                winner = "Black" if self.board.turn == chess.WHITE else "White"
                                loser = "White" if self.board.turn == chess.WHITE else "Black"
                                self.game_result_text = f"Game Over: {winner} wins ({loser} Resigned)"
                                continue # 아래의 체스판 클릭 로직 무시
                            
                            bx, by = event.pos[0] - self.board_x, event.pos[1] - self.board_y
                            if 0 <= bx < self.SQUARE_SIZE * 8 and 0 <= by < self.SQUARE_SIZE * 8:
                                clicked_square = chess.square(bx // self.SQUARE_SIZE, 7 - (by // self.SQUARE_SIZE))
                                
                                if self.selected_square is None:
                                    piece = self.board.piece_at(clicked_square)
                                    if piece and piece.color == self.board.turn:
                                        self.selected_square = clicked_square
                                        self.dragging = True
                                        self.drag_pos = event.pos
                                        self.clicked_already_selected = False
                                else:
                                    if clicked_square == self.selected_square:
                                        self.dragging = True
                                        self.drag_pos = event.pos
                                        self.clicked_already_selected = True 
                                    else:
                                        piece = self.board.piece_at(clicked_square)
                                        if piece and piece.color == self.board.turn:
                                            self.selected_square = clicked_square
                                            self.dragging = True
                                            self.drag_pos = event.pos
                                            self.clicked_already_selected = False
                                        else:
                                            if self._try_make_move(self.selected_square, clicked_square):
                                                self.selected_square = None
                                            self.clicked_already_selected = False
                        elif event.button == 3: 
                            self.selected_square, self.dragging, self.clicked_already_selected = None, False, False
                    elif event.type == pygame.MOUSEMOTION:
                        if self.dragging: self.drag_pos = event.pos
                    elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                        if self.dragging:
                            self.dragging = False
                            if self.selected_square is not None:
                                bx, by = event.pos[0] - self.board_x, event.pos[1] - self.board_y
                                if 0 <= bx < self.SQUARE_SIZE * 8 and 0 <= by < self.SQUARE_SIZE * 8:
                                    released_square = chess.square(bx // self.SQUARE_SIZE, 7 - (by // self.SQUARE_SIZE))
                                    if released_square != self.selected_square:
                                        if self._try_make_move(self.selected_square, released_square): self.selected_square = None
                                    elif getattr(self, 'clicked_already_selected', False): self.selected_square = None
                                else: self.selected_square = None
                                self.clicked_already_selected = False

            if not self.ai_thinking and not current_player.is_human() and not self.promotion_pending and not self.game_over:
                self.handle_ai_move()

            self.draw_board()
            self.clock.tick(60)
        pygame.quit()