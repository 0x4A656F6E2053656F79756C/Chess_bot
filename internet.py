import time
import datetime
import os
import chess
import chess.pgn
import torch
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from AI import TwoHeadChessCNN, ChessMCTS

class ChessDotComAutoBot:
    def __init__(self):
        os.makedirs("model", exist_ok=True)
        os.makedirs("notes", exist_ok=True)
        
        model_name = input("사용할 모델 파일 이름 (model/ 폴더 내, 기본: model_v2.pth): ").strip()
        if not model_name:
            model_name = "model_v2.pth"
            
        self.model_path = os.path.join("model", model_name)

        color = input("어떤 진영으로 플레이하시나요? (W: 백 / B: 흑): ").strip().upper()
        self.play_as_white = (color != 'B')
        print(f"✅ 설정 완료: {'백(White)' if self.play_as_white else '흑(Black)'}으로 플레이합니다.")
        print("=====================================\n")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TwoHeadChessCNN().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        self.mcts = ChessMCTS(model=self.model, device=self.device, num_simulations=100)
        self.board = chess.Board()

        print("🌐 브라우저를 엽니다...")
        self.driver = webdriver.Chrome(service=Service("./chromedriver.exe"))
        self.driver.get("https://www.chess.com/play/computer")
        
    def uci_to_square_class(self, square_str):
        file_to_num = {'a': '1', 'b': '2', 'c': '3', 'd': '4', 'e': '5', 'f': '6', 'g': '7', 'h': '8'}
        return f"square-{file_to_num[square_str[0]]}{square_str[1]}"

    def click_square(self, uci_move):
        from_sq, to_sq = uci_move[:2], uci_move[2:4]
        try:
            from_element = self.driver.find_element(By.XPATH, f"//div[contains(@class, '{self.uci_to_square_class(from_sq)}')]")
            square_size = from_element.size['width']
            files, ranks = 'abcdefgh', '12345678'
            
            offset_x = (files.index(to_sq[0]) - files.index(from_sq[0])) * square_size
            offset_y = -(ranks.index(to_sq[1]) - ranks.index(from_sq[1])) * square_size
            if not self.play_as_white: offset_x, offset_y = -offset_x, -offset_y

            ActionChains(self.driver).drag_and_drop_by_offset(from_element, offset_x, offset_y).perform()
            print(f"🤖 자동 드래그 앤 드롭 완료: {uci_move}")

            if len(uci_move) == 5:
                try:
                    prom_class = f"{'w' if self.play_as_white else 'b'}{uci_move[4]}" 
                    prom_element = WebDriverWait(self.driver, 3).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, f"div.promotion-piece.{prom_class}"))
                    )
                    time.sleep(0.2) 
                    ActionChains(self.driver).move_to_element(prom_element).click().perform()
                    print(f"👑 폰 승급 물리적 자동 클릭 완료: {prom_class}")
                except Exception as e:
                    print(f"⚠️ 승급 자동 클릭 실패! 직접 화면을 눌러주세요: {e}")
        except Exception as e:
            print(f"이동 에러 발생: {e}")

    def get_opponent_move(self):
        target_ply = len(self.board.move_stack) + 1
        print(f"👀 상대방의 턴 대기 중... (목표 기보 번호: {target_ply})")
        while True:
            try:
                move_elements = self.driver.find_elements(By.CSS_SELECTOR, "span.node-highlight-content")
                if len(move_elements) >= target_ply:
                    san_move = move_elements[target_ply - 1].text.strip()
                    if san_move:
                        move = self.board.parse_san(san_move)
                        self.board.push(move)
                        print(f"🎯 상대방이 둔 수 파악 완료: {san_move} ({move.uci()})")
                        break 
            except ValueError: break
            except Exception: pass
            time.sleep(0.5)
            
    def run(self):
        print(f"\n⏳ 봇을 선택하고, 게임 설정에서 '{'흑' if not self.play_as_white else '백'}'을 고르세요.")
        input("게임이 시작되었다면 [Enter]를 누르세요...")

        while not self.board.is_game_over() and not self.board.can_claim_draw():
            if (self.board.turn == chess.WHITE and self.play_as_white) or (self.board.turn == chess.BLACK and not self.play_as_white):
                print("\n🤔 MCTS가 수를 계산 중입니다...")
                best_move = self.mcts.search(self.board)
                self.click_square(best_move.uci())
                self.board.push(best_move)
            else:
                self.get_opponent_move()
        
        print(f"🏁 게임 종료! 결과: {'1/2-1/2 (무승부)' if self.board.can_claim_draw() else self.board.result()}")
        
        # 기보 저장
        os.makedirs("notes", exist_ok=True)
        game_pgn = chess.pgn.Game.from_board(self.board)
        game_pgn.headers["Event"] = "Chess.com AutoBot Match"
        filename = datetime.datetime.now().strftime("internet_match_%Y%m%d_%H%M%S.pgn")
        with open("notes/" + filename, "w", encoding="utf-8") as f:
            f.write(str(game_pgn))
        print(f"✅ 인터넷 대국 기보 저장 완료: notes/{filename}")

if __name__ == "__main__":
    bot = ChessDotComAutoBot()
    bot.run()