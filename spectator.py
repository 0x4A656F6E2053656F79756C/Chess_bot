import os
import time
import threading
import chess
import chess.pgn
import datetime
import torch

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

from AI import TwoHeadChessCNN, ChessMCTS
from GUI import ChessGame, Player
from config import device

# 🌟 SpectatorPlayer는 관전 모드 및 로컬 분석 모드 모두에서 AI 수읽기 정보를 GUI에 전달하는 역할
class SpectatorPlayer(Player):
    def __init__(self, mcts):
        self.mcts = mcts

    def is_human(self):
        # 모드 2에서는 사람이 두므로 HumanPlayer와 같은 역할
        return True 

    def get_move(self, board):
        # GUI에서 마우스 입력을 받으므로 여기서는 None을 반환
        return None

class ChessSpectator:
    def __init__(self):
        os.makedirs("model", exist_ok=True)
        # 다중 스레드 꼬임(Race Condition)을 방지할 락 객체 생성
        self.lock = threading.Lock() 
        
        print("=== 📺 체스 AI 관전 및 분석 모드 ===")
        print("1. chess.com 관전 (웹 크롤링)")
        print("2. 로컬 직접 두기 (사람 vs 사람, 무한 분석)")
        self.mode = input("모드를 선택하세요 (1 또는 2) [기본: 1]: ").strip()
        if self.mode not in ['1', '2']:
            self.mode = '1'

        model_name = input("사용할 모델 파일 이름 (model/ 폴더 내, 기본: model_v5.pth): ").strip()
        if not model_name:
            model_name = "model_v5.pth"
        self.model_path = os.path.join("model", model_name)

        self.device = device
        print(f"🤖 AI 모델 로딩 중... ({self.device})")
        
        self.model = TwoHeadChessCNN().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        self.mcts = ChessMCTS(model=self.model, device=self.device, num_simulations=10)
        self.game = None 
        self.running = False

        if self.mode == '1':
            chrome_options = Options()
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--start-maximized")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option("useAutomationExtension", False)

            chrome_data_path = os.path.join(os.getcwd(), "chrome_data")
            first_run = False

            if not os.path.exists(chrome_data_path):
                os.makedirs(chrome_data_path)
                first_run = True
                print("📁 chrome_data 폴더 생성 완료")

            chrome_options.add_argument(f"--user-data-dir={chrome_data_path}")
            
            print("\n🌐 크롬 드라이버 설정")
            print("1. 로컬 드라이버 사용 (현재 폴더의 chromedriver.exe 사용)")
            print("2. 자동 설치 (ChromeDriverManager 사용)")
            driver_choice = input("드라이버 방식을 선택하세요 (1 또는 2) [기본: 1]: ").strip()

            print("브라우저 실행 중...")
            try:
                if driver_choice == '2':
                    service = Service(ChromeDriverManager().install())
                    self.driver = webdriver.Chrome(service=service, options=chrome_options)
                else:
                    service = Service(executable_path="./chromedriver.exe")
                    self.driver = webdriver.Chrome(service=service, options=chrome_options)
            except Exception as e:
                print(f"❌ 크롬 드라이버 실행 실패: {e}")
                exit()

            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.driver.get("https://www.chess.com/play")
            print("✅ chess.com 페이지 로딩 완료")

            if first_run:
                print("\n🔑 첫 실행입니다.")
                print("브라우저에서 chess.com 로그인 후 Enter를 눌러주세요.")
                input()
        else:
            print("\n✅ 로컬 직접 두기 모드로 시작합니다.")

    # 🌟 수 인식 및 MCTS 동기화 로직을 하나로 통합
    def monitor_moves(self):
        last_move_count = 0
        while self.running and not self.game.game_over:
            try:
                if self.mode == '1':
                    # chess.com 관전 모드: 웹 크롤링
                    moves_elements = self.driver.find_elements(By.CSS_SELECTOR, "span.node-highlight-content")
                    current_move_count = len(moves_elements)
                    if current_move_count > last_move_count:
                        for i in range(last_move_count, current_move_count):
                            san = moves_elements[i].text.strip()
                            if not san: continue
                            san = san.replace("0-0-0", "O-O-O").replace("0-0", "O-O")
                            try:
                                with self.lock: 
                                    move = self.game.board.parse_san(san)
                                    self.game.board.push(move)
                                    self.mcts.update_with_move(move)
                                print(f"🎯 실시간 수 인식: {san}")
                            except ValueError as e:
                                print(f"⚠️ 기보 파싱 에러: '{san}' -> {e}")
                        last_move_count = current_move_count
                else:
                    # 로컬 분석 모드: GUI 보드 감시
                    with self.lock:
                        current_move_count = len(self.game.board.move_stack)
                        if current_move_count > last_move_count:
                            for i in range(last_move_count, current_move_count):
                                move = self.game.board.move_stack[i]
                                self.mcts.update_with_move(move)
                                print(f"🎯 둔 수 인식: {self.game.board.san(move) if i < len(self.game.board.move_stack)-1 else move}")
                            last_move_count = current_move_count
            except Exception as e:
                print(f"⚠️ 수 인식 에러: {e}")
            time.sleep(0.1 if self.mode == '2' else 0.5) # 분석 모드는 더 빠르게 수 인식

    # 🌟 연속 분석 로직 (히트맵 갱신을 위해 상시 실행)
    def analyze_continuously(self):
        while self.running and not self.game.game_over:
            with self.lock:
                if not self.game.board.is_game_over():
                    board_copy = self.game.board.copy()
                    try:
                        self.mcts.search(board_copy, add_noise=False, temperature=0.0)
                    except Exception as e:
                        print(f"⚠️ 분석 스레드 에러: {e}")
            time.sleep(0.05) # 히트맵이 자연스럽게 갱신되도록 대기

    def start(self):
        print("\n✅ 준비 완료!")
        if self.mode == '1':
            print("1. 열린 크롬 창에서 관전할 라이브 대국 페이지로 들어가세요.")
            print("2. 대국 창이 보이면 아래에서 [Enter] 키를 누르세요.")
            input("시작하려면 Enter를 누르세요...")

        print("\n🖥️ 체스판 GUI를 띄웁니다...")
        
        # 🌟 둘 다 SpectatorPlayer 사용
        self.spectator_player = SpectatorPlayer(self.mcts)
        self.game = ChessGame(
            white_player=self.spectator_player, 
            black_player=self.spectator_player, 
            model_path=self.model_path,
            spectator_mode=(self.mode == '1') 
        )

        if self.mode == '2':
            # 1. 주사율 강제 다운 (10Hz)
            orig_tick = self.game.clock.tick
            self.game.clock.tick = lambda fps: orig_tick(10)

            # 2. 1번 모드처럼 컨트롤 패널 강제 렌더링
            orig_draw = self.game.draw
            def patched_draw():
                orig_draw()
                self.game.draw_control_panel()
            self.game.draw = patched_draw
        
        self.game.show_heatmap = True
        self.game.show_eval = True
        self.running = True

        # 분석 스레드 및 수 인식 스레드 실행
        analyze_thread = threading.Thread(target=self.analyze_continuously, daemon=True)
        analyze_thread.start()
        monitor_thread = threading.Thread(target=self.monitor_moves, daemon=True)
        monitor_thread.start()

        self.game.run()
        
        self.running = False
        
        if self.mode == '1':
            self.driver.quit()
            print("📺 관전을 종료합니다.")
        else:
            # 로컬 분석 모드 종료 시 기보 저장
            os.makedirs("notes", exist_ok=True)
            pgn_game = chess.pgn.Game.from_board(self.game.board)
            pgn_game.headers["Event"] = "Local Human Analysis Match"
            pgn_game.headers["White"] = "Human"
            pgn_game.headers["Black"] = "Human"
            pgn_game.headers["Result"] = self.game.board.result()
            filename = datetime.datetime.now().strftime("match_%Y%m%d_%H%M%S.pgn")
            with open("notes/" + filename, "w", encoding="utf-8") as f:
                f.write(str(pgn_game))
            print(f"📺 분석을 종료합니다. 기보 저장 완료: notes/{filename}")

if __name__ == "__main__":
    app = ChessSpectator()
    app.start()