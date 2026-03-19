import os
import time
import threading
import chess
import torch

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

from AI import TwoHeadChessCNN, ChessMCTS
from GUI import ChessGame, Player

class SpectatorPlayer(Player):
    def __init__(self, mcts):
        self.mcts = mcts

    def is_human(self):
        return True 

    def get_move(self, board):
        return None

class ChessSpectator:
    def __init__(self):
        os.makedirs("model", exist_ok=True)
        # 🌟 다중 스레드 꼬임(Race Condition)을 방지할 락 객체 생성
        self.lock = threading.Lock() 
        
        print("=== 📺 체스 AI 관전(중계) 모드 ===")
        model_name = input("사용할 모델 파일 이름 (model/ 폴더 내, 기본: model_v5.pth): ").strip()
        if not model_name:
            model_name = "model_v5.pth"
        self.model_path = os.path.join("model", model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🤖 AI 모델 로딩 중... ({self.device})")
        
        self.model = TwoHeadChessCNN().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        self.mcts = ChessMCTS(model=self.model, device=self.device, num_simulations=10)
        self.game = None 
        self.running = False

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

    def monitor_moves(self):
        last_move_count = 0
        while self.running and not self.game.game_over:
            try:
                moves_elements = self.driver.find_elements(By.CSS_SELECTOR, "span.node-highlight-content")
                current_move_count = len(moves_elements)
                
                if current_move_count > last_move_count:
                    for i in range(last_move_count, current_move_count):
                        san = moves_elements[i].text.strip()
                        if not san: continue
                            
                        san = san.replace("0-0-0", "O-O-O").replace("0-0", "O-O")
                        
                        try:
                            move = self.game.board.parse_san(san)
                            # 🌟 락을 걸고 안전하게 보드와 트리 동시 갱신
                            with self.lock: 
                                self.game.board.push(move)
                                self.mcts.update_with_move(move)
                            print(f"🎯 실시간 수 인식: {san}")
                        except ValueError as e:
                            print(f"⚠️ 기보 파싱 에러: '{san}' -> {e}")
                            
                    last_move_count = current_move_count
            except Exception:
                pass
            time.sleep(0.5)

    def analyze_continuously(self):
        while self.running and not self.game.game_over:
            # 🌟 락을 걸어 트리가 갱신되는 동안엔 접근하지 못하도록 보호
            with self.lock:
                if not self.game.board.is_game_over():
                    board_copy = self.game.board.copy()
                    try:
                        self.mcts.search(board_copy, add_noise=False, temperature=0.0)
                    except Exception as e:
                        print(f"⚠️ 분석 스레드 에러: {e}")
            
            # 락을 풀고 1초 대기 (이 타이밍에 monitor_moves가 트리를 갱신할 수 있음)
            time.sleep(0.05)

    def start(self):
        print("\n✅ 준비 완료!")
        print("1. 열린 크롬 창에서 관전할 라이브 대국 페이지로 들어가세요.")
        print("2. 대국 창이 보이면 아래에서 [Enter] 키를 누르세요.")
        input("시작하려면 Enter를 누르세요...")

        print("\n🖥️ 체스판 GUI를 띄웁니다...")
        
        self.spectator_player = SpectatorPlayer(self.mcts)
        # 🌟 spectator_mode=True 인자 전달!
        self.game = ChessGame(
            white_player=self.spectator_player, 
            black_player=self.spectator_player, 
            model_path=self.model_path,
            spectator_mode=True 
        )
        
        self.game.show_heatmap = True
        self.game.show_eval = True
        self.running = True

        monitor_thread = threading.Thread(target=self.monitor_moves, daemon=True)
        analyze_thread = threading.Thread(target=self.analyze_continuously, daemon=True)
        
        monitor_thread.start()
        analyze_thread.start()

        self.game.run()
        
        self.running = False
        self.driver.quit()
        print("📺 관전을 종료합니다.")

if __name__ == "__main__":
    app = ChessSpectator()
    app.start()