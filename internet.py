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
from selenium.webdriver.chrome.options import Options

# 작성하신 AI.py에서 클래스들을 불러옵니다.
from AI import TwoHeadChessCNN, ChessMCTS

class ChessDotComAutoBot:
    def __init__(self):
        os.makedirs("model", exist_ok=True)
        os.makedirs("notes", exist_ok=True)

        # -----------------------------
        # 1. 모델 및 시뮬레이션 설정
        # -----------------------------
        model_name = input("사용할 모델 파일 이름 (model/ 폴더 내, 기본: model_v4.pth): ").strip()
        if not model_name:
            model_name = "model_v4.pth"
        self.model_path = os.path.join("model", model_name)

        color = input("어떤 진영으로 플레이하시나요? (W: 백 / B: 흑) [기본: W]: ").strip().upper()
        # 입력이 비어있으면(그냥 엔터를 치면) 자동으로 'W'로 설정합니다.
        if not color:
            color = "W"
        
        self.play_as_white = (color != "B")

        # 시뮬레이션 횟수 입력 추가
        sim_input = input("MCTS 시뮬레이션 횟수를 입력하세요 (기본: 100): ").strip()
        num_sim = int(sim_input) if sim_input.isdigit() else 100

        print(f"✅ 설정 완료: {'백(White)' if self.play_as_white else '흑(Black)'}으로 플레이합니다. (시뮬레이션: {num_sim}회)")
        print("=====================================\n")

        # -----------------------------
        # 2. Chrome 옵션 및 프로필 설정
        # -----------------------------
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

        # -----------------------------
        # 3. AI 모델 및 MCTS 로딩
        # -----------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TwoHeadChessCNN().to(self.device)
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        self.mcts = ChessMCTS(
            model=self.model,
            device=self.device,
            num_simulations=num_sim  # 입력받은 횟수 적용
        )
        self.board = chess.Board()

        # -----------------------------
        # 4. Chrome 브라우저 실행
        # -----------------------------
        print("\n🌐 크롬 드라이버 설정")
        print("1. 로컬 드라이버 사용 (현재 폴더의 chromedriver.exe 사용)")
        print("2. 자동 설치 (ChromeDriverManager 사용")
        driver_choice = input("드라이버 방식을 선택하세요 (1 또는 2) [기본: 1]: ").strip()

        print("브라우저 실행 중...")
        try:
            if driver_choice == '2':
                # 자동 설치 사용
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
            else:
                # 로컬 드라이버 사용
                driver_path = "./chromedriver.exe"
                service = Service(executable_path=driver_path)
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
                
        except Exception as e:
            print(f"\n❌ 크롬 드라이버 실행 중 에러가 발생했습니다: {e}")
            print("드라이버 버전을 확인하거나 자동 설치 모드(2)를 사용해 보세요.")
            exit()

        # 자동화 탐지 방지 우회
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        self.driver.get("https://www.chess.com/play/computer")
        print("✅ chess.com 페이지 로딩 완료")

        if first_run:
            print("\n🔑 첫 실행입니다.")
            print("브라우저에서 chess.com 로그인 후 Enter를 눌러주세요.")
            input()

    # -----------------------------
    # 체스 좌표 변환 유틸리티
    # -----------------------------
    def uci_to_square_class(self, square_str):
        file_to_num = {"a": "1", "b": "2", "c": "3", "d": "4", "e": "5", "f": "6", "g": "7", "h": "8"}
        return f"square-{file_to_num[square_str[0]]}{square_str[1]}"

    # -----------------------------
    # 실제 기물 이동 (웹 조작)
    # -----------------------------
    def click_square(self, uci_move):
        from_sq, to_sq = uci_move[:2], uci_move[2:4]

        try:
            from_element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, f"//div[contains(@class,'{self.uci_to_square_class(from_sq)}')]"))
            )

            square_size = from_element.size["width"]
            files = "abcdefgh"
            ranks = "12345678"

            offset_x = (files.index(to_sq[0]) - files.index(from_sq[0])) * square_size
            offset_y = -(ranks.index(to_sq[1]) - ranks.index(from_sq[1])) * square_size

            if not self.play_as_white:
                offset_x = -offset_x
                offset_y = -offset_y

            ActionChains(self.driver).drag_and_drop_by_offset(from_element, offset_x, offset_y).perform()
            
            # [수정됨] 폰 승급 로직 복구
            if len(uci_move) == 5:
                try:
                    prom_class = f"{'w' if self.play_as_white else 'b'}{uci_move[4]}" 
                    prom_element = WebDriverWait(self.driver, 3).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, f"div.promotion-piece.{prom_class}"))
                    )
                    time.sleep(0.2) 
                    ActionChains(self.driver).move_to_element(prom_element).click().perform()
                    print(f"👑 폰 승급 자동 클릭 완료: {prom_class}")
                except Exception as e:
                    print(f"⚠️ 승급 자동 클릭 실패! 직접 화면을 눌러주세요: {e}")

            print(f"🤖 자동 이동 완료: {uci_move}")
            return True  # [수정됨] 성공 여부 반환

        except Exception as e:
            print(f"❌ 이동 실패: {e}")
            return False # [수정됨] 실패 여부 반환

    # -----------------------------
    # 상대방 수 읽기
    # -----------------------------
    def get_opponent_move(self):
        target_ply = len(self.board.move_stack) + 1
        print(f"👀 상대 수 대기중... (목표 수: {target_ply})")

        while True:
            try:
                moves = self.driver.find_elements(By.CSS_SELECTOR, "span.node-highlight-content")
                if len(moves) >= target_ply:
                    san = moves[target_ply - 1].text.strip()
                    move = self.board.parse_san(san)
                    self.board.push(move)
                    
                    # 🌟 추가된 부분: 파악한 상대방의 수를 MCTS 트리에 반영
                    self.mcts.update_with_move(move) 
                    
                    print(f"🎯 상대 수 파악: {san}")
                    break
            except Exception as e:
                # [수정됨] StaleElement 에러 등 흔한 대기 에러는 무시하되, 치명적 에러 확인을 위해 필터링 가능
                if "stale" not in str(e).lower():
                    pass
            time.sleep(0.5)

    # -----------------------------
    # 메인 게임 루프
    # -----------------------------
    def run(self):
        print("\n⏳ 봇을 선택하고 게임을 시작하세요.")
        input("게임 시작 후 Enter를 누르세요...")
        time.sleep(2)

        # [수정됨] 무승부 종료 조건 복구
        while not self.board.is_game_over() and not self.board.can_claim_draw():
            
            if (self.board.turn == chess.WHITE and self.play_as_white) or \
               (self.board.turn == chess.BLACK and not self.play_as_white):
                
                print("\n🤔 AI 계산중...")
                temp = 0.1 if len(self.board.move_stack) < 6 else 0.0
                
                move = self.mcts.search(
                    self.board,
                    add_noise=False, 
                    temperature=temp
                )

                # [수정됨] 웹 브라우저에서 실제 이동이 성공했을 때만 내부 board 업데이트
                if self.click_square(move.uci()):
                    self.board.push(move)
                    
                    # 🌟 추가된 부분: AI가 실제로 둔 수를 MCTS 트리에 반영
                    self.mcts.update_with_move(move) 
                    
                else:
                    print("⚠️ 웹 이동에 실패하여 재시도합니다.")
                    time.sleep(1)
            else:
                self.get_opponent_move()

        # 결과 출력
        result = "1/2-1/2 (무승부)" if self.board.can_claim_draw() else self.board.result()
        print(f"\n🏁 게임 종료! 결과: {result}")

        # 기보 저장
        game = chess.pgn.Game.from_board(self.board)
        filename = datetime.datetime.now().strftime("internet_match_%Y%m%d_%H%M%S.pgn")
        with open(f"notes/{filename}", "w", encoding="utf-8") as f:
            f.write(str(game))
        print(f"📄 기보 저장 완료: notes/{filename}")

if __name__ == "__main__":
    bot = ChessDotComAutoBot()
    bot.run()