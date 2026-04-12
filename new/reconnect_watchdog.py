import subprocess
import time
import sys

# ==========================================
# ⚙️ 접속 설정
# ==========================================
SSH_COMMAND = ["ssh", "-o", "ServerAliveInterval=30", "-p", "9001", "cyphy@143.248.56.161"]
# 서버에서 실행할 심박동 명령 (1초마다 시간 출력)
HEARTBEAT_CMD = "while true; do echo \"[Heartbeat] $(date +%H:%M:%S)\"; sleep 1; done"
# ==========================================

def start_reconnect_loop():
    print("🚀 서버 감시 및 자동 재접속 프로세스를 시작합니다.")
    
    while True:
        try:
            print(f"\n[info] {time.strftime('%Y-%m-%d %H:%M:%S')} 서버 연결 시도 중...")
            
            # SSH 세션을 열고 하트비트 명령 실행
            # stderr를 stdout으로 통합하여 연결 끊김 감지
            process = subprocess.Popen(
                SSH_COMMAND + [HEARTBEAT_CMD],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            # 서버의 출력을 실시간으로 모니터링
            while True:
                line = process.stdout.readline()
                if not line: # 연결이 끊기면 line이 없음
                    break
                print(line.strip(), end='\r') # 같은 줄에 타이머 갱신
                sys.stdout.flush()

            print("\n⚠️ 연결이 끊겼습니다. 재접속을 시도합니다.")
            process.wait()

        except Exception as e:
            print(f"\n❌ 에러 발생: {e}")
        
        time.sleep(5) # 재접속 시도 전 대기 시간

if __name__ == "__main__":
    start_reconnect_loop()