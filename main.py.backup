import threading
import time
import os
import sys
from flask import Flask
from milvus_manager import MilvusManager
from obsidian_processor import ObsidianProcessor
from watcher import start_watcher
from web_interface import app as web_app
from api import api
import config

# Import the CMD-optimized progress monitor
from progress_monitor_cmd import ProgressMonitor

# FastMCP API 서버 임포트
from fastmcp_api import run_server_in_background

def check_fastmcp_connection():
    """FastMCP API 서버 연결 상태 확인"""
    import requests
    from colorama import Fore, Style
    
    try:
        # 서버 상태 확인 - 직접 API 호출
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'Obsidian-Milvus-Integration/2.0'
        }
        
        # API 키 설정
        if hasattr(config, 'FASTMCP_API_KEY') and config.FASTMCP_API_KEY:
            headers['Authorization'] = f'Bearer {config.FASTMCP_API_KEY}'
        
        # 헬스 체크 API 호출
        response = requests.get(
            f"{config.FASTMCP_URL}/api/health",
            headers=headers,
            timeout=5
        )
        
        # 응답 확인
        if response.status_code == 200:
            print(f"{Fore.GREEN}✓ FastMCP API 서버 연결 성공: {config.FASTMCP_URL}{Style.RESET_ALL}")
            return True
        else:
            print(f"{Fore.RED}✗ FastMCP API 서버 연결 실패: 상태 코드 {response.status_code}{Style.RESET_ALL}")
            return False
    except requests.exceptions.ConnectionError:
        # 연결 오류는 서버가 아직 실행되지 않았을 가능성이 높음
        print(f"{Fore.YELLOW}FastMCP API 서버가 실행되고 있지 않습니다. 서버를 시작합니다.{Style.RESET_ALL}")
        return False
    except Exception as e:
        print(f"{Fore.RED}✗ FastMCP API 서버 연결 실패: {e}{Style.RESET_ALL}")
        return False

def initialize():
    """시스템 초기화"""
    from colorama import Fore, Style
    print("Obsidian-Milvus-FastMCP system initializing...")
    
    # Milvus 연결 및 컬렉션 준비
    milvus_manager = MilvusManager()
    
    # Obsidian 프로세서 초기화
    processor = ObsidianProcessor(milvus_manager)
    
    # 초기 인덱싱 여부 확인
    results = milvus_manager.query("id >= 0", limit=1)
    if not results:
        print("데이터가 없습니다. 웹 인터페이스에서 '전체 재색인' 버튼을 클릭하여 인덱싱을 시작하세요.")
    else:
        print(f"기존 데이터가 있습니다. 웹 인터페이스에서 추가 인덱싱을 실행할 수 있습니다.")
    
    # Flask 앱에 API 블루프린트 등록
    web_app.register_blueprint(api, url_prefix='/api')
    
    # 정적 파일 및 템플릿 디렉토리 생성
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # FastMCP API 서버 시작 (먼저 시작해야 함)
    try:
        print(f"{Fore.BLUE}Starting FastMCP API server...{Style.RESET_ALL}")
        start_fastmcp_api_server()
        print(f"{Fore.GREEN}FastMCP API server started successfully{Style.RESET_ALL}")
        # 서버가 완전히 시작될 시간 여유 제공
        time.sleep(2)
        # 연결 확인
        fastmcp_connected = check_fastmcp_connection()
    except Exception as e:
        print(f"{Fore.YELLOW}FastMCP API server error: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}이 오류는 무시하고 계속 진행할 수 있습니다.{Style.RESET_ALL}")
        fastmcp_connected = False
    
    # Claude Desktop 통합 초기화
    # Claude Desktop 통합 - 실패해도 계속 진행
    if fastmcp_connected:
        try:
            from claude_integration import ClaudeIntegration
            print("Initializing Claude Desktop integration with FastMCP 2.0...")
            claude = ClaudeIntegration()
            chat_id = claude.setup_chat()
            if chat_id:
                print(f"{Fore.GREEN}Claude Desktop chat session created with ID: {chat_id}{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}Claude Desktop chat session creation failed.{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}FastMCP 2.0 API가 호환되지 않거나 사용할 수 없습니다.{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}이 오류는 무시하고 계속 진행할 수 있습니다.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}Claude Desktop integration error: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}FastMCP 2.0 API가 호환되지 않거나 사용할 수 없습니다.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}이 오류는 무시하고 계속 진행할 수 있습니다.{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}FastMCP 2.0 연결에 실패했습니다.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}이 오류는 무시하고 계속 진행할 수 있습니다.{Style.RESET_ALL}")
    
    return processor

def start_fastmcp_api_server():
    """Run FastMCP API server in the background with error handling"""
    from colorama import Fore, Style
    import threading
    
    print(f"Starting FastMCP API server at http://{config.FASTMCP_HOST}:{config.FASTMCP_PORT}")
    
    # FastMCP API 서버를 별도 스레드에서 실행
    from fastmcp_api import run_server_in_background
    
    try:
        # 백그라운드에서 서버 실행
        server_thread = run_server_in_background()
        
        # 서버가 시작될 시간 여유 제공
        time.sleep(2)
        
        # 서버 상태 확인
        if check_fastmcp_connection():
            print(f"{Fore.GREEN}FastMCP API server started successfully{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}FastMCP API server may not be running correctly{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}이 오류는 무시하고 계속 진행할 수 있습니다.{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}FastMCP API server error: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}이 오류는 무시하고 계속 진행할 수 있습니다.{Style.RESET_ALL}")

def open_claude_desktop():
    """Open Claude Desktop in the default browser"""
    import webbrowser
    claude_url = "https://claude.ai"
    print(f"Opening Claude Desktop at {claude_url}")
    webbrowser.open(claude_url)

def perform_full_embedding(processor):
    """Perform full embedding (reindex all files)"""
    from colorama import Fore, Style
    
    # 화면 전체 지우기
    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # Unix/Linux/MacOS
        os.system('clear')
    
    print(f"\n{Fore.RED}{Style.BRIGHT}Starting full embedding process...{Style.RESET_ALL}")
    print("This may take a long time depending on the size of your vault.")
    print("Progress will be displayed in the terminal.\n")
    
    # 임베딩 프로세스 시작
    try:
        # 임베딩 진행 중 상태 확인
        if hasattr(processor, 'embedding_in_progress') and processor.embedding_in_progress:
            print(f"{Fore.YELLOW}Warning: Embedding process is already in progress.{Style.RESET_ALL}")
            choice = input("Do you want to restart the embedding process? (y/n): ")
            if choice.lower() != 'y':
                return
        
        # 전체 재처리의 경우 Milvus 컬렉션 삭제 및 재생성
        print(f"\n{Fore.YELLOW}Do you want to completely delete and recreate the Milvus collection?{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}If you select 'Yes', all existing embedding data will be deleted and all files will be re-embedded.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}If you select 'No', existing embedding data will be preserved, but all files will still be re-embedded.{Style.RESET_ALL}")
        recreate_choice = input("Delete and recreate collection? (y/n): ")
        
        if recreate_choice.lower() == 'y':
            print(f"\n{Fore.CYAN}[DEBUG] Recreating Milvus collection...{Style.RESET_ALL}")
            # Milvus 컬렉션 삭제 및 재생성
            try:
                processor.milvus_manager.recreate_collection()
                print(f"{Fore.GREEN}Successfully recreated Milvus collection.{Style.RESET_ALL}")
                # 임베딩 진행 정보에 전체 재처리 모드 표시
                processor.embedding_progress["is_full_reindex"] = True
                print(f"{Fore.GREEN}Set to process all files.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error recreating Milvus collection: {e}{Style.RESET_ALL}")
                import traceback
                print(f"{Fore.RED}Stack trace:\n{traceback.format_exc()}{Style.RESET_ALL}")
                # 오류가 발생해도 계속 진행
        else:
            # 사용자가 컬렉션을 재생성하지 않더라도 전체 재처리이미로 모든 파일 처리
            processor.embedding_progress["is_full_reindex"] = True
            print(f"{Fore.GREEN}Set to process all files without recreating collection.{Style.RESET_ALL}")
        
        # 임베딩 시작 전 디버깅 메시지
        print(f"\n{Fore.CYAN}[DEBUG] Starting embedding process...{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[DEBUG] Obsidian vault path: {processor.vault_path}{Style.RESET_ALL}")
        
        # 경로 존재 확인
        if not os.path.exists(processor.vault_path):
            print(f"\n{Fore.RED}Error: Obsidian vault path does not exist: {processor.vault_path}{Style.RESET_ALL}")
            return
            
        # 경로에 파일 존재 확인
        md_files = [f for f in os.listdir(processor.vault_path) if f.endswith('.md')]
        print(f"{Fore.CYAN}[DEBUG] Found {len(md_files)} markdown files in vault root{Style.RESET_ALL}")
        
        # 임베딩 시작
        print(f"{Fore.CYAN}[DEBUG] Calling processor.process_all_files(){Style.RESET_ALL}")
        result = processor.process_all_files()
        print(f"{Fore.CYAN}[DEBUG] process_all_files returned: {result}{Style.RESET_ALL}")
        
        # 여기서 결과 확인 및 처리 추가
        if result == 0:
            print(f"\n{Fore.RED}{Style.BRIGHT}Embedding process failed! Check the logs above for details.{Style.RESET_ALL}")
            return
        elif result is None:
            print(f"\n{Fore.YELLOW}{Style.BRIGHT}Warning: Embedding process returned None. This may indicate an issue.{Style.RESET_ALL}")
            # None이 반환되어도 계속 진행
        
        # 임베딩 완료 대기 (성공 시에만 실행)
        print(f"{Fore.CYAN}[DEBUG] Waiting for embedding process to complete...{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[DEBUG] Current embedding_in_progress = {processor.embedding_in_progress}{Style.RESET_ALL}")
        
        while hasattr(processor, 'embedding_in_progress') and processor.embedding_in_progress:
            # 1초마다 상태 확인
            time.sleep(1)
            print(f"{Fore.CYAN}[DEBUG] Still waiting... embedding_in_progress = {processor.embedding_in_progress}{Style.RESET_ALL}")
        
        print(f"\n{Fore.GREEN}{Style.BRIGHT}Full embedding completed successfully!{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Error during embedding process: {e}{Style.RESET_ALL}")
        import traceback
        print(f"\n{Fore.RED}Stack trace:\n{traceback.format_exc()}{Style.RESET_ALL}")
    finally:
        # 임베딩이 중단된 경우에도 상태 정리
        if hasattr(processor, 'embedding_in_progress'):
            processor.embedding_in_progress = False
    
def perform_incremental_embedding(processor):
    """Perform incremental embedding (only new/modified files)"""
    from colorama import Fore, Style
    import time as time_module
    
    # 화면 전체 지우기
    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # Unix/Linux/MacOS
        os.system('clear')
    
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}Starting incremental embedding process...{Style.RESET_ALL}")
    print("This will process only new or modified files.")
    print("Progress will be displayed in the terminal.\n")
    
    # 임베딩 프로세스 시작
    try:
        # 임베딩 진행 중 상태 확인
        if hasattr(processor, 'embedding_in_progress') and processor.embedding_in_progress:
            print(f"{Fore.YELLOW}Warning: Embedding process is already in progress.{Style.RESET_ALL}")
            choice = input("Do you want to restart the embedding process? (y/n): ")
            if choice.lower() != 'y':
                return
        
        # 임베딩 시작
        processor.process_updated_files()
        
        # 임베딩 완료 대기
        while hasattr(processor, 'embedding_in_progress') and processor.embedding_in_progress:
            # 1초마다 상태 확인
            time_module.sleep(1)
        
        print(f"\n{Fore.GREEN}{Style.BRIGHT}Incremental embedding completed successfully!{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Error during embedding process: {e}{Style.RESET_ALL}")
    finally:
        # 임베딩이 중단된 경우에도 상태 정리
        if hasattr(processor, 'embedding_in_progress'):
            processor.embedding_in_progress = False

def show_menu():
    """Display the command-line menu with colorful options"""
    from colorama import Fore, Style, Back
    
    print("\n" + "="*60)
    print(f"{Style.BRIGHT}{Fore.CYAN}Obsidian-Milvus-Claude Desktop Command Menu{Style.RESET_ALL}")
    print("="*60)
    
    # Option 1: Full Embedding (Red background for caution)
    print(f"{Fore.WHITE}{Back.RED}{Style.BRIGHT} 1 {Style.RESET_ALL} {Fore.RED}{Style.BRIGHT}Full Embedding{Style.RESET_ALL} (reindex all files)")
    
    # Option 2: Incremental Embedding (Yellow background for moderate caution)
    print(f"{Fore.BLACK}{Back.YELLOW}{Style.BRIGHT} 2 {Style.RESET_ALL} {Fore.YELLOW}{Style.BRIGHT}Incremental Embedding{Style.RESET_ALL} (only new/modified files)")
    
    # Option 3: Open Claude Desktop (Green background for safe option)
    print(f"{Fore.WHITE}{Back.GREEN}{Style.BRIGHT} 3 {Style.RESET_ALL} {Fore.GREEN}{Style.BRIGHT}Open Claude Desktop{Style.RESET_ALL}")
    
    # Option 4: Exit (Blue for neutral option)
    print(f"{Fore.WHITE}{Back.BLUE}{Style.BRIGHT} 4 {Style.RESET_ALL} {Fore.BLUE}{Style.BRIGHT}Exit{Style.RESET_ALL}")
    
    print("="*60)
    choice = input(f"{Fore.CYAN}Enter your choice (1-4): {Style.RESET_ALL}")
    return choice

def main():
    """Main execution function with command-line interface"""
    # 출력 필터 적용 (메뉴 표시 전에 필터링 시작)
    from progress_monitor_cmd import OutputFilter
    import sys
    # 디버깅을 위해 필터 비활성화
    # sys.stdout = OutputFilter(sys.__stdout__)
    sys.stdout = sys.__stdout__  # Use original stdout for debugging
    
    processor = initialize()
    
    # Start file change detection thread
    watcher_thread = threading.Thread(target=start_watcher, args=(processor,))
    watcher_thread.daemon = True
    watcher_thread.start()
    
    # FastMCP API 서버 시작
    api_thread = threading.Thread(target=start_fastmcp_api_server)
    api_thread.daemon = True
    api_thread.start()
    print(f"FastMCP API server is running in the background at http://localhost:{config.FLASK_PORT + 1}")
    
    # Start web server in a separate thread with error handling
    def run_web_server(processor):
        try:
            web_app.run(host='0.0.0.0', port=config.FLASK_PORT, debug=False, use_reloader=False)
        except OSError as e:
            error_msg = f"Web server error: {e}"
            print(f"\n{Fore.RED}{error_msg}{Style.RESET_ALL}")
            if hasattr(processor, 'monitor') and hasattr(processor.monitor, 'add_error_log'):
                processor.monitor.add_error_log(error_msg)
            # Try with 127.0.0.1 instead of 0.0.0.0
            try:
                print(f"Retrying with 127.0.0.1 instead of 0.0.0.0...")
                web_app.run(host='127.0.0.1', port=config.FLASK_PORT, debug=False, use_reloader=False)
            except Exception as e2:
                print(f"\n{Fore.RED}Failed to start web server: {e2}{Style.RESET_ALL}")
    
    web_thread = threading.Thread(target=lambda: run_web_server(processor))
    web_thread.daemon = True
    web_thread.start()
    print(f"Web server started at http://localhost:{config.FLASK_PORT}")
    
    # 서버가 시작될 때까지 잠시 대기 (Flask 메시지가 메뉴와 섞이지 않도록)
    time.sleep(2)
    
    # Command-line interface loop
    while True:
        choice = show_menu()
        
        if choice == '1':
            perform_full_embedding(processor)
        elif choice == '2':
            perform_incremental_embedding(processor)
        elif choice == '3':
            # Claude Desktop 열기
            open_claude_desktop()
        elif choice == '4':
            print("Exiting program...")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")
        
        # Brief pause to allow user to read output
        time.sleep(1)

if __name__ == "__main__":
    main()