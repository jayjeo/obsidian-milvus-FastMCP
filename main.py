import threading
import time
import os
import sys
from milvus_manager import MilvusManager
from obsidian_processor import ObsidianProcessor
from watcher import start_watcher
import config

# Import the CMD-optimized progress monitor
from progress_monitor_cmd import ProgressMonitor

def initialize():
    """시스템 초기화"""
    from colorama import Fore, Style
    print("Obsidian-Milvus-MCP system initializing...")
    
    # Milvus 연결 및 컬렉션 준비
    milvus_manager = MilvusManager()
    
    # Obsidian 프로세서 초기화
    processor = ObsidianProcessor(milvus_manager)
    
    # 초기 인덱싱 여부 확인
    results = milvus_manager.query("id >= 0", limit=1)
    if not results:
        print("데이터가 없습니다. MCP 서버를 통해 인덱싱을 시작하세요.")
    else:
        print(f"기존 데이터가 있습니다. 총 {milvus_manager.count_entities()}개 문서가 인덱싱되어 있습니다.")
    
    return processor

def perform_full_embedding(processor):
    """Perform full embedding (reindex all files)"""
    from colorama import Fore, Style
    import subprocess
    
    print(f"\n{Fore.RED}{Style.BRIGHT}Starting full embedding process...{Style.RESET_ALL}")
    print("This may take a long time depending on the size of your vault.")
    print("Progress will be displayed in the terminal.\n")
    
    try:
        # 전체 재처리의 경우 Milvus 컬렉션 삭제 및 재생성
        print(f"\n{Fore.YELLOW}Do you want to completely delete and recreate the Milvus collection?{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}If you select 'Yes', all existing embedding data will be deleted and all files will be re-embedded.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}If you select 'No', existing embedding data will be preserved, but all files will still be re-embedded.{Style.RESET_ALL}")
        recreate_choice = input("Delete and recreate collection? (y/n): ")
        
        if recreate_choice.lower() == 'y':
            print(f"\n{Fore.CYAN}[FORCE DELETE] Running powerful collection reset script...{Style.RESET_ALL}")
            
            # 강제 삭제 스크립트 실행
            try:
                script_path = os.path.join(os.path.dirname(__file__), "force_reset_collection.py")
                
                # 스크립트가 존재하는지 확인
                if not os.path.exists(script_path):
                    print(f"{Fore.RED}Error: force_reset_collection.py not found at {script_path}{Style.RESET_ALL}")
                    return
                
                print(f"{Fore.BLUE}Executing: python {script_path}{Style.RESET_ALL}")
                
                # subprocess로 스크립트 실행 (자동으로 'y' 입력)
                process = subprocess.Popen(
                    ["python", script_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=os.path.dirname(__file__)
                )
                
                # 자동으로 'y' 입력하여 확인
                output, _ = process.communicate(input='y\n')
                
                # 출력 표시
                print(output)
                
                if process.returncode == 0:
                    print(f"{Fore.GREEN}✅ Force collection reset completed successfully!{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}❌ Force collection reset failed with return code: {process.returncode}{Style.RESET_ALL}")
                    return
                    
            except Exception as e:
                print(f"{Fore.RED}Error running force reset script: {e}{Style.RESET_ALL}")
                # 폴백: 기존 방식으로 시도
                print(f"{Fore.YELLOW}Falling back to regular collection recreation...{Style.RESET_ALL}")
                try:
                    processor.milvus_manager.recreate_collection()
                    print(f"{Fore.GREEN}Successfully recreated Milvus collection using fallback method.{Style.RESET_ALL}")
                except Exception as e2:
                    print(f"{Fore.RED}Fallback method also failed: {e2}{Style.RESET_ALL}")
                    return
        else:
            # 사용자가 컬렉션을 재생성하지 않더라도 전체 재처리임을 표시
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
    
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}Starting incremental embedding process...{Style.RESET_ALL}")
    print("This will process only new or modified files.")
    print("Progress will be displayed in the terminal.\n")
    
    try:
        processor.process_updated_files()
        
        while hasattr(processor, 'embedding_in_progress') and processor.embedding_in_progress:
            time.sleep(1)
        
        print(f"\n{Fore.GREEN}{Style.BRIGHT}Incremental embedding completed successfully!{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Error during embedding process: {e}{Style.RESET_ALL}")
    finally:
        if hasattr(processor, 'embedding_in_progress'):
            processor.embedding_in_progress = False

def start_mcp_server():
    """MCP 서버 시작"""
    from colorama import Fore, Style
    
    print(f"\n{Fore.CYAN}{Style.BRIGHT}Starting MCP Server for Claude Desktop...{Style.RESET_ALL}")
    print(f"Server name: {config.FASTMCP_SERVER_NAME}")
    print(f"Transport: {config.FASTMCP_TRANSPORT}")
    
    # MCP 서버 실행
    os.system(f'python "{os.path.join(os.path.dirname(__file__), "mcp_server.py")}"')

def show_menu():
    """Display the command-line menu with colorful options"""
    from colorama import Fore, Style, Back
    
    print("\n" + "="*60)
    print(f"{Style.BRIGHT}{Fore.CYAN}Obsidian-Milvus-Claude Desktop Command Menu{Style.RESET_ALL}")
    print("="*60)
    
    # Option 1: Start MCP Server (Green background for main option)
    print(f"{Fore.WHITE}{Back.GREEN}{Style.BRIGHT} 1 {Style.RESET_ALL} {Fore.GREEN}{Style.BRIGHT}Start MCP Server{Style.RESET_ALL} (for Claude Desktop)")
    
    # Option 2: Full Embedding (Red background for caution)
    print(f"{Fore.WHITE}{Back.RED}{Style.BRIGHT} 2 {Style.RESET_ALL} {Fore.RED}{Style.BRIGHT}Full Embedding{Style.RESET_ALL} (reindex all files)")
    
    # Option 3: Incremental Embedding (Yellow background for moderate caution)
    print(f"{Fore.BLACK}{Back.YELLOW}{Style.BRIGHT} 3 {Style.RESET_ALL} {Fore.YELLOW}{Style.BRIGHT}Incremental Embedding{Style.RESET_ALL} (only new/modified files)")
    
    # Option 4: Exit (Blue for neutral option)
    print(f"{Fore.WHITE}{Back.BLUE}{Style.BRIGHT} 4 {Style.RESET_ALL} {Fore.BLUE}{Style.BRIGHT}Exit{Style.RESET_ALL}")
    
    print("="*60)
    choice = input(f"{Fore.CYAN}Enter your choice (1-4): {Style.RESET_ALL}")
    return choice

def main():
    """Main execution function with command-line interface"""
    from colorama import Fore, Style
    
    processor = initialize()
    
    # Start file change detection thread
    watcher_thread = threading.Thread(target=start_watcher, args=(processor,))
    watcher_thread.daemon = True
    watcher_thread.start()
    
    print(f"{Fore.GREEN}✅ File watcher started{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📁 Monitoring: {config.OBSIDIAN_VAULT_PATH}{Style.RESET_ALL}")
    
    # Command-line interface loop
    while True:
        choice = show_menu()
        
        if choice == '1':
            start_mcp_server()
        elif choice == '2':
            perform_full_embedding(processor)
        elif choice == '3':
            perform_incremental_embedding(processor)
        elif choice == '4':
            print("Exiting program...")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")
        
        time.sleep(1)

if __name__ == "__main__":
    main()
