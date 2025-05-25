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

def perform_complete_physical_reset():
    """Complete physical reset of Milvus data - FIXED VERSION"""
    from colorama import Fore, Style
    import subprocess
    import shutil
    
    try:
        # 1단계: Milvus 서비스 중지 (더 강력한 정리)
        print(f"{Fore.BLUE}[1/6] Stopping all Milvus containers forcefully...{Style.RESET_ALL}")
        
        # 강제 중지 및 삭제 명령들 (더 포괄적)
        cleanup_commands = [
            "podman stop --all --timeout 5",  # 모든 컨테이너 중지
            "podman stop milvus-standalone milvus-minio milvus-etcd --timeout 5",  # 개별 중지
            "podman rm --force milvus-standalone milvus-minio milvus-etcd",  # 강제 삭제
            "podman container prune --force",  # 중지된 컨테이너 모두 삭제
            "podman pod stop --all",  # 모든 pod 중지
            "podman pod rm --all --force",  # 모든 pod 강제 삭제
        ]
        
        for cmd in cleanup_commands:
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                print(f"{Fore.CYAN}Executed: {cmd}{Style.RESET_ALL}")
                if result.returncode == 0:
                    print(f"{Fore.GREEN}✅ Success{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}⚠️ Command completed with warnings (this is normal){Style.RESET_ALL}")
                    if result.stderr:
                        print(f"{Fore.YELLOW}Warning: {result.stderr.strip()}{Style.RESET_ALL}")
            except subprocess.TimeoutExpired:
                print(f"{Fore.YELLOW}⚠️ {cmd} - Timeout (this is normal){Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️ {cmd} - {e} (this is normal if containers weren't running){Style.RESET_ALL}")
        
        # 2단계: Podman 시스템 정리
        print(f"\n{Fore.BLUE}[2/6] Cleaning up Podman system...{Style.RESET_ALL}")
        
        system_cleanup_commands = [
            "podman system prune --all --force --volumes",  # 시스템 전체 정리
            "podman volume rm milvus-etcd-data milvus-minio-data milvus-db-data --force",  # 볼륨 삭제
            "podman network rm milvus milvus-network --force",  # 네트워크 삭제
        ]
        
        for cmd in system_cleanup_commands:
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                print(f"{Fore.CYAN}Executed: {cmd}{Style.RESET_ALL}")
                if result.returncode == 0:
                    print(f"{Fore.GREEN}✅ Success{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}⚠️ Command completed with warnings (volumes/networks may not exist){Style.RESET_ALL}")
            except subprocess.TimeoutExpired:
                print(f"{Fore.YELLOW}⚠️ {cmd} - Timeout (this is normal){Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️ {cmd} - {e} (this is normal if volumes/networks don't exist){Style.RESET_ALL}")
        
        # 3단계: MilvusData 폴더 완전 삭제 (영구 보존 데이터)
        print(f"\n{Fore.BLUE}[3/6] Deleting permanent embedding data (MilvusData)...{Style.RESET_ALL}")
        
        milvus_data_path = os.path.join(os.path.dirname(__file__), "MilvusData")
        
        if os.path.exists(milvus_data_path):
            print(f"{Fore.CYAN}Deleting: {milvus_data_path}{Style.RESET_ALL}")
            try:
                # Windows에서 읽기 전용 파일 처리를 위한 함수
                def handle_remove_readonly(func, path, exc):
                    import stat
                    if os.path.exists(path):
                        os.chmod(path, stat.S_IWRITE)
                        func(path)
                
                shutil.rmtree(milvus_data_path, onerror=handle_remove_readonly)
                print(f"{Fore.GREEN}✅ MilvusData folder deleted successfully{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}❌ Error deleting MilvusData folder: {e}{Style.RESET_ALL}")
                return False
        else:
            print(f"{Fore.YELLOW}⚠️ MilvusData folder not found (already clean){Style.RESET_ALL}")
        
        # 4단계: volumes 폴더 완전 삭제 (컨테이너 데이터)
        print(f"\n{Fore.BLUE}[4/6] Deleting container data (volumes)...{Style.RESET_ALL}")
        
        volumes_path = os.path.join(os.path.dirname(__file__), "volumes")
        
        if os.path.exists(volumes_path):
            print(f"{Fore.CYAN}Deleting: {volumes_path}{Style.RESET_ALL}")
            try:
                def handle_remove_readonly(func, path, exc):
                    import stat
                    if os.path.exists(path):
                        os.chmod(path, stat.S_IWRITE)
                        func(path)
                
                shutil.rmtree(volumes_path, onerror=handle_remove_readonly)
                print(f"{Fore.GREEN}✅ volumes folder deleted successfully{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}❌ Error deleting volumes folder: {e}{Style.RESET_ALL}")
                return False
        else:
            print(f"{Fore.YELLOW}⚠️ volumes folder not found (already clean){Style.RESET_ALL}")
        
        # 5단계: 완전한 대기 시간
        print(f"\n{Fore.BLUE}[5/6] Waiting for system cleanup to complete...{Style.RESET_ALL}")
        time.sleep(10)  # 시스템이 완전히 정리될 때까지 대기
        
        # 6단계: Milvus 재시작 (수정된 방법)
        print(f"\n{Fore.BLUE}[6/6] Restarting Milvus...{Style.RESET_ALL}")
        
        start_script = os.path.join(os.path.dirname(__file__), "start-milvus.bat")
        if os.path.exists(start_script):
            print(f"{Fore.CYAN}Executing: {start_script}{Style.RESET_ALL}")
            try:
                # 백그라운드에서 시작 스크립트 실행 (더 긴 대기 시간)
                process = subprocess.Popen([start_script], shell=True, cwd=os.path.dirname(__file__))
                
                # 프로세스가 시작될 때까지 잠시 대기
                time.sleep(5)
                
                # 프로세스가 실행 중인지 확인
                if process.poll() is None:
                    print(f"{Fore.GREEN}✅ Milvus restart initiated{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}⚠️ Start script completed quickly (this may be normal){Style.RESET_ALL}")
                
                # Milvus 서비스가 실제로 시작될 때까지 대기
                print(f"{Fore.CYAN}Waiting for Milvus services to fully initialize...{Style.RESET_ALL}")
                time.sleep(30)  # 더 긴 대기 시간
                
            except Exception as e:
                print(f"{Fore.RED}❌ Error starting Milvus: {e}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Please manually run: start-milvus.bat{Style.RESET_ALL}")
                return False
        else:
            print(f"{Fore.YELLOW}⚠️ start-milvus.bat not found. Please start Milvus manually.{Style.RESET_ALL}")
        
        print(f"\n{Fore.GREEN}🎉 Complete physical reset finished successfully!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}💾 All old data has been permanently deleted{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🔥 Both MilvusData (embedding data) AND volumes (container data) deleted{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🚀 Milvus is starting with a clean state{Style.RESET_ALL}")
        
        return True
        
    except Exception as e:
        print(f"{Fore.RED}❌ Error during complete physical reset: {e}{Style.RESET_ALL}")
        import traceback
        print(f"{Fore.RED}Stack trace:\n{traceback.format_exc()}{Style.RESET_ALL}")
        return False

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
            print(f"\n{Fore.RED}[COMPLETE PHYSICAL RESET] Starting complete Milvus reset...{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}This will:{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}  1. Stop all Milvus containers{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}  2. Delete all physical data files (MilvusData folder){Style.RESET_ALL}")
            print(f"{Fore.YELLOW}  3. Restart Milvus with clean state{Style.RESET_ALL}")
            
            # 완전한 물리적 리셋 실행
            success = perform_complete_physical_reset()
            
            if not success:
                print(f"{Fore.RED}❌ Complete physical reset failed{Style.RESET_ALL}")
                return
            
            print(f"{Fore.GREEN}✅ Complete physical reset completed successfully!{Style.RESET_ALL}")
            
            # 물리적 리셋 후 새로운 MilvusManager 인스턴스 생성
            print(f"{Fore.CYAN}Reconnecting to fresh Milvus instance...{Style.RESET_ALL}")
            try:
                # 잠시 대기 후 재연결 - 더 긴 대기 시간
                print(f"{Fore.CYAN}Waiting for Milvus services to fully initialize...{Style.RESET_ALL}")
                time.sleep(45)  # 45초 대기
                
                # 기존 MilvusManager의 모니터링 중지
                if hasattr(processor, 'milvus_manager') and hasattr(processor.milvus_manager, 'stop_monitoring'):
                    processor.milvus_manager.stop_monitoring()
                
                # 새로운 MilvusManager 인스턴스 생성
                processor.milvus_manager = MilvusManager()
                print(f"{Fore.GREEN}✅ Successfully connected to fresh Milvus instance{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}❌ Error reconnecting to Milvus: {e}{Style.RESET_ALL}")
                
                # 오류 유형에 따른 구체적인 안내
                if "nodes not enough" in str(e):
                    print(f"{Fore.YELLOW}🕰️ This error means Milvus services are still starting up.{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}Solutions:{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}  1. Wait 2-3 minutes and try again{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}  2. Run 'start-milvus.bat' and wait for completion{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}  3. Check if all containers are running: podman ps{Style.RESET_ALL}")
                elif "already in use" in str(e) or "container name" in str(e).lower():
                    print(f"{Fore.YELLOW}🔄 This appears to be a container name conflict.{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}Please run 'complete-reset.bat' to clean up all containers, then try again.{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}Please wait a moment and try again{Style.RESET_ALL}")
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
    """Perform incremental embedding with automatic cleanup of deleted files"""
    from colorama import Fore, Style
    
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}Starting incremental embedding & deleted cleanup process...{Style.RESET_ALL}")
    print("This will process new/modified files and automatically clean up deleted files.")
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

def perform_cleanup_deleted_files(processor):
    """삭제된 파일의 embedding 정리"""
    from colorama import Fore, Style
    
    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}Starting cleanup of deleted files...{Style.RESET_ALL}")
    print("This will detect files that have been deleted from your vault")
    print("but still exist in the Milvus embedding database.\n")
    
    try:
        # 삭제된 파일 탐지
        deleted_files = processor.detect_deleted_files()
        
        if not deleted_files:
            print(f"\n{Fore.GREEN}{Style.BRIGHT}✅ No deleted files found. Your database is clean!{Style.RESET_ALL}")
            return
        
        # 삭제될 파일 목록 표시
        print(f"\n{Fore.YELLOW}Found {len(deleted_files)} deleted files:{Style.RESET_ALL}")
        
        # 처음 10개 파일만 표시
        display_count = min(10, len(deleted_files))
        for i, file_path in enumerate(deleted_files[:display_count]):
            print(f"{Fore.YELLOW}  {i+1:2d}. {file_path}{Style.RESET_ALL}")
        
        if len(deleted_files) > display_count:
            print(f"{Fore.YELLOW}  ... and {len(deleted_files) - display_count} more files{Style.RESET_ALL}")
        
        # 사용자 확인
        print(f"\n{Fore.RED}{Style.BRIGHT}⚠️  WARNING: This will permanently remove embedding data for these files!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}If you have backups of these files and want to restore them later,")
        print(f"you will need to run embedding again.{Style.RESET_ALL}")
        
        confirm = input(f"\n{Fore.YELLOW}Do you want to proceed with cleanup? (y/N): {Style.RESET_ALL}")
        
        if confirm.lower() in ['y', 'yes']:
            print(f"\n{Fore.CYAN}Starting cleanup process...{Style.RESET_ALL}")
            
            # 삭제 실행
            success_count = processor.cleanup_deleted_embeddings(deleted_files)
            
            if success_count > 0:
                print(f"\n{Fore.GREEN}{Style.BRIGHT}🎉 Cleanup completed successfully!{Style.RESET_ALL}")
                print(f"{Fore.GREEN}✅ Removed embeddings for {success_count} deleted files{Style.RESET_ALL}")
                
                if success_count < len(deleted_files):
                    failed_count = len(deleted_files) - success_count
                    print(f"{Fore.YELLOW}⚠️  {failed_count} files could not be removed (may require manual cleanup){Style.RESET_ALL}")
            else:
                print(f"\n{Fore.RED}{Style.BRIGHT}❌ Cleanup failed. No files were removed.{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Please check the error messages above.{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.CYAN}Cleanup cancelled by user.{Style.RESET_ALL}")
            
    except Exception as e:
        print(f"\n{Fore.RED}{Style.BRIGHT}Error during cleanup process: {e}{Style.RESET_ALL}")
        import traceback
        print(f"\n{Fore.RED}Stack trace:\n{traceback.format_exc()}{Style.RESET_ALL}")
    finally:
        print(f"\n{Fore.CYAN}Cleanup process finished.{Style.RESET_ALL}")

def start_mcp_server():
    """MCP Server startup only - exits after starting"""
    from colorama import Fore, Style
    import subprocess
    
    print(f"\n{Fore.CYAN}{Style.BRIGHT}Starting MCP Server for Claude Desktop...{Style.RESET_ALL}")
    print(f"Server name: {config.FASTMCP_SERVER_NAME}")
    print(f"Transport: {config.FASTMCP_TRANSPORT}")
    
    # Start MCP server and exit the main program
    try:
        mcp_script_path = os.path.join(os.path.dirname(__file__), "mcp_server.py")
        print(f"{Fore.GREEN}Starting MCP server: {mcp_script_path}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}The main program will now exit to allow MCP server to run independently.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}This is the intended behavior - MCP server should run continuously.{Style.RESET_ALL}")
        
        # Execute MCP server directly and exit
        os.execv(sys.executable, [sys.executable, mcp_script_path])
        
    except Exception as e:
        print(f"{Fore.RED}Error starting MCP server: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Falling back to system call...{Style.RESET_ALL}")
        os.system(f'python "{mcp_script_path}"')
        sys.exit(0)  # Exit after starting MCP server

def show_menu():
    """Display the command-line menu with colorful options"""
    from colorama import Fore, Style, Back
    
    print("\n" + "="*70)
    print(f"{Style.BRIGHT}{Fore.CYAN}Obsidian-Milvus-Claude Desktop Command Menu{Style.RESET_ALL}")
    print("="*70)
    
    # Option 1: Start MCP Server (Green background for main option)
    print(f"{Fore.WHITE}{Back.GREEN}{Style.BRIGHT} 1 {Style.RESET_ALL} {Fore.GREEN}{Style.BRIGHT}Start MCP Server{Style.RESET_ALL} (for Claude Desktop)")
    
    # Option 2: Full Embedding (Red background for caution)
    print(f"{Fore.WHITE}{Back.RED}{Style.BRIGHT} 2 {Style.RESET_ALL} {Fore.RED}{Style.BRIGHT}Full Embedding{Style.RESET_ALL} (reindex all files)")
    
    # Option 3: Incremental Embedding with Cleanup (Yellow background for moderate caution)
    print(f"{Fore.BLACK}{Back.YELLOW}{Style.BRIGHT} 3 {Style.RESET_ALL} {Fore.YELLOW}{Style.BRIGHT}Incremental & Deleted Cleanup{Style.RESET_ALL} (process new/modified + remove deleted)")
    
    # Option 4: Cleanup Deleted Files (Magenta background for maintenance)
    print(f"{Fore.WHITE}{Back.MAGENTA}{Style.BRIGHT} 4 {Style.RESET_ALL} {Fore.MAGENTA}{Style.BRIGHT}Cleanup Deleted Files{Style.RESET_ALL} (remove orphaned embeddings)")
    
    # Option 5: Exit (Blue for neutral option)
    print(f"{Fore.WHITE}{Back.BLUE}{Style.BRIGHT} 5 {Style.RESET_ALL} {Fore.BLUE}{Style.BRIGHT}Exit{Style.RESET_ALL}")
    
    print("="*70)
    choice = input(f"{Fore.CYAN}Enter your choice (1-5): {Style.RESET_ALL}")
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
            # Option 1: Start MCP Server - this will exit the program
            start_mcp_server()
            # This line should never be reached due to os.execv in start_mcp_server
            break
        elif choice == '2':
            perform_full_embedding(processor)
        elif choice == '3':
            perform_incremental_embedding(processor)
        elif choice == '4':
            perform_cleanup_deleted_files(processor)
        elif choice == '5':
            print("Exiting program...")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")
        
        time.sleep(1)

if __name__ == "__main__":
    main()
