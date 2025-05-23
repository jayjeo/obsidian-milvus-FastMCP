#!/usr/bin/env python3
"""
Safe Milvus Podman Controller - 데이터 보존 버전
기존 embedding 데이터를 보존하면서 Milvus 컨테이너를 안전하게 재생성합니다.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

class Colors:
    """Terminal color codes"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(message, color=Colors.ENDC):
    """Print with color"""
    print(f"{color}{message}{Colors.ENDC}")

def print_header(title):
    """Print header"""
    print_colored(f"\n{'='*60}", Colors.HEADER)
    print_colored(f"🔧 {title}", Colors.HEADER)
    print_colored(f"{'='*60}", Colors.HEADER)

def input_colored(prompt, color=Colors.OKCYAN):
    """Colored input"""
    return input(f"{color}{prompt}{Colors.ENDC}")

def get_podman_path():
    """Find Podman executable path"""
    possible_paths = [
        "podman",  # In PATH
        "/usr/bin/podman",  # Linux default path
        "/opt/homebrew/bin/podman",  # macOS Homebrew
        "/usr/local/bin/podman",  # macOS other
        "C:\\Program Files\\RedHat\\Podman\\podman.exe",  # Windows
    ]
    
    for path in possible_paths:
        try:
            result = subprocess.run([path, "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                return path
        except:
            continue
    
    return None

class SafeMilvusPodmanController:
    """Safe Milvus controller - 데이터 보존 중심"""
    
    def __init__(self, podman_path, data_base_path=None):
        self.podman_path = podman_path
        self.network = "milvus-network"
        
        # 기본 데이터 저장 경로 설정
        if data_base_path is None:
            # 프로젝트 디렉토리 내에 안전한 저장소 만들기
            self.project_dir = Path(__file__).parent.resolve()
            self.data_base_path = self.project_dir / "milvus_persistent_data"
        else:
            self.data_base_path = Path(data_base_path)
        
        # 각 서비스별 데이터 경로
        self.data_paths = {
            "etcd": self.data_base_path / "etcd_data",
            "minio": self.data_base_path / "minio_data",
            "milvus": self.data_base_path / "milvus_data"
        }
        
        # 컨테이너 이미지
        self.images = {
            "etcd": "quay.io/coreos/etcd:v3.5.5",
            "minio": "minio/minio:RELEASE.2023-03-20T20-16-18Z",
            "milvus": "milvusdb/milvus:v2.3.3"
        }
        
        self.api_port = "19530"
        self.web_port = "9091"
    
    def run_command(self, cmd):
        """Execute command"""
        try:
            if isinstance(cmd, str):
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            else:
                result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0, result.stdout, result.stderr
        except Exception as e:
            return False, "", str(e)
    
    def check_existing_data(self):
        """기존 데이터 확인"""
        print_colored("🔍 기존 데이터 확인 중...", Colors.OKBLUE)
        
        existing_data = {}
        for service, path in self.data_paths.items():
            if path.exists() and any(path.iterdir()):
                existing_data[service] = path
                print_colored(f"  ✅ {service} 데이터 발견: {path}", Colors.OKGREEN)
            else:
                print_colored(f"  ⚪ {service} 데이터 없음: {path}", Colors.ENDC)
        
        return existing_data
    
    def migrate_existing_data(self):
        """기존 데이터 마이그레이션"""
        print_colored("🔄 기존 데이터 마이그레이션 확인...", Colors.OKBLUE)
        
        # 기존 위치에서 데이터 찾기
        old_data_locations = [
            "G:/JJ Dropbox/J J/PythonWorks/milvus/obsidian-milvus-openwebui/EmbeddingResult",
            "G:/JJ Dropbox/J J/PythonWorks/milvus/obsidian-milvus-FastMCP/EmbeddingResult",
            self.project_dir / "EmbeddingResult"
        ]
        
        migrated = False
        for old_location in old_data_locations:
            old_path = Path(old_location)
            if old_path.exists():
                print_colored(f"📂 기존 데이터 발견: {old_path}", Colors.WARNING)
                
                # 각 서비스별 데이터 마이그레이션
                for service in ["etcd", "minio", "milvus"]:
                    old_service_path = old_path / service
                    new_service_path = self.data_paths[service]
                    
                    if old_service_path.exists() and any(old_service_path.iterdir()):
                        if not new_service_path.exists():
                            print_colored(f"  🔄 {service} 데이터 복사 중...", Colors.OKBLUE)
                            
                            # 새 경로 생성
                            new_service_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            # 데이터 복사
                            import shutil
                            shutil.copytree(old_service_path, new_service_path)
                            print_colored(f"  ✅ {service} 데이터 복사 완료", Colors.OKGREEN)
                            migrated = True
                        else:
                            print_colored(f"  ⚪ {service} 데이터 이미 존재", Colors.ENDC)
                
                if migrated:
                    print_colored(f"📋 원본 데이터는 {old_path}에 그대로 보존됩니다.", Colors.OKGREEN)
                    break
        
        return migrated
    
    def create_data_directories(self):
        """데이터 디렉토리 생성"""
        print_colored("📁 데이터 디렉토리 준비 중...", Colors.OKBLUE)
        
        # 베이스 디렉토리 생성
        self.data_base_path.mkdir(parents=True, exist_ok=True)
        print_colored(f"  ✅ 베이스 디렉토리: {self.data_base_path}", Colors.OKGREEN)
        
        # 각 서비스별 디렉토리 생성
        for service, path in self.data_paths.items():
            path.mkdir(parents=True, exist_ok=True)
            print_colored(f"  ✅ {service} 디렉토리: {path}", Colors.OKGREEN)
        
        return True
    
    def backup_existing_data(self):
        """기존 데이터 백업"""
        print_colored("💾 데이터 백업 생성 중...", Colors.OKBLUE)
        
        backup_base = self.data_base_path.parent / f"milvus_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        import shutil
        if self.data_base_path.exists():
            shutil.copytree(self.data_base_path, backup_base)
            print_colored(f"  ✅ 백업 생성 완료: {backup_base}", Colors.OKGREEN)
            return backup_base
        else:
            print_colored("  ⚪ 백업할 데이터 없음", Colors.ENDC)
            return None
    
    def start_machine(self):
        """Start Podman machine (if needed)"""
        if os.name == 'nt' or sys.platform == 'darwin':
            print_colored("🔧 Podman 머신 시작 중...", Colors.OKBLUE)
            success, _, _ = self.run_command([self.podman_path, "machine", "start"])
            if success:
                print_colored("✅ Podman 머신 시작 완료", Colors.OKGREEN)
            else:
                print_colored("⚠️ Podman 머신 시작 실패 (이미 실행 중일 수 있음)", Colors.WARNING)
            time.sleep(2)
    
    def create_network(self):
        """Create network"""
        print_colored(f"🌐 네트워크 '{self.network}' 생성 중...", Colors.OKBLUE)
        success, _, _ = self.run_command([self.podman_path, "network", "exists", self.network])
        if not success:
            success, _, _ = self.run_command([self.podman_path, "network", "create", self.network])
            if success:
                print_colored("✅ 네트워크 생성 완료", Colors.OKGREEN)
            else:
                print_colored("❌ 네트워크 생성 실패", Colors.FAIL)
                return False
        else:
            print_colored("✅ 네트워크 이미 존재", Colors.OKGREEN)
        return True
    
    def stop_containers(self):
        """기존 컨테이너 정리 (데이터는 보존)"""
        print_colored("🧹 기존 컨테이너 정리 중...", Colors.OKBLUE)
        containers = ["milvus-standalone", "milvus-minio", "milvus-etcd"]
        
        for container in containers:
            # 컨테이너 중지
            success, _, _ = self.run_command([self.podman_path, "stop", container])
            if success:
                print_colored(f"  ✅ {container} 중지됨", Colors.OKGREEN)
            
            # 컨테이너 삭제 (볼륨은 보존)
            success, _, _ = self.run_command([self.podman_path, "rm", container])
            if success:
                print_colored(f"  ✅ {container} 삭제됨", Colors.OKGREEN)
        
        print_colored("💡 데이터는 안전하게 보존됩니다!", Colors.OKGREEN)
    
    def start_etcd(self):
        """Start etcd container with persistent data"""
        print_colored("[1/3] 📊 etcd 시작 중...", Colors.OKBLUE)
        
        # 절대 경로로 변환
        etcd_data_path = str(self.data_paths["etcd"].resolve())
        
        cmd = [
            self.podman_path, "run", "-d", "--name", "milvus-etcd", 
            "--network", self.network,
            "-v", f"{etcd_data_path}:/etcd",
            "-e", "ETCD_AUTO_COMPACTION_MODE=revision",
            "-e", "ETCD_AUTO_COMPACTION_RETENTION=1000", 
            "-e", "ETCD_QUOTA_BACKEND_BYTES=4294967296",
            "--user", "0:0",
            self.images["etcd"],
            "etcd", "-advertise-client-urls=http://127.0.0.1:2379",
            "-listen-client-urls", "http://0.0.0.0:2379", "--data-dir", "/etcd"
        ]
        
        success, _, stderr = self.run_command(cmd)
        if success:
            print_colored("  ✅ etcd 시작 완료", Colors.OKGREEN)
        else:
            print_colored(f"  ❌ etcd 시작 실패: {stderr}", Colors.FAIL)
        return success
    
    def start_minio(self):
        """Start MinIO container with persistent data"""
        print_colored("[2/3] 🗄️ MinIO 시작 중...", Colors.OKBLUE)
        
        # 절대 경로로 변환
        minio_data_path = str(self.data_paths["minio"].resolve())
        
        cmd = [
            self.podman_path, "run", "-d", "--name", "milvus-minio",
            "--network", self.network,
            "-v", f"{minio_data_path}:/minio_data",
            "-e", "MINIO_ACCESS_KEY=minioadmin",
            "-e", "MINIO_SECRET_KEY=minioadmin", 
            "--user", "0:0",
            self.images["minio"],
            "server", "/minio_data"
        ]
        
        success, _, stderr = self.run_command(cmd)
        if success:
            print_colored("  ✅ MinIO 시작 완료", Colors.OKGREEN)
        else:
            print_colored(f"  ❌ MinIO 시작 실패: {stderr}", Colors.FAIL)
        return success
    
    def start_milvus(self):
        """Start Milvus container with persistent data"""
        print_colored("[3/3] 🚀 Milvus 시작 중...", Colors.OKBLUE)
        
        # 절대 경로로 변환
        milvus_data_path = str(self.data_paths["milvus"].resolve())
        
        cmd = [
            self.podman_path, "run", "-d", "--name", "milvus-standalone",
            "--network", self.network,
            "-v", f"{milvus_data_path}:/var/lib/milvus",
            "-p", f"{self.api_port}:{self.api_port}",
            "-p", f"{self.web_port}:{self.web_port}",
            "-e", "ETCD_ENDPOINTS=milvus-etcd:2379",
            "-e", "MINIO_ADDRESS=milvus-minio:9000",
            "--user", "0:0",
            self.images["milvus"],
            "milvus", "run", "standalone"
        ]
        
        success, _, stderr = self.run_command(cmd)
        if success:
            print_colored("  ✅ Milvus 시작 완료", Colors.OKGREEN)
        else:
            print_colored(f"  ❌ Milvus 시작 실패: {stderr}", Colors.FAIL)
        return success
    
    def check_status(self):
        """Check container status"""
        print_colored("\n📊 컨테이너 상태:", Colors.OKBLUE)
        success, stdout, _ = self.run_command([
            self.podman_path, "ps", 
            "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        ])
        if success:
            print_colored(stdout, Colors.ENDC)
        return success
    
    def show_data_info(self):
        """데이터 저장 정보 표시"""
        print_colored("\n💾 데이터 저장 정보:", Colors.OKBLUE)
        print_colored(f"📂 베이스 경로: {self.data_base_path}", Colors.ENDC)
        
        total_size = 0
        for service, path in self.data_paths.items():
            if path.exists():
                # 디렉토리 크기 계산
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                size_mb = size / (1024 * 1024)
                total_size += size_mb
                print_colored(f"  📁 {service}: {path} ({size_mb:.1f}MB)", Colors.ENDC)
            else:
                print_colored(f"  📁 {service}: {path} (비어있음)", Colors.ENDC)
        
        print_colored(f"📊 총 데이터 크기: {total_size:.1f}MB", Colors.OKGREEN)
    
    def wait_for_ready(self, max_wait_time=120):
        """Wait for Milvus to be ready"""
        print_colored(f"⏳ Milvus 서비스 준비 대기 중 (최대 {max_wait_time}초)...", Colors.WARNING)
        
        for i in range(max_wait_time):
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex(('localhost', int(self.api_port)))
                sock.close()
                
                if result == 0:
                    print_colored(f"\n✅ Milvus 서비스 준비 완료! ({i+1}초 소요)", Colors.OKGREEN)
                    return True
            except:
                pass
            
            time.sleep(1)
            if i % 10 == 0:
                print(f"\n[{i}초]", end="")
            print(".", end="", flush=True)
        
        print_colored(f"\n⚠️ {max_wait_time}초 대기 후 서비스 준비 확인 실패", Colors.WARNING)
        return False
    
    def safe_start_all(self):
        """안전한 전체 시작 프로세스"""
        print_header("안전한 Milvus 시작 (데이터 보존)")
        
        # 1. 기존 데이터 확인
        existing_data = self.check_existing_data()
        
        # 2. 데이터 마이그레이션 (필요시)
        self.migrate_existing_data()
        
        # 3. 사용자 확인
        if existing_data:
            print_colored("📋 기존 embedding 데이터가 발견되었습니다.", Colors.WARNING)
            print_colored("🔒 이 데이터는 안전하게 보존됩니다.", Colors.OKGREEN)
            
            choice = input_colored("계속 진행하시겠습니까? (y/n): ")
            if choice.lower() != 'y':
                print_colored("작업이 취소되었습니다.", Colors.WARNING)
                return False
        
        # 4. 백업 생성
        backup_path = self.backup_existing_data()
        
        # 5. 디렉토리 준비
        if not self.create_data_directories():
            return False
        
        # 6. Podman 머신 시작
        self.start_machine()
        
        # 7. 기존 컨테이너 정리
        self.stop_containers()
        
        # 8. 네트워크 생성
        if not self.create_network():
            return False
        
        # 9. 서비스 시작
        if not self.start_etcd():
            return False
        
        if not self.start_minio():
            return False
        
        print_colored("⏳ 의존성 서비스 준비 대기...", Colors.WARNING)
        time.sleep(15)
        
        if not self.start_milvus():
            return False
        
        # 10. 상태 확인
        self.check_status()
        
        # 11. 서비스 준비 대기
        if self.wait_for_ready():
            print_colored("\n" + "="*60, Colors.OKGREEN)
            print_colored("🎉 안전한 Milvus 시작 완료! 🎉", Colors.OKGREEN)
            print_colored("="*60, Colors.OKGREEN)
            print_colored(f"🌐 Milvus API:    http://localhost:{self.api_port}", Colors.OKGREEN)
            print_colored(f"🌐 Web Interface: http://localhost:{self.web_port}", Colors.OKGREEN)
            print_colored("💾 모든 데이터가 안전하게 보존되었습니다!", Colors.OKGREEN)
            
            if backup_path:
                print_colored(f"📋 백업 위치: {backup_path}", Colors.OKGREEN)
            
            self.show_data_info()
            print_colored("="*60, Colors.OKGREEN)
            
            return True
        else:
            print_colored("⚠️ 서비스 시작은 완료되었으나 준비 상태 확인에 실패했습니다.", Colors.WARNING)
            print_colored("수동으로 확인해보세요: http://localhost:19530", Colors.OKBLUE)
            return False

def main():
    """메인 함수"""
    print_header("Safe Milvus Podman Controller")
    
    # Podman 경로 찾기
    podman_path = get_podman_path()
    if not podman_path:
        print_colored("❌ Podman을 찾을 수 없습니다.", Colors.FAIL)
        print_colored("Podman을 설치하고 다시 시도해주세요.", Colors.OKBLUE)
        return
    
    print_colored(f"✅ Podman 발견: {podman_path}", Colors.OKGREEN)
    
    # 데이터 경로 설정
    print_colored("\n📂 데이터 저장 경로를 선택하세요:", Colors.OKBLUE)
    print_colored("1. 기본 경로 (프로젝트 내 milvus_persistent_data)", Colors.ENDC)
    print_colored("2. 커스텀 경로 지정", Colors.ENDC)
    
    choice = input_colored("선택 (1-2): ")
    
    data_path = None
    if choice == "2":
        custom_path = input_colored("저장할 경로를 입력하세요: ")
        if custom_path:
            data_path = custom_path
    
    # 컨트롤러 생성 및 실행
    controller = SafeMilvusPodmanController(podman_path, data_path)
    
    try:
        if controller.safe_start_all():
            print_colored("\n✅ 모든 작업이 성공적으로 완료되었습니다!", Colors.OKGREEN)
        else:
            print_colored("\n❌ 일부 작업이 실패했습니다.", Colors.FAIL)
    except KeyboardInterrupt:
        print_colored("\n\n⏹️ 사용자에 의해 중단되었습니다.", Colors.WARNING)
    except Exception as e:
        print_colored(f"\n❌ 예상치 못한 오류: {e}", Colors.FAIL)

if __name__ == "__main__":
    main()
