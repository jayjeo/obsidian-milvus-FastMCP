#!/usr/bin/env python3
"""
Interactive Milvus MCP Test - Step-by-step testing with automatic problem resolution
Users can directly select each step to test, and the system automatically resolves issues when they occur.
"""

import asyncio
import json
import os
import shutil
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

def print_step(step_num, title):
    """Print step title"""
    print_colored(f"\n{step_num}. {title}", Colors.OKBLUE)
    print_colored("-" * 40, Colors.OKBLUE)

def input_colored(prompt, color=Colors.OKCYAN):
    """Colored input"""
    return input(f"{color}{prompt}{Colors.ENDC}")

def install_package(package_name, import_name=None):
    """Automatic package installation"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print_colored(f"✅ {package_name} already installed", Colors.OKGREEN)
        return True
    
    except ImportError:
        print_colored(f"⚠️ {package_name} package not found. Attempting installation...", Colors.WARNING)
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print_colored(f"✅ {package_name} installation complete", Colors.OKGREEN)
            return True
        except subprocess.CalledProcessError as e:
            print_colored(f"❌ {package_name} installation failed: {e}", Colors.FAIL)
            return False

def check_milvus_server():
    """Check Milvus server status"""
    # Check multiple endpoints
    endpoints = [
        "http://localhost:19530/health",
        "http://localhost:9091/healthz", 
        "http://localhost:19530"
    ]
    
    for endpoint in endpoints:
        try:
            import requests
            response = requests.get(endpoint, timeout=10)
            if response.status_code == 200:
                return True
        except:
            continue
    
    # Check TCP port connection
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('localhost', 19530))
        sock.close()
        return result == 0
    except:
        return False

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

def check_podman():
    """Check Podman installation and status"""
    podman_path = get_podman_path()
    if not podman_path:
        return False, None
    
    try:
        result = subprocess.run([podman_path, "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            return True, podman_path
    except:
        pass
    
    return False, None

class MilvusPodmanController:
    """Safe Milvus controller - 데이터 보존 중심"""
    
    def __init__(self, podman_path):
        self.podman_path = podman_path
        self.network = "milvus-network"
        
        # 프로젝트 디렉토리 내에 안전한 저장소 만들기 (config.py와 동일한 경로 사용)
        self.project_dir = Path(__file__).parent.resolve()
        
        # config.py에서 external storage path 가져오기
        try:
            import sys
            sys.path.insert(0, str(self.project_dir))
            import config
            # config.py에서 정의된 경로 사용
            self.data_base_path = Path(config.get_external_storage_path())
        except Exception as e:
            print_colored(f"Warning: Could not import config.py: {e}", Colors.WARNING)
            # 폴백: 기본값 사용
            self.data_base_path = self.project_dir / "MilvusData"
        
        # 각 서비스별 데이터 경로 (현재 compose 파일과 일치하도록 수정)
        self.volumes_base_path = self.project_dir / "volumes"  # 컨테이너 데이터
        self.data_paths = {
            "etcd": self.volumes_base_path / "etcd",           # volumes/etcd (컨테이너 데이터)
            "minio": self.data_base_path / "minio",            # MilvusData/minio (영구 데이터)
            "milvus": self.data_base_path / "milvus"           # MilvusData/milvus (영구 데이터)
        }
        
        # 기존 데이터 위치도 확인 (현재는 없음)
        self.legacy_data_paths = []
        
        self.images = {
            "etcd": "quay.io/coreos/etcd:v3.5.5",
            "minio": "minio/minio:RELEASE.2023-03-20T20-16-18Z",
            "milvus": "milvusdb/milvus:v2.3.3"
        }
        self.api_port = "19530"
        self.web_port = "9091"
    
    def show_data_info(self):
        """데이터 저장 정보 표시"""
        print_colored("\n💾 데이터 저장 정보:", Colors.OKBLUE)
        print_colored(f"📂 베이스 경로: {self.data_base_path}", Colors.ENDC)
        
        total_size = 0
        for service, path in self.data_paths.items():
            if path.exists():
                # 디렉토리 크기 계산
                try:
                    size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                    size_mb = size / (1024 * 1024)
                    total_size += size_mb
                    print_colored(f"  📁 {service}: {path} ({size_mb:.1f}MB)", Colors.ENDC)
                except:
                    print_colored(f"  📁 {service}: {path} (크기 계산 실패)", Colors.ENDC)
            else:
                print_colored(f"  📁 {service}: {path} (비어있음)", Colors.ENDC)
        
        print_colored(f"📊 총 데이터 크기: {total_size:.1f}MB", Colors.OKGREEN)
    
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
    
    def start_machine(self):
        """Start Podman machine (if needed)"""
        # Windows/macOS may need Podman machine
        if os.name == 'nt' or sys.platform == 'darwin':
            print_colored("🔧 Starting Podman machine...", Colors.OKBLUE)
            success, _, _ = self.run_command([self.podman_path, "machine", "start"])
            if success:
                print_colored("✅ Podman machine start complete", Colors.OKGREEN)
            else:
                print_colored("⚠️ Podman machine start failed (may already be running)", Colors.WARNING)
            time.sleep(2)
    
    def create_network(self):
        """Create network"""
        print_colored(f"🌐 Creating network '{self.network}'...", Colors.OKBLUE)
        # Check existing network
        success, _, _ = self.run_command([self.podman_path, "network", "exists", self.network])
        if not success:
            success, _, _ = self.run_command([self.podman_path, "network", "create", self.network])
            if success:
                print_colored("✅ Network creation complete", Colors.OKGREEN)
            else:
                print_colored("❌ Network creation failed", Colors.FAIL)
                return False
        else:
            print_colored("✅ Network already exists", Colors.OKGREEN)
        return True
    

    def stop_containers(self):
        """기존 컴테이너 정리 (데이터는 보존)"""
        print_colored("🧹 기존 컴테이너 정리 중...", Colors.OKBLUE)
        containers = ["milvus-standalone", "milvus-minio", "milvus-etcd"]
        
        for container in containers:
            # 컴테이너 중지
            success, _, _ = self.run_command([self.podman_path, "stop", container])
            if success:
                print_colored(f"  ✅ {container} 중지됨", Colors.OKGREEN)
            
            # 컴테이너 삭제 (볼륨은 보존)
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
            print_colored(f"  💾 데이터 위치: {etcd_data_path}", Colors.ENDC)
        else:
            print_colored(f"  ❌ etcd 시작 실패: {stderr}", Colors.FAIL)
        return success
    
    def check_and_migrate_data(self):
        """기존 데이터 확인 및 마이그레이션"""
        print_colored("🔍 기존 embedding 데이터 확인 중...", Colors.OKBLUE)
        
        # 기존 데이터 확인
        existing_data = False
        migration_source = None
        
        for legacy_path in self.legacy_data_paths:
            if legacy_path.exists():
                print_colored(f"📂 기존 데이터 발견: {legacy_path}", Colors.WARNING)
                
                # 각 서비스 데이터 확인
                for service in ["etcd", "minio", "milvus"]:
                    service_path = legacy_path / service
                    if service_path.exists() and any(service_path.iterdir()):
                        existing_data = True
                        migration_source = legacy_path
                        print_colored(f"  ✅ {service} 데이터 있음", Colors.OKGREEN)
                
                if existing_data:
                    break
        
        if existing_data:
            print_colored("📋 기존 embedding 데이터가 발견되었습니다!", Colors.WARNING)
            print_colored("🔒 이 데이터를 안전하게 보존하고 새 위치로 복사합니다.", Colors.OKGREEN)
            
            choice = input_colored("계속 진행하시겠습니까? (y/n): ")
            if choice.lower() != 'y':
                print_colored("작업이 취소되었습니다.", Colors.WARNING)
                return False
            
            # 데이터 마이그레이션
            self.migrate_data(migration_source)
        
        # 데이터 디렉토리 준비
        self.create_data_directories()
        return True
    
    def migrate_data(self, source_path):
        """데이터 마이그레이션"""
        print_colored("🔄 데이터 마이그레이션 시작...", Colors.OKBLUE)
        
        import shutil
        
        for service in ["etcd", "minio", "milvus"]:
            source_service_path = source_path / service
            target_service_path = self.data_paths[service]
            
            if source_service_path.exists() and any(source_service_path.iterdir()):
                if not target_service_path.exists():
                    print_colored(f"  🔄 {service} 데이터 복사 중...", Colors.OKBLUE)
                    target_service_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(source_service_path, target_service_path)
                    print_colored(f"  ✅ {service} 데이터 복사 완료", Colors.OKGREEN)
                else:
                    print_colored(f"  ⚪ {service} 데이터 이미 존재", Colors.ENDC)
        
        print_colored(f"📋 원본 데이터는 {source_path}에 그대로 보존됩니다.", Colors.OKGREEN)
    
    def create_data_directories(self):
        """데이터 디렉토리 생성"""
        print_colored("📁 데이터 디렉토리 준비 중...", Colors.OKBLUE)
        
        # 베이스 디렉토리 생성 (MilvusData - 영구 데이터)
        self.data_base_path.mkdir(parents=True, exist_ok=True)
        print_colored(f"  ✅ MilvusData 디렉토리: {self.data_base_path}", Colors.OKGREEN)
        
        # volumes 디렉토리 생성 (컨테이너 데이터)
        self.volumes_base_path.mkdir(parents=True, exist_ok=True)
        print_colored(f"  ✅ volumes 디렉토리: {self.volumes_base_path}", Colors.OKGREEN)
        
        # 각 서비스별 디렉토리 생성
        for service, path in self.data_paths.items():
            path.mkdir(parents=True, exist_ok=True)
            if service == "etcd":
                print_colored(f"  ✅ {service} 디렉토리: {path} (컨테이너 데이터)", Colors.OKGREEN)
            else:
                print_colored(f"  ✅ {service} 디렉토리: {path} (영구 데이터)", Colors.OKGREEN)
        
        return True
    
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
            print_colored(f"  💾 데이터 위치: {minio_data_path}", Colors.ENDC)
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
            print_colored("  ✅ Milvus 시작 완룼", Colors.OKGREEN)
            print_colored(f"  💾 데이터 위치: {milvus_data_path}", Colors.ENDC)
        else:
            print_colored(f"  ❌ Milvus 시작 실패: {stderr}", Colors.FAIL)
        return success
    
    def check_status(self):
        """Check container status"""
        print_colored("\n📊 Container status:", Colors.OKBLUE)
        success, stdout, _ = self.run_command([
            self.podman_path, "ps", 
            "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        ])
        if success:
            print_colored(stdout, Colors.ENDC)
        return success
    
    def check_container_logs(self, container_name):
        """Check container logs"""
        print_colored(f"📋 Checking {container_name} logs...", Colors.OKBLUE)
        success, stdout, stderr = self.run_command([self.podman_path, "logs", "--tail", "20", container_name])
        if success:
            print_colored(f"📋 {container_name} recent logs:", Colors.OKBLUE)
            print_colored(stdout, Colors.ENDC)
            if stderr:
                print_colored("🔴 Error logs:", Colors.WARNING)
                print_colored(stderr, Colors.ENDC)
        return success
    
    def diagnose_milvus_issues(self):
        """Diagnose Milvus issues"""
        print_colored("\n🔍 Diagnosing Milvus issues...", Colors.OKBLUE)
        
        # 1. Check container status
        success, stdout, _ = self.run_command([self.podman_path, "ps", "-a", "--filter", "name=milvus"])
        if success:
            print_colored("📊 Milvus-related container status:", Colors.OKBLUE)
            print_colored(stdout, Colors.ENDC)
        
        # 2. Check individual container logs
        containers = ["milvus-etcd", "milvus-minio", "milvus-standalone"]
        for container in containers:
            # Check if container exists
            success, _, _ = self.run_command([self.podman_path, "container", "exists", container])
            if success:
                self.check_container_logs(container)
                print("-" * 50)
        
        # 3. Check ports
        print_colored("🔌 Checking port usage:", Colors.OKBLUE)
        try:
            import socket
            ports_to_check = [19530, 9091, 2379, 9000]
            for port in ports_to_check:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                if result == 0:
                    print_colored(f"  ✅ Port {port}: Open", Colors.OKGREEN)
                else:
                    print_colored(f"  ❌ Port {port}: Closed", Colors.FAIL)
        except Exception as e:
            print_colored(f"Port check error: {e}", Colors.WARNING)
        
        # 4. Check network
        print_colored("🌐 Network status:", Colors.OKBLUE)
        success, stdout, _ = self.run_command([self.podman_path, "network", "inspect", self.network])
        if success:
            print_colored("  ✅ Network normal", Colors.OKGREEN)
        else:
            print_colored("  ❌ Network issues", Colors.FAIL)
    
    def restart_milvus_container(self):
        """Restart Milvus container only"""
        print_colored("🔄 Restarting Milvus container...", Colors.OKBLUE)
        
        # Stop Milvus container
        self.run_command([self.podman_path, "stop", "milvus-standalone"])
        self.run_command([self.podman_path, "rm", "milvus-standalone"])
        
        # Wait briefly
        time.sleep(5)
        
        # Restart Milvus
        if self.start_milvus():
            print_colored("✅ Milvus container restart complete", Colors.OKGREEN)
            return True
        else:
            print_colored("❌ Milvus container restart failed", Colors.FAIL)
            return False
    
    def wait_for_milvus_ready(self, max_wait_time=180):
        """Wait for Milvus ready state (extended wait time with diagnostics)"""
        print_colored(f"⏳ Waiting for Milvus service ready (max {max_wait_time} seconds)...", Colors.WARNING)
        
        for i in range(max_wait_time):
            if check_milvus_server():
                print_colored(f"\n✅ Milvus server ready! (took {i+1} seconds)", Colors.OKGREEN)
                return True
            
            # Status check every 30 seconds
            if i > 0 and i % 30 == 0:
                print_colored(f"\n⏳ {i} seconds elapsed... Checking status", Colors.WARNING)
                self.check_status()
            
            time.sleep(1)
            if i % 10 == 0:
                print(f"\n[{i}s]", end="")
            print(".", end="", flush=True)
        
        print_colored(f"\n⚠️ Milvus not ready after {max_wait_time} seconds wait.", Colors.WARNING)
        
        # Run diagnostics
        self.diagnose_milvus_issues()
        
        # Offer additional wait option
        choice = input_colored("\n🔧 Would you like to wait an additional 60 seconds? (y/n): ")
        if choice.lower() == 'y':
            return self.wait_for_milvus_ready(60)
        
        return False
    
    def start_all(self):
        """Start complete Milvus stack"""
        print_colored("="*60, Colors.HEADER)
        print_colored("    안전한 Milvus 시작 (데이터 보존)", Colors.HEADER)
        print_colored("="*60, Colors.HEADER)
        
        # 1. 기존 데이터 확인 및 마이그레이션
        if not self.check_and_migrate_data():
            return False
        
        # 2. Podman 머신 시작
        self.start_machine()
        
        # 3. 기존 컴테이너 정리
        self.stop_containers()
        
        if not self.create_network():
            return False
        
        # Start services
        if not self.start_etcd():
            return False
        
        if not self.start_minio():
            return False
        
        print_colored("⏳ Waiting for dependency services to be ready...", Colors.WARNING)
        time.sleep(15)
        
        if not self.start_milvus():
            return False
        
        print_colored("\n⏳ Waiting for service readiness...", Colors.WARNING)
        time.sleep(20)
        
        # Final status check
        self.check_status()
        
        print_colored("\n" + "="*60, Colors.OKGREEN)
        print_colored("                    🎉 Success! 🎉", Colors.OKGREEN)
        print_colored("="*60, Colors.OKGREEN)
        print_colored(f"🌐 Milvus API:    http://localhost:{self.api_port}", Colors.OKGREEN)
        print_colored(f"🌐 Web Interface: http://localhost:{self.web_port}", Colors.OKGREEN)
        print_colored("💾 Data persists after restart", Colors.OKGREEN)
        print_colored("="*60, Colors.OKGREEN)
        
        return True

def start_milvus_server():
    """Attempt to start Milvus server (using Podman)"""
    print_colored("🚀 Attempting to start Milvus server...", Colors.WARNING)
    
    # Check Podman
    podman_available, podman_path = check_podman()
    
    if podman_available:
        print_colored(f"📦 Starting Milvus using Podman...", Colors.OKBLUE)
        print_colored(f"   Podman path: {podman_path}", Colors.ENDC)
        
        try:
            controller = MilvusPodmanController(podman_path)
            
            if controller.start_all():
                # Enhanced server startup wait and verification
                if controller.wait_for_milvus_ready():
                    print_colored("✅ Milvus server is fully ready!", Colors.OKGREEN)
                    return True
                else:
                    print_colored("⚠️ Milvus containers are running but service is not fully ready.", Colors.WARNING)
                    
                    # Offer restart option
                    choice = input_colored("🔄 Would you like to restart the Milvus container? (y/n): ")
                    if choice.lower() == 'y':
                        if controller.restart_milvus_container():
                            return controller.wait_for_milvus_ready(120)  # 2 minute additional wait
                    
                    print_colored("💡 Manual verification methods:", Colors.OKBLUE)
                    print_colored("1. Check container logs: podman logs milvus-standalone", Colors.ENDC)
                    print_colored("2. Check ports: netstat -an | grep 19530", Colors.ENDC)
                    print_colored("3. Check web interface: http://localhost:9091", Colors.ENDC)
                    print_colored("4. Try connection test again after some time", Colors.ENDC)
                    
                    return False
            else:
                return False
            
        except Exception as e:
            print_colored(f"❌ Milvus startup with Podman failed: {e}", Colors.FAIL)
            return False
    else:
        print_colored("❌ Podman not found.", Colors.FAIL)
        print_colored("💡 Solution:", Colors.OKBLUE)
        print_colored("1. Install Podman:", Colors.ENDC)
        print_colored("   - Windows: https://github.com/containers/podman/blob/main/docs/tutorials/podman-for-windows.md", Colors.ENDC)
        print_colored("   - macOS: brew install podman", Colors.ENDC)
        print_colored("   - Linux: Use distribution package manager", Colors.ENDC)
        print_colored("2. Or install Milvus directly: https://milvus.io/docs/install_standalone-docker.md", Colors.ENDC)
        return False

class MilvusTest:
    def __init__(self):
        self.test_results = {}
        # Check current project directory
        self.project_dir = Path(__file__).parent.resolve()
        self.mcp_server_path = self.project_dir / "mcp_server.py"
    
    def test_dependencies(self):
        """1. Install and check required packages"""
        print_step(1, "Install and check required packages")
        
        packages = [
            ("mcp", "mcp.server.fastmcp"),
            ("pymilvus", "pymilvus"),
            ("requests", "requests"),
            ("numpy", "numpy")
        ]
        
        all_installed = True
        for package_name, import_name in packages:
            if not install_package(package_name, import_name):
                all_installed = False
        
        if all_installed:
            print_colored("✅ All required packages are installed!", Colors.OKGREEN)
        else:
            print_colored("❌ Some package installations failed.", Colors.FAIL)
            print_colored("💡 Try manual installation: pip install mcp pymilvus requests numpy", Colors.OKBLUE)
        
        self.test_results["dependencies"] = all_installed
        return all_installed
    
    def test_milvus_connection(self):
        """2. Milvus connection test"""
        print_step(2, "Milvus connection test")
        
        # First check server status
        if not check_milvus_server():
            print_colored("❌ Cannot connect to Milvus server.", Colors.FAIL)
            
            choice = input_colored("🔧 Would you like to automatically start Milvus server? (y/n): ")
            if choice.lower() == 'y':
                if start_milvus_server():
                    print_colored("✅ Milvus server has been started!", Colors.OKGREEN)
                else:
                    print_colored("❌ Milvus server startup failed.", Colors.FAIL)
                    self.test_results["milvus_connection"] = False
                    return False
            else:
                print_colored("💡 Please start Milvus server manually.", Colors.OKBLUE)
                self.test_results["milvus_connection"] = False
                return False
        
        try:
            from pymilvus import connections, utility
            
            # Connect to Milvus server
            connections.connect(
                alias="default",
                host='localhost',
                port='19530'
            )
            
            if connections.has_connection("default"):
                print_colored("✅ Milvus connection successful!", Colors.OKGREEN)
                
                # Display server info
                try:
                    print_colored(f"📊 Milvus server info:", Colors.OKBLUE)
                    collections = utility.list_collections()
                    print_colored(f"   Existing collections: {len(collections)}", Colors.ENDC)
                    if collections:
                        for col in collections:
                            print_colored(f"   - {col}", Colors.ENDC)
                except:
                    pass
                
                self.test_results["milvus_connection"] = True
                return True
            else:
                print_colored("❌ Milvus connection failed", Colors.FAIL)
                self.test_results["milvus_connection"] = False
                return False
                
        except Exception as e:
            print_colored(f"❌ Milvus connection error: {e}", Colors.FAIL)
            print_colored("💡 Solution:", Colors.OKBLUE)
            print_colored("1. Check if Milvus server is running", Colors.ENDC)
            print_colored("2. Check if port 19530 is available", Colors.ENDC)
            print_colored("3. Check firewall settings", Colors.ENDC)
            
            self.test_results["milvus_connection"] = False
            return False
    
    def test_collection_operations(self):
        """3. Collection creation and manipulation test"""
        print_step(3, "Collection creation and manipulation test")
        
        if not self.test_results.get("milvus_connection", False):
            print_colored("⚠️ Milvus connection required. Please run test 2 first.", Colors.WARNING)
            return False
        
        try:
            from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility
            import numpy as np
            
            collection_name = "test_obsidian_notes"
            
            # Drop existing collection if exists
            if utility.has_collection(collection_name):
                choice = input_colored(f"🗑️ Delete existing test collection '{collection_name}'? (y/n): ")
                if choice.lower() == 'y':
                    utility.drop_collection(collection_name)
                    print_colored(f"✅ Existing collection deleted", Colors.OKGREEN)
                else:
                    collection_name = f"test_obsidian_notes_{int(time.time())}"
                    print_colored(f"📝 Using new collection name: {collection_name}", Colors.OKBLUE)
            
            # Define field schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
            ]
            
            # Create collection schema
            schema = CollectionSchema(fields, f"Test collection for Obsidian notes")
            
            # Create collection
            collection = Collection(collection_name, schema)
            print_colored(f"✅ Collection '{collection_name}' created successfully", Colors.OKGREEN)
            
            # Generate test data
            test_data = [
                ["test.md", "# Test Document\n\nThis is a test document.", np.random.rand(384).tolist()],
                ["example.md", "# Example Document\n\nExample content.", np.random.rand(384).tolist()],
                ["sample.md", "# Sample Note\n\nSample content.", np.random.rand(384).tolist()]
            ]
            
            entities = [
                [item[0] for item in test_data],  # file_path
                [item[1] for item in test_data],  # content  
                [item[2] for item in test_data]   # embedding
            ]
            
            # Insert data
            insert_result = collection.insert(entities)
            print_colored(f"✅ Test data insertion successful: {len(insert_result.primary_keys)} items", Colors.OKGREEN)
            
            # Create index
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index("embedding", index_params)
            print_colored("✅ Vector index creation successful", Colors.OKGREEN)
            
            # Load collection
            collection.load()
            print_colored("✅ Collection memory load successful", Colors.OKGREEN)
            
            # Search test
            search_vectors = [np.random.rand(384).tolist()]
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            
            results = collection.search(
                search_vectors,
                "embedding",
                search_params,
                limit=3,
                output_fields=["file_path", "content"]
            )
            
            print_colored(f"✅ Search test successful: {len(results[0])} results", Colors.OKGREEN)
            for i, hit in enumerate(results[0]):
                print_colored(f"   {i+1}. {hit.entity.get('file_path')}: distance {hit.distance:.4f}", Colors.ENDC)
            
            self.test_results["collection_operations"] = True
            return True
            
        except Exception as e:
            print_colored(f"❌ Collection operation error: {e}", Colors.FAIL)
            print_colored("💡 Solution:", Colors.OKBLUE)
            print_colored("1. Check if Milvus server has sufficient memory", Colors.ENDC)
            print_colored("2. Check if collection name is valid", Colors.ENDC)
            print_colored("3. Check if data types and schema are correct", Colors.ENDC)
            
            self.test_results["collection_operations"] = False
            return False
    
    def test_mcp_server_file(self):
        """4. Local MCP server file test"""
        print_step(4, "Local MCP server file test")
        
        # Check local MCP server file
        if not self.mcp_server_path.exists():
            print_colored(f"❌ MCP server file not found: {self.mcp_server_path}", Colors.FAIL)
            print_colored("💡 Check if mcp_server.py file exists in the project directory.", Colors.OKBLUE)
            self.test_results["mcp_server_file"] = False
            return False
        
        print_colored(f"✅ MCP server file found: {self.mcp_server_path}", Colors.OKGREEN)
        
        # File syntax check
        try:
            with open(self.mcp_server_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic syntax check
            compile(content, str(self.mcp_server_path), 'exec')
            print_colored("✅ MCP server file syntax check passed", Colors.OKGREEN)
            
            # Check important imports and functions
            required_elements = [
                "from mcp.server.fastmcp import FastMCP",
                "@mcp.tool()",
                "search_documents",
                "get_document_content"
            ]
            
            missing_elements = []
            for element in required_elements:
                if element not in content:
                    missing_elements.append(element)
            
            if missing_elements:
                print_colored("⚠️ Some required elements are missing:", Colors.WARNING)
                for missing in missing_elements:
                    print_colored(f"   - {missing}", Colors.WARNING)
            else:
                print_colored("✅ All required MCP elements are included", Colors.OKGREEN)
            
            self.test_results["mcp_server_file"] = True
            return True
            
        except SyntaxError as e:
            print_colored(f"❌ MCP server file syntax error: {e}", Colors.FAIL)
            print_colored(f"   Line {e.lineno}: {e.text}", Colors.FAIL)
            self.test_results["mcp_server_file"] = False
            return False
        except Exception as e:
            print_colored(f"❌ MCP server file check error: {e}", Colors.FAIL)
            self.test_results["mcp_server_file"] = False
            return False
    
    def test_claude_desktop_config(self):
        """5. Generate Claude Desktop configuration file (using config.py project path)"""
        print_step(5, "Generate Claude Desktop configuration file (Auto-path from config.py)")
        
        # Determine config file path
        if os.name == 'nt':  # Windows
            config_dir = Path(os.environ.get('APPDATA', '')) / 'Claude'
        else:  # macOS/Linux
            config_dir = Path.home() / 'Library' / 'Application Support' / 'Claude'
        
        config_file = config_dir / 'claude_desktop_config.json'
        
        print_colored(f"📍 Config file path: {config_file}", Colors.OKBLUE)
        
        # Create config directory if it doesn't exist
        config_dir.mkdir(parents=True, exist_ok=True)
        print_colored(f"✅ Config directory ready: {config_dir}", Colors.OKGREEN)
        
        # Read existing config
        existing_config = {"mcpServers": {}}
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    existing_config = json.load(f)
                
                # Create mcpServers key if it doesn't exist
                if 'mcpServers' not in existing_config:
                    existing_config['mcpServers'] = {}
                
                print_colored(f"✅ Existing config loaded. Current MCP servers: {len(existing_config['mcpServers'])}", Colors.OKGREEN)
                
                # Display existing server list
                if existing_config['mcpServers']:
                    print_colored("📋 Existing MCP servers:", Colors.OKBLUE)
                    for server_name in existing_config['mcpServers'].keys():
                        print_colored(f"   • {server_name}", Colors.ENDC)
                
                # Auto backup existing config
                backup_file = config_dir / f'claude_desktop_config_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                shutil.copy2(config_file, backup_file)
                print_colored(f"📋 Auto backup created: {backup_file}", Colors.OKGREEN)
                
            except Exception as e:
                print_colored(f"⚠️ Error reading existing config: {e}", Colors.WARNING)
                print_colored("Starting with new config.", Colors.OKBLUE)
                existing_config = {"mcpServers": {}}
        else:
            print_colored("📝 Creating new config file.", Colors.OKBLUE)
        
        # Clean up any problematic existing configs
        problematic_servers = []
        for server_name, server_config in existing_config['mcpServers'].items():
            # Check for old incorrect configs
            if (server_name in ["milvus-obsidian", "obsidian-milvus", "obsidian-assistant"] and 
                (server_config.get("args", []) == ["-m", "milvus_mcp.server"] or
                 "milvus_mcp.server" in str(server_config.get("args", [])))):
                problematic_servers.append(server_name)
        
        if problematic_servers:
            print_colored("🔧 Found old/incorrect Milvus configs, removing them:", Colors.WARNING)
            for server in problematic_servers:
                print_colored(f"   🗑️ Removing: {server}", Colors.WARNING)
                del existing_config['mcpServers'][server]
        
        # Import config to get the actual project path
        try:
            import sys
            sys.path.insert(0, str(self.project_dir))
            import config
            
            # Use the project path from config.py (auto-detected)
            project_path_str = config.get_project_absolute_path()
            mcp_server_path_str = os.path.join(project_path_str, "mcp_server.py")
            
            print_colored(f"📍 Using project path from config.py: {project_path_str}", Colors.OKBLUE)
            
        except Exception as e:
            print_colored(f"⚠️ Could not import config.py: {e}", Colors.WARNING)
            print_colored("Falling back to detected project directory", Colors.WARNING)
            # Fallback to detected path
            project_path_str = str(self.project_dir)
            mcp_server_path_str = str(self.mcp_server_path)
        
        # Use the exact server name from your config
        milvus_server_name = "obsidian-assistant"
        
        milvus_config = {
            "command": "python",
            "args": [
                mcp_server_path_str  # Raw path without extra escaping
            ],
            "env": {
                "PYTHONPATH": project_path_str  # Raw path without extra escaping
            }
        }
        
        # Show what we're adding
        print_colored(f"➕ Adding '{milvus_server_name}' server with config:", Colors.OKGREEN)
        print_colored(f"   Command: {milvus_config['command']}", Colors.ENDC)
        print_colored(f"   Script: {mcp_server_path_str}", Colors.ENDC)
        print_colored(f"   Environment: PYTHONPATH = {project_path_str}", Colors.ENDC)
        print_colored(f"🎯 Project path auto-detected from config.py!", Colors.OKGREEN)
        
        existing_config['mcpServers'][milvus_server_name] = milvus_config
        
        # Save config
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(existing_config, f, indent=2, ensure_ascii=False)
            
            print_colored(f"✅ Claude Desktop configuration saved successfully!", Colors.OKGREEN)
            print_colored(f"📋 Total MCP servers: {len(existing_config['mcpServers'])}", Colors.OKGREEN)
            
            # Display the exact config that was written
            print_colored(f"\n📋 Final configuration for '{milvus_server_name}':", Colors.OKBLUE)
            config_json = json.dumps({milvus_server_name: milvus_config}, indent=2)
            print_colored(config_json, Colors.ENDC)
            
            # Verify the config was written correctly
            print_colored(f"\n🔍 Verifying configuration file...", Colors.OKBLUE)
            with open(config_file, 'r', encoding='utf-8') as f:
                verification_config = json.load(f)
            
            if milvus_server_name in verification_config.get('mcpServers', {}):
                saved_config = verification_config['mcpServers'][milvus_server_name]
                if (saved_config.get('command') == 'python' and 
                    len(saved_config.get('args', [])) > 0 and
                    'mcp_server.py' in saved_config['args'][0]):
                    print_colored("✅ Configuration verified successfully!", Colors.OKGREEN)
                    
                    # Show the exact matching config format requested
                    print_colored(f"\n🎯 EXACT CONFIG INSTALLED (using config.py path):", Colors.OKGREEN)
                    exact_config = {
                        "mcpServers": {
                            "obsidian-assistant": {
                                "command": "python",
                                "args": [
                                    mcp_server_path_str
                                ],
                                "env": {
                                    "PYTHONPATH": project_path_str
                                }
                            }
                        }
                    }
                    print_colored(json.dumps(exact_config, indent=2), Colors.ENDC)
                    print_colored(f"\n📁 Project path source: config.get_project_absolute_path()", Colors.OKBLUE)
                    print_colored(f"📋 Actual path: {project_path_str}", Colors.OKBLUE)
                    
                else:
                    print_colored("⚠️ Configuration saved but may not be correct", Colors.WARNING)
            else:
                print_colored("❌ Configuration verification failed", Colors.FAIL)
                return False
            
            # Final instructions
            print_colored("\n" + "="*60, Colors.OKGREEN)
            print_colored("🎉 SETUP COMPLETE! 🎉", Colors.OKGREEN)
            print_colored("="*60, Colors.OKGREEN)
            print_colored("📋 Next steps:", Colors.OKBLUE)
            print_colored("1. 🔄 Restart Claude Desktop application", Colors.ENDC)
            print_colored("   ⚠️ IMPORTANT: Don't just click the X button to close!", Colors.WARNING)
            print_colored("   ⚠️ Use the menu option to quit properly for complete restart.", Colors.WARNING)
            print_colored("2. 🔧 Make sure Milvus server is running", Colors.ENDC)
            print_colored("3. 💬 In Claude Desktop, you can now use:", Colors.ENDC)
            print_colored("   • search_documents()", Colors.ENDC)
            print_colored("   • get_document_content()", Colors.ENDC)
            print_colored("   • intelligent_search()", Colors.ENDC)
            print_colored("   • advanced_filter_search()", Colors.ENDC)
            print_colored("   • And many more Obsidian search tools!", Colors.ENDC)
            print_colored("="*60, Colors.OKGREEN)
            
            self.test_results["claude_desktop_config"] = True
            return True
            
        except Exception as e:
            print_colored(f"❌ Config save error: {e}", Colors.FAIL)
            print_colored("💡 Solutions to try:", Colors.OKBLUE)
            print_colored("1. Close Claude Desktop if it's running", Colors.ENDC)
            print_colored("2. Run this script as administrator (Windows)", Colors.ENDC)
            print_colored("3. Check if you have write permissions", Colors.ENDC)
            print_colored("4. Check available disk space", Colors.ENDC)
            
            # Try to create a local config file as fallback
            try:
                local_config_file = self.project_dir / 'claude_desktop_config.json'
                with open(local_config_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_config, f, indent=2, ensure_ascii=False)
                
                print_colored(f"\n💾 Fallback: Config saved to project directory:", Colors.WARNING)
                print_colored(f"   {local_config_file}", Colors.WARNING)
                print_colored(f"📋 Manually copy this file to:", Colors.OKBLUE)
                print_colored(f"   {config_file}", Colors.OKBLUE)
                
            except Exception as e2:
                print_colored(f"❌ Fallback save also failed: {e2}", Colors.FAIL)
            
            self.test_results["claude_desktop_config"] = False
            return False

def perform_complete_physical_reset():
    """Perform complete physical reset - Hard deletion of all data"""
    print_header("⚠️ COMPLETE PHYSICAL RESET - HARD DELETION ⚠️")
    print_colored("This will PERMANENTLY DELETE ALL data:", Colors.FAIL)
    print_colored("• All Milvus collections and vector data", Colors.FAIL)
    print_colored("• All embedding data", Colors.FAIL)
    print_colored("• All container data", Colors.FAIL)
    print_colored("• All persistent storage", Colors.FAIL)
    print_colored("\n⚠️ THIS ACTION CANNOT BE UNDONE! ⚠️", Colors.FAIL)
    
    # Triple confirmation
    confirm1 = input_colored("\nType 'DELETE' to confirm hard deletion: ", Colors.FAIL)
    if confirm1 != 'DELETE':
        print_colored("Reset cancelled.", Colors.OKGREEN)
        return
    
    confirm2 = input_colored("Type 'YES' to confirm you understand data will be lost: ", Colors.FAIL)
    if confirm2 != 'YES':
        print_colored("Reset cancelled.", Colors.OKGREEN)
        return
    
    confirm3 = input_colored("Type 'RESET' for final confirmation: ", Colors.FAIL)
    if confirm3 != 'RESET':
        print_colored("Reset cancelled.", Colors.OKGREEN)
        return
    
    print_colored("\n🔥 Starting complete physical reset...", Colors.WARNING)
    
    # Check if Podman is available
    podman_available, podman_path = check_podman()
    
    if podman_available:
        print_colored("🐳 Stopping and removing all Milvus containers...", Colors.WARNING)
        
        # Stop and remove containers
        containers = ["milvus-standalone", "milvus-minio", "milvus-etcd"]
        for container in containers:
            try:
                subprocess.run([podman_path, "stop", container], capture_output=True)
                subprocess.run([podman_path, "rm", "-f", container], capture_output=True)
                print_colored(f"  ✅ Container {container} removed", Colors.OKGREEN)
            except Exception as e:
                print_colored(f"  ⚠️ Error removing {container}: {e}", Colors.WARNING)
        
        # Remove network
        try:
            subprocess.run([podman_path, "network", "rm", "milvus-network"], capture_output=True)
            print_colored("  ✅ Network milvus-network removed", Colors.OKGREEN)
        except Exception as e:
            print_colored(f"  ⚠️ Error removing network: {e}", Colors.WARNING)
        
        # Remove volumes
        try:
            result = subprocess.run([podman_path, "volume", "ls", "-q"], capture_output=True, text=True)
            volumes = result.stdout.strip().split('\n')
            milvus_volumes = [v for v in volumes if 'milvus' in v.lower()]
            
            for volume in milvus_volumes:
                subprocess.run([podman_path, "volume", "rm", "-f", volume], capture_output=True)
                print_colored(f"  ✅ Volume {volume} removed", Colors.OKGREEN)
        except Exception as e:
            print_colored(f"  ⚠️ Error removing volumes: {e}", Colors.WARNING)
    
    # Delete all data directories
    print_colored("\n🗂️ Removing all data directories...", Colors.WARNING)
    
    # Get project directory
    project_dir = Path(__file__).parent.resolve()
    
    # Import config to get data paths
    try:
        import sys
        sys.path.insert(0, str(project_dir))
        import config
        data_base_path = Path(config.get_external_storage_path())
    except Exception as e:
        print_colored(f"Warning: Could not import config.py: {e}", Colors.WARNING)
        data_base_path = project_dir / "MilvusData"
    
    # Data directories to remove
    data_dirs_to_remove = [
        data_base_path,
        project_dir / "MilvusData",
        project_dir / "embedding_cache",
        project_dir / "__pycache__",
    ]
    
    for data_dir in data_dirs_to_remove:
        if data_dir.exists():
            try:
                if data_dir.is_dir():
                    shutil.rmtree(data_dir)
                    print_colored(f"  ✅ Directory removed: {data_dir}", Colors.OKGREEN)
                else:
                    data_dir.unlink()
                    print_colored(f"  ✅ File removed: {data_dir}", Colors.OKGREEN)
            except Exception as e:
                print_colored(f"  ❌ Error removing {data_dir}: {e}", Colors.FAIL)
        else:
            print_colored(f"  ⚪ Not found: {data_dir}", Colors.ENDC)
    
    # Remove any .pyc files
    print_colored("\n🧹 Cleaning Python cache files...", Colors.WARNING)
    try:
        for pyc_file in project_dir.rglob("*.pyc"):
            pyc_file.unlink()
        for pyo_file in project_dir.rglob("*.pyo"):
            pyo_file.unlink()
        print_colored("  ✅ Python cache files cleaned", Colors.OKGREEN)
    except Exception as e:
        print_colored(f"  ⚠️ Error cleaning cache: {e}", Colors.WARNING)
    
    # Remove any temporary files
    print_colored("\n🧹 Cleaning temporary files...", Colors.WARNING)
    temp_patterns = ["*.tmp", "*.temp", "*.log", "*.bak"]
    for pattern in temp_patterns:
        try:
            for temp_file in project_dir.rglob(pattern):
                temp_file.unlink()
            print_colored(f"  ✅ Cleaned {pattern} files", Colors.OKGREEN)
        except Exception as e:
            print_colored(f"  ⚠️ Error cleaning {pattern}: {e}", Colors.WARNING)
    
    print_colored("\n" + "="*60, Colors.FAIL)
    print_colored("🔥 COMPLETE PHYSICAL RESET FINISHED 🔥", Colors.FAIL)
    print_colored("="*60, Colors.FAIL)
    print_colored("✅ All containers stopped and removed", Colors.OKGREEN)
    print_colored("✅ All data directories deleted", Colors.OKGREEN)
    print_colored("✅ All cache files cleaned", Colors.OKGREEN)
    print_colored("✅ All temporary files removed", Colors.OKGREEN)
    print_colored("\n💡 System is now in clean state", Colors.OKBLUE)
    print_colored("💡 Run setup again to start fresh", Colors.OKBLUE)
    print_colored("="*60, Colors.FAIL)

def show_menu():
    """Display main menu"""
    print_header("Milvus MCP Interactive Test")
    print_colored("Select test to run:", Colors.OKBLUE)
    print_colored("1. Install and check required packages", Colors.ENDC)
    print_colored("2. Milvus connection test", Colors.ENDC)  
    print_colored("3. Collection creation and manipulation test", Colors.ENDC)
    print_colored("4. Local MCP server file test", Colors.ENDC)
    print_colored("5. Generate Claude Desktop configuration file", Colors.ENDC)
    print_colored("6. View all results", Colors.ENDC)
    print_colored("7. Run all tests automatically", Colors.ENDC)
    print_colored("8. ⚠️ Complete Physical Reset (Hard Delete All Data)", Colors.FAIL)
    print_colored("0. Exit", Colors.ENDC)

def show_results(test_results):
    """Display test results"""
    print_header("Test Results Summary")
    
    tests = [
        ("Required packages", "dependencies"),
        ("Milvus connection", "milvus_connection"),
        ("Collection operations", "collection_operations"),
        ("MCP server file", "mcp_server_file"),
        ("Claude Desktop config", "claude_desktop_config")
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_key in tests:
        result = test_results.get(test_key, None)
        if result is True:
            status = "✅ Passed"
            passed += 1
        elif result is False:
            status = "❌ Failed"
        else:
            status = "⏸️ Not run"
        
        print_colored(f"{test_name:<20} {status}", Colors.ENDC)
    
    print_colored(f"\nTotal {passed}/{total} tests passed", Colors.OKBLUE)
    
    if passed == total:
        print_colored("\n🎉 All tests successful!", Colors.OKGREEN)
        print_colored("Restart Claude Desktop and try using Milvus features!", Colors.OKGREEN)
        print_colored("⚠️ IMPORTANT: Use menu > quit, not the X button for proper restart!", Colors.WARNING)
    elif passed > 0:
        print_colored(f"\n⚠️ {total - passed} tests not yet completed.", Colors.WARNING)
    else:
        print_colored("\n❌ No tests have been run yet.", Colors.WARNING)

def run_all_tests(tester):
    """Run all tests automatically"""
    print_header("Run All Tests Automatically")
    
    tests = [
        ("1. Install required packages", tester.test_dependencies),
        ("2. Milvus connection test", tester.test_milvus_connection),
        ("3. Collection operations test", tester.test_collection_operations),
        ("4. MCP server file test", tester.test_mcp_server_file),
        ("5. Claude Desktop config", tester.test_claude_desktop_config)
    ]
    
    for test_name, test_func in tests:
        print_colored(f"\n▶️ Running {test_name}...", Colors.OKBLUE)
        test_func()
        
        # Brief wait after each test
        time.sleep(1)
    
    show_results(tester.test_results)

def main():
    """Main function"""
    tester = MilvusTest()
    
    print_header("Obsidian-Milvus FastMCP Test Tool")
    print_colored(f"📂 Project directory: {tester.project_dir}", Colors.OKBLUE)
    print_colored(f"📄 MCP server file: {tester.mcp_server_path}", Colors.OKBLUE)
    
    # Pre-check if project files exist
    required_files = ['mcp_server.py', 'config.py']
    missing_files = []
    for file in required_files:
        if not (tester.project_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print_colored(f"❌ Required files missing: {', '.join(missing_files)}", Colors.FAIL)
        print_colored(f"Check if current directory is the correct project folder.", Colors.WARNING)
        print_colored(f"Project folder should contain these files:", Colors.OKBLUE)
        for file in required_files:
            status = "✅" if file not in missing_files else "❌"
            print_colored(f"   {status} {file}", Colors.ENDC)
        print_colored("\nPlease run script from correct project folder.", Colors.WARNING)
        input_colored("\nPress Enter to continue...")
        return
    
    while True:
        show_menu()
        
        choice = input_colored("\nSelect (0-8): ")
        
        try:
            choice = int(choice)
            
            if choice == 0:
                print_colored("👋 Exiting program.", Colors.OKGREEN)
                break
            elif choice == 1:
                tester.test_dependencies()
            elif choice == 2:
                tester.test_milvus_connection()
            elif choice == 3:
                tester.test_collection_operations()
            elif choice == 4:
                tester.test_mcp_server_file()
            elif choice == 5:
                tester.test_claude_desktop_config()
            elif choice == 6:
                show_results(tester.test_results)
            elif choice == 7:
                run_all_tests(tester)
            elif choice == 8:
                perform_complete_physical_reset()
            else:
                print_colored("❌ Invalid selection. Please enter a number between 0-8.", Colors.FAIL)
        
        except ValueError:
            print_colored("❌ Please enter a number.", Colors.FAIL)
        except KeyboardInterrupt:
            print_colored("\n\n⏹️ Interrupted by user.", Colors.WARNING)
            break
        except Exception as e:
            print_colored(f"\n❌ Unexpected error: {e}", Colors.FAIL)
        
        # Wait for next selection
        input_colored("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
