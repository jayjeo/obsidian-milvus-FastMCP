#!/usr/bin/env python3
"""
인터랙티브 Milvus MCP 테스트 - 단계별 테스트 및 자동 문제 해결
사용자가 직접 각 단계를 선택하여 테스트하고, 문제 발생 시 자동으로 해결합니다.
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
    """터미널 색상 코드"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(message, color=Colors.ENDC):
    """색상이 있는 출력"""
    print(f"{color}{message}{Colors.ENDC}")

def print_header(title):
    """헤더 출력"""
    print_colored(f"\n{'='*60}", Colors.HEADER)
    print_colored(f"🔧 {title}", Colors.HEADER)
    print_colored(f"{'='*60}", Colors.HEADER)

def print_step(step_num, title):
    """단계 제목 출력"""
    print_colored(f"\n{step_num}. {title}", Colors.OKBLUE)
    print_colored("-" * 40, Colors.OKBLUE)

def input_colored(prompt, color=Colors.OKCYAN):
    """색상이 있는 입력"""
    return input(f"{color}{prompt}{Colors.ENDC}")

def install_package(package_name, import_name=None):
    """패키지 자동 설치"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print_colored(f"✅ {package_name} 이미 설치됨", Colors.OKGREEN)
        return True
    except ImportError:
        print_colored(f"⚠️ {package_name} 패키지가 없습니다. 설치를 시도합니다...", Colors.WARNING)
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print_colored(f"✅ {package_name} 설치 완료", Colors.OKGREEN)
            return True
        except subprocess.CalledProcessError as e:
            print_colored(f"❌ {package_name} 설치 실패: {e}", Colors.FAIL)
            return False

def check_milvus_server():
    """Milvus 서버 상태 확인"""
    # 여러 엔드포인트로 확인
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
    
    # TCP 포트 연결 확인
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
    """Podman 실행 파일 경로 찾기"""
    possible_paths = [
        "podman",  # PATH에 있는 경우
        "/usr/bin/podman",  # Linux 기본 경로
        "/opt/homebrew/bin/podman",  # macOS Homebrew
        "/usr/local/bin/podman",  # macOS 기타
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
    """Podman 설치 및 상태 확인"""
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
    """Podman을 사용한 Milvus 컨트롤러"""
    
    def __init__(self, podman_path):
        self.podman_path = podman_path
        self.network = "milvus-network"
        self.volumes = {
            "etcd": "milvus-etcd-vol",
            "minio": "milvus-minio-vol", 
            "milvus": "milvus-data-vol"
        }
        self.images = {
            "etcd": "quay.io/coreos/etcd:v3.5.5",
            "minio": "minio/minio:RELEASE.2023-03-20T20-16-18Z",
            "milvus": "milvusdb/milvus:v2.3.3"
        }
        self.api_port = "19530"
        self.web_port = "9091"
    
    def run_command(self, cmd):
        """명령어 실행"""
        try:
            if isinstance(cmd, str):
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            else:
                result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0, result.stdout, result.stderr
        except Exception as e:
            return False, "", str(e)
    
    def start_machine(self):
        """Podman 머신 시작 (필요시)"""
        # Windows/macOS에서는 Podman 머신이 필요할 수 있음
        if os.name == 'nt' or sys.platform == 'darwin':
            print_colored("🔧 Podman 머신을 시작합니다...", Colors.OKBLUE)
            success, _, _ = self.run_command([self.podman_path, "machine", "start"])
            if success:
                print_colored("✅ Podman 머신 시작 완료", Colors.OKGREEN)
            else:
                print_colored("⚠️ Podman 머신 시작 실패 (이미 실행 중일 수 있음)", Colors.WARNING)
            time.sleep(2)
    
    def create_network(self):
        """네트워크 생성"""
        print_colored(f"🌐 네트워크 '{self.network}' 생성 중...", Colors.OKBLUE)
        # 기존 네트워크 확인
        success, _, _ = self.run_command([self.podman_path, "network", "exists", self.network])
        if not success:
            success, _, _ = self.run_command([self.podman_path, "network", "create", self.network])
            if success:
                print_colored("✅ 네트워크 생성 완료", Colors.OKGREEN)
            else:
                print_colored("❌ 네트워크 생성 실패", Colors.FAIL)
                return False
        else:
            print_colored("✅ 네트워크가 이미 존재합니다", Colors.OKGREEN)
        return True
    
    def create_volumes(self):
        """볼륨 생성"""
        print_colored("💾 영구 볼륨들을 생성합니다...", Colors.OKBLUE)
        for name, volume in self.volumes.items():
            success, _, _ = self.run_command([self.podman_path, "volume", "exists", volume])
            if not success:
                success, _, _ = self.run_command([self.podman_path, "volume", "create", volume])
                if success:
                    print_colored(f"  ✅ 볼륨 {volume} 생성 완료", Colors.OKGREEN)
                else:
                    print_colored(f"  ❌ 볼륨 {volume} 생성 실패", Colors.FAIL)
                    return False
            else:
                print_colored(f"  ✅ 볼륨 {volume} 이미 존재", Colors.OKGREEN)
        return True
    
    def stop_containers(self):
        """기존 컨테이너 정리"""
        print_colored("🧹 기존 컨테이너들을 정리합니다...", Colors.OKBLUE)
        containers = ["milvus-standalone", "milvus-minio", "milvus-etcd"]
        for container in containers:
            self.run_command([self.podman_path, "stop", container])
            self.run_command([self.podman_path, "rm", container])
    
    def start_etcd(self):
        """etcd 컨테이너 시작"""
        print_colored("[1/3] 📊 etcd 시작 중...", Colors.OKBLUE)
        cmd = [
            self.podman_path, "run", "-d", "--name", "milvus-etcd", 
            "--network", self.network,
            "-v", f"{self.volumes['etcd']}:/etcd",
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
        """MinIO 컨테이너 시작"""
        print_colored("[2/3] 🗄️ MinIO 시작 중...", Colors.OKBLUE)
        cmd = [
            self.podman_path, "run", "-d", "--name", "milvus-minio",
            "--network", self.network,
            "-v", f"{self.volumes['minio']}:/minio_data",
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
        """Milvus 컨테이너 시작"""
        print_colored("[3/3] 🚀 Milvus 시작 중...", Colors.OKBLUE)
        cmd = [
            self.podman_path, "run", "-d", "--name", "milvus-standalone",
            "--network", self.network,
            "-v", f"{self.volumes['milvus']}:/var/lib/milvus",
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
        """컨테이너 상태 확인"""
        print_colored("\n📊 컨테이너 상태:", Colors.OKBLUE)
        success, stdout, _ = self.run_command([
            self.podman_path, "ps", 
            "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        ])
        if success:
            print_colored(stdout, Colors.ENDC)
        return success
    
    def check_container_logs(self, container_name):
        """컨테이너 로그 확인"""
        print_colored(f"📋 {container_name} 로그 확인 중...", Colors.OKBLUE)
        success, stdout, stderr = self.run_command([self.podman_path, "logs", "--tail", "20", container_name])
        if success:
            print_colored(f"📋 {container_name} 최근 로그:", Colors.OKBLUE)
            print_colored(stdout, Colors.ENDC)
            if stderr:
                print_colored("🔴 에러 로그:", Colors.WARNING)
                print_colored(stderr, Colors.ENDC)
        return success
    
    def diagnose_milvus_issues(self):
        """Milvus 문제 진단"""
        print_colored("\n🔍 Milvus 문제 진단 중...", Colors.OKBLUE)
        
        # 1. 컨테이너 상태 확인
        success, stdout, _ = self.run_command([self.podman_path, "ps", "-a", "--filter", "name=milvus"])
        if success:
            print_colored("📊 Milvus 관련 컨테이너 상태:", Colors.OKBLUE)
            print_colored(stdout, Colors.ENDC)
        
        # 2. 개별 컨테이너 로그 확인
        containers = ["milvus-etcd", "milvus-minio", "milvus-standalone"]
        for container in containers:
            # 컨테이너가 실행 중인지 확인
            success, _, _ = self.run_command([self.podman_path, "container", "exists", container])
            if success:
                self.check_container_logs(container)
                print("-" * 50)
        
        # 3. 포트 확인
        print_colored("🔌 포트 사용 상황 확인:", Colors.OKBLUE)
        try:
            import socket
            ports_to_check = [19530, 9091, 2379, 9000]
            for port in ports_to_check:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                if result == 0:
                    print_colored(f"  ✅ 포트 {port}: 열림", Colors.OKGREEN)
                else:
                    print_colored(f"  ❌ 포트 {port}: 닫힘", Colors.FAIL)
        except Exception as e:
            print_colored(f"포트 확인 오류: {e}", Colors.WARNING)
        
        # 4. 네트워크 확인
        print_colored("🌐 네트워크 상태:", Colors.OKBLUE)
        success, stdout, _ = self.run_command([self.podman_path, "network", "inspect", self.network])
        if success:
            print_colored("  ✅ 네트워크 정상", Colors.OKGREEN)
        else:
            print_colored("  ❌ 네트워크 문제", Colors.FAIL)
    
    def restart_milvus_container(self):
        """Milvus 컨테이너만 재시작"""
        print_colored("🔄 Milvus 컨테이너를 재시작합니다...", Colors.OKBLUE)
        
        # Milvus 컨테이너 정지
        self.run_command([self.podman_path, "stop", "milvus-standalone"])
        self.run_command([self.podman_path, "rm", "milvus-standalone"])
        
        # 잠시 대기
        time.sleep(5)
        
        # Milvus 재시작
        if self.start_milvus():
            print_colored("✅ Milvus 컨테이너 재시작 완료", Colors.OKGREEN)
            return True
        else:
            print_colored("❌ Milvus 컨테이너 재시작 실패", Colors.FAIL)
            return False
    
    def wait_for_milvus_ready(self, max_wait_time=180):
        """Milvus 준비 상태까지 대기 (확장된 대기 시간과 진단)"""
        print_colored(f"⏳ Milvus 서비스 준비 대기 중 (최대 {max_wait_time}초)...", Colors.WARNING)
        
        for i in range(max_wait_time):
            if check_milvus_server():
                print_colored(f"\n✅ Milvus 서버 준비 완료! ({i+1}초 소요)", Colors.OKGREEN)
                return True
            
            # 30초마다 상태 체크
            if i > 0 and i % 30 == 0:
                print_colored(f"\n⏳ {i}초 경과... 상태 확인 중", Colors.WARNING)
                self.check_status()
            
            time.sleep(1)
            if i % 10 == 0:
                print(f"\n[{i}s]", end="")
            print(".", end="", flush=True)
        
        print_colored(f"\n⚠️ {max_wait_time}초 대기 후에도 Milvus가 준비되지 않았습니다.", Colors.WARNING)
        
        # 진단 실행
        self.diagnose_milvus_issues()
        
        # 추가 대기 옵션 제공
        choice = input_colored("\n🔧 추가로 60초 더 대기하시겠습니까? (y/n): ")
        if choice.lower() == 'y':
            return self.wait_for_milvus_ready(60)
        
        return False
    
    def start_all(self):
        """전체 Milvus 스택 시작"""
        print_colored("="*60, Colors.HEADER)
        print_colored("         Milvus with Podman 시작", Colors.HEADER)
        print_colored("="*60, Colors.HEADER)
        
        # Podman 머신 시작 (필요시)
        self.start_machine()
        
        # 인프라 설정
        self.stop_containers()
        
        if not self.create_network():
            return False
        
        if not self.create_volumes():
            return False
        
        # 서비스 시작
        if not self.start_etcd():
            return False
        
        if not self.start_minio():
            return False
        
        print_colored("⏳ 의존성 서비스 준비 대기 중...", Colors.WARNING)
        time.sleep(15)
        
        if not self.start_milvus():
            return False
        
        print_colored("\n⏳ 서비스 준비 완료 대기 중...", Colors.WARNING)
        time.sleep(20)
        
        # 최종 상태 확인
        self.check_status()
        
        print_colored("\n" + "="*60, Colors.OKGREEN)
        print_colored("                    🎉 성공! 🎉", Colors.OKGREEN)
        print_colored("="*60, Colors.OKGREEN)
        print_colored(f"🌐 Milvus API:    http://localhost:{self.api_port}", Colors.OKGREEN)
        print_colored(f"🌐 웹 인터페이스: http://localhost:{self.web_port}", Colors.OKGREEN)
        print_colored("💾 데이터는 재시작 후에도 유지됩니다", Colors.OKGREEN)
        print_colored("="*60, Colors.OKGREEN)
        
        return True

def start_milvus_server():
    """Milvus 서버 시작 시도 (Podman 사용)"""
    print_colored("🚀 Milvus 서버를 시작하려고 시도합니다...", Colors.WARNING)
    
    # Podman 확인
    podman_available, podman_path = check_podman()
    
    if podman_available:
        print_colored(f"📦 Podman을 사용하여 Milvus를 시작합니다...", Colors.OKBLUE)
        print_colored(f"   Podman 경로: {podman_path}", Colors.ENDC)
        
        try:
            controller = MilvusPodmanController(podman_path)
            
            if controller.start_all():
                # 개선된 서버 시작 대기 및 확인
                if controller.wait_for_milvus_ready():
                    print_colored("✅ Milvus 서버가 완전히 준비되었습니다!", Colors.OKGREEN)
                    return True
                else:
                    print_colored("⚠️ Milvus 컨테이너는 실행 중이지만 서비스가 완전히 준비되지 않았습니다.", Colors.WARNING)
                    
                    # 재시작 옵션 제공
                    choice = input_colored("🔄 Milvus 컨테이너를 재시작해보시겠습니까? (y/n): ")
                    if choice.lower() == 'y':
                        if controller.restart_milvus_container():
                            return controller.wait_for_milvus_ready(120)  # 2분 추가 대기
                    
                    print_colored("💡 수동 확인 방법:", Colors.OKBLUE)
                    print_colored("1. 컨테이너 로그 확인: podman logs milvus-standalone", Colors.ENDC)
                    print_colored("2. 포트 확인: netstat -an | grep 19530", Colors.ENDC)
                    print_colored("3. 웹 인터페이스 확인: http://localhost:9091", Colors.ENDC)
                    print_colored("4. 시간이 지난 후 다시 연결 테스트 시도", Colors.ENDC)
                    
                    return False
            else:
                return False
            
        except Exception as e:
            print_colored(f"❌ Podman을 사용한 Milvus 시작 실패: {e}", Colors.FAIL)
            return False
    else:
        print_colored("❌ Podman을 찾을 수 없습니다.", Colors.FAIL)
        print_colored("💡 해결 방법:", Colors.OKBLUE)
        print_colored("1. Podman을 설치하세요:", Colors.ENDC)
        print_colored("   - Windows: https://github.com/containers/podman/blob/main/docs/tutorials/podman-for-windows.md", Colors.ENDC)
        print_colored("   - macOS: brew install podman", Colors.ENDC)
        print_colored("   - Linux: 배포판별 패키지 매니저 사용", Colors.ENDC)
        print_colored("2. 또는 Milvus를 직접 설치하세요: https://milvus.io/docs/install_standalone-docker.md", Colors.ENDC)
        return False

class MilvusTest:
    def __init__(self):
        self.test_results = {}
        # 현재 프로젝트 디렉토리 확인
        self.project_dir = Path(__file__).parent.resolve()
        self.mcp_server_path = self.project_dir / "mcp_server.py"
    
    def test_dependencies(self):
        """1. 필수 패키지 설치 및 확인"""
        print_step(1, "필수 패키지 설치 및 확인")
        
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
            print_colored("✅ 모든 필수 패키지가 설치되었습니다!", Colors.OKGREEN)
        else:
            print_colored("❌ 일부 패키지 설치에 실패했습니다.", Colors.FAIL)
            print_colored("💡 수동으로 설치해보세요: pip install mcp pymilvus requests numpy", Colors.OKBLUE)
        
        self.test_results["dependencies"] = all_installed
        return all_installed
    
    def test_milvus_connection(self):
        """2. Milvus 연결 테스트"""
        print_step(2, "Milvus 연결 테스트")
        
        # 먼저 서버 상태 확인
        if not check_milvus_server():
            print_colored("❌ Milvus 서버에 연결할 수 없습니다.", Colors.FAIL)
            
            choice = input_colored("🔧 Milvus 서버를 자동으로 시작하시겠습니까? (y/n): ")
            if choice.lower() == 'y':
                if start_milvus_server():
                    print_colored("✅ Milvus 서버가 시작되었습니다!", Colors.OKGREEN)
                else:
                    print_colored("❌ Milvus 서버 시작에 실패했습니다.", Colors.FAIL)
                    self.test_results["milvus_connection"] = False
                    return False
            else:
                print_colored("💡 Milvus 서버를 수동으로 시작해주세요.", Colors.OKBLUE)
                self.test_results["milvus_connection"] = False
                return False
        
        try:
            from pymilvus import connections, utility
            
            # Milvus 서버 연결
            connections.connect(
                alias="default",
                host='localhost',
                port='19530'
            )
            
            if connections.has_connection("default"):
                print_colored("✅ Milvus 연결 성공!", Colors.OKGREEN)
                
                # 서버 정보 출력
                try:
                    print_colored(f"📊 Milvus 서버 정보:", Colors.OKBLUE)
                    collections = utility.list_collections()
                    print_colored(f"   기존 컬렉션 수: {len(collections)}", Colors.ENDC)
                    if collections:
                        for col in collections:
                            print_colored(f"   - {col}", Colors.ENDC)
                except:
                    pass
                
                self.test_results["milvus_connection"] = True
                return True
            else:
                print_colored("❌ Milvus 연결 실패", Colors.FAIL)
                self.test_results["milvus_connection"] = False
                return False
                
        except Exception as e:
            print_colored(f"❌ Milvus 연결 오류: {e}", Colors.FAIL)
            print_colored("💡 해결 방법:", Colors.OKBLUE)
            print_colored("1. Milvus 서버가 실행 중인지 확인하세요", Colors.ENDC)
            print_colored("2. 포트 19530이 사용 가능한지 확인하세요", Colors.ENDC)
            print_colored("3. 방화벽 설정을 확인하세요", Colors.ENDC)
            
            self.test_results["milvus_connection"] = False
            return False
    
    def test_collection_operations(self):
        """3. 컬렉션 생성 및 조작 테스트"""
        print_step(3, "컬렉션 생성 및 조작 테스트")
        
        if not self.test_results.get("milvus_connection", False):
            print_colored("⚠️ Milvus 연결이 필요합니다. 먼저 2번 테스트를 실행하세요.", Colors.WARNING)
            return False
        
        try:
            from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility
            import numpy as np
            
            collection_name = "test_obsidian_notes"
            
            # 기존 컬렉션이 있다면 삭제
            if utility.has_collection(collection_name):
                choice = input_colored(f"🗑️ 기존 테스트 컬렉션 '{collection_name}'을 삭제하시겠습니까? (y/n): ")
                if choice.lower() == 'y':
                    utility.drop_collection(collection_name)
                    print_colored(f"✅ 기존 컬렉션 삭제 완료", Colors.OKGREEN)
                else:
                    collection_name = f"test_obsidian_notes_{int(time.time())}"
                    print_colored(f"📝 새로운 컬렉션 이름 사용: {collection_name}", Colors.OKBLUE)
            
            # 필드 스키마 정의
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
            ]
            
            # 컬렉션 스키마 생성
            schema = CollectionSchema(fields, f"Test collection for Obsidian notes")
            
            # 컬렉션 생성
            collection = Collection(collection_name, schema)
            print_colored(f"✅ 컬렉션 '{collection_name}' 생성 성공", Colors.OKGREEN)
            
            # 테스트 데이터 생성
            test_data = [
                ["test.md", "# 테스트 문서\n\n이것은 테스트 문서입니다.", np.random.rand(384).tolist()],
                ["example.md", "# 예제 문서\n\n예제 내용입니다.", np.random.rand(384).tolist()],
                ["sample.md", "# 샘플 노트\n\n샘플 내용입니다.", np.random.rand(384).tolist()]
            ]
            
            entities = [
                [item[0] for item in test_data],  # file_path
                [item[1] for item in test_data],  # content  
                [item[2] for item in test_data]   # embedding
            ]
            
            # 데이터 삽입
            insert_result = collection.insert(entities)
            print_colored(f"✅ 테스트 데이터 삽입 성공: {len(insert_result.primary_keys)}개 항목", Colors.OKGREEN)
            
            # 인덱스 생성
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index("embedding", index_params)
            print_colored("✅ 벡터 인덱스 생성 성공", Colors.OKGREEN)
            
            # 컬렉션 로드
            collection.load()
            print_colored("✅ 컬렉션 메모리 로드 성공", Colors.OKGREEN)
            
            # 검색 테스트
            search_vectors = [np.random.rand(384).tolist()]
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            
            results = collection.search(
                search_vectors,
                "embedding",
                search_params,
                limit=3,
                output_fields=["file_path", "content"]
            )
            
            print_colored(f"✅ 검색 테스트 성공: {len(results[0])}개 결과", Colors.OKGREEN)
            for i, hit in enumerate(results[0]):
                print_colored(f"   {i+1}. {hit.entity.get('file_path')}: 거리 {hit.distance:.4f}", Colors.ENDC)
            
            self.test_results["collection_operations"] = True
            return True
            
        except Exception as e:
            print_colored(f"❌ 컬렉션 조작 오류: {e}", Colors.FAIL)
            print_colored("💡 해결 방법:", Colors.OKBLUE)
            print_colored("1. Milvus 서버 메모리가 충분한지 확인하세요", Colors.ENDC)
            print_colored("2. 컬렉션 이름이 유효한지 확인하세요", Colors.ENDC)
            print_colored("3. 데이터 타입과 스키마가 올바른지 확인하세요", Colors.ENDC)
            
            self.test_results["collection_operations"] = False
            return False
    
    def test_mcp_server_file(self):
        """4. 로컬 MCP 서버 파일 테스트"""
        print_step(4, "로컬 MCP 서버 파일 테스트")
        
        # 로컬 MCP 서버 파일 확인
        if not self.mcp_server_path.exists():
            print_colored(f"❌ MCP 서버 파일을 찾을 수 없습니다: {self.mcp_server_path}", Colors.FAIL)
            print_colored("💡 mcp_server.py 파일이 프로젝트 디렉토리에 있는지 확인하세요.", Colors.OKBLUE)
            self.test_results["mcp_server_file"] = False
            return False
        
        print_colored(f"✅ MCP 서버 파일 발견: {self.mcp_server_path}", Colors.OKGREEN)
        
        # 파일 구문 검사
        try:
            with open(self.mcp_server_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 기본 구문 검사
            compile(content, str(self.mcp_server_path), 'exec')
            print_colored("✅ MCP 서버 파일 구문 검사 통과", Colors.OKGREEN)
            
            # 중요한 임포트와 함수 확인
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
                print_colored("⚠️ 일부 필수 요소가 누락되었습니다:", Colors.WARNING)
                for missing in missing_elements:
                    print_colored(f"   - {missing}", Colors.WARNING)
            else:
                print_colored("✅ 모든 필수 MCP 요소가 포함되어 있습니다", Colors.OKGREEN)
            
            self.test_results["mcp_server_file"] = True
            return True
            
        except SyntaxError as e:
            print_colored(f"❌ MCP 서버 파일 구문 오류: {e}", Colors.FAIL)
            print_colored(f"   라인 {e.lineno}: {e.text}", Colors.FAIL)
            self.test_results["mcp_server_file"] = False
            return False
        except Exception as e:
            print_colored(f"❌ MCP 서버 파일 확인 오류: {e}", Colors.FAIL)
            self.test_results["mcp_server_file"] = False
            return False
    
    def test_claude_desktop_config(self):
        """5. Claude Desktop 설정 파일 생성 (로컬 MCP 서버 사용)"""
        print_step(5, "Claude Desktop 설정 파일 생성")
        
        # MCP 서버 파일이 존재하는지 확인
        if not self.test_results.get("mcp_server_file", False):
            print_colored("⚠️ 먼저 4번 테스트(MCP 서버 파일 테스트)를 실행하세요.", Colors.WARNING)
            return False
        
        # 설정 파일 경로 결정
        if os.name == 'nt':  # Windows
            config_dir = Path(os.environ.get('APPDATA', '')) / 'Claude'
        else:  # macOS/Linux
            config_dir = Path.home() / 'Library' / 'Application Support' / 'Claude'
        
        config_file = config_dir / 'claude_desktop_config.json'
        
        print_colored(f"📍 설정 파일 경로: {config_file}", Colors.OKBLUE)
        
        # 설정 디렉토리가 없으면 생성
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # 기존 설정 읽기
        existing_config = {"mcpServers": {}}
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    existing_config = json.load(f)
                
                # mcpServers 키가 없으면 생성
                if 'mcpServers' not in existing_config:
                    existing_config['mcpServers'] = {}
                
                print_colored(f"✅ 기존 설정 로드. 현재 MCP 서버: {len(existing_config['mcpServers'])}개", Colors.OKGREEN)
                
                # 기존 서버 목록 출력
                if existing_config['mcpServers']:
                    print_colored("📋 보존되는 기존 MCP 서버:", Colors.OKBLUE)
                    for server_name in existing_config['mcpServers'].keys():
                        print_colored(f"   • {server_name}", Colors.ENDC)
                
                # 백업 생성 확인
                choice = input_colored("💾 기존 설정을 백업하시겠습니까? (y/n): ")
                if choice.lower() == 'y':
                    backup_file = config_dir / f'claude_desktop_config_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                    shutil.copy2(config_file, backup_file)
                    print_colored(f"📋 백업 생성: {backup_file}", Colors.OKGREEN)
                
            except Exception as e:
                print_colored(f"⚠️ 기존 설정 읽기 오류: {e}", Colors.WARNING)
                print_colored("새로운 설정으로 시작합니다.", Colors.OKBLUE)
                existing_config = {"mcpServers": {}}
        else:
            print_colored("📝 새로운 설정 파일을 생성합니다.", Colors.OKBLUE)
        
        # 잘못된 기존 설정 제거 확인
        problematic_servers = []
        for server_name, server_config in existing_config['mcpServers'].items():
            if server_name == "milvus-obsidian" and server_config.get("args", []) == ["-m", "milvus_mcp.server"]:
                problematic_servers.append(server_name)
        
        if problematic_servers:
            print_colored("🔧 잘못된 기존 Milvus 설정을 발견했습니다:", Colors.WARNING)
            for server in problematic_servers:
                print_colored(f"   - {server}: python -m milvus_mcp.server (잘못된 모듈 경로)", Colors.WARNING)
            
            choice = input_colored("🗑️ 이 설정들을 제거하고 새로운 설정으로 교체하시겠습니까? (y/n): ")
            if choice.lower() == 'y':
                for server in problematic_servers:
                    del existing_config['mcpServers'][server]
                    print_colored(f"🗑️ 제거됨: {server}", Colors.OKGREEN)
        
        # 올바른 Milvus MCP 서버 설정
        milvus_server_name = "obsidian-milvus"
        
        # Windows 경로를 JSON에서 사용할 수 있도록 이스케이프 처리
        escaped_mcp_path = str(self.mcp_server_path).replace('\\', '\\\\')
        escaped_project_path = str(self.project_dir).replace('\\', '\\\\')
        
        milvus_config = {
            "command": "python",
            "args": [escaped_mcp_path],
            "env": {
                "PYTHONPATH": escaped_project_path,
                "MILVUS_HOST": "localhost",
                "MILVUS_PORT": "19530",
                "LOG_LEVEL": "INFO"
            }
        }
        
        # 설정 옵션 제공
        if milvus_server_name in existing_config['mcpServers']:
            print_colored(f"⚠️ '{milvus_server_name}' 서버가 이미 존재합니다.", Colors.WARNING)
            choice = input_colored("🔄 기존 설정을 업데이트하시겠습니까? (y/n): ")
            if choice.lower() != 'y':
                print_colored("⏭️ Milvus 서버 설정을 건너뜁니다.", Colors.WARNING)
                self.test_results["claude_desktop_config"] = True
                return True
            print_colored(f"🔄 '{milvus_server_name}' 서버 설정 업데이트", Colors.OKGREEN)
        else:
            print_colored(f"➕ '{milvus_server_name}' 서버 추가", Colors.OKGREEN)
        
        existing_config['mcpServers'][milvus_server_name] = milvus_config
        
        # 설정 저장
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(existing_config, f, indent=2, ensure_ascii=False)
            
            print_colored(f"✅ Claude Desktop 설정 완료!", Colors.OKGREEN)
            print_colored(f"📋 총 MCP 서버: {len(existing_config['mcpServers'])}개", Colors.OKGREEN)
            
            # 최종 서버 목록
            print_colored("\n📋 설정된 모든 MCP 서버:", Colors.OKBLUE)
            for server_name in existing_config['mcpServers'].keys():
                marker = " [새로 추가/업데이트]" if server_name == milvus_server_name else ""
                print_colored(f"   • {server_name}{marker}", Colors.ENDC)
            
            # 설정 세부 정보 표시
            print_colored(f"\n🔧 '{milvus_server_name}' 서버 설정:", Colors.OKBLUE)
            print_colored(f"   명령어: python", Colors.ENDC)
            print_colored(f"   스크립트: {self.mcp_server_path}", Colors.ENDC)
            print_colored(f"   프로젝트 경로: {self.project_dir}", Colors.ENDC)
            print_colored(f"   Milvus 호스트: localhost:19530", Colors.ENDC)
            
            print_colored("\n🎉 Claude Desktop을 재시작하여 변경사항을 적용하세요!", Colors.OKGREEN)
            
            self.test_results["claude_desktop_config"] = True
            return True
            
        except Exception as e:
            print_colored(f"❌ 설정 저장 오류: {e}", Colors.FAIL)
            print_colored("💡 해결 방법:", Colors.OKBLUE)
            print_colored("1. Claude Desktop이 실행 중이 아닌지 확인하세요", Colors.ENDC)
            print_colored("2. 설정 디렉토리에 쓰기 권한이 있는지 확인하세요", Colors.ENDC)
            print_colored("3. 디스크 공간이 충분한지 확인하세요", Colors.ENDC)
            
            self.test_results["claude_desktop_config"] = False
            return False

def show_menu():
    """메인 메뉴 표시"""
    print_header("Milvus MCP 인터랙티브 테스트")
    print_colored("다음 중 실행할 테스트를 선택하세요:", Colors.OKBLUE)
    print_colored("1. 필수 패키지 설치 및 확인", Colors.ENDC)
    print_colored("2. Milvus 연결 테스트", Colors.ENDC)  
    print_colored("3. 컬렉션 생성 및 조작 테스트", Colors.ENDC)
    print_colored("4. 로컬 MCP 서버 파일 테스트", Colors.ENDC)
    print_colored("5. Claude Desktop 설정 파일 생성", Colors.ENDC)
    print_colored("6. 전체 결과 보기", Colors.ENDC)
    print_colored("7. 전체 테스트 자동 실행", Colors.ENDC)
    print_colored("0. 종료", Colors.ENDC)

def show_results(test_results):
    """테스트 결과 표시"""
    print_header("테스트 결과 요약")
    
    tests = [
        ("필수 패키지", "dependencies"),
        ("Milvus 연결", "milvus_connection"),
        ("컬렉션 조작", "collection_operations"),
        ("MCP 서버 파일", "mcp_server_file"),
        ("Claude Desktop 설정", "claude_desktop_config")
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_key in tests:
        result = test_results.get(test_key, None)
        if result is True:
            status = "✅ 통과"
            passed += 1
        elif result is False:
            status = "❌ 실패"
        else:
            status = "⏸️ 미실행"
        
        print_colored(f"{test_name:<20} {status}", Colors.ENDC)
    
    print_colored(f"\n총 {passed}/{total}개 테스트 통과", Colors.OKBLUE)
    
    if passed == total:
        print_colored("\n🎉 모든 테스트가 성공했습니다!", Colors.OKGREEN)
        print_colored("Claude Desktop을 재시작하고 Milvus 기능을 사용해보세요!", Colors.OKGREEN)
    elif passed > 0:
        print_colored(f"\n⚠️ {total - passed}개 테스트가 아직 완료되지 않았습니다.", Colors.WARNING)
    else:
        print_colored("\n❌ 아직 실행된 테스트가 없습니다.", Colors.WARNING)

def run_all_tests(tester):
    """전체 테스트 자동 실행"""
    print_header("전체 테스트 자동 실행")
    
    tests = [
        ("1. 필수 패키지 설치", tester.test_dependencies),
        ("2. Milvus 연결 테스트", tester.test_milvus_connection),
        ("3. 컬렉션 조작 테스트", tester.test_collection_operations),
        ("4. MCP 서버 파일 테스트", tester.test_mcp_server_file),
        ("5. Claude Desktop 설정", tester.test_claude_desktop_config)
    ]
    
    for test_name, test_func in tests:
        print_colored(f"\n▶️ {test_name} 실행 중...", Colors.OKBLUE)
        test_func()
        
        # 각 테스트 후 잠시 대기
        time.sleep(1)
    
    show_results(tester.test_results)

def main():
    """메인 함수"""
    tester = MilvusTest()
    
    print_header("Obsidian-Milvus FastMCP 테스트 도구")
    print_colored(f"📂 프로젝트 디렉토리: {tester.project_dir}", Colors.OKBLUE)
    print_colored(f"📄 MCP 서버 파일: {tester.mcp_server_path}", Colors.OKBLUE)
    
    # 프로젝트 파일들이 존재하는지 미리 확인
    required_files = ['mcp_server.py', 'config.py']
    missing_files = []
    for file in required_files:
        if not (tester.project_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print_colored(f"❌ 필수 파일이 누락되었습니다: {', '.join(missing_files)}", Colors.FAIL)
        print_colored(f"현재 디렉토리가 올바른 프로젝트 폴더인지 확인해주세요.", Colors.WARNING)
        print_colored(f"프로젝트 폴더에는 다음 파일들이 있어야 합니다:", Colors.OKBLUE)
        for file in required_files:
            status = "✅" if file not in missing_files else "❌"
            print_colored(f"   {status} {file}", Colors.ENDC)
        print_colored("\n올바른 프로젝트 폴더에서 스크립트를 실행해주세요.", Colors.WARNING)
        input_colored("\n계속하려면 Enter를 누르세요...")
        return
    
    while True:
        show_menu()
        
        choice = input_colored("\n선택하세요 (0-7): ")
        
        try:
            choice = int(choice)
            
            if choice == 0:
                print_colored("👋 프로그램을 종료합니다.", Colors.OKGREEN)
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
            else:
                print_colored("❌ 잘못된 선택입니다. 0-7 사이의 숫자를 입력하세요.", Colors.FAIL)
        
        except ValueError:
            print_colored("❌ 숫자를 입력해주세요.", Colors.FAIL)
        except KeyboardInterrupt:
            print_colored("\n\n⏹️ 사용자에 의해 중단되었습니다.", Colors.WARNING)
            break
        except Exception as e:
            print_colored(f"\n❌ 예상치 못한 오류: {e}", Colors.FAIL)
        
        # 다음 선택을 위한 대기
        input_colored("\n계속하려면 Enter를 누르세요...")

if __name__ == "__main__":
    main()