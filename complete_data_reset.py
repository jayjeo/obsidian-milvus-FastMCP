#!/usr/bin/env python3
"""
완전한 Milvus 데이터 리셋 스크립트
- Milvus 컨테이너 중지
- MilvusData 폴더 완전 삭제
- 컨테이너 재시작
- 새로운 컬렉션 생성
"""

import sys
import os
import shutil
import subprocess
import time
from pathlib import Path

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from colorama import Fore, Style, init
from pymilvus import connections, utility

# colorama 초기화
init()

def print_colored(message, color=Fore.WHITE):
    """컬러 출력"""
    print(f"{color}{message}{Style.RESET_ALL}")

def get_podman_path():
    """Podman 경로 가져오기"""
    try:
        return config.get_podman_path()
    except:
        # 기본 경로들 시도
        possible_paths = [
            "podman",
            "C:\\Program Files\\RedHat\\Podman\\podman.exe"
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run([path, "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    return path
            except:
                continue
        
        raise FileNotFoundError("Podman을 찾을 수 없습니다!")

def stop_milvus_containers():
    """Milvus 컨테이너들 중지"""
    print_colored("🛑 Milvus 컨테이너들 중지 중...", Fore.YELLOW)
    
    podman_path = get_podman_path()
    containers = ["milvus-standalone", "milvus-minio", "milvus-etcd"]
    
    for container in containers:
        try:
            print_colored(f"  - {container} 중지 중...", Fore.BLUE)
            subprocess.run([podman_path, "stop", container], 
                         capture_output=True, text=True, timeout=30)
            print_colored(f"  ✅ {container} 중지됨", Fore.GREEN)
        except Exception as e:
            print_colored(f"  ⚠️ {container} 중지 실패 (무시): {e}", Fore.YELLOW)
    
    time.sleep(5)  # 완전 중지 대기

def delete_milvus_data():
    """config.py의 EXTERNAL_STORAGE_PATH에 설정된 폴더 완전 삭제"""
    print_colored("🗑️ Milvus 데이터 폴더 완전 삭제 중...", Fore.RED)
    
    # config.py에서 설정된 경로 사용
    milvus_data_path = Path(config.get_external_storage_path())
    
    print_colored(f"📂 삭제 대상: {milvus_data_path}", Fore.WHITE)
    print_colored(f"📋 설정 출처: config.EXTERNAL_STORAGE_PATH", Fore.BLUE)
    
    if milvus_data_path.exists():
        # 폴더 크기 확인
        try:
            total_size = sum(f.stat().st_size for f in milvus_data_path.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            print_colored(f"📊 삭제될 데이터 크기: {size_mb:.1f}MB", Fore.YELLOW)
        except:
            print_colored("📊 데이터 크기를 계산할 수 없습니다.", Fore.YELLOW)
        
        # 완전 삭제
        try:
            shutil.rmtree(milvus_data_path)
            print_colored(f"✅ {milvus_data_path.name} 폴더 완전 삭제 완료!", Fore.GREEN)
            return True
        except Exception as e:
            print_colored(f"❌ {milvus_data_path.name} 폴더 삭제 실패: {e}", Fore.RED)
            return False
    else:
        print_colored(f"⚠️ {milvus_data_path.name} 폴더가 존재하지 않습니다.", Fore.YELLOW)
        return True

def recreate_milvus_data_folders():
    """config.py의 EXTERNAL_STORAGE_PATH에 설정된 폴더 재생성"""
    print_colored("📁 Milvus 데이터 폴더 재생성 중...", Fore.BLUE)
    
    # config.py에서 설정된 경로 사용
    milvus_data_path = Path(config.get_external_storage_path())
    
    print_colored(f"📂 생성 대상: {milvus_data_path}", Fore.WHITE)
    
    # 베이스 폴더 생성
    milvus_data_path.mkdir(parents=True, exist_ok=True)
    
    # 하위 폴더들 생성
    subdirs = ["etcd", "minio", "milvus"]
    for subdir in subdirs:
        subdir_path = milvus_data_path / subdir
        subdir_path.mkdir(exist_ok=True)
        print_colored(f"  ✅ {subdir} 폴더 생성", Fore.GREEN)
    
    print_colored(f"✅ {milvus_data_path.name} 폴더 재생성 완료!", Fore.GREEN)

def start_milvus_containers():
    """Milvus 컨테이너들 재시작"""
    print_colored("🚀 Milvus 컨테이너들 재시작 중...", Fore.BLUE)
    
    try:
        podman_path = get_podman_path()
        project_dir = Path(__file__).parent.resolve()
        compose_file = project_dir / "milvus-podman-compose.yml"
        
        if not compose_file.exists():
            print_colored("❌ milvus-podman-compose.yml 파일을 찾을 수 없습니다!", Fore.RED)
            return False
        
        # Podman compose로 재시작
        result = subprocess.run([
            podman_path, "compose", "-f", str(compose_file), "up", "-d"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print_colored("✅ Milvus 컨테이너들 재시작 완료!", Fore.GREEN)
            return True
        else:
            print_colored(f"❌ 컨테이너 재시작 실패: {result.stderr}", Fore.RED)
            return False
            
    except Exception as e:
        print_colored(f"❌ 컨테이너 재시작 중 오류: {e}", Fore.RED)
        return False

def wait_for_milvus_ready():
    """Milvus 서비스 준비 대기"""
    print_colored("⏳ Milvus 서비스 준비 대기 중...", Fore.YELLOW)
    
    import socket
    max_wait = 60
    for i in range(max_wait):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', 19530))
            sock.close()
            
            if result == 0:
                print_colored(f"✅ Milvus 서비스 준비 완료! ({i+1}초)", Fore.GREEN)
                time.sleep(5)  # 추가 안정화 시간
                return True
        except:
            pass
        
        if i % 10 == 0:
            print_colored(f"  - 대기 중... ({i}/{max_wait}초)", Fore.BLUE)
        
        time.sleep(1)
    
    print_colored("⚠️ Milvus 서비스 준비 시간이 초과되었습니다.", Fore.YELLOW)
    return False

def create_new_collection():
    """새로운 컬렉션 생성"""
    print_colored("🔧 새로운 컬렉션 생성 중...", Fore.BLUE)
    
    try:
        # Milvus 연결
        connections.connect(
            alias="default",
            host=config.MILVUS_HOST,
            port=config.MILVUS_PORT
        )
        
        # MilvusManager로 새 컬렉션 생성
        from milvus_manager import MilvusManager
        milvus_manager = MilvusManager()
        
        print_colored("✅ 새로운 컬렉션 생성 완료!", Fore.GREEN)
        
        # 확인
        total_entities = milvus_manager.count_entities()
        print_colored(f"📊 새 컬렉션에는 {total_entities}개 문서가 있습니다.", Fore.GREEN)
        
        return True
        
    except Exception as e:
        print_colored(f"❌ 새 컬렉션 생성 중 오류: {e}", Fore.RED)
        return False
    finally:
        try:
            connections.disconnect("default")
        except:
            pass

def main():
    """메인 함수"""
    print_colored("🔥 완전한 Milvus 데이터 리셋", Fore.CYAN)
    print_colored("=" * 60, Fore.CYAN)
    
    # 현재 설정 정보
    print_colored("📋 현재 설정:", Fore.BLUE)
    print_colored(f"  호스트: {config.MILVUS_HOST}:{config.MILVUS_PORT}", Fore.WHITE)
    print_colored(f"  컬렉션: {config.COLLECTION_NAME}", Fore.WHITE)
    
    # config.py에서 설정된 경로 사용
    milvus_data_path = Path(config.get_external_storage_path())
    print_colored(f"  데이터 경로: {milvus_data_path}", Fore.WHITE)
    print_colored(f"  설정 출처: config.EXTERNAL_STORAGE_PATH", Fore.BLUE)
    
    # 현재 데이터 폴더 크기 확인
    if milvus_data_path.exists():
        try:
            total_size = sum(f.stat().st_size for f in milvus_data_path.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            print_colored(f"📊 현재 {milvus_data_path.name} 크기: {size_mb:.1f}MB", Fore.YELLOW)
        except:
            print_colored("📊 현재 데이터 크기를 계산할 수 없습니다.", Fore.YELLOW)
    else:
        print_colored(f"📊 {milvus_data_path.name} 폴더가 존재하지 않습니다.", Fore.YELLOW)
    
    # 사용자 확인
    print_colored(f"\n🔥 모든 Milvus 데이터를 완전히 삭제하고 초기화하시겠습니까?", Fore.YELLOW)
    print_colored("⚠️  이 작업은 되돌릴 수 없습니다!", Fore.RED)
    choice = input("⚠️  모든 embedding 데이터가 영구적으로 삭제됩니다! (y/N): ")
    
    if choice.lower() != 'y':
        print_colored("❌ 작업이 취소되었습니다.", Fore.RED)
        return
    
    print_colored("\n" + "=" * 60, Fore.CYAN)
    
    # 1단계: 컨테이너 중지
    print_colored("1️⃣ Milvus 컨테이너 중지", Fore.CYAN)
    stop_milvus_containers()
    
    # 2단계: 데이터 완전 삭제
    print_colored(f"\n2️⃣ {milvus_data_path.name} 폴더 완전 삭제", Fore.CYAN)
    if not delete_milvus_data():
        print_colored("❌ 데이터 삭제 실패", Fore.RED)
        return
    
    # 3단계: 폴더 재생성
    print_colored(f"\n3️⃣ {milvus_data_path.name} 폴더 재생성", Fore.CYAN)
    recreate_milvus_data_folders()
    
    # 4단계: 컨테이너 재시작
    print_colored("\n4️⃣ Milvus 컨테이너 재시작", Fore.CYAN)
    if not start_milvus_containers():
        print_colored("❌ 컨테이너 재시작 실패", Fore.RED)
        return
    
    # 5단계: 서비스 준비 대기
    print_colored("\n5️⃣ Milvus 서비스 준비 대기", Fore.CYAN)
    if not wait_for_milvus_ready():
        print_colored("⚠️ 서비스 준비 시간 초과, 수동으로 확인해주세요.", Fore.YELLOW)
    
    # 6단계: 새 컬렉션 생성
    print_colored("\n6️⃣ 새로운 컬렉션 생성", Fore.CYAN)
    if create_new_collection():
        print_colored("6️⃣ ✅ 새로운 컬렉션 생성 완료!", Fore.GREEN)
    else:
        print_colored("6️⃣ ❌ 새 컬렉션 생성 실패", Fore.RED)
    
    # 완료
    print_colored("\n" + "=" * 60, Fore.GREEN)
    print_colored("🎉 완전한 데이터 리셋 완료!", Fore.GREEN)
    
    # 최종 크기 확인
    if milvus_data_path.exists():
        try:
            total_size = sum(f.stat().st_size for f in milvus_data_path.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            print_colored(f"📊 리셋 후 {milvus_data_path.name} 크기: {size_mb:.1f}MB", Fore.GREEN)
        except:
            print_colored("📊 리셋 후 데이터 크기를 계산할 수 없습니다.", Fore.YELLOW)
    
    print_colored("\n📝 다음 단계:", Fore.CYAN)
    print_colored("  1. python main.py 실행", Fore.WHITE)
    print_colored("  2. 2번 (Full Embedding) 선택", Fore.WHITE)
    print_colored("  3. n 입력 (컬렉션은 이미 새로 만들었음)", Fore.WHITE)
    print_colored("=" * 60, Fore.GREEN)

if __name__ == "__main__":
    main()
