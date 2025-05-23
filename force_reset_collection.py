#!/usr/bin/env python3
"""
Milvus 컬렉션 강제 완전 삭제 스크립트
"""

import sys
import os

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from colorama import Fore, Style, init
from pymilvus import connections, utility, Collection

# colorama 초기화
init()

def print_colored(message, color=Fore.WHITE):
    """컬러 출력"""
    print(f"{color}{message}{Style.RESET_ALL}")

def force_delete_collection():
    """컬렉션 강제 삭제"""
    try:
        # Milvus 연결
        print_colored("📡 Milvus 서버에 연결 중...", Fore.BLUE)
        connections.connect(
            alias="default",
            host=config.MILVUS_HOST,
            port=config.MILVUS_PORT
        )
        print_colored("✅ Milvus 연결 성공", Fore.GREEN)
        
        collection_name = config.COLLECTION_NAME
        
        # 모든 컬렉션 목록 확인
        print_colored("📋 현재 존재하는 모든 컬렉션:", Fore.CYAN)
        all_collections = utility.list_collections()
        for i, col in enumerate(all_collections, 1):
            print_colored(f"  {i}. {col}", Fore.WHITE)
        
        if not all_collections:
            print_colored("📋 존재하는 컬렉션이 없습니다.", Fore.YELLOW)
            return True
        
        # 대상 컬렉션 확인
        if collection_name in all_collections:
            print_colored(f"🎯 대상 컬렉션 '{collection_name}' 발견!", Fore.YELLOW)
            
            try:
                # 컬렉션 로드 해제 (중요!)
                print_colored(f"⏸️ 컬렉션 '{collection_name}' 언로드 중...", Fore.BLUE)
                collection = Collection(collection_name)
                collection.release()
                print_colored("✅ 컬렉션 언로드 완료", Fore.GREEN)
            except Exception as e:
                print_colored(f"⚠️ 컬렉션 언로드 중 오류 (무시): {e}", Fore.YELLOW)
            
            # 컬렉션 삭제
            print_colored(f"🗑️ 컬렉션 '{collection_name}' 삭제 중...", Fore.RED)
            utility.drop_collection(collection_name)
            print_colored(f"✅ 컬렉션 '{collection_name}' 삭제 완료!", Fore.GREEN)
            
        else:
            print_colored(f"⚠️ 컬렉션 '{collection_name}'을 찾을 수 없습니다.", Fore.YELLOW)
        
        # 삭제 후 확인
        print_colored("🔍 삭제 후 컬렉션 목록 확인:", Fore.CYAN)
        remaining_collections = utility.list_collections()
        if remaining_collections:
            for i, col in enumerate(remaining_collections, 1):
                print_colored(f"  {i}. {col}", Fore.WHITE)
        else:
            print_colored("📋 모든 컬렉션이 삭제되었습니다.", Fore.GREEN)
        
        # 컬렉션이 정말 삭제되었는지 재확인
        if collection_name not in remaining_collections:
            print_colored(f"✅ 컬렉션 '{collection_name}' 완전 삭제 확인!", Fore.GREEN)
            return True
        else:
            print_colored(f"❌ 컬렉션 '{collection_name}'이 아직 존재합니다!", Fore.RED)
            return False
            
    except Exception as e:
        print_colored(f"❌ 오류 발생: {e}", Fore.RED)
        import traceback
        print_colored(f"상세 오류:\n{traceback.format_exc()}", Fore.RED)
        return False
    
    finally:
        # 연결 해제
        try:
            connections.disconnect("default")
            print_colored("📡 Milvus 연결 해제", Fore.BLUE)
        except:
            pass

def create_new_collection():
    """새로운 컬렉션 생성"""
    try:
        print_colored("🔧 새로운 컬렉션 생성 중...", Fore.BLUE)
        
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
        import traceback
        print_colored(f"상세 오류:\n{traceback.format_exc()}", Fore.RED)
        return False

def main():
    """메인 함수"""
    print_colored("🔥 Milvus 컬렉션 강제 완전 삭제 및 재생성", Fore.CYAN)
    print_colored("=" * 60, Fore.CYAN)
    
    # 현재 설정 정보
    print_colored("📋 현재 설정:", Fore.BLUE)
    print_colored(f"  호스트: {config.MILVUS_HOST}:{config.MILVUS_PORT}", Fore.WHITE)
    print_colored(f"  컬렉션: {config.COLLECTION_NAME}", Fore.WHITE)
    print_colored(f"  Obsidian 경로: {config.OBSIDIAN_VAULT_PATH}", Fore.WHITE)
    
    # 경로 존재 확인
    if os.path.exists(config.OBSIDIAN_VAULT_PATH):
        print_colored("✅ Obsidian 볼트 경로 확인됨", Fore.GREEN)
    else:
        print_colored("❌ Obsidian 볼트 경로를 찾을 수 없습니다!", Fore.RED)
        return
    
    # 사용자 확인
    print_colored(f"\n🔥 컬렉션 '{config.COLLECTION_NAME}'을 강제로 완전히 삭제하고 재생성하시겠습니까?", Fore.YELLOW)
    choice = input("⚠️  모든 기존 데이터가 영구적으로 삭제됩니다! (y/N): ")
    
    if choice.lower() != 'y':
        print_colored("❌ 작업이 취소되었습니다.", Fore.RED)
        return
    
    print_colored("\n" + "=" * 60, Fore.CYAN)
    
    # 1단계: 강제 삭제
    print_colored("1️⃣ 컬렉션 강제 삭제 시작", Fore.CYAN)
    if force_delete_collection():
        print_colored("1️⃣ ✅ 컬렉션 강제 삭제 완료!", Fore.GREEN)
    else:
        print_colored("1️⃣ ❌ 컬렉션 삭제 실패", Fore.RED)
        return
    
    # 2단계: 새 컬렉션 생성
    print_colored("\n2️⃣ 새로운 컬렉션 생성 시작", Fore.CYAN)
    if create_new_collection():
        print_colored("2️⃣ ✅ 새로운 컬렉션 생성 완료!", Fore.GREEN)
    else:
        print_colored("2️⃣ ❌ 새 컬렉션 생성 실패", Fore.RED)
        return
    
    # 완료
    print_colored("\n" + "=" * 60, Fore.GREEN)
    print_colored("🎉 모든 작업 완료!", Fore.GREEN)
    print_colored("📝 다음 단계:", Fore.CYAN)
    print_colored("  1. python main.py 실행", Fore.WHITE)
    print_colored("  2. 2번 (Full Embedding) 선택", Fore.WHITE)
    print_colored("  3. n 입력 (컬렉션은 이미 새로 만들었음)", Fore.WHITE)
    print_colored("=" * 60, Fore.GREEN)

if __name__ == "__main__":
    main()
