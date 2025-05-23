#!/usr/bin/env python3
"""
Milvus 컬렉션 완전 삭제 및 재생성 스크립트
"""

import sys
import os

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from milvus_manager import MilvusManager
import config
from colorama import Fore, Style, init

# colorama 초기화
init()

def print_colored(message, color=Fore.WHITE):
    """컬러 출력"""
    print(f"{color}{message}{Style.RESET_ALL}")

def main():
    """메인 함수"""
    print_colored("🗑️ Milvus 컬렉션 완전 삭제 및 재생성", Fore.CYAN)
    print_colored("=" * 50, Fore.CYAN)
    
    try:
        # Milvus 매니저 초기화
        print_colored("📡 Milvus 연결 중...", Fore.BLUE)
        milvus_manager = MilvusManager()
        
        # 기존 데이터 확인
        try:
            total_entities = milvus_manager.count_entities()
            print_colored(f"📊 현재 컬렉션 '{config.COLLECTION_NAME}'에 {total_entities}개 문서가 있습니다.", Fore.YELLOW)
        except Exception as e:
            print_colored(f"⚠️ 기존 컬렉션 확인 중 오류: {e}", Fore.YELLOW)
        
        # 사용자 확인
        choice = input(f"\n🔥 컬렉션 '{config.COLLECTION_NAME}'을 완전히 삭제하고 재생성하시겠습니까? (y/N): ")
        
        if choice.lower() != 'y':
            print_colored("❌ 작업이 취소되었습니다.", Fore.RED)
            return
        
        # 컬렉션 삭제 및 재생성
        print_colored(f"\n🗑️ 컬렉션 '{config.COLLECTION_NAME}' 삭제 중...", Fore.RED)
        milvus_manager.recreate_collection()
        print_colored("✅ 컬렉션이 성공적으로 삭제되고 재생성되었습니다!", Fore.GREEN)
        
        # 확인
        try:
            total_entities = milvus_manager.count_entities()
            print_colored(f"📊 새로운 컬렉션에는 {total_entities}개 문서가 있습니다.", Fore.GREEN)
        except Exception as e:
            print_colored(f"📊 새로운 컬렉션이 비어있습니다. (정상)", Fore.GREEN)
        
        print_colored("\n🎉 컬렉션 재생성 완료!", Fore.GREEN)
        print_colored("💡 이제 main.py에서 전체 재인덱싱을 실행하세요.", Fore.CYAN)
        
    except Exception as e:
        print_colored(f"❌ 오류 발생: {e}", Fore.RED)
        import traceback
        print_colored(f"상세 오류:\n{traceback.format_exc()}", Fore.RED)

if __name__ == "__main__":
    main()
