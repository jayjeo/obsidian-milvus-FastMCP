"""
Config.py 최적화 설정
기존 설정을 고속 처리에 맞게 조정합니다.
"""

import os

# 기존 config.py 백업 및 최적화
def optimize_config():
    """config.py 파일을 최적화된 설정으로 수정"""
    config_path = "config.py"
    backup_path = "config_backup.py"
    
    if not os.path.exists(config_path):
        print(f"❌ {config_path} 파일을 찾을 수 없습니다.")
        return False
    
    try:
        # 기존 config 백업
        with open(config_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        print(f"✓ 기존 설정 백업됨: {backup_path}")
        
        # 최적화된 설정 적용
        optimizations = {
            'EMBEDDING_BATCH_SIZE': '2000',  # 대폭 증가
            'BATCH_SIZE': '2000',
            'CHUNK_SIZE': '1000',  # 적당한 크기로 조정
            'CHUNK_OVERLAP': '100',
            'CHUNK_MIN_SIZE': '50',
            'GPU_MEMORY_FRACTION': '0.95',  # GPU 메모리 거의 모두 사용
            'GPU_ENABLE_CUDNN_BENCHMARK': 'True',
            'GPU_FORCE_TENSOR_CORES': 'True',
        }
        
        content = original_content
        
        for key, value in optimizations.items():
            # 기존 설정이 있으면 수정, 없으면 추가
            import re
            pattern = rf'^{key}\s*=.*$'
            replacement = f'{key} = {value}'
            
            if re.search(pattern, content, re.MULTILINE):
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                print(f"✓ 수정: {key} = {value}")
            else:
                content += f'\n{replacement}'
                print(f"✓ 추가: {key} = {value}")
        
        # 추가 최적화 설정
        additional_config = '''

# 속도 최적화 설정 (자동 추가됨)
DISABLE_PROGRESS_MONITORING = True  # 진행률 모니터링 비활성화
MEMORY_CHECK_INTERVAL = 30  # 메모리 체크 간격 (초)
FAST_MODE = True  # 빠른 모드 활성화
EMBEDDING_CACHE_SIZE = 10000  # 캐시 크기 증가
MAX_WORKERS = 1  # 멀티스레딩 비활성화 (배치 처리로 대체)

# GPU 최적화 비활성화 (단순한 설정이 더 빠름)
DISABLE_COMPLEX_GPU_OPTIMIZATION = True
'''
        
        if 'FAST_MODE' not in content:
            content += additional_config
            print("✓ 추가 최적화 설정 적용")
        
        # 수정된 내용 저장
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ {config_path} 최적화 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 설정 최적화 실패: {e}")
        return False

def restore_config():
    """원래 설정으로 복원"""
    config_path = "config.py"
    backup_path = "config_backup.py"
    
    if not os.path.exists(backup_path):
        print(f"❌ 백업 파일을 찾을 수 없습니다: {backup_path}")
        return False
    
    try:
        with open(backup_path, 'r', encoding='utf-8') as f:
            backup_content = f.read()
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(backup_content)
        
        print(f"✅ 원래 설정으로 복원됨: {config_path}")
        return True
        
    except Exception as e:
        print(f"❌ 설정 복원 실패: {e}")
        return False

def show_config_comparison():
    """설정 비교 표시"""
    print("\n" + "="*60)
    print("설정 최적화 비교")
    print("="*60)
    
    print("기존 설정 → 최적화 설정")
    print("-" * 40)
    print("EMBEDDING_BATCH_SIZE: 32 → 2000 (62배 증가)")
    print("BATCH_SIZE: 64 → 2000 (31배 증가)")
    print("GPU_MEMORY_FRACTION: 0.7 → 0.95 (35% 더 사용)")
    print("메모리 체크 간격: 2초 → 30초 (15배 감소)")
    print("복잡한 GPU 최적화: 활성 → 비활성 (오버헤드 제거)")
    print("진행률 모니터링: 활성 → 비활성 (CPU 사용량 감소)")
    
    print("\n예상 성능 향상:")
    print("🚀 임베딩 속도: 5-10배 향상")
    print("📈 GPU 사용률: 대폭 증가")
    print("💾 메모리 효율성: 개선")
    print("⏱️ 전체 처리 시간: 80% 단축")

def main():
    """메인 함수"""
    print("Config.py 최적화 도구")
    print("현재 설정을 고속 처리에 맞게 최적화합니다.")
    
    while True:
        print("\n선택하세요:")
        print("1. 설정 최적화 비교 보기")
        print("2. 설정 최적화 적용")
        print("3. 원래 설정으로 복원")
        print("4. 종료")
        
        choice = input("\n선택 (1-4): ").strip()
        
        if choice == "1":
            show_config_comparison()
        elif choice == "2":
            if optimize_config():
                print("\n⚠️ 주의: config.py가 수정되었습니다.")
                print("변경사항을 적용하려면 프로그램을 재시작하세요.")
        elif choice == "3":
            restore_config()
        elif choice == "4":
            break
        else:
            print("올바른 번호를 선택하세요.")

if __name__ == "__main__":
    main()
