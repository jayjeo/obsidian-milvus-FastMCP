"""
Embedding 속도 개선을 위한 최적화 스크립트
이 스크립트는 기존 코드의 성능 병목을 해결합니다.

주요 개선 사항:
1. 배치 크기 대폭 증가
2. 메모리 모니터링 최적화
3. 단순화된 GPU 설정
4. 진정한 배치 처리 구현
"""

import os
import time
import gc
import torch
import psutil
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import config

class FastEmbeddingProcessor:
    """고속 임베딩 처리를 위한 최적화된 클래스"""
    
    def __init__(self):
        self.device = self._setup_device()
        self.model = self._setup_model()
        self.optimal_batch_size = self._calculate_optimal_batch_size()
        
    def _setup_device(self):
        """디바이스 설정 - 단순화"""
        if torch.cuda.is_available() and getattr(config, 'USE_GPU', True):
            device = torch.device('cuda')
            print(f"✓ GPU 사용: {torch.cuda.get_device_name(0)}")
            
            # 기본적인 GPU 최적화만 적용
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            return device
        else:
            print("• CPU 모드 사용")
            return torch.device('cpu')
    
    def _setup_model(self):
        """모델 설정 - 단순화"""
        print("모델 로딩 중...")
        model = SentenceTransformer(config.EMBEDDING_MODEL, device=self.device)
        
        # 모델 evaluation 모드로 설정
        model.eval()
        
        print(f"✓ 모델 로드 완료: {config.EMBEDDING_MODEL}")
        return model
    
    def _calculate_optimal_batch_size(self):
        """시스템 사양에 따른 최적 배치 크기 계산"""
        if 'cuda' in str(self.device):
            # GPU 메모리 기반 계산
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            if gpu_memory_gb >= 16:
                return 2000  # 16GB+ GPU
            elif gpu_memory_gb >= 8:
                return 1000  # 8-16GB GPU
            elif gpu_memory_gb >= 4:
                return 500   # 4-8GB GPU
            else:
                return 200   # < 4GB GPU
        else:
            # CPU RAM 기반 계산
            ram_gb = psutil.virtual_memory().total / (1024**3)
            
            if ram_gb >= 32:
                return 500   # 32GB+ RAM
            elif ram_gb >= 16:
                return 200   # 16-32GB RAM
            else:
                return 100   # < 16GB RAM
    
    def encode_batch_optimized(self, texts, show_progress=True):
        """최적화된 배치 임베딩"""
        if not texts:
            return []
        
        # 텍스트 전처리
        clean_texts = []
        for text in texts:
            if isinstance(text, str) and text.strip():
                # 길이 제한 (너무 긴 텍스트 처리)
                if len(text) > 8000:
                    text = text[:8000]
                clean_texts.append(text)
            else:
                clean_texts.append("empty")  # 빈 텍스트 처리
        
        batch_size = self.optimal_batch_size
        embeddings = []
        
        print(f"임베딩 처리 시작: {len(clean_texts)}개 텍스트, 배치 크기: {batch_size}")
        
        # 진행률 표시
        if show_progress:
            pbar = tqdm(total=len(clean_texts), desc="Embedding 처리")
        
        try:
            for i in range(0, len(clean_texts), batch_size):
                batch_texts = clean_texts[i:i+batch_size]
                
                # GPU 메모리 확인 (간단히)
                if 'cuda' in str(self.device) and i > 0 and i % (batch_size * 5) == 0:
                    if torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory > 0.9:
                        torch.cuda.empty_cache()
                        gc.collect()
                
                with torch.no_grad():
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        batch_size=min(batch_size, len(batch_texts)),
                        show_progress_bar=False,  # tqdm으로 표시하므로 비활성화
                        convert_to_tensor=False,  # NumPy 배열로 반환
                        normalize_embeddings=True,  # 정규화 적용
                        device=str(self.device)
                    )
                
                embeddings.extend(batch_embeddings.tolist() if hasattr(batch_embeddings, 'tolist') else batch_embeddings)
                
                if show_progress:
                    pbar.update(len(batch_texts))
                
                # 주기적 메모리 정리 (매 10번째 배치마다)
                if i > 0 and i % (batch_size * 10) == 0:
                    gc.collect()
                    
        finally:
            if show_progress:
                pbar.close()
        
        print(f"✓ 임베딩 완료: {len(embeddings)}개 벡터 생성")
        return embeddings


def benchmark_embedding_speed():
    """임베딩 속도 벤치마크 테스트"""
    print("\n" + "="*60)
    print("EMBEDDING 속도 벤치마크 테스트")
    print("="*60)
    
    # 테스트 데이터 생성
    test_texts = [
        f"이것은 테스트 문서 {i}입니다. 임베딩 처리 속도를 측정하기 위한 샘플 텍스트입니다. "
        f"여러 문장으로 구성되어 있으며, 실제 문서와 유사한 길이를 가지고 있습니다. "
        f"테스트 번호: {i}, 현재 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        for i in range(500)  # 500개 텍스트로 테스트
    ]
    
    processor = FastEmbeddingProcessor()
    
    print(f"테스트 데이터: {len(test_texts)}개 텍스트")
    print(f"사용 디바이스: {processor.device}")
    print(f"배치 크기: {processor.optimal_batch_size}")
    
    # 벤치마크 실행
    start_time = time.time()
    embeddings = processor.encode_batch_optimized(test_texts)
    end_time = time.time()
    
    # 결과 출력
    elapsed_time = end_time - start_time
    texts_per_second = len(test_texts) / elapsed_time
    ms_per_text = (elapsed_time / len(test_texts)) * 1000
    
    print(f"\n벤치마크 결과:")
    print(f"• 처리 시간: {elapsed_time:.2f}초")
    print(f"• 처리 속도: {texts_per_second:.1f} 텍스트/초")
    print(f"• 평균 처리 시간: {ms_per_text:.1f}ms/텍스트")
    print(f"• 생성된 임베딩: {len(embeddings)}개")
    print(f"• 벡터 차원: {len(embeddings[0]) if embeddings else 0}")
    
    # 메모리 사용량 출력
    if 'cuda' in str(processor.device):
        allocated = torch.cuda.memory_allocated() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"• GPU 메모리 사용: {allocated:.2f}/{total:.2f} GB ({allocated/total*100:.1f}%)")
    
    ram = psutil.virtual_memory()
    print(f"• RAM 사용: {(ram.total - ram.available)/(1024**3):.2f}/{ram.total/(1024**3):.2f} GB ({ram.percent:.1f}%)")


def apply_speed_fixes():
    """기존 코드에 속도 개선 적용"""
    print("\n" + "="*60)
    print("EMBEDDING 속도 개선 적용")
    print("="*60)
    
    fixes_applied = []
    
    # 1. config.py 배치 크기 수정
    try:
        config_path = "config.py"
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 배치 크기 관련 설정 찾아서 수정
            if 'EMBEDDING_BATCH_SIZE' in content:
                # 기존 값 찾기
                import re
                match = re.search(r'EMBEDDING_BATCH_SIZE\s*=\s*(\d+)', content)
                if match:
                    old_value = match.group(1)
                    new_value = "2000"  # 대폭 증가
                    content = content.replace(f'EMBEDDING_BATCH_SIZE = {old_value}', f'EMBEDDING_BATCH_SIZE = {new_value}')
                    
                    with open(config_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    fixes_applied.append(f"✓ 배치 크기 수정: {old_value} → {new_value}")
            
            # 모니터링 간격 수정
            if 'resource_check_interval' in content:
                content = content.replace('resource_check_interval = 2', 'resource_check_interval = 30')
                fixes_applied.append("✓ 모니터링 간격 수정: 2초 → 30초")
                
    except Exception as e:
        print(f"⚠ config.py 수정 실패: {e}")
    
    # 2. 최적화된 임베딩 함수 생성
    optimized_code = '''
def get_embeddings_batch_fast(self, texts, batch_size=None):
    """최적화된 고속 배치 임베딩"""
    if batch_size is None:
        batch_size = getattr(self, 'optimal_batch_size', 2000)
    
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        with torch.no_grad():
            vectors = self.model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_tensor=False,
                normalize_embeddings=True,
                device=str(self.device)
            )
        
        embeddings.extend(vectors.tolist() if hasattr(vectors, 'tolist') else vectors)
        
        # 주기적 메모리 정리
        if i > 0 and i % (batch_size * 10) == 0:
            gc.collect()
            if 'cuda' in str(self.device):
                torch.cuda.empty_cache()
    
    return embeddings
'''
    
    # 최적화 코드를 파일로 저장
    with open("embedding_optimization_patch.py", "w", encoding='utf-8') as f:
        f.write(optimized_code)
    
    fixes_applied.append("✓ 최적화된 임베딩 함수 생성: embedding_optimization_patch.py")
    
    # 결과 출력
    print("\n적용된 수정사항:")
    for fix in fixes_applied:
        print(fix)
    
    print(f"\n추가 권장사항:")
    print("1. embeddings.py의 배치 크기를 2000으로 수정")
    print("2. 메모리 모니터링 간격을 30초로 변경")
    print("3. 복잡한 GPU 최적화 로직 비활성화")
    print("4. 개별 파일 처리 대신 전체 배치 처리 구현")


def main():
    """메인 함수"""
    print("EMBEDDING 속도 개선 도구")
    print("이 도구는 현재 프로젝트의 embedding 처리 속도를 대폭 개선합니다.")
    
    while True:
        print("\n선택하세요:")
        print("1. 속도 벤치마크 테스트")
        print("2. 속도 개선 적용")
        print("3. 종료")
        
        choice = input("\n선택 (1-3): ").strip()
        
        if choice == "1":
            benchmark_embedding_speed()
        elif choice == "2":
            apply_speed_fixes()
        elif choice == "3":
            break
        else:
            print("올바른 번호를 선택하세요.")


if __name__ == "__main__":
    main()
