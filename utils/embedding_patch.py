"""
embeddings.py용 속도 최적화 패치

사용법:
1. 이 파일을 embeddings.py와 같은 폴더에 저장
2. embeddings.py에 다음 코드 추가:

from embedding_patch import get_embeddings_batch_optimized
EmbeddingModel.get_embeddings_batch_optimized = get_embeddings_batch_optimized
"""

import torch
import gc
import numpy as np
from tqdm import tqdm

def get_embeddings_batch_optimized(self, texts, batch_size=None):
    """최적화된 고속 배치 임베딩"""
    if not texts:
        return []
    
    # 기본 배치 크기 설정
    if batch_size is None:
        batch_size = getattr(self, 'optimal_batch_size', 2000)
    
    print(f"🚀 고속 배치 임베딩 시작: {len(texts)}개 텍스트, 배치 크기: {batch_size}")
    
    # 텍스트 전처리
    clean_texts = []
    for text in texts:
        if isinstance(text, str) and text.strip():
            if len(text) > 8000:
                text = text[:8000]  # 길이 제한
            clean_texts.append(text)
        else:
            clean_texts.append("empty")
    
    embeddings = []
    
    # 메모리 정리
    gc.collect()
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 배치 처리
    with tqdm(total=len(clean_texts), desc="임베딩 처리") as pbar:
        for i in range(0, len(clean_texts), batch_size):
            batch_texts = clean_texts[i:i+batch_size]
            
            try:
                with torch.no_grad():
                    # SentenceTransformer 모델 직접 사용
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        batch_size=len(batch_texts),
                        show_progress_bar=False,  # tqdm 사용
                        convert_to_tensor=False,
                        normalize_embeddings=True,
                        device=str(self.device) if hasattr(self, 'device') else None
                    )
                
                # 결과 처리
                if isinstance(batch_embeddings, np.ndarray):
                    embeddings.extend(batch_embeddings.tolist())
                else:
                    embeddings.extend(batch_embeddings)
                
                pbar.update(len(batch_texts))
                
                # 주기적 메모리 정리 (10배치마다)
                if i > 0 and i % (batch_size * 10) == 0:
                    gc.collect()
                    if hasattr(torch, 'cuda') and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                print(f"\n⚠️ 배치 {i//batch_size + 1} 처리 오류: {e}")
                # 개별 처리로 폴백
                for text in batch_texts:
                    try:
                        vector = self.get_embedding(text)
                        embeddings.append(vector)
                    except:
                        # 기본 차원의 0 벡터 추가
                        embeddings.append([0] * 384)  # 기본 차원
                pbar.update(len(batch_texts))
    
    print(f"✅ 임베딩 완료: {len(embeddings)}개 벡터 생성")
    return embeddings

# 패치 적용 함수
def apply_embedding_patch():
    """임베딩 패치를 EmbeddingModel에 적용"""
    try:
        from embeddings import EmbeddingModel
        
        # 기존 메서드 백업
        if not hasattr(EmbeddingModel, '_original_get_embeddings_batch'):
            EmbeddingModel._original_get_embeddings_batch = EmbeddingModel.get_embeddings_batch
        
        # 최적화된 메서드로 교체
        EmbeddingModel.get_embeddings_batch = get_embeddings_batch_optimized
        
        print("✅ 임베딩 속도 패치 적용 완료")
        return True
        
    except Exception as e:
        print(f"❌ 임베딩 패치 적용 실패: {e}")
        return False

if __name__ == "__main__":
    apply_embedding_patch()
