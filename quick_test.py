"""
빠른 속도 테스트 스크립트
최적화 효과를 즉시 확인할 수 있습니다.
"""

import time
import sys
import os

# 패치 적용
try:
    from embedding_patch import apply_embedding_patch
    apply_embedding_patch()
    print("🚀 고속 패치 적용됨")
except:
    print("⚠️ 패치 적용 실패 - 기본 설정 사용")

def quick_embedding_test():
    """빠른 임베딩 테스트"""
    try:
        from embeddings import EmbeddingModel
        
        # 테스트 데이터
        test_texts = [
            f"테스트 문서 {i}: 이것은 임베딩 속도 테스트를 위한 샘플 텍스트입니다. "
            f"여러 문장으로 구성되어 실제 문서와 유사한 구조를 가지고 있습니다."
            for i in range(100)
        ]
        
        print(f"\n📊 속도 테스트 시작")
        print(f"테스트 데이터: {len(test_texts)}개 텍스트")
        
        # 임베딩 모델 초기화
        model = EmbeddingModel()
        
        # 배치 처리 테스트
        start_time = time.time()
        
        if hasattr(model, 'get_embeddings_batch_optimized'):
            print("🚀 최적화된 배치 처리 사용")
            embeddings = model.get_embeddings_batch_optimized(test_texts)
        else:
            print("📦 기본 배치 처리 사용")
            embeddings = model.get_embeddings_batch(test_texts)
        
        end_time = time.time()
        
        # 결과 출력
        elapsed = end_time - start_time
        speed = len(test_texts) / elapsed
        
        print(f"\n📈 테스트 결과:")
        print(f"   ⏱️  처리 시간: {elapsed:.2f}초")
        print(f"   🚀 처리 속도: {speed:.1f} 텍스트/초")
        print(f"   📊 생성 벡터: {len(embeddings)}개")
        
        if speed > 20:
            print(f"   ✅ 우수한 성능! (20+ texts/sec)")
        elif speed > 10:
            print(f"   👍 양호한 성능 (10+ texts/sec)")
        elif speed > 5:
            print(f"   ⚠️  개선 필요 (5+ texts/sec)")
        else:
            print(f"   ❌ 성능 부족 (< 5 texts/sec)")
            print(f"      → config 최적화 확인 필요")
        
        return speed
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    quick_embedding_test()
