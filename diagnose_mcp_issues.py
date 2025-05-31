#!/usr/bin/env python3
"""
MCP 서버 도구 문제 진단 스크립트
작동하지 않는 도구들의 원인을 파악합니다.
"""

import os
import sys
import traceback

# 환경 변수 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("=== MCP 서버 도구 진단 시작 ===\n")

# 1. 기본 모듈 임포트 테스트
print("1. 기본 모듈 임포트 테스트...")
try:
    import config
    print("✓ config 모듈 임포트 성공")
except Exception as e:
    print(f"✗ config 모듈 임포트 실패: {e}")
    sys.exit(1)

try:
    from milvus_manager import MilvusManager
    print("✓ MilvusManager 임포트 성공")
except Exception as e:
    print(f"✗ MilvusManager 임포트 실패: {e}")
    sys.exit(1)

try:
    from search_engine import SearchEngine
    print("✓ SearchEngine 임포트 성공")
except Exception as e:
    print(f"✗ SearchEngine 임포트 실패: {e}")
    sys.exit(1)

try:
    from enhanced_search_engine import EnhancedSearchEngine
    print("✓ EnhancedSearchEngine 임포트 성공")
except Exception as e:
    print(f"✗ EnhancedSearchEngine 임포트 실패: {e}")
    sys.exit(1)

# 2. 고급 모듈 임포트 테스트
print("\n2. 고급 모듈 임포트 테스트...")
hnsw_optimizer = None
rag_engine = None

try:
    from hnsw_optimizer import HNSWOptimizer
    print("✓ HNSWOptimizer 임포트 성공")
except Exception as e:
    print(f"✗ HNSWOptimizer 임포트 실패: {e}")
    traceback.print_exc()

try:
    from advanced_rag import AdvancedRAGEngine
    print("✓ AdvancedRAGEngine 임포트 성공")
except Exception as e:
    print(f"✗ AdvancedRAGEngine 임포트 실패: {e}")
    traceback.print_exc()

# 3. 컴포넌트 초기화 테스트
print("\n3. 컴포넌트 초기화 테스트...")
milvus_manager = None
search_engine = None
enhanced_search = None

try:
    print("- MilvusManager 초기화 중...")
    milvus_manager = MilvusManager()
    print("✓ MilvusManager 초기화 성공")
except Exception as e:
    print(f"✗ MilvusManager 초기화 실패: {e}")
    traceback.print_exc()

if milvus_manager:
    try:
        print("- SearchEngine 초기화 중...")
        search_engine = SearchEngine(milvus_manager)
        print("✓ SearchEngine 초기화 성공")
    except Exception as e:
        print(f"✗ SearchEngine 초기화 실패: {e}")
        traceback.print_exc()

    try:
        print("- EnhancedSearchEngine 초기화 중...")
        enhanced_search = EnhancedSearchEngine(milvus_manager)
        print("✓ EnhancedSearchEngine 초기화 성공")
    except Exception as e:
        print(f"✗ EnhancedSearchEngine 초기화 실패: {e}")
        traceback.print_exc()

    # HNSWOptimizer 초기화 테스트
    if 'HNSWOptimizer' in globals():
        try:
            print("- HNSWOptimizer 초기화 중...")
            hnsw_optimizer = HNSWOptimizer(milvus_manager)
            print("✓ HNSWOptimizer 초기화 성공")
        except Exception as e:
            print(f"✗ HNSWOptimizer 초기화 실패: {e}")
            traceback.print_exc()

    # AdvancedRAGEngine 초기화 테스트
    if 'AdvancedRAGEngine' in globals() and enhanced_search:
        try:
            print("- AdvancedRAGEngine 초기화 중...")
            rag_engine = AdvancedRAGEngine(milvus_manager, enhanced_search)
            print("✓ AdvancedRAGEngine 초기화 성공")
        except Exception as e:
            print(f"✗ AdvancedRAGEngine 초기화 실패: {e}")
            traceback.print_exc()

# 4. 메서드 존재 확인
print("\n4. 중요 메서드 존재 확인...")

if milvus_manager:
    # _sanitize_query_expr 메서드 확인
    if hasattr(milvus_manager, '_sanitize_query_expr'):
        print("✓ milvus_manager._sanitize_query_expr 메서드 존재")
    else:
        print("✗ milvus_manager._sanitize_query_expr 메서드 없음")
    
    # search_with_params 메서드 확인
    if hasattr(milvus_manager, 'search_with_params'):
        print("✓ milvus_manager.search_with_params 메서드 존재")
    else:
        print("✗ milvus_manager.search_with_params 메서드 없음")

if search_engine:
    # hybrid_search 메서드 확인
    if hasattr(search_engine, 'hybrid_search'):
        print("✓ search_engine.hybrid_search 메서드 존재")
    else:
        print("✗ search_engine.hybrid_search 메서드 없음")

if enhanced_search:
    # advanced_filter_search 메서드 확인
    if hasattr(enhanced_search, 'advanced_filter_search'):
        print("✓ enhanced_search.advanced_filter_search 메서드 존재")
    else:
        print("✗ enhanced_search.advanced_filter_search 메서드 없음")

# 5. 간단한 기능 테스트
print("\n5. 간단한 기능 테스트...")

if milvus_manager:
    try:
        # count_entities 테스트
        count = milvus_manager.count_entities()
        print(f"✓ count_entities 테스트 성공: {count} 문서")
    except Exception as e:
        print(f"✗ count_entities 테스트 실패: {e}")

    try:
        # _sanitize_query_expr 테스트
        test_expr = "path == '/test/path'"
        sanitized = milvus_manager._sanitize_query_expr(test_expr)
        print(f"✓ _sanitize_query_expr 테스트 성공: '{test_expr}' -> '{sanitized}'")
    except Exception as e:
        print(f"✗ _sanitize_query_expr 테스트 실패: {e}")

if search_engine:
    try:
        # _keyword_search 메서드 테스트 (빈 결과라도 에러만 안나면 됨)
        results = search_engine._keyword_search("test", limit=5)
        print(f"✓ _keyword_search 테스트 성공: {len(results)} 결과")
    except Exception as e:
        print(f"✗ _keyword_search 테스트 실패: {e}")
        traceback.print_exc()

if enhanced_search:
    try:
        # advanced_filter_search 간단 테스트
        results = enhanced_search.advanced_filter_search("test", limit=5)
        print(f"✓ advanced_filter_search 테스트 성공: {len(results)} 결과")
    except Exception as e:
        print(f"✗ advanced_filter_search 테스트 실패: {e}")
        traceback.print_exc()

# 6. 진단 요약
print("\n=== 진단 요약 ===")
print(f"- MilvusManager: {'✓ 정상' if milvus_manager else '✗ 실패'}")
print(f"- SearchEngine: {'✓ 정상' if search_engine else '✗ 실패'}")
print(f"- EnhancedSearchEngine: {'✓ 정상' if enhanced_search else '✗ 실패'}")
print(f"- HNSWOptimizer: {'✓ 정상' if hnsw_optimizer else '✗ 실패 또는 미설치'}")
print(f"- AdvancedRAGEngine: {'✓ 정상' if rag_engine else '✗ 실패 또는 미설치'}")

print("\n진단 완료!")

# 리소스 정리
if milvus_manager:
    try:
        milvus_manager.stop_monitoring()
    except:
        pass
