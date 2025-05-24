#!/usr/bin/env python3
"""
전체 검색을 위한 헬퍼 함수들
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger('OptimizedMCP')

def batch_search_all_documents(
    milvus_manager,
    query_vector: List[float] = None,
    filter_expr: str = None,
    batch_size: int = 1000,
    max_total_results: int = 2000,
    similarity_threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """
    전체 컬렉션을 배치로 스캔하여 모든 관련 문서 검색
    
    Args:
        milvus_manager: Milvus 매니저 인스턴스
        query_vector: 검색할 벡터 (None이면 전체 스캔)
        filter_expr: 필터 표현식
        batch_size: 배치 크기
        max_total_results: 최대 결과 수
        similarity_threshold: 유사도 임계값
    
    Returns:
        모든 매칭 결과의 리스트
    """
    try:
        all_results = []
        offset = 0
        
        # 전체 문서 수 확인
        total_entities = milvus_manager.count_entities()
        logger.info(f"전체 문서 수: {total_entities}")
        
        if query_vector is None:
            # 벡터 검색이 아닌 경우, 전체 문서 조회
            while offset < total_entities and len(all_results) < max_total_results:
                try:
                    batch_results = milvus_manager.query(
                        expr=filter_expr if filter_expr else "id >= 0",
                        output_fields=["id", "path", "title", "chunk_text", "content", "file_type", "tags", "created_at", "updated_at", "chunk_index"],
                        limit=min(batch_size, max_total_results - len(all_results)),
                        offset=offset
                    )
                    
                    if not batch_results:
                        break
                    
                    # 결과 포맷팅
                    for result in batch_results:
                        formatted_result = {
                            "id": result.get("id", ""),
                            "path": result.get("path", ""),
                            "title": result.get("title", "제목 없음"),
                            "chunk_text": result.get("chunk_text", ""),
                            "content": result.get("content", ""),
                            "file_type": result.get("file_type", ""),
                            "tags": result.get("tags", []),
                            "created_at": result.get("created_at", ""),
                            "updated_at": result.get("updated_at", ""),
                            "chunk_index": result.get("chunk_index", 0),
                            "score": 1.0,  # 기본 점수
                            "source": "full_scan"
                        }
                        all_results.append(formatted_result)
                    
                    offset += len(batch_results)
                    
                    # 프로그레스 로깅
                    if offset % (batch_size * 5) == 0:
                        logger.info(f"배치 처리 진행: {offset}/{total_entities}")
                    
                except Exception as e:
                    logger.error(f"배치 {offset} 처리 중 오류: {e}")
                    offset += batch_size
                    continue
        
        else:
            # 벡터 유사도 검색
            search_limit = min(max_total_results, total_entities)
            
            try:
                search_results = milvus_manager.search(
                    vectors=[query_vector],
                    limit=search_limit,
                    expr=filter_expr
                )
                
                for hit in search_results[0]:  # search returns list of lists
                    if hit.score >= similarity_threshold:
                        result = {
                            "id": hit.id,
                            "path": hit.entity.get('path', ''),
                            "title": hit.entity.get('title', '제목 없음'),
                            "chunk_text": hit.entity.get('chunk_text', ''),
                            "content": hit.entity.get('content', ''),
                            "file_type": hit.entity.get('file_type', ''),
                            "tags": hit.entity.get('tags', []),
                            "created_at": hit.entity.get('created_at', ''),
                            "updated_at": hit.entity.get('updated_at', ''),
                            "chunk_index": hit.entity.get('chunk_index', 0),
                            "score": float(hit.score),
                            "source": "vector_search"
                        }
                        all_results.append(result)
                        
            except Exception as e:
                logger.error(f"벡터 검색 중 오류: {e}")
                # 폴백: 전체 스캔으로 전환
                return batch_search_all_documents(
                    milvus_manager, None, filter_expr, batch_size, max_total_results, similarity_threshold
                )
        
        logger.info(f"전체 검색 완료: {len(all_results)}개 결과 발견")
        return all_results
        
    except Exception as e:
        logger.error(f"전체 검색 중 오류: {e}")
        return []

def smart_limit_adjustment(
    query: str,
    search_all: bool = False,
    default_limit: int = 100,
    max_limit: int = 2000
) -> int:
    """
    쿼리와 검색 모드에 따라 적절한 limit 값을 결정
    
    Args:
        query: 검색 쿼리
        search_all: 전체 검색 모드 여부
        default_limit: 기본 limit
        max_limit: 최대 limit
    
    Returns:
        조정된 limit 값
    """
    if search_all:
        return max_limit
    
    # 쿼리 복잡도에 따른 동적 조정
    query_words = len(query.split())
    
    if query_words <= 2:
        # 간단한 쿼리 - 더 많은 결과 필요할 수 있음
        return min(default_limit * 2, max_limit)
    elif query_words <= 5:
        # 중간 복잡도
        return default_limit
    else:
        # 복잡한 쿼리 - 더 정확한 결과 우선
        return min(default_limit, 50)

def optimize_search_results(
    results: List[Dict[str, Any]],
    query: str,
    max_results: int = 100,
    remove_duplicates: bool = True
) -> List[Dict[str, Any]]:
    """
    검색 결과 최적화 및 중복 제거
    
    Args:
        results: 검색 결과 리스트
        query: 원본 쿼리
        max_results: 최대 반환할 결과 수
        remove_duplicates: 중복 제거 여부
    
    Returns:
        최적화된 결과 리스트
    """
    if not results:
        return []
    
    try:
        # 중복 제거 (같은 path의 문서)
        if remove_duplicates:
            seen_paths = set()
            unique_results = []
            
            for result in results:
                path = result.get('path', '')
                if path and path not in seen_paths:
                    seen_paths.add(path)
                    unique_results.append(result)
                elif not path:  # path가 없는 경우도 포함
                    unique_results.append(result)
            
            results = unique_results
        
        # 점수 기반 정렬
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # 쿼리 관련성 재점수화 (간단한 키워드 매칭)
        query_keywords = set(query.lower().split())
        
        for result in results:
            title = result.get('title', '').lower()
            content = result.get('chunk_text', '').lower()
            
            # 키워드 매칭 보너스
            title_matches = sum(1 for keyword in query_keywords if keyword in title)
            content_matches = sum(1 for keyword in query_keywords if keyword in content)
            
            # 기존 점수에 키워드 매칭 보너스 추가
            original_score = result.get('score', 0)
            keyword_bonus = (title_matches * 0.1) + (content_matches * 0.05)
            result['score'] = min(original_score + keyword_bonus, 1.0)
        
        # 재정렬
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # 상위 결과만 반환
        return results[:max_results]
        
    except Exception as e:
        logger.error(f"결과 최적화 중 오류: {e}")
        return results[:max_results]

def enhanced_text_search(
    results: List[Dict[str, Any]],
    query: str,
    search_fields: List[str] = ["title", "chunk_text", "content"]
) -> List[Dict[str, Any]]:
    """
    향상된 텍스트 검색으로 벡터 검색 결과 보완
    
    Args:
        results: 기존 검색 결과
        query: 검색 쿼리
        search_fields: 검색할 필드들
    
    Returns:
        텍스트 검색으로 보완된 결과
    """
    if not query or not results:
        return results
    
    try:
        query_terms = query.lower().split()
        enhanced_results = []
        
        for result in results:
            relevance_score = 0
            matched_terms = set()
            
            for field in search_fields:
                field_content = str(result.get(field, '')).lower()
                
                for term in query_terms:
                    if term in field_content:
                        matched_terms.add(term)
                        # 필드별 가중치
                        if field == "title":
                            relevance_score += 0.4
                        elif field == "chunk_text":
                            relevance_score += 0.3
                        elif field == "content":
                            relevance_score += 0.2
            
            # 텍스트 매칭 점