#!/usr/bin/env python3
"""
기본 검색 엔진 클래스 - Milvus의 기능을 활용한 하이브리드 검색
기존 milvus_manager.py의 고급 기능들과 연동
"""

import time
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from embeddings import EmbeddingModel
import config

logger = logging.getLogger('SearchEngine')

class SearchEngine:
    def __init__(self, milvus_manager):
        self.milvus_manager = milvus_manager
        self.embedding_model = EmbeddingModel()
        self.recent_queries = []  # 최근 쿼리 캐시
        
    def hybrid_search(self, query: str, limit: int = 10, filter_params: Optional[Dict] = None) -> Tuple[List[Dict], Dict]:
        """하이브리드 검색: 벡터 검색 + 키워드 검색 결합"""
        try:
            start_time = time.time()
            
            # 최근 쿼리에 추가
            self.recent_queries.append(query)
            if len(self.recent_queries) > 50:
                self.recent_queries = self.recent_queries[-50:]
            
            # 1. 벡터 검색 수행
            vector_results = self._vector_search(query, limit * 2, filter_params)
            
            # 2. 키워드 검색 수행
            keyword_results = self._keyword_search(query, limit, filter_params)
            
            # 3. 결과 융합
            fused_results = self._fuse_search_results(vector_results, keyword_results, query)
            
            # 4. 상위 결과만 반환
            final_results = fused_results[:limit]
            
            search_time = time.time() - start_time
            
            search_info = {
                "query": query,
                "search_type": "hybrid",
                "vector_results_count": len(vector_results),
                "keyword_results_count": len(keyword_results),
                "final_results_count": len(final_results),
                "search_time_ms": round(search_time * 1000, 2),
                "total_count": len(final_results)
            }
            
            return final_results, search_info
            
        except Exception as e:
            logger.error(f"하이브리드 검색 중 오류: {e}")
            # 폴백: 벡터 검색만 수행
            try:
                vector_results = self._vector_search(query, limit, filter_params)
                search_info = {
                    "query": query,
                    "search_type": "vector_fallback",
                    "error": str(e),
                    "total_count": len(vector_results)
                }
                return vector_results, search_info
            except Exception as e2:
                logger.error(f"폴백 검색도 실패: {e2}")
                return [], {"query": query, "error": str(e2), "total_count": 0}
    
    def _vector_search(self, query: str, limit: int, filter_params: Optional[Dict] = None) -> List[Dict]:
        """벡터 유사도 검색"""
        try:
            # 쿼리를 벡터로 변환
            query_vector = self.embedding_model.get_embedding(query)
            
            # 필터 표현식 구성
            filter_expr = self._build_filter_expr(filter_params) if filter_params else None
            
            # milvus_manager의 고급 검색 기능 사용
            if hasattr(self.milvus_manager, 'search_with_params'):
                # 최적화된 검색 파라미터 사용
                search_params = self._get_optimized_search_params(query)
                raw_results = self.milvus_manager.search_with_params(
                    vector=query_vector,
                    limit=limit,
                    filter_expr=filter_expr,
                    search_params=search_params
                )
            else:
                # 기본 검색 사용
                raw_results = self.milvus_manager.search(query_vector, limit, filter_expr)
            
            # 결과 포맷팅
            formatted_results = []
            for hit in raw_results:
                result = {
                    "id": hit.id,
                    "path": hit.entity.get('path', ''),
                    "title": hit.entity.get('title', '제목 없음'),
                    "chunk_text": hit.entity.get('chunk_text', ''),
                    "content": hit.entity.get('content', ''),
                    "score": float(hit.score),
                    "source": "vector",
                    "file_type": hit.entity.get('file_type', ''),
                    "tags": hit.entity.get('tags', []),
                    "chunk_index": hit.entity.get('chunk_index', 0),
                    "created_at": hit.entity.get('created_at', ''),
                    "updated_at": hit.entity.get('updated_at', '')
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"벡터 검색 중 오류: {e}")
            return []
    
    def _keyword_search(self, query: str, limit: int, filter_expr: Optional[str] = None) -> List[Dict]:
        """키워드 기반 검색 (메모리에서 필터링)"""
        try:
            # 전체 문서 조회 (페이지네이션 사용)
            all_docs = []
            offset = 0
            batch_size = 1000
            
            while True:
                batch = self.milvus_manager.query(
                    expr=filter_expr or "id >= 0",
                    output_fields=["id", "path", "title", "chunk_text", "content", "file_type", "tags", "chunk_index", "created_at", "updated_at"],
                    limit=batch_size,
                    offset=offset
                )
                
                if not batch:
                    break
                    
                all_docs.extend(batch)
                offset += batch_size
                
                if len(batch) < batch_size:
                    break
            
            # 키워드 점수 계산
            query_terms = re.findall(r'[\w가-힣]+', query.lower())
            scored_results = []
            
            for doc in all_docs:
                score = self._calculate_keyword_score(doc, query_terms)
                if score > 0:
                    result = {
                        "id": doc.get('id', ''),
                        "path": doc.get('path', ''),
                        "title": doc.get('title', '제목 없음'),
                        "chunk_text": doc.get('chunk_text', ''),
                        "content": doc.get('content', ''),
                        "score": score,
                        "source": "keyword",
                        "file_type": doc.get('file_type', ''),
                        "tags": doc.get('tags', []),
                        "chunk_index": doc.get('chunk_index', 0),
                        "created_at": doc.get('created_at', ''),
                        "updated_at": doc.get('updated_at', '')
                    }
                    scored_results.append(result)
            
            # 점수 순으로 정렬하고 상위 결과 반환
            scored_results.sort(key=lambda x: x['score'], reverse=True)
            return scored_results[:limit]
            
        except Exception as e:
            logger.error(f"키워드 검색 중 오류: {e}")
            return []
    
    def _calculate_keyword_score(self, doc: Dict, query_terms: List[str]) -> float:
        """키워드 매칭 점수 계산"""
        score = 0.0
        
        path = doc.get('path', '').lower()
        title = doc.get('title', '').lower()
        content = doc.get('chunk_text', '').lower()
        
        for term in query_terms:
            # 경로에서 매칭 (높은 점수)
            if term in path:
                score += 5.0
            
            # 제목에서 매칭 (중간 점수)
            if term in title:
                score += 3.0
                
            # 내용에서 매칭 (기본 점수)
            content_matches = content.count(term)
            score += content_matches * 1.0
        
        return score
    
    def _fuse_search_results(self, vector_results: List[Dict], keyword_results: List[Dict], query: str) -> List[Dict]:
        """벡터 검색과 키워드 검색 결과 융합"""
        # 문서 ID별로 결과 수집
        result_dict = {}
        
        # 벡터 검색 결과 추가 (가중치 0.7)
        for result in vector_results:
            doc_id = result['id']
            result_dict[doc_id] = result.copy()
            result_dict[doc_id]['vector_score'] = result['score'] * 0.7
            result_dict[doc_id]['keyword_score'] = 0.0
            result_dict[doc_id]['sources'] = ['vector']
        
        # 키워드 검색 결과 추가 (가중치 0.3)
        for result in keyword_results:
            doc_id = result['id']
            if doc_id in result_dict:
                # 이미 있는 경우 키워드 점수 추가
                result_dict[doc_id]['keyword_score'] = result['score'] * 0.3
                result_dict[doc_id]['sources'].append('keyword')
            else:
                # 새로운 결과인 경우
                result_dict[doc_id] = result.copy()
                result_dict[doc_id]['vector_score'] = 0.0
                result_dict[doc_id]['keyword_score'] = result['score'] * 0.3
                result_dict[doc_id]['sources'] = ['keyword']
        
        # 최종 점수 계산 및 정렬
        fused_results = []
        for doc_id, result in result_dict.items():
            # 하이브리드 점수 = 벡터 점수 + 키워드 점수 + 다중 소스 보너스
            hybrid_score = result['vector_score'] + result['keyword_score']
            if len(result['sources']) > 1:
                hybrid_score *= 1.2  # 다중 소스 보너스
            
            result['score'] = hybrid_score
            result['source'] = '+'.join(result['sources'])
            
            # 불필요한 임시 필드 제거
            if 'vector_score' in result:
                del result['vector_score']
            if 'keyword_score' in result:
                del result['keyword_score']
            if 'sources' in result:
                del result['sources']
                
            fused_results.append(result)
        
        # 최종 점수로 정렬
        fused_results.sort(key=lambda x: x['score'], reverse=True)
        return fused_results
    
    def _build_filter_expr(self, filter_params: Dict) -> Optional[str]:
        """필터 파라미터를 Milvus 표현식으로 변환"""
        if not filter_params:
            return None
            
        expressions = []
        
        # 파일 타입 필터
        if 'file_types' in filter_params:
            file_types = filter_params['file_types']
            if len(file_types) == 1:
                expressions.append(f"file_type == '{file_types[0]}'")
            else:
                type_conditions = " or ".join([f"file_type == '{ft}'" for ft in file_types])
                expressions.append(f"({type_conditions})")
        
        # 태그 필터
        if 'tags' in filter_params:
            tags = filter_params['tags']
            for tag in tags:
                expressions.append(f"tags like '%{tag}%'")
        
        # 시간 범위 필터
        if 'date_range' in filter_params:
            start_date, end_date = filter_params['date_range']
            expressions.append(f"created_at >= '{start_date}' and created_at <= '{end_date}'")
        
        return " and ".join(expressions) if expressions else None
    
    def _get_optimized_search_params(self, query: str) -> Dict:
        """쿼리 복잡도에 따른 최적화된 검색 파라미터"""
        query_length = len(query.split())
        
        if query_length <= 3:
            # 간단한 쿼리 - 빠른 검색
            complexity = "simple"
        elif query_length <= 8:
            # 중간 복잡도 쿼리
            complexity = "medium"
        else:
            # 복잡한 쿼리 - 정확한 검색
            complexity = "complex"
        
        # GPU/CPU에 따른 파라미터 설정
        if config.USE_GPU:
            params_map = {
                "simple": {"nprobe": 8},
                "medium": {"nprobe": 16},
                "complex": {"nprobe": 32}
            }
            return {
                "metric_type": "COSINE",
                "params": params_map[complexity]
            }
        else:
            params_map = {
                "simple": {"ef": 64},
                "medium": {"ef": 128},
                "complex": {"ef": 256}
            }
            return {
                "metric_type": "COSINE",
                "params": params_map[complexity]
            }
    
    def semantic_similarity_search(self, query: str, similarity_threshold: float = 0.7) -> List[Dict]:
        """의미적 유사도 기반 검색"""
        try:
            # 높은 정확도로 검색
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 256} if not config.USE_GPU else {"nprobe": 32}
            }
            
            query_vector = self.embedding_model.get_embedding(query)
            
            if hasattr(self.milvus_manager, 'search_with_params'):
                raw_results = self.milvus_manager.search_with_params(
                    vector=query_vector,
                    limit=100,
                    search_params=search_params
                )
            else:
                raw_results = self.milvus_manager.search(query_vector, 100)
            
            # 유사도 임계값 적용
            filtered_results = []
            for hit in raw_results:
                if hit.score >= similarity_threshold:
                    result = {
                        "id": hit.id,
                        "path": hit.entity.get('path', ''),
                        "title": hit.entity.get('title', '제목 없음'),
                        "chunk_text": hit.entity.get('chunk_text', ''),
                        "score": float(hit.score),
                        "semantic_score": float(hit.score),
                        "source": "semantic",
                        "file_type": hit.entity.get('file_type', ''),
                        "tags": hit.entity.get('tags', []),
                        "created_at": hit.entity.get('created_at', ''),
                        "updated_at": hit.entity.get('updated_at', '')
                    }
                    filtered_results.append(result)
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"의미적 검색 중 오류: {e}")
            return []
