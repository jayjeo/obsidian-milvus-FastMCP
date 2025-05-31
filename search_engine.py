#!/usr/bin/env python3
"""
기본 검색 엔진 클래스 - Milvus의 기능을 활용한 하이브리드 검색
기존 milvus_manager.py의 고급 기능들과 연동
"""

import time
import re
from typing import List, Dict, Any, Optional, Tuple
from embeddings import EmbeddingModel
import config

# Import centralized logging system
from logger import get_logger

# Get logger for this module
logger = get_logger('search_engine')

class SearchEngine:
    def __init__(self, milvus_manager):
        self.milvus_manager = milvus_manager
        self.embedding_model = EmbeddingModel()
        self.recent_queries = []  # 최근 쿼리 캐시
        
    def ensure_json_serializable(self, obj):
        """객체가 JSON 직렬화 가능하도록 변환"""
        if isinstance(obj, dict):
            return {k: self.ensure_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.ensure_json_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
        
    def hybrid_search(self, query: str, limit: int = 10, filter_params: Optional[Dict] = None) -> Tuple[List[Dict], Dict]:
        """하이브리드 검색: 벡터 검색 + 키워드 검색 결합"""
        logger.info(f"Starting hybrid search for query: '{query}' with limit: {limit}")
        
        # 특수문자 확인 - 잠재적 문제 사전 감지
        has_special_chars = any(c in query for c in "'\"()[]{},;")
        if has_special_chars:
            logger.debug(f"Query contains special characters that might require special handling: '{query}'")
            
        try:
            start_time = time.time()
            
            # 최근 쿼리에 추가
            self.recent_queries.append(query)
            if len(self.recent_queries) > 50:
                self.recent_queries = self.recent_queries[-50:]
                logger.debug("Trimmed recent queries cache to 50 items")
            
            # 1. 벡터 검색 수행
            logger.debug(f"Performing vector search with doubled limit: {limit * 2}")
            vector_start = time.time()
            vector_results = self._vector_search(query, limit * 2, filter_params)
            logger.debug(f"Vector search completed in {time.time() - vector_start:.3f}s with {len(vector_results)} results")
            
            # 2. 키워드 검색 수행
            logger.debug(f"Performing keyword search with limit: {limit}")
            keyword_start = time.time()
            # 필터 표현식 구성
            filter_expr = self._build_filter_expr(filter_params) if filter_params else None
            if filter_expr:
                logger.debug(f"Using filter expression: {filter_expr}")
                
            keyword_results = self._keyword_search(query, limit, filter_expr)
            logger.debug(f"Keyword search completed in {time.time() - keyword_start:.3f}s with {len(keyword_results)} results")
            
            # 3. 결과 육합
            logger.debug("Fusing search results from vector and keyword searches")
            fusion_start = time.time()
            fused_results = self._fuse_search_results(vector_results, keyword_results, query)
            logger.debug(f"Results fusion completed in {time.time() - fusion_start:.3f}s with {len(fused_results)} combined results")
            
            # 4. 상위 결과만 반환
            final_results = fused_results[:limit]
            
            search_time = time.time() - start_time
            logger.info(f"Hybrid search completed in {search_time:.3f}s with {len(final_results)} final results")
            
            search_info = {
                "query": query,
                "search_type": "hybrid",
                "vector_results_count": len(vector_results),
                "keyword_results_count": len(keyword_results),
                "final_results_count": len(final_results),
                "search_time_ms": round(search_time * 1000, 2),
                "total_count": len(final_results)
            }
            
            # 결과를 JSON 직렬화 가능하게 변환
            logger.debug("Ensuring results are JSON serializable")
            return self.ensure_json_serializable(final_results), self.ensure_json_serializable(search_info)
            
        except Exception as e:
            logger.error(f"하이브리드 검색 중 오류: {e}", exc_info=True)
            logger.warning("Falling back to vector search only")
            # 폴백: 벡터 검색만 수행
            try:
                logger.debug("Attempting vector search fallback")
                fallback_start = time.time()
                vector_results = self._vector_search(query, limit, filter_params)
                fallback_time = time.time() - fallback_start
                
                logger.info(f"Vector search fallback completed successfully in {fallback_time:.3f}s with {len(vector_results)} results")
                
                search_info = {
                    "query": query,
                    "search_type": "vector_fallback",
                    "error": str(e),
                    "fallback_time_ms": round(fallback_time * 1000, 2),
                    "total_count": len(vector_results)
                }
                return self.ensure_json_serializable(vector_results), self.ensure_json_serializable(search_info)
            except Exception as e2:
                logger.error(f"폴백 검색도 실패: {e2}", exc_info=True)
                # 모든 검색 방법이 실패한 경우 빈 결과 반환
                logger.critical(f"All search methods failed for query: '{query}'")
                return [], self.ensure_json_serializable({"query": query, "error": str(e2), "total_count": 0})
    
    def _vector_search(self, query: str, limit: int, filter_params: Optional[Dict] = None) -> List[Dict]:
        """벡터 유사도 검색"""
        logger.debug(f"Starting vector search for query: '{query}' with limit: {limit}")
        
        # 특수문자 확인
        has_special_chars = any(c in query for c in "'\"()[]{},;")
        if has_special_chars:
            logger.debug(f"Vector search query contains special characters: '{query}'")
            
        try:
            # 쿼리를 벡터로 변환
            embedding_start = time.time()
            logger.debug("Generating embedding for query")
            query_vector = self.embedding_model.get_embedding(query)
            logger.debug(f"Query embedding generated in {time.time() - embedding_start:.3f}s")
            
            # 필터 표현식 구성
            filter_expr = self._build_filter_expr(filter_params) if filter_params else None
            if filter_expr:
                logger.debug(f"Using filter expression for vector search: {filter_expr}")
            
            # milvus_manager의 고급 검색 기능 사용
            search_start = time.time()
            if hasattr(self.milvus_manager, 'search_with_params'):
                # 최적화된 검색 파라미터 사용
                logger.debug("Using advanced search_with_params method")
                search_params = self._get_optimized_search_params(query)
                logger.debug(f"Optimized search parameters: {search_params}")
                
                raw_results = self.milvus_manager.search_with_params(
                    vector=query_vector,
                    limit=limit,
                    filter_expr=filter_expr,
                    search_params=search_params
                )
            else:
                # 기본 검색 사용
                logger.debug("Using basic search method")
                raw_results = self.milvus_manager.search(query_vector, limit, filter_expr)
                
            search_duration = time.time() - search_start
            logger.debug(f"Milvus search completed in {search_duration:.3f}s with {len(raw_results)} raw results")
            
            # 결과 포맷팅
            format_start = time.time()
            logger.debug("Formatting search results")
            formatted_results = []
            format_errors = 0
            
            for hit in raw_results:
                try:
                    # 안전하게 값 추출
                    try:
                        hit_id = hit.id
                    except Exception as e:
                        logger.debug(f"Could not extract ID directly, using fallback: {e}")
                        hit_id = str(getattr(hit, 'id', 'unknown_id'))
                        
                    # entity가 딕셔너리가 아닐 경우 대비
                    entity = getattr(hit, 'entity', {})
                    if not isinstance(entity, dict):
                        logger.debug(f"Entity is not a dictionary, converting type: {type(entity)}")
                        entity = {}
                        
                    # 점수가 직렬화 가능한지 확인
                    try:
                        score = float(hit.score)
                    except Exception as e:
                        logger.debug(f"Could not convert score to float: {e}")
                        score = 0.0
                    
                    result = {
                        "id": hit_id,
                        "path": entity.get('path', ''),
                        "title": entity.get('title', '제목 없음'),
                        "chunk_text": entity.get('chunk_text', ''),
                        "content": entity.get('content', ''),
                        "score": score,
                        "source": "vector",
                        "file_type": entity.get('file_type', ''),
                        "tags": entity.get('tags', []),
                        "chunk_index": entity.get('chunk_index', 0),
                        "created_at": entity.get('created_at', ''),
                        "updated_at": entity.get('updated_at', '')
                    }
                    formatted_results.append(result)
                except Exception as e:
                    logger.error(f"벡터 결과 포맷팅 오류: {e}", exc_info=True)
                    format_errors += 1
                    continue
            
            format_duration = time.time() - format_start
            logger.debug(f"Results formatting completed in {format_duration:.3f}s - {len(formatted_results)} success, {format_errors} errors")
            
            # 결과를 JSON 직렬화 가능하게 변환
            logger.debug(f"Returning {len(formatted_results)} vector search results")
            return self.ensure_json_serializable(formatted_results)
            
        except Exception as e:
            logger.error(f"벡터 검색 중 오류: {e}", exc_info=True)
            return []
    
    def _keyword_search(self, query: str, limit: int, filter_expr: Optional[str] = None) -> List[Dict]:
        """키워드 기반 검색 (메모리에서 필터링)"""
        logger.debug(f"Starting keyword search for query: '{query}' with limit: {limit}")
        
        # 특수문자 확인
        has_special_chars = any(c in query for c in "'\"()[]{},;")
        if has_special_chars:
            logger.debug(f"Keyword search query contains special characters: '{query}'")
        
        # 쿼리 토큰화
        tokens = re.findall(r'\w+', query.lower())
        logger.debug(f"Extracted {len(tokens)} search tokens: {tokens}")
        
        if not tokens:
            logger.warning("No valid search tokens found in query")
            return []
        try:
            # 전체 문서 조회 (페이지네이션 사용)
            query_start = time.time()
            all_docs = []
            offset = 0
            batch_size = 1000
            batch_count = 0
            
            logger.debug(f"Starting batched document retrieval with batch size {batch_size}")
            if filter_expr:
                logger.debug(f"Using filter expression: {filter_expr}")
            
            while True:
                batch_start = time.time()
                logger.debug(f"Retrieving batch at offset {offset}")
                
                # 특수문자 확인 - 인용 부호 등이 잘못되면 오류 발생 가능
                query_expr = filter_expr or "id >= 0"
                if has_special_chars and "path" in query_expr:
                    logger.debug(f"Query with potential special characters in path filter: {query_expr}")
                
                batch = self.milvus_manager.query(
                    expr=query_expr,
                    output_fields=["id", "path", "title", "chunk_text", "content", "file_type", "tags", "chunk_index", "created_at", "updated_at"],
                    limit=batch_size,
                    offset=offset
                )
                
                batch_count += 1
                logger.debug(f"Batch {batch_count} retrieved {len(batch)} documents in {time.time() - batch_start:.3f}s")
                
                if not batch:
                    logger.debug("Empty batch received, ending retrieval")
                    break
                    
                all_docs.extend(batch)
                offset += batch_size
                
                if len(batch) < batch_size:
                    logger.debug(f"Retrieved partial batch ({len(batch)}/{batch_size}), ending retrieval")
                    break
            
            query_retrieval_time = time.time() - query_start
            logger.info(f"Document retrieval completed: {len(all_docs)} documents in {query_retrieval_time:.3f}s")
            
            # 키워드 점수 계산
            scoring_start = time.time()
            logger.debug("Starting keyword scoring process")
            
            # 한글 포함 토큰화 패턴
            query_terms = re.findall(r'[\w가-힣]+', query.lower())
            logger.debug(f"Using {len(query_terms)} query terms for scoring: {query_terms}")
            
            scored_results = []
            special_char_paths = 0
            excalidraw_files = 0
            
            for doc in all_docs:
                # 특수 문자 경로 검색 - 이전에 문제를 일으켰던 부분
                path = doc.get('path', '')
                has_special_path = any(c in path for c in "'\"()[]{},;")
                is_excalidraw = "excalidraw" in path.lower()
                
                if has_special_path:
                    special_char_paths += 1
                    if len(special_char_paths) <= 5:  # 로그 과도화 방지
                        logger.debug(f"Processing document with special characters in path: {path}")
                        
                if is_excalidraw:
                    excalidraw_files += 1
                    if excalidraw_files <= 5:  # 로그 과도화 방지
                        logger.debug(f"Processing Excalidraw file: {path}")
                
                # 점수 계산
                score = self._calculate_keyword_score(doc, query_terms)
                if score > 0:
                    result = {
                        "id": doc.get('id', ''),
                        "path": path,
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
            
            scoring_time = time.time() - scoring_start
            logger.debug(f"Keyword scoring completed in {scoring_time:.3f}s - found {len(scored_results)} relevant documents")
            
            if special_char_paths > 0:
                logger.info(f"Processed {special_char_paths} documents with special characters in paths")
            if excalidraw_files > 0:
                logger.info(f"Processed {excalidraw_files} Excalidraw files")
            
            # 점수 순으로 정렬하고 상위 결과 반환
            logger.debug("Sorting results by score")
            scored_results.sort(key=lambda x: x['score'], reverse=True)
            limited_results = scored_results[:limit]
            
            total_time = time.time() - query_start
            logger.info(f"Keyword search completed in {total_time:.3f}s with {len(limited_results)} final results")
            
            # 결과를 JSON 직렬화 가능하게 변환
            logger.debug("Ensuring results are JSON serializable")
            return self.ensure_json_serializable(limited_results)
            
        except Exception as e:
            # 특수 문자로 인한 오류인지 확인
            if has_special_chars:
                logger.error(f"키워드 검색 중 오류 (특수 문자 지정 쿼리): {e}", exc_info=True)
                logger.warning("Special characters in query may have caused the error")
            else:
                logger.error(f"키워드 검색 중 오류: {e}", exc_info=True)
            return []
    
    def _calculate_keyword_score(self, doc: Dict, query_terms: List[str]) -> float:
        """키워드 매칭 점수 계산"""
        # 특수 문자 경로 처리를 위한 확인
        path = doc.get('path', '').lower()
        doc_id = doc.get('id', 'unknown_id')
        has_special_path = any(c in path for c in "'\"()[]{},;")
        is_excalidraw = "excalidraw" in path.lower()
        
        # 고유한 로깅은 DEBUG 레벨에서만 필요
        if has_special_path or is_excalidraw:
            logger.debug(f"Calculating score for document with special path: {doc_id}: {path}")
            
        score = 0.0
        
        title = doc.get('title', '').lower()
        content = doc.get('chunk_text', '').lower()
        
        term_matches = []
        
        for term in query_terms:
            term_score = 0.0
            
            # 경로에서 매칭 (높은 점수)
            if term in path:
                term_score += 5.0
                term_matches.append(f"{term}(path:+5.0)")
            
            # 제목에서 매칭 (중간 점수)
            if term in title:
                term_score += 3.0
                term_matches.append(f"{term}(title:+3.0)")
                
            # 내용에서 매칭 (기본 점수)
            content_matches = content.count(term)
            if content_matches > 0:
                content_score = content_matches * 1.0
                term_score += content_score
                term_matches.append(f"{term}(content:{content_matches}:+{content_score:.1f})")
                
            score += term_score
            
        # 많은 로깅은 피해야 하지만, 특수한 경로는 로깅
        if has_special_path or is_excalidraw:
            if score > 0:
                logger.debug(f"Document {doc_id} with special path scored {score:.2f} - matches: {', '.join(term_matches)}")
            else:
                logger.debug(f"Document {doc_id} with special path had no matches")
        
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
                try:
                    if hit.score >= similarity_threshold:
                        # 안전하게 값 추출
                        try:
                            hit_id = hit.id
                        except:
                            hit_id = str(getattr(hit, 'id', 'unknown_id'))
                            
                        # entity가 딕셔너리가 아닐 경우 대비
                        entity = getattr(hit, 'entity', {})
                        if not isinstance(entity, dict):
                            entity = {}
                            
                        # 점수가 직렬화 가능한지 확인
                        try:
                            score = float(hit.score)
                        except:
                            score = 0.0
                            
                        result = {
                            "id": hit_id,
                            "path": entity.get('path', ''),
                            "title": entity.get('title', '제목 없음'),
                            "chunk_text": entity.get('chunk_text', ''),
                            "score": score,
                            "semantic_score": score,
                            "source": "semantic",
                            "file_type": entity.get('file_type', ''),
                            "chunk_index": entity.get('chunk_index', 0),
                            "tags": entity.get('tags', []),
                            "created_at": entity.get('created_at', ''),
                            "updated_at": entity.get('updated_at', '')
                        }
                        filtered_results.append(result)
                except Exception as e:
                    logger.error(f"의미적 검색 결과 포맷팅 오류: {e}")
                    continue
            
            # 결과를 JSON 직렬화 가능하게 변환
            return self.ensure_json_serializable(filtered_results)
            
        except Exception as e:
            logger.error(f"의미적 검색 중 오류: {e}")
            return []
