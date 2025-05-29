#!/usr/bin/env python3
"""
향상된 검색 엔진 - 기존 milvus_manager.py의 고급 기능들과 통합
Milvus의 메타데이터 필터링을 최대한 활용하는 향상된 검색 엔진
"""

import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from search_engine import SearchEngine
import config

# Import centralized logger
from logger import get_logger

# Initialize module logger
logger = get_logger(__name__)

class EnhancedSearchEngine(SearchEngine):
    def __init__(self, milvus_manager):
        super().__init__(milvus_manager)
        self.metadata_cache = {}
        
    def advanced_filter_search(self, query, **filters):
        """고급 메타데이터 필터 검색 - milvus_manager의 고급 기능 활용"""
        search_start = time.time()
        logger.info(f"Starting advanced filter search for query: '{query}' with {len(filters)} filters")
        logger.debug(f"Filter parameters: {filters}")
        
        try:
            # milvus_manager의 고급 메타데이터 검색 기능 사용
            embedding_start = time.time()
            query_vector = self.embedding_model.get_embedding(query)
            logger.debug(f"Generated query embedding in {time.time() - embedding_start:.3f}s")
            
            if hasattr(self.milvus_manager, 'advanced_metadata_search'):
                # 기존 고급 메타데이터 검색 사용
                logger.debug("Using advanced_metadata_search method")
                search_start_time = time.time()
                raw_results = self.milvus_manager.advanced_metadata_search(query_vector, filters)
                logger.debug(f"Advanced metadata search completed in {time.time() - search_start_time:.3f}s")
            else:
                # 폴백: 일반 검색으로 대체
                logger.debug("Falling back to standard search with filter expression")
                filter_expr = self._build_complex_filter_expr(filters)
                limit = filters.get('limit', 20)
                
                search_start_time = time.time()
                if hasattr(self.milvus_manager, 'search_with_params'):
                    logger.debug(f"Using search_with_params method with limit {limit}")
                    raw_results = self.milvus_manager.search_with_params(
                        vector=query_vector,
                        limit=limit,
                        filter_expr=filter_expr
                    )
                else:
                    logger.debug(f"Using standard search method with limit {limit}")
                    raw_results = self.milvus_manager.search(
                        query_vector,
                        limit,
                        filter_expr
                    )
                logger.debug(f"Search completed in {time.time() - search_start_time:.3f}s")
            
            format_start = time.time()
            results = self._format_enhanced_results(raw_results, query)
            logger.debug(f"Formatted {len(results)} results in {time.time() - format_start:.3f}s")
            
            total_time = time.time() - search_start
            logger.info(f"Advanced filter search completed in {total_time:.3f}s with {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"고급 필터 검색 오류: {e}", exc_info=True)
            return []
    
    def _build_complex_filter_expr(self, filters):
        """복잡한 필터 조건을 Milvus 표현식으로 변환"""
        logger.debug(f"Building complex filter expression from {len(filters)} filter parameters")
        filter_expressions = []
        
        # 시간 기반 필터링
        if filters.get('time_range'):
            start, end = filters['time_range']
            time_expr = f"created_at >= {start} && created_at <= {end}"
            filter_expressions.append(time_expr)
            logger.debug(f"Added time range filter: {time_expr}")
        
        # 파일 크기 필터링
        if filters.get('file_size_range'):
            min_size, max_size = filters['file_size_range']
            size_expr = f"file_size >= {min_size} && file_size <= {max_size}"
            filter_expressions.append(size_expr)
            logger.debug(f"Added file size filter: {size_expr}")
        
        # 콘텐츠 품질 필터링
        if filters.get('min_content_quality'):
            quality = filters['min_content_quality']
            quality_expr = f"content_quality >= {quality}"
            filter_expressions.append(quality_expr)
            logger.debug(f"Added content quality filter: {quality_expr}")
        
        # 관련성 점수 필터링
        if filters.get('min_relevance_score'):
            score = filters['min_relevance_score']
            score_expr = f"user_relevance_score >= {score}"
            filter_expressions.append(score_expr)
            logger.debug(f"Added relevance score filter: {score_expr}")
        
        # 복합 태그 필터링
        if filters.get('tag_logic'):
            logger.debug("Building complex tag filter expression")
            tag_expr = self._build_complex_tag_filter(filters['tag_logic'])
            if tag_expr:
                filter_expressions.append(tag_expr)
                logger.debug(f"Added tag logic filter: {tag_expr}")
        
        final_expr = " && ".join(filter_expressions) if filter_expressions else None
        logger.info(f"Built filter expression with {len(filter_expressions)} conditions: {final_expr}")
        return final_expr
    
    def _build_complex_tag_filter(self, tag_logic):
        """복잡한 태그 로직 구성"""
        logger.debug(f"Building complex tag filter from tag logic: {tag_logic}")
        
        if not tag_logic:
            logger.debug("Empty tag logic provided, returning None")
            return None
        
        expressions = []
        
        # AND 조건
        if tag_logic.get('and'):
            and_tags = tag_logic['and']
            logger.debug(f"Processing AND condition with {len(and_tags)} tags: {and_tags}")
            and_expressions = []
            for tag in and_tags:
                and_expressions.append(f"tags like '%{tag}%'")
            if and_expressions:
                and_expr = f"({' && '.join(and_expressions)})"
                expressions.append(and_expr)
                logger.debug(f"Added AND expression: {and_expr}")
        
        # OR 조건
        if tag_logic.get('or'):
            or_tags = tag_logic['or']
            logger.debug(f"Processing OR condition with {len(or_tags)} tags: {or_tags}")
            or_expressions = []
            for tag in or_tags:
                or_expressions.append(f"tags like '%{tag}%'")
            if or_expressions:
                or_expr = f"({' || '.join(or_expressions)})"
                expressions.append(or_expr)
                logger.debug(f"Added OR expression: {or_expr}")
        
        # NOT 조건
        if tag_logic.get('not'):
            not_tags = tag_logic['not']
            logger.debug(f"Processing NOT condition with {len(not_tags)} tags: {not_tags}")
            not_expressions = []
            for tag in not_tags:
                not_expressions.append(f"!(tags like '%{tag}%')")
            if not_expressions:
                not_expr = f"({' && '.join(not_expressions)})"
                expressions.append(not_expr)
                logger.debug(f"Added NOT expression: {not_expr}")
        
        final_expr = " && ".join(expressions) if expressions else None
        logger.info(f"Built tag filter expression with {len(expressions)} conditions: {final_expr}")
        return final_expr
    
    def multi_modal_search(self, query, include_attachments=True):
        """다중 모달 검색 (텍스트 + 첨부파일)"""
        search_start = time.time()
        logger.info(f"Starting multi-modal search for query: '{query}' with include_attachments={include_attachments}")
        
        try:
            # 기본 하이브리드 검색
            logger.debug("Performing base hybrid text search")
            hybrid_start = time.time()
            text_results, stats = self.hybrid_search(query)
            logger.debug(f"Hybrid text search completed in {time.time() - hybrid_start:.3f}s with {len(text_results)} results")
            logger.debug(f"Hybrid search stats: {stats}")
            
            if include_attachments:
                # 첨부파일 필터링
                logger.debug("Performing attachment search")
                attachment_filter = "file_type in ['pdf', 'docx', 'pptx', 'xlsx']"
                logger.debug(f"Using attachment filter: {attachment_filter}")
                
                embedding_start = time.time()
                query_vector = self.embedding_model.get_embedding(query)
                logger.debug(f"Generated attachment query embedding in {time.time() - embedding_start:.3f}s")
                
                attach_search_start = time.time()
                if hasattr(self.milvus_manager, 'search_with_params'):
                    logger.debug("Using search_with_params method for attachment search")
                    attachment_results = self.milvus_manager.search_with_params(
                        vector=query_vector,
                        limit=20,
                        filter_expr=attachment_filter
                    )
                else:
                    logger.debug("Using standard search method for attachment search")
                    attachment_results = self.milvus_manager.search(
                        query_vector, 20, attachment_filter
                    )
                logger.debug(f"Attachment search completed in {time.time() - attach_search_start:.3f}s")
                
                # 첨부파일 결과 변환
                format_start = time.time()
                attachment_results = self._format_enhanced_results(attachment_results, query)
                logger.debug(f"Formatted {len(attachment_results)} attachment results in {time.time() - format_start:.3f}s")
                
                # 첨부파일 결과에 태그 붙이기
                for result in attachment_results:
                    result['source'] = 'attachment'
                logger.debug(f"Tagged {len(attachment_results)} results as 'attachment'")
                
                # 텍스트와 첨부파일 결과 합치기
                combined_results = text_results + attachment_results
                logger.debug(f"Combined {len(text_results)} text results with {len(attachment_results)} attachment results")
                
                # 점수순 정렬
                sort_start = time.time()
                combined_results.sort(key=lambda x: x['score'], reverse=True)
                logger.debug(f"Sorted {len(combined_results)} combined results in {time.time() - sort_start:.3f}s")
                
                total_time = time.time() - search_start
                logger.info(f"Multi-modal search completed in {total_time:.3f}s with {len(combined_results)} total results ({len(text_results)} text, {len(attachment_results)} attachments)")
                return combined_results
            
            total_time = time.time() - search_start
            logger.info(f"Text-only search completed in {total_time:.3f}s with {len(text_results)} results")
            return text_results
            
        except Exception as e:
            total_time = time.time() - search_start
            logger.error(f"다중 모달 검색 오류 ({total_time:.3f}s 후 실패): {e}", exc_info=True)
            return []
    
    def contextual_search(self, query, context_docs=None, expand_context=True):
        """컨텍스트 기반 확장 검색"""
        try:
            search_vectors = [self.embedding_model.get_embedding(query)]
            
            # 컨텍스트 문서들의 벡터 추가
            if context_docs:
                for doc_id in context_docs:
                    doc_vector = self._get_document_vector(doc_id)
                    if doc_vector:
                        search_vectors.append(doc_vector)
            
            # 멀티 벡터 검색
            all_results = []
            for vector in search_vectors:
                if hasattr(self.milvus_manager, 'search_with_params'):
                    results = self.milvus_manager.search_with_params(
                        vector=vector,
                        limit=50,
                        search_params={"metric_type": "COSINE", "params": {"ef": 128}}
                    )
                else:
                    results = self.milvus_manager.search(vector, 50)
                
                all_results.extend(results)
            
            # 중복 제거 및 컨텍스트 점수 계산
            unique_results = self._deduplicate_and_score_context(all_results, query)
            
            if expand_context:
                # 관련 문서들의 주변 청크 포함
                expanded_results = self._expand_context_chunks(unique_results)
                return expanded_results
            
            return unique_results
            
        except Exception as e:
            logger.error(f"컨텍스트 검색 오류: {e}")
            return []
    
    def _get_document_vector(self, doc_id):
        """문서 ID로 벡터 조회"""
        try:
            results = self.milvus_manager.query(
                expr=f"id == {doc_id}",
                output_fields=["vector"],
                limit=1
            )
            if results and "vector" in results[0]:
                return results[0]["vector"]
        except Exception as e:
            logger.error(f"문서 벡터 조회 오류: {e}")
        return None
    
    def _deduplicate_and_score_context(self, all_results, query):
        """중복 제거 및 컨텍스트 점수 계산"""
        result_map = {}
        
        for hit in all_results:
            doc_id = hit.id
            if doc_id not in result_map:
                result_map[doc_id] = {
                    "id": doc_id,
                    "path": hit.entity.get('path', ''),
                    "title": hit.entity.get('title', '제목 없음'),
                    "chunk_text": hit.entity.get('chunk_text', ''),
                    "score": float(hit.score),
                    "source": "contextual",
                    "file_type": hit.entity.get('file_type', ''),
                    "tags": hit.entity.get('tags', []),
                    "created_at": hit.entity.get('created_at', ''),
                    "updated_at": hit.entity.get('updated_at', ''),
                    "context_scores": [float(hit.score)]
                }
            else:
                result_map[doc_id]["context_scores"].append(float(hit.score))
        
        # 컨텍스트 점수 계산
        for doc_id, result in result_map.items():
            context_scores = result["context_scores"]
            avg_score = sum(context_scores) / len(context_scores)
            frequency_bonus = min(len(context_scores) * 0.1, 0.3)
            result["final_score"] = avg_score + frequency_bonus
            result["score"] = result["final_score"]
            del result["context_scores"]
            del result["final_score"]
        
        # 점수로 정렬
        unique_results = list(result_map.values())
        unique_results.sort(key=lambda x: x["score"], reverse=True)
        
        return unique_results
    
    def _expand_context_chunks(self, results):
        """관련 문서들의 주변 청크 포함"""
        expanded_results = []
        
        for result in results[:10]:  # 상위 10개만 확장
            expanded_results.append(result)
            
            # 같은 문서의 인접 청크들 찾기
            try:
                path = result["path"]
                adjacent_chunks = self.milvus_manager.query(
                    expr=f'path == "{path}"',
                    output_fields=["id", "path", "title", "chunk_text", "chunk_index", "file_type", "tags", "created_at", "updated_at"],
                    limit=50
                )
                
                # 청크 인덱스로 정렬
                adjacent_chunks.sort(key=lambda x: x.get("chunk_index", 0))
                
                for chunk in adjacent_chunks:
                    if chunk["id"] != result["id"]:
                        chunk_result = {
                            "id": chunk["id"],
                            "path": chunk["path"],
                            "title": chunk["title"],
                            "chunk_text": chunk["chunk_text"],
                            "score": result["score"] * 0.8,
                            "source": "adjacent_chunk",
                            "file_type": chunk["file_type"],
                            "tags": chunk["tags"],
                            "created_at": chunk["created_at"],
                            "updated_at": chunk["updated_at"]
                        }
                        expanded_results.append(chunk_result)
                        
            except Exception as e:
                logger.error(f"청크 확장 중 오류: {e}")
                continue
        
        return expanded_results
    
    def _merge_multimodal_results(self, text_results, attachment_results):
        """다중 모달 결과 병합"""
        # 텍스트 결과에 높은 가중치
        for result in text_results:
            result["multimodal_score"] = result["score"] * 1.0
        
        # 첨부파일 결과에 중간 가중치
        for result in attachment_results:
            result["multimodal_score"] = result["score"] * 0.8
        
        # 모든 결과 병합
        all_results = text_results + attachment_results
        
        # 다중 모달 점수로 정렬
        all_results.sort(key=lambda x: x["multimodal_score"], reverse=True)
        
        return all_results
    
    def _format_enhanced_results(self, results, query):
        """향상된 결과 포맷팅"""
        formatted_results = []
        for hit in results:
            result = {
                "id": hit.id,
                "path": hit.entity.get('path', ''),
                "title": hit.entity.get('title', '제목 없음'),
                "chunk_text": hit.entity.get('chunk_text', ''),
                "score": float(hit.score),
                "source": "enhanced",
                "file_type": hit.entity.get('file_type', ''),
                "tags": hit.entity.get('tags', []),
                "created_at": hit.entity.get('created_at', ''),
                "updated_at": hit.entity.get('updated_at', ''),
                "relevance_explanation": self._generate_relevance_explanation(hit, query)
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def _generate_relevance_explanation(self, hit, query):
        """관련성 설명 생성"""
        explanation_parts = []
        
        # 제목 매칭
        title = hit.entity.get('title', '')
        if any(word.lower() in title.lower() for word in query.split()):
            explanation_parts.append("제목에서 쿼리 키워드 발견")
        
        # 콘텐츠 매칭
        content = hit.entity.get('chunk_text', '')
        query_words = query.lower().split()
        content_matches = sum(1 for word in query_words if word in content.lower())
        if content_matches > 0:
            explanation_parts.append(f"콘텐츠에서 {content_matches}개 키워드 매칭")
        
        # 점수 기반 설명
        score = float(hit.score)
        if score > 0.9:
            explanation_parts.append("매우 높은 의미적 유사도")
        elif score > 0.7:
            explanation_parts.append("높은 의미적 유사도")
        elif score > 0.5:
            explanation_parts.append("중간 의미적 유사도")
        
        return " | ".join(explanation_parts) if explanation_parts else "벡터 유사도 기반 매칭"
