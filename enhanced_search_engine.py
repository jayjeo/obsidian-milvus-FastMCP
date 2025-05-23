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
import logging

logger = logging.getLogger('EnhancedSearchEngine')

class EnhancedSearchEngine(SearchEngine):
    def __init__(self, milvus_manager):
        super().__init__(milvus_manager)
        self.metadata_cache = {}
        
    def advanced_filter_search(self, query, **filters):
        """고급 메타데이터 필터 검색 - milvus_manager의 고급 기능 활용"""
        try:
            # milvus_manager의 고급 메타데이터 검색 기능 사용
            query_vector = self.embedding_model.get_embedding(query)
            
            if hasattr(self.milvus_manager, 'advanced_metadata_search'):
                # 기존 고급 메타데이터 검색 사용
                raw_results = self.milvus_manager.advanced_metadata_search(query_vector, filters)
            else:
                # 폴백: 일반 검색으로 대체
                filter_expr = self._build_complex_filter_expr(filters)
                if hasattr(self.milvus_manager, 'search_with_params'):
                    raw_results = self.milvus_manager.search_with_params(
                        vector=query_vector,
                        limit=filters.get('limit', 20),
                        filter_expr=filter_expr
                    )
                else:
                    raw_results = self.milvus_manager.search(
                        query_vector,
                        filters.get('limit', 20),
                        filter_expr
                    )
            
            return self._format_enhanced_results(raw_results, query)
            
        except Exception as e:
            logger.error(f"고급 필터 검색 오류: {e}")
            return []
    
    def _build_complex_filter_expr(self, filters):
        """복잡한 필터 조건을 Milvus 표현식으로 변환"""
        filter_expressions = []
        
        # 시간 기반 필터링
        if filters.get('time_range'):
            start, end = filters['time_range']
            filter_expressions.append(f"created_at >= {start} && created_at <= {end}")
        
        # 파일 크기 필터링
        if filters.get('file_size_range'):
            min_size, max_size = filters['file_size_range']
            filter_expressions.append(f"file_size >= {min_size} && file_size <= {max_size}")
        
        # 콘텐츠 품질 필터링
        if filters.get('min_content_quality'):
            quality = filters['min_content_quality']
            filter_expressions.append(f"content_quality >= {quality}")
        
        # 관련성 점수 필터링
        if filters.get('min_relevance_score'):
            score = filters['min_relevance_score']
            filter_expressions.append(f"user_relevance_score >= {score}")
        
        # 복합 태그 필터링
        if filters.get('tag_logic'):
            tag_expr = self._build_complex_tag_filter(filters['tag_logic'])
            if tag_expr:
                filter_expressions.append(tag_expr)
        
        return " && ".join(filter_expressions) if filter_expressions else None
    
    def _build_complex_tag_filter(self, tag_logic):
        """복잡한 태그 로직 구성"""
        if not tag_logic:
            return None
        
        expressions = []
        
        # AND 조건
        if tag_logic.get('and'):
            and_expressions = []
            for tag in tag_logic['and']:
                and_expressions.append(f"tags like '%{tag}%'")
            if and_expressions:
                expressions.append(f"({' && '.join(and_expressions)})")
        
        # OR 조건
        if tag_logic.get('or'):
            or_expressions = []
            for tag in tag_logic['or']:
                or_expressions.append(f"tags like '%{tag}%'")
            if or_expressions:
                expressions.append(f"({' || '.join(or_expressions)})")
        
        # NOT 조건
        if tag_logic.get('not'):
            not_expressions = []
            for tag in tag_logic['not']:
                not_expressions.append(f"!(tags like '%{tag}%')")
            if not_expressions:
                expressions.append(f"({' && '.join(not_expressions)})")
        
        return " && ".join(expressions) if expressions else None
    
    def multi_modal_search(self, query, include_attachments=True):
        """다중 모달 검색 (텍스트 + 첨부파일)"""
        try:
            # 기본 하이브리드 검색
            text_results, _ = self.hybrid_search(query)
            
            if include_attachments:
                # 첨부파일 필터링
                attachment_filter = "file_type in ['pdf', 'docx', 'pptx', 'xlsx']"
                query_vector = self.embedding_model.get_embedding(query)
                
                if hasattr(self.milvus_manager, 'search_with_params'):
                    attachment_results = self.milvus_manager.search_with_params(
                        vector=query_vector,
                        limit=20,
                        filter_expr=attachment_filter
                    )
                else:
                    attachment_results = self.milvus_manager.search(
                        query_vector, 20, attachment_filter
                    )
                
                # 첨부파일 결과 포맷팅
                formatted_attachment_results = []
                for hit in attachment_results:
                    result = {
                        "id": hit.id,
                        "path": hit.entity.get('path', ''),
                        "title": hit.entity.get('title', '제목 없음'),
                        "chunk_text": hit.entity.get('chunk_text', ''),
                        "score": float(hit.score),
                        "source": "attachment",
                        "file_type": hit.entity.get('file_type', ''),
                        "tags": hit.entity.get('tags', []),
                        "created_at": hit.entity.get('created_at', ''),
                        "updated_at": hit.entity.get('updated_at', '')
                    }
                    formatted_attachment_results.append(result)
                
                # 결과 병합
                combined_results = self._merge_multimodal_results(
                    text_results, formatted_attachment_results
                )
                return combined_results
            
            return text_results
            
        except Exception as e:
            logger.error(f"다중 모달 검색 오류: {e}")
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
