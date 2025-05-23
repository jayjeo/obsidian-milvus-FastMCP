#!/usr/bin/env python3
"""
고급 RAG 엔진 - 기존 milvus_manager.py의 지식 그래프 기능과 통합
Milvus의 강력한 기능을 활용한 고급 RAG 패턴
"""

import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import defaultdict
import config

logger = logging.getLogger('AdvancedRAGEngine')

class AdvancedRAGEngine:
    def __init__(self, milvus_manager, search_engine):
        self.milvus_manager = milvus_manager
        self.search_engine = search_engine
        
    def hierarchical_retrieval(self, query, max_depth=3):
        """계층적 검색 - 문서 → 섹션 → 청크"""
        
        # 1단계: 문서 수준 검색
        doc_results = self._search_by_level(query, level="document", limit=10)
        
        # 2단계: 관련 문서의 섹션 검색  
        section_results = []
        for doc in doc_results[:5]:  # 상위 5개 문서
            try:
                sections = self._search_by_level(
                    query, 
                    level="section", 
                    filter_expr=f"path like '{doc['path']}%'",
                    limit=5
                )
                section_results.extend(sections)
            except Exception as e:
                logger.error(f"섹션 검색 오류: {e}")
                continue
        
        # 3단계: 관련 섹션의 상세 청크 검색
        chunk_results = []
        for section in section_results[:10]:  # 상위 10개 섹션
            try:
                chunks = self._search_by_level(
                    query,
                    level="chunk", 
                    filter_expr=f"path == '{section['path']}'",
                    limit=3
                )
                chunk_results.extend(chunks)
            except Exception as e:
                logger.error(f"청크 검색 오류: {e}")
                continue
        
        return {
            "documents": doc_results,
            "sections": section_results, 
            "chunks": chunk_results,
            "hierarchy_path": self._build_hierarchy_path(chunk_results)
        }
    
    def multi_query_fusion(self, queries, fusion_method="weighted"):
        """다중 쿼리 융합 검색"""
        
        all_results = {}
        query_weights = {}
        
        # 각 쿼리별 검색 실행
        for i, query in enumerate(queries):
            try:
                results, _ = self.search_engine.hybrid_search(query, limit=50)
                
                # 쿼리 중요도 계산 (길이, 복잡도 기반)
                weight = self._calculate_query_weight(query)
                query_weights[i] = weight
                
                # 결과를 문서 ID별로 그룹화
                for result in results:
                    doc_id = result['id']
                    if doc_id not in all_results:
                        all_results[doc_id] = {
                            'document': result,
                            'scores': [],
                            'queries': []
                        }
                    
                    all_results[doc_id]['scores'].append(result['score'] * weight)
                    all_results[doc_id]['queries'].append(i)
                    
            except Exception as e:
                logger.error(f"쿼리 '{query}' 검색 오류: {e}")
                continue
        
        # 융합 점수 계산
        fused_results = []
        for doc_id, data in all_results.items():
            if fusion_method == "weighted":
                # 가중 평균
                fused_score = sum(data['scores']) / len(data['scores'])
            elif fusion_method == "max":
                # 최대값
                fused_score = max(data['scores'])
            elif fusion_method == "reciprocal_rank":
                # 역순위 융합
                fused_score = sum(1.0 / (rank + 1) for rank in range(len(data['scores'])))
            else:
                # 기본값: 가중 평균
                fused_score = sum(data['scores']) / len(data['scores'])
            
            data['document']['fused_score'] = fused_score
            data['document']['query_coverage'] = len(data['queries']) / len(queries)
            fused_results.append(data['document'])
        
        # 융합 점수로 정렬
        fused_results.sort(key=lambda x: x['fused_score'], reverse=True)
        
        return fused_results
    
    def adaptive_chunk_retrieval(self, query, context_size="dynamic"):
        """적응적 청크 검색 - 쿼리 복잡도에 따른 청크 크기 조정"""
        
        # 쿼리 복잡도 분석
        complexity = self._analyze_query_complexity(query)
        
        if context_size == "dynamic":
            if complexity["type"] == "factual":
                # 사실적 질문: 작은 청크로 정확한 답변
                chunk_size = "small"
                limit = 5
            elif complexity["type"] == "analytical": 
                # 분석적 질문: 중간 청크로 충분한 컨텍스트
                chunk_size = "medium"
                limit = 10
            elif complexity["type"] == "comprehensive":
                # 포괄적 질문: 큰 청크로 넓은 컨텍스트
                chunk_size = "large"
                limit = 15
            else:
                # 기본값
                chunk_size = "medium"
                limit = 10
        else:
            chunk_size = context_size
            limit = 10
        
        try:
            # 기본 하이브리드 검색 사용
            results, search_info = self.search_engine.hybrid_search(query, limit=limit)
            
            # 결과 포맷팅
            formatted_results = []
            for result in results:
                formatted_result = {
                    "id": result['id'],
                    "path": result['path'],
                    "title": result['title'],
                    "chunk_text": result['chunk_text'],
                    "score": result['score'],
                    "source": "adaptive",
                    "file_type": result['file_type'],
                    "tags": result['tags'],
                    "created_at": result['created_at'],
                    "updated_at": result['updated_at'],
                    "chunk_index": result.get('chunk_index', 0)
                }
                formatted_results.append(formatted_result)
            
            # 인접 청크 포함 (컨텍스트 확장)
            expanded_results = self._expand_with_adjacent_chunks(formatted_results)
            
            return {
                "primary_chunks": formatted_results,
                "expanded_chunks": expanded_results,
                "complexity_analysis": complexity,
                "recommended_chunk_size": chunk_size
            }
            
        except Exception as e:
            logger.error(f"적응적 청크 검색 오류: {e}")
            return {
                "primary_chunks": [],
                "expanded_chunks": [],
                "complexity_analysis": complexity,
                "error": str(e)
            }
    
    def semantic_graph_retrieval(self, query, max_hops=2):
        """의미적 그래프 기반 검색 - milvus_manager의 지식 그래프 기능 활용"""
        
        try:
            # 1단계: 초기 관련 문서 검색
            initial_results, _ = self.search_engine.hybrid_search(query, limit=10)
            
            if not initial_results:
                return {
                    "direct_matches": [],
                    "connected_documents": [],
                    "graph_ranked_results": [],
                    "connection_depth": max_hops
                }
            
            # 2단계: milvus_manager의 지식 그래프 구축 기능 사용
            if hasattr(self.milvus_manager, 'build_knowledge_graph'):
                # 첫 번째 결과를 시작점으로 지식 그래프 구축
                start_doc_id = initial_results[0]['id']
                knowledge_graph = self.milvus_manager.build_knowledge_graph(
                    start_doc_id=start_doc_id,
                    max_depth=max_hops,
                    similarity_threshold=0.7
                )
                
                # 연결된 문서 ID 추출
                connected_docs = [node["id"] for node in knowledge_graph.get("nodes", [])]
                
            else:
                # 폴백: 수동으로 의미적 연결 탐색
                connected_docs = set()
                current_docs = [r['id'] for r in initial_results]
                
                for hop in range(max_hops):
                    next_level_docs = set()
                    
                    for doc_id in current_docs:
                        # 현재 문서와 의미적으로 유사한 문서들 찾기
                        doc_vector = self._get_document_vector(doc_id)
                        if doc_vector:
                            try:
                                if hasattr(self.milvus_manager, 'search_with_params'):
                                    similar_docs = self.milvus_manager.search_with_params(
                                        vector=doc_vector,
                                        limit=5,
                                        search_params={"metric_type": "COSINE", "params": {"ef": 128}}
                                    )
                                else:
                                    similar_docs = self.milvus_manager.search(doc_vector, 5)
                                
                                for hit in similar_docs:
                                    if hit.score > 0.7:  # 높은 유사도만
                                        next_level_docs.add(hit.id)
                                        
                            except Exception as e:
                                logger.error(f"유사 문서 검색 오류: {e}")
                                continue
                    
                    connected_docs.update(next_level_docs)
                    current_docs = list(next_level_docs - connected_docs)
                    
                    if not current_docs:  # 더 이상 연결된 문서가 없으면 중단
                        break
                
                connected_docs = list(connected_docs)
            
            # 3단계: 그래프 기반 랭킹
            graph_ranked_results = self._rank_by_graph_centrality(
                initial_results, connected_docs, query
            )
            
            return {
                "direct_matches": initial_results,
                "connected_documents": connected_docs,
                "graph_ranked_results": graph_ranked_results,
                "connection_depth": max_hops
            }
            
        except Exception as e:
            logger.error(f"의미적 그래프 검색 오류: {e}")
            return {
                "direct_matches": [],
                "connected_documents": [],
                "graph_ranked_results": [],
                "error": str(e)
            }
    
    def temporal_aware_retrieval(self, query, time_weight=0.3):
        """시간 인식 검색 - 최신성과 관련성의 균형"""
        
        try:
            # 기본 관련성 검색
            relevance_results, _ = self.search_engine.hybrid_search(query, limit=50)
            
            # 시간 가중치 적용
            current_time = time.time()
            time_weighted_results = []
            
            for result in relevance_results:
                created_time = result.get('created_at', current_time)
                if isinstance(created_time, str):
                    try:
                        created_time = datetime.fromisoformat(created_time).timestamp()
                    except:
                        created_time = current_time
                
                # 시간 점수 계산 (최신일수록 높은 점수)
                time_diff = current_time - created_time
                days_old = time_diff / (24 * 3600)
                time_score = 1.0 / (1.0 + days_old * 0.1)  # 시간 감쇠
                
                # 관련성과 최신성 결합
                combined_score = (
                    result['score'] * (1 - time_weight) + 
                    time_score * time_weight
                )
                
                result['time_score'] = time_score
                result['combined_score'] = combined_score
                time_weighted_results.append(result)
            
            # 결합 점수로 재정렬
            time_weighted_results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            return time_weighted_results
            
        except Exception as e:
            logger.error(f"시간 인식 검색 오류: {e}")
            return []
    
    def _search_by_level(self, query, level, filter_expr=None, limit=10):
        """수준별 검색 실행"""
        try:
            # 기본 하이브리드 검색 사용
            if filter_expr:
                # 필터가 있는 경우 벡터 검색만 사용
                query_vector = self.search_engine.embedding_model.get_embedding(query)
                if hasattr(self.milvus_manager, 'search_with_params'):
                    results = self.milvus_manager.search_with_params(
                        vector=query_vector,
                        limit=limit,
                        filter_expr=filter_expr
                    )
                else:
                    results = self.milvus_manager.search(query_vector, limit, filter_expr)
                
                return self._format_results(results)
            else:
                # 필터가 없는 경우 하이브리드 검색 사용
                results, _ = self.search_engine.hybrid_search(query, limit=limit)
                return results
            
        except Exception as e:
            logger.error(f"레벨별 검색 오류: {e}")
            return []
    
    def _calculate_query_weight(self, query):
        """쿼리 중요도 계산"""
        # 길이, 키워드 밀도, 특수 용어 등을 고려
        base_weight = 1.0
        
        # 길이 가중치
        length_weight = min(len(query.split()) / 10.0, 1.5)
        
        # 특수 키워드 가중치
        special_keywords = ['중요', 'critical', '긴급', 'urgent', '필수', 'important']
        keyword_weight = 1.0 + sum(0.2 for keyword in special_keywords if keyword in query.lower())
        
        return base_weight * length_weight * keyword_weight
    
    def _analyze_query_complexity(self, query):
        """쿼리 복잡도 분석"""
        words = query.split()
        
        # 질문 유형 분류
        factual_keywords = ['what', 'who', 'when', 'where', '무엇', '누구', '언제', '어디']
        analytical_keywords = ['why', 'how', 'analyze', '왜', '어떻게', '분석']
        comprehensive_keywords = ['compare', 'summarize', 'overview', '비교', '요약', '개요']
        
        if any(word in query.lower() for word in factual_keywords):
            query_type = "factual"
        elif any(word in query.lower() for word in analytical_keywords):
            query_type = "analytical"
        elif any(word in query.lower() for word in comprehensive_keywords):
            query_type = "comprehensive"
        else:
            query_type = "general"
        
        return {
            "type": query_type,
            "word_count": len(words),
            "complexity_score": len(words) * 0.1 + (1.0 if query_type == "comprehensive" else 0.5)
        }
    
    def _format_results(self, results):
        """결과 포맷팅"""
        formatted_results = []
        for hit in results:
            formatted_result = {
                "id": hit.id,
                "path": hit.entity.get('path', ''),
                "title": hit.entity.get('title', '제목 없음'),
                "chunk_text": hit.entity.get('chunk_text', ''),
                "score": float(hit.score),
                "source": "formatted",
                "file_type": hit.entity.get('file_type', ''),
                "tags": hit.entity.get('tags', []),
                "created_at": hit.entity.get('created_at', ''),
                "updated_at": hit.entity.get('updated_at', '')
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def _expand_with_adjacent_chunks(self, results):
        """인접 청크로 확장"""
        expanded_results = []
        
        for result in results[:5]:  # 상위 5개 결과만 확장
            try:
                # 현재 결과 추가
                expanded_results.append(result)
                
                # 같은 문서의 인접 청크들 찾기
                path = result["path"]
                current_chunk_index = result.get("chunk_index", 0)
                
                adjacent_chunks = self.milvus_manager.query(
                    expr=f'path == "{path}"',
                    output_fields=["id", "path", "title", "chunk_text", "chunk_index", "file_type", "tags", "created_at", "updated_at"],
                    limit=50
                )
                
                # 인접한 청크들만 선택 (현재 청크 인덱스 ±2 범위)
                for chunk in adjacent_chunks:
                    chunk_index = chunk.get("chunk_index", 0)
                    if (chunk["id"] != result["id"] and 
                        abs(chunk_index - current_chunk_index) <= 2):
                        
                        chunk_result = {
                            "id": chunk["id"],
                            "path": chunk["path"],
                            "title": chunk["title"],
                            "chunk_text": chunk["chunk_text"],
                            "score": result["score"] * 0.8,  # 약간 낮은 점수
                            "source": "adjacent_chunk",
                            "file_type": chunk["file_type"],
                            "tags": chunk["tags"],
                            "created_at": chunk["created_at"],
                            "updated_at": chunk["updated_at"],
                            "chunk_index": chunk["chunk_index"]
                        }
                        expanded_results.append(chunk_result)
                        
            except Exception as e:
                logger.error(f"청크 확장 중 오류: {e}")
                expanded_results.append(result)  # 오류 시 원본 결과만 추가
                continue
        
        return expanded_results
    
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
    
    def _rank_by_graph_centrality(self, initial_results, connected_docs, query):
        """그래프 중심성 기반 랭킹"""
        # 간단한 연결 기반 중심성 계산
        centrality_scores = defaultdict(float)
        
        # 초기 결과들에 기본 점수 부여
        for result in initial_results:
            centrality_scores[result['id']] = result['score']
        
        # 연결된 문서들에 연결성 보너스 부여
        for doc_id in connected_docs:
            connection_count = len([r for r in initial_results if r['id'] == doc_id])
            centrality_scores[doc_id] += connection_count * 0.1
        
        # 점수로 정렬
        ranked_results = []
        for result in initial_results:
            result['centrality_score'] = centrality_scores[result['id']]
            ranked_results.append(result)
        
        ranked_results.sort(key=lambda x: x['centrality_score'], reverse=True)
        
        return ranked_results
    
    def _build_hierarchy_path(self, chunk_results):
        """계층 경로 구축"""
        paths = []
        for chunk in chunk_results[:5]:  # 상위 5개만
            path_parts = chunk.get('path', '').split('/')
            if len(path_parts) > 1:
                hierarchy = " → ".join(path_parts[-3:])  # 마지막 3개 부분만
                paths.append({
                    "chunk_id": chunk['id'],
                    "hierarchy": hierarchy,
                    "title": chunk.get('title', '제목 없음')
                })
        return paths
