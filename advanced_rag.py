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
from logger import get_logger

# 중앙 집중식 로깅 시스템 사용
logger = get_logger('AdvancedRAGEngine')

class AdvancedRAGEngine:
    def __init__(self, milvus_manager, search_engine):
        self.milvus_manager = milvus_manager
        self.search_engine = search_engine
        
    def hierarchical_retrieval(self, query, max_depth=3):
        """계층적 검색 - 문서 → 섹션 → 청크"""
        start_time = time.time()
        logger.info(f"계층적 검색 시작 - 쿼리: '{query}', 최대 깊이: {max_depth}")
        
        try:
            # 1단계: 문서 수준 검색
            doc_start_time = time.time()
            logger.debug(f"1단계: 문서 수준 검색 시작 (limit=10)")
            doc_results = self._search_by_level(query, level="document", limit=10)
            doc_time = time.time() - doc_start_time
            
            # 특수 문자 처리를 위한 문서 경로 분석
            special_char_paths = []
            for doc in doc_results:
                path = doc.get('path', '')
                if path and any(c in path for c in "'\"()[]{},;"): 
                    special_char_paths.append(path)
                    logger.debug(f"특수 문자가 포함된 문서 경로 발견: {path}")
            
            logger.debug(f"1단계 완료: {len(doc_results)} 문서 검색됨 ({doc_time:.3f}초), 특수 문자 경로: {len(special_char_paths)}")
            
            # 2단계: 관련 문서의 섹션 검색  
            section_start_time = time.time()
            logger.debug(f"2단계: 섹션 검색 시작 - 상위 5개 문서 사용")
            section_results = []
            section_errors = 0
            
            for i, doc in enumerate(doc_results[:5]):  # 상위 5개 문서
                try:
                    doc_path = doc['path']
                    logger.debug(f"문서[{i+1}/5] 섹션 검색: {doc_path}")
                    
                    # 특수 문자 처리를 위한 필터 표현식 보호
                    safe_path = doc_path.replace("'", "''")
                    filter_expr = f"path like '{safe_path}%'"
                    
                    sections = self._search_by_level(
                        query, 
                        level="section", 
                        filter_expr=filter_expr,
                        limit=5
                    )
                    section_results.extend(sections)
                    logger.debug(f"문서[{i+1}/5] 섹션 {len(sections)}개 검색됨")
                    
                except Exception as e:
                    section_errors += 1
                    logger.error(f"섹션 검색 오류 (문서: {doc.get('path', 'unknown')}): {e}", exc_info=True)
                    continue
            
            section_time = time.time() - section_start_time
            logger.debug(f"2단계 완료: {len(section_results)} 섹션 검색됨 ({section_time:.3f}초), 오류: {section_errors}")
            
            # 3단계: 관련 섹션의 상세 청크 검색
            chunk_start_time = time.time()
            logger.debug(f"3단계: 청크 검색 시작 - 상위 10개 섹션 사용")
            chunk_results = []
            chunk_errors = 0
            
            for i, section in enumerate(section_results[:10]):  # 상위 10개 섹션
                try:
                    section_path = section['path']
                    logger.debug(f"섹션[{i+1}/10] 청크 검색: {section_path}")
                    
                    # 특수 문자 처리를 위한 필터 표현식 보호
                    safe_path = section_path.replace("'", "''")
                    filter_expr = f"path == '{safe_path}'"
                    
                    chunks = self._search_by_level(
                        query,
                        level="chunk", 
                        filter_expr=filter_expr,
                        limit=3
                    )
                    chunk_results.extend(chunks)
                    logger.debug(f"섹션[{i+1}/10] 청크 {len(chunks)}개 검색됨")
                    
                except Exception as e:
                    chunk_errors += 1
                    logger.error(f"청크 검색 오류 (섹션: {section.get('path', 'unknown')}): {e}", exc_info=True)
                    continue
            
            chunk_time = time.time() - chunk_start_time
            logger.debug(f"3단계 완료: {len(chunk_results)} 청크 검색됨 ({chunk_time:.3f}초), 오류: {chunk_errors}")
            
            # 계층 구조 경로 구축
            hierarchy_path = self._build_hierarchy_path(chunk_results)
            
            total_time = time.time() - start_time
            logger.info(f"계층적 검색 완료: 문서 {len(doc_results)}개, 섹션 {len(section_results)}개, 청크 {len(chunk_results)}개 (총 {total_time:.3f}초)")
            
            return {
                "documents": doc_results,
                "sections": section_results, 
                "chunks": chunk_results,
                "hierarchy_path": hierarchy_path,
                "stats": {
                    "total_time": total_time,
                    "document_search_time": doc_time,
                    "section_search_time": section_time,
                    "chunk_search_time": chunk_time,
                    "section_errors": section_errors,
                    "chunk_errors": chunk_errors
                }
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"계층적 검색 실패 ({total_time:.3f}초): {e}", exc_info=True)
            return {
                "documents": [],
                "sections": [], 
                "chunks": [],
                "hierarchy_path": [],
                "error": str(e)
            }
    
    def multi_query_fusion(self, queries, fusion_method="weighted"):
        """다중 쿼리 융합 검색"""
        start_time = time.time()
        logger.info(f"다중 쿼리 융합 검색 시작 - {len(queries)}개 쿼리, 융합 방법: {fusion_method}")
        
        try:
            all_results = {}
            query_weights = {}
            query_times = {}
            query_result_counts = {}
            special_char_paths = []
            
            # 각 쿼리별 검색 실행
            for i, query in enumerate(queries):
                query_start = time.time()
                logger.debug(f"쿼리[{i+1}/{len(queries)}] 검색 시작: '{query}'")
                
                try:
                    results, search_info = self.search_engine.hybrid_search(query, limit=50)
                    query_times[i] = time.time() - query_start
                    query_result_counts[i] = len(results)
                    
                    # 쿼리 중요도 계산 (길이, 복잡도 기반)
                    weight = self._calculate_query_weight(query)
                    query_weights[i] = weight
                    logger.debug(f"쿼리[{i+1}/{len(queries)}] 가중치: {weight:.4f}, 결과: {len(results)}개")
                    
                    # 특수 문자 있는 경로 확인
                    for result in results:
                        path = result.get('path', '')
                        if path and any(c in path for c in "'\"()[]{},;"):
                            if path not in special_char_paths:
                                special_char_paths.append(path)
                                logger.debug(f"특수 문자가 포함된 경로 발견: {path}")
                    
                    # 결과를 문서 ID별로 그룹화
                    for result in results:
                        doc_id = result['id']
                        if doc_id not in all_results:
                            all_results[doc_id] = {
                                'document': result,
                                'scores': [],
                                'queries': []
                            }
                        
                        # 가중치 적용한 점수 추가
                        weighted_score = result['score'] * weight
                        all_results[doc_id]['scores'].append(weighted_score)
                        all_results[doc_id]['queries'].append(i)
                        
                except Exception as e:
                    logger.error(f"쿼리[{i+1}/{len(queries)}] '{query}' 검색 오류: {e}", exc_info=True)
                    continue
            
            # 결과 통계 로깅
            query_coverage = len(query_times)
            if query_coverage > 0:
                avg_query_time = sum(query_times.values()) / query_coverage
                avg_results_per_query = sum(query_result_counts.values()) / query_coverage
                logger.debug(
                    f"쿼리 실행 통계: {query_coverage}/{len(queries)} 쿼리 성공, "
                    f"평균 쿼리 시간: {avg_query_time:.3f}초, "
                    f"평균 결과 수: {avg_results_per_query:.1f}개"
                )
            
            # 융합 점수 계산
            fusion_start = time.time()
            logger.debug(f"결과 융합 시작: {len(all_results)}개 문서, 방법: {fusion_method}")
            
            fused_results = []
            for doc_id, data in all_results.items():
                if fusion_method == "weighted":
                    # 가중 평균
                    fused_score = sum(data['scores']) / len(data['scores'])
                    method_desc = "가중 평균"
                elif fusion_method == "max":
                    # 최대값
                    fused_score = max(data['scores'])
                    method_desc = "최대값"
                elif fusion_method == "reciprocal_rank":
                    # 역순위 융합
                    fused_score = sum(1.0 / (rank + 1) for rank in range(len(data['scores'])))
                    method_desc = "역순위 융합"
                else:
                    # 기본값: 가중 평균
                    fused_score = sum(data['scores']) / len(data['scores'])
                    method_desc = "기본 가중 평균"
                
                # 쿼리 커버리지 계산 (전체 쿼리 중 바로 이 문서를 찾은 쿼리 비율)
                query_coverage = len(data['queries']) / len(queries)
                
                data['document']['fused_score'] = fused_score
                data['document']['query_coverage'] = query_coverage
                data['document']['fusion_method'] = method_desc
                data['document']['matching_queries'] = data['queries']
                fused_results.append(data['document'])
            
            fusion_time = time.time() - fusion_start
            logger.debug(f"결과 융합 완료: {len(fused_results)}개 문서 ({fusion_time:.3f}초)")
            
            # 육합 점수로 정렬
            fused_results.sort(key=lambda x: x['fused_score'], reverse=True)
            
            total_time = time.time() - start_time
            logger.info(f"다중 쿼리 융합 검색 완료: {query_coverage}/{len(queries)} 쿼리 성공, {len(fused_results)}개 결과 ({total_time:.3f}초)")
            
            return {
                "results": fused_results,
                "stats": {
                    "total_time": total_time,
                    "query_times": query_times,
                    "fusion_time": fusion_time,
                    "query_weights": query_weights,
                    "special_paths_count": len(special_char_paths),
                    "query_coverage": query_coverage
                }
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"다중 쿼리 융합 검색 실패 ({total_time:.3f}초): {e}", exc_info=True)
            return {
                "results": [],
                "error": str(e)
            }
    
    def adaptive_chunk_retrieval(self, query, context_size="dynamic"):
        """적응적 청크 검색 - 쿼리 복잡도에 따른 청크 크기 조정"""
        start_time = time.time()
        logger.info(f"적응적 청크 검색 시작 - 쿼리: '{query}', 컨텍스트 크기 모드: {context_size}")
        
        try:
            # 쿼리 복잡도 분석
            complexity_start = time.time()
            logger.debug(f"쿼리 복잡도 분석 시작: '{query}'")
            complexity = self._analyze_query_complexity(query)
            complexity_time = time.time() - complexity_start
            
            logger.debug(f"쿼리 복잡도 분석 완료 ({complexity_time:.3f}초): "
                         f"유형={complexity['type']}, "
                         f"점수={complexity.get('score', 0):.2f}, "
                         f"키워드={complexity.get('keywords', [])}")
            
            # 청크 크기 및 한계 결정
            if context_size == "dynamic":
                if complexity["type"] == "factual":
                    # 사실적 질문: 작은 청크로 정확한 답변
                    chunk_size = "small"
                    limit = 5
                    logger.debug(f"사실적 질문 감지: 작은 청크 크기 사용, limit={limit}")
                elif complexity["type"] == "analytical": 
                    # 분석적 질문: 중간 청크로 충분한 컨텍스트
                    chunk_size = "medium"
                    limit = 10
                    logger.debug(f"분석적 질문 감지: 중간 청크 크기 사용, limit={limit}")
                elif complexity["type"] == "comprehensive":
                    # 포괄적 질문: 큰 청크로 넓은 컨텍스트
                    chunk_size = "large"
                    limit = 15
                    logger.debug(f"포괄적 질문 감지: 큰 청크 크기 사용, limit={limit}")
                else:
                    # 기본값
                    chunk_size = "medium"
                    limit = 10
                    logger.debug(f"기본 질문 유형: 중간 청크 크기 사용, limit={limit}")
            else:
                chunk_size = context_size
                limit = 10
                logger.debug(f"직접 지정된 청크 크기 사용: {chunk_size}, limit={limit}")
            
            # 기본 하이브리드 검색 실행
            search_start = time.time()
            logger.debug(f"하이브리드 검색 시작: limit={limit}")
            results, search_info = self.search_engine.hybrid_search(query, limit=limit)
            search_time = time.time() - search_start
            
            logger.debug(f"하이브리드 검색 완료 ({search_time:.3f}초): {len(results)}개 결과")
            
            # 특수 문자 처리를 위한 경로 분석
            special_char_paths = []
            for result in results:
                path = result.get('path', '')
                if path and any(c in path for c in "'\"()[]{},;"):
                    if path not in special_char_paths:
                        special_char_paths.append(path)
                        logger.debug(f"특수 문자가 포함된 경로 발견: {path}")
            
            # 결과 포맷팅
            format_start = time.time()
            logger.debug("검색 결과 포맷팅 시작")
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
            
            format_time = time.time() - format_start
            logger.debug(f"결과 포맷팅 완료 ({format_time:.3f}초)")
            
            # 인접 청크 포함 (컨텍스트 확장)
            expansion_start = time.time()
            logger.debug(f"인접 청크 확장 시작: {len(formatted_results)}개 청크")
            expanded_results = self._expand_with_adjacent_chunks(formatted_results)
            expansion_time = time.time() - expansion_start
            
            added_context = len(expanded_results) - len(formatted_results)
            logger.debug(f"인접 청크 확장 완료 ({expansion_time:.3f}초): {added_context}개 추가 청크")
            
            total_time = time.time() - start_time
            logger.info(f"적응적 청크 검색 완료 ({total_time:.3f}초): "
                       f"쿼리 유형={complexity['type']}, "
                       f"청크 크기={chunk_size}, "
                       f"기본 결과={len(formatted_results)}개, "
                       f"확장 결과={len(expanded_results)}개")
            
            return {
                "primary_chunks": formatted_results,
                "expanded_chunks": expanded_results,
                "complexity_analysis": complexity,
                "recommended_chunk_size": chunk_size,
                "stats": {
                    "total_time": total_time,
                    "complexity_analysis_time": complexity_time,
                    "search_time": search_time,
                    "format_time": format_time,
                    "expansion_time": expansion_time,
                    "special_paths_count": len(special_char_paths)
                }
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"적응적 청크 검색 오류 ({total_time:.3f}초): {e}", exc_info=True)
            return {
                "primary_chunks": [],
                "expanded_chunks": [],
                "complexity_analysis": complexity if 'complexity' in locals() else {"type": "unknown", "error": str(e)},
                "error": str(e)
            }
    
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
    
    def semantic_graph_retrieval(self, query, max_hops=2):
        """의미적 그래프 기반 검색"""
        start_time = time.time()
        logger.info(f"의미적 그래프 검색 시작 - 쿼리: '{query}', 최대 홉: {max_hops}")
        
        try:
            # 시작 노드 찾기 (쿼리와 가장 유사한 문서)
            initial_start = time.time()
            logger.debug(f"시작 노드 검색 중 (limit=5)")
            start_results, search_info = self.search_engine.hybrid_search(query, limit=5)
            initial_time = time.time() - initial_start
            
            if not start_results:
                logger.warning(f"쿼리 '{query}'에 대한 초기 결과를 찾을 수 없음 ({initial_time:.3f}초)")
                return {'results': [], 'graph': {}, 'message': "No initial results found"}
            
            logger.debug(f"시작 노드 {len(start_results)}개 찾음 ({initial_time:.3f}초)")
            
            # 특수 문자 경로 확인
            special_char_paths = []
            for result in start_results:
                path = result.get('path', '')
                if path and any(c in path for c in "'\"()[]{},;"):
                    special_char_paths.append(path)
                    logger.debug(f"특수 문자가 포함된 시작 노드 경로 발견: {path}")
            
            # 그래프 탐색 준비
            graph_start = time.time()
            logger.debug("그래프 탐색 초기화")
            visited = set()  # 방문한 노드 ID
            nodes = {}  # 노드 ID -> 노드 정보
            edges = []  # 엣지 목록
            frontier = []  # 탐색 대기열
            
            # 시작 노드 추가
            for result in start_results:
                node_id = result['id']
                nodes[node_id] = {
                    'id': node_id,
                    'title': result['title'],
                    'path': result['path'],
                    'score': result['score'],
                    'hop': 0
                }
                frontier.append((node_id, 0))  # (node_id, hop_count)
                visited.add(node_id)
            
            logger.debug(f"그래프 탐색 시작: {len(frontier)}개 시작 노드, 최대 홉 {max_hops}")
            
            # 그래프 탐색 통계
            exploration_stats = {
                "total_explored": 0,
                "by_hop": {h: 0 for h in range(max_hops + 1)},
                "errors": 0
            }
            
            # 그래프 탐색 (BFS)
            while frontier and len(nodes) < 50:  # 최대 50개 노드
                node_id, hop_count = frontier.pop(0)
                exploration_stats["total_explored"] += 1
                exploration_stats["by_hop"][hop_count] += 1
                
                if hop_count >= max_hops:
                    logger.debug(f"최대 홉 {max_hops}에 도달하여 노드 {node_id} 탐색 중단")
                    continue
                
                logger.debug(f"노드 탐색 중: ID={node_id}, 홉={hop_count}/{max_hops}")
                    
                # 현재 노드와 연결된 노드 찾기
                try:
                    node_start = time.time()
                    related_nodes = self._find_related_nodes(node_id, query)
                    node_time = time.time() - node_start
                    
                    logger.debug(f"노드 {node_id}에서 {len(related_nodes)}개의 관련 노드 발견 ({node_time:.3f}초)")
                    
                    for related in related_nodes:
                        rel_id = related['id']
                        
                        # 노드 추가
                        if rel_id not in visited:
                            nodes[rel_id] = {
                                'id': rel_id,
                                'title': related['title'],
                                'path': related['path'],
                                'score': related.get('score', 0.0),
                                'hop': hop_count + 1
                            }
                            visited.add(rel_id)
                            frontier.append((rel_id, hop_count + 1))
                            
                            # 특수 문자 경로 확인
                            path = related.get('path', '')
                            if path and any(c in path for c in "'\"()[]{},;"):
                                if path not in special_char_paths:
                                    special_char_paths.append(path)
                                    logger.debug(f"특수 문자가 포함된 관련 노드 경로 발견: {path}")
                        
                        # 엣지 추가
                        edges.append({
                            'source': node_id,
                            'target': rel_id,
                            'type': related.get('relation_type', 'related'),
                            'weight': related.get('score', 0.5)
                        })
                    
                except Exception as e:
                    exploration_stats["errors"] += 1
                    logger.error(f"노드 {node_id} 탐색 오류: {e}", exc_info=True)
            
            graph_time = time.time() - graph_start
            logger.debug(f"그래프 탐색 완료 ({graph_time:.3f}초): {len(nodes)}개 노드, {len(edges)}개 엣지")
            
            # 최종 결과 구성
            results = [{'id': k, **v} for k, v in nodes.items()]
            results.sort(key=lambda x: x['score'], reverse=True)
            
            total_time = time.time() - start_time
            logger.info(f"의미적 그래프 검색 완료 ({total_time:.3f}초): {len(nodes)}개 노드, {len(edges)}개 엣지, 결과 {min(len(results), 20)}개")
            
            return {
                'results': results[:20],  # 상위 20개 결과
                'graph': {
                    'nodes': list(nodes.values()),
                    'edges': edges
                },
                'stats': {
                    'total_time': total_time,
                    'initial_search_time': initial_time,
                    'graph_exploration_time': graph_time,
                    'node_count': len(nodes),
                    'edge_count': len(edges),
                    'max_hop': max_hops,
                    'exploration_stats': exploration_stats,
                    'special_paths_count': len(special_char_paths)
                }
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"의미적 그래프 검색 오류 ({total_time:.3f}초): {e}", exc_info=True)
            return {
                'results': [],
                'graph': {'nodes': [], 'edges': []},
                'error': str(e)
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
            try:
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
                    
                formatted_result = {
                    "id": hit_id,
                    "path": entity.get('path', ''),
                    "title": entity.get('title', '제목 없음'),
                    "chunk_text": entity.get('chunk_text', ''),
                    "score": score,
                    "source": "formatted",
                    "file_type": entity.get('file_type', ''),
                    "tags": entity.get('tags', []),
                    "created_at": entity.get('created_at', ''),
                    "updated_at": entity.get('updated_at', '')
                }
                formatted_results.append(formatted_result)
            except Exception as e:
                logger.error(f"결과 포맷팅 오류: {e}")
                continue
        
        # 결과를 JSON 직렬화 가능하게 변환
        return self.ensure_json_serializable(formatted_results)
    
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
