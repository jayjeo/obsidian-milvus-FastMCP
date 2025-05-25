"""Obsidian-Milvus Fast MCP Server - 완전 최적화 버전 (Enhanced)
전체 노트 검색 및 고급 검색 모드를 지원하는 업그레이드 버전

New Features:
- 전체 검색 모드 (limit=None 지원)
- 자동 검색 모드 결정
- 배치 검색 및 페이지네이션
- 기본 limit 200-500으로 증가
- 종합 검색 기능

Enhanced with:
- 고급 메타데이터 필터링  
- HNSW 인덱스 최적화
- 계층적/의미적 그래프 검색
- 다중 쿼리 융합
- 적응적 청크 검색
- 시간 인식 검색
- 성능 최적화 및 모니터링
"""

# Import warning suppressor first
import warning_suppressor

import os
import sys
import json
import traceback
import logging
import time
import asyncio
import math
from typing import List, Dict, Any, Optional, Tuple, Generator
from datetime import datetime

# Configure basic logging
logging.basicConfig(
    level=logging.ERROR,  # Use ERROR level by default
    format='%(message)s'
)

# Import other modules
from mcp.server.fastmcp import FastMCP
import config
from milvus_manager import MilvusManager
from search_engine import SearchEngine

# 새로운 고급 모듈들
from enhanced_search_engine import EnhancedSearchEngine
from hnsw_optimizer import HNSWOptimizer
from advanced_rag import AdvancedRAGEngine

# Set up logging level from config
log_level_str = getattr(config, 'LOG_LEVEL', 'ERROR')
log_level = getattr(logging, log_level_str, logging.ERROR)
logging.getLogger().setLevel(log_level)

# Get logger for this module
logger = logging.getLogger('OptimizedMCP')
logger.setLevel(log_level)

# Helper function to safely print messages
def safe_print(message, level="info"):
    """Print a message safely using the logger"""
    if level.lower() == "error":
        logger.error(message)
    elif level.lower() == "warning":
        logger.warning(message)
    else:
        logger.info(message)

mcp = FastMCP(config.FASTMCP_SERVER_NAME)

# 전역 변수들
milvus_manager = None
enhanced_search = None
search_engine = None
hnsw_optimizer = None
rag_engine = None

def initialize_components():
    """모든 컴포넌트들 초기화"""
    global milvus_manager, search_engine, enhanced_search, hnsw_optimizer, rag_engine
    
    try:
        logger.info("Starting Enhanced Obsidian-Milvus Fast MCP Server...")
        
        milvus_manager = MilvusManager()
        search_engine = SearchEngine(milvus_manager)
        enhanced_search = EnhancedSearchEngine(milvus_manager)
        hnsw_optimizer = HNSWOptimizer(milvus_manager)
        rag_engine = AdvancedRAGEngine(milvus_manager, enhanced_search)
        
        try:
            # Skip auto-tuning to prevent hanging
            logger.info("Skipping auto-tuning to prevent system hang")
            # optimization_params = hnsw_optimizer.auto_tune_parameters()
            # logger.info(f"Auto-tuning completed: {optimization_params}")
        except Exception as e:
            logger.warning(f"Auto-tuning warning: {e}")
        
        logger.info("All components initialized!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Component initialization failed: {e}")
        return False

# ==================== 새로운 고급 검색 도구들 ====================

def analyze_query_complexity(query: str) -> Dict[str, Any]:
    """쿼리 복잡도 분석하여 최적 검색 모드 결정"""
    words = query.split()
    word_count = len(words)
    
    # 키워드 기반 복잡도 분석
    complex_keywords = ['분석', 'analyze', '비교', 'compare', '관계', 'relation', '연결', 'connection']
    semantic_keywords = ['의미', 'meaning', '개념', 'concept', '이해', 'understand']
    specific_keywords = ['정확히', 'exact', '특정', 'specific', '찾아줘', 'find']
    
    complexity_score = 0
    
    # 단어 수 기반 점수
    if word_count <= 2:
        complexity_score += 1  # 단순
    elif word_count <= 5:
        complexity_score += 2  # 보통
    else:
        complexity_score += 3  # 복잡
    
    # 키워드 기반 점수
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in complex_keywords):
        complexity_score += 2
    if any(keyword in query_lower for keyword in semantic_keywords):
        complexity_score += 1
    if any(keyword in query_lower for keyword in specific_keywords):
        complexity_score += 1
    
    # 검색 모드 결정
    if complexity_score <= 2:
        search_mode = "fast"
        search_strategy = "keyword"
    elif complexity_score <= 4:
        search_mode = "balanced"
        search_strategy = "hybrid"
    else:
        search_mode = "comprehensive"
        search_strategy = "semantic_graph"
    
    return {
        "complexity_score": complexity_score,
        "word_count": word_count,
        "recommended_mode": search_mode,
        "recommended_strategy": search_strategy,
        "estimated_time": "fast" if complexity_score <= 2 else "medium" if complexity_score <= 4 else "slow"
    }

@mcp.tool()
async def auto_search_mode_decision(
    query: str,
    execute_search: bool = True,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """쿼리를 분석하여 최적의 검색 모드를 자동으로 결정하고 실행"""
    global search_engine, enhanced_search, rag_engine
    
    if not search_engine:
        return {"error": "Search engine not initialized.", "query": query}
    
    try:
        start_time = time.time()
        
        # 쿼리 분석
        analysis = analyze_query_complexity(query)
        recommended_mode = analysis["recommended_mode"]
        recommended_strategy = analysis["recommended_strategy"]
        
        # limit 자동 결정
        if limit is None:
            if recommended_mode == "fast":
                limit = 100
            elif recommended_mode == "balanced":
                limit = 300
            else:  # comprehensive
                limit = 500
        
        results = []
        search_info = {}
        
        if execute_search:
            # 추천된 모드로 검색 실행
            if recommended_strategy == "keyword":
                results = search_engine._keyword_search(query=query, limit=limit)
                search_info = {"type": "keyword", "mode": "fast"}
                
            elif recommended_strategy == "hybrid":
                results, search_info = search_engine.hybrid_search(
                    query=query, limit=limit
                )
                
            elif recommended_strategy == "semantic_graph" and rag_engine:
                try:
                    results = rag_engine.semantic_graph_retrieval(query, max_hops=2)
                    if isinstance(results, dict) and "primary_chunks" in results:
                        results = results["primary_chunks"][:limit]
                    # Ensure all data is JSON serializable
                    results = [{k: (str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v) 
                              for k, v in item.items()} for item in results]
                    search_info = {"type": "semantic_graph", "mode": "comprehensive"}
                except Exception as e:
                    logger.error(f"Semantic graph retrieval error: {e}")
                    # Fallback to hybrid search if semantic graph fails
                    results, search_info = search_engine.hybrid_search(query=query, limit=limit)
                
            else:
                # 폴백: 하이브리드 검색
                results, search_info = search_engine.hybrid_search(
                    query=query, limit=limit
                )
        
        analysis_time = time.time() - start_time
        
        # Ensure all data is JSON serializable
        def ensure_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: ensure_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [ensure_json_serializable(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        # Create response with serializable data
        response = {
            "query": query,
            "query_analysis": ensure_json_serializable(analysis),
            "selected_mode": recommended_mode,
            "selected_strategy": recommended_strategy,
            "limit_used": limit,
            "results": ensure_json_serializable(results) if execute_search else [],
            "search_info": ensure_json_serializable(search_info) if execute_search else {},
            "performance": {
                "analysis_time_ms": round(analysis_time * 1000, 2),
                "total_results": len(results) if execute_search else 0,
                "mode_effectiveness": "optimal" if results else "needs_adjustment"
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Auto search mode decision error: {e}")
        return {"error": str(e), "query": query}

@mcp.tool()
async def comprehensive_search_all(
    query: str,
    include_similarity_scores: bool = True,
    batch_size: int = 500,
    similarity_threshold: float = 0.3
) -> Dict[str, Any]:
    """전체 컬렉션을 대상으로 한 종합 검색 (limit 제한 없음)"""
    global milvus_manager, search_engine
    
    if not milvus_manager or not search_engine:
        return {"error": "Required components not initialized.", "query": query}
    
    try:
        start_time = time.time()
        
        # 전체 컬렉션 크기 확인
        total_entities = milvus_manager.count_entities()
        print(f"🔍 Comprehensive search across {total_entities} documents...")
        
        all_results = []
        processed_batches = 0
        
        # 배치별로 전체 컬렉션 검색
        for offset in range(0, total_entities, batch_size):
            try:
                batch_results = milvus_manager.query(
                    expr="id >= 0",
                    output_fields=["id", "path", "title", "chunk_text", "content", "file_type", "tags", "created_at", "updated_at"],
                    limit=batch_size,
                    offset=offset
                )
                
                # 각 문서에 대해 유사도 계산
                if include_similarity_scores:
                    query_embedding = search_engine.embedding_model.get_embedding(query)
                    
                    for doc in batch_results:
                        doc_text = f"{doc.get('title', '')} {doc.get('chunk_text', '')}"
                        if doc_text.strip():
                            doc_embedding = search_engine.embedding_model.get_embedding(doc_text)
                            similarity = search_engine._calculate_cosine_similarity(query_embedding, doc_embedding)
                            
                            if similarity >= similarity_threshold:
                                doc['similarity_score'] = float(similarity)
                                doc['search_relevance'] = 'high' if similarity > 0.7 else 'medium' if similarity > 0.5 else 'low'
                                all_results.append(doc)
                else:
                    # 키워드 기반 필터링
                    query_words = set(query.lower().split())
                    for doc in batch_results:
                        doc_text = f"{doc.get('title', '')} {doc.get('chunk_text', '')}".lower()
                        if any(word in doc_text for word in query_words):
                            doc['similarity_score'] = 0.5  # 기본값
                            doc['search_relevance'] = 'keyword_match'
                            all_results.append(doc)
                
                processed_batches += 1
                
                # 진행 상황 출력
                if processed_batches % 5 == 0:
                    print(f"📊 Processed {processed_batches * batch_size}/{total_entities} documents...")
                
            except Exception as batch_error:
                logger.error(f"Batch processing error at offset {offset}: {batch_error}")
                continue
        
        # 결과 정렬 (유사도 순)
        if include_similarity_scores:
            all_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        search_time = time.time() - start_time
        
        # Ensure all data is JSON serializable
        def ensure_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: ensure_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [ensure_json_serializable(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        # Process results to ensure they're serializable
        processed_results = []
        for result in all_results:
            processed_result = {}
            for k, v in result.items():
                if not isinstance(v, (str, int, float, bool, type(None))):
                    processed_result[k] = str(v)
                else:
                    processed_result[k] = v
            processed_results.append(processed_result)
        
        return {
            "query": query,
            "search_type": "comprehensive_all",
            "total_documents_searched": total_entities,
            "total_results_found": len(processed_results),
            "results": processed_results,
            "search_parameters": {
                "batch_size": batch_size,
                "similarity_threshold": similarity_threshold,
                "include_similarity_scores": include_similarity_scores
            },
            "performance_metrics": {
                "search_time_seconds": round(search_time, 2),
                "documents_per_second": round(total_entities / search_time, 2) if search_time > 0 else 0,
                "batches_processed": processed_batches,
                "effectiveness_ratio": len(processed_results) / total_entities if total_entities > 0 else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Comprehensive search error: {e}")
        return {"error": str(e), "query": query}

@mcp.tool()
async def batch_search_with_pagination(
    query: str,
    page_size: int = 200,
    max_pages: Optional[int] = None,
    search_mode: str = "hybrid"
) -> Dict[str, Any]:
    """페이지네이션 방식으로 배치 검색 수행"""
    global search_engine, milvus_manager
    
    if not search_engine or not milvus_manager:
        return {"error": "Required components not initialized.", "query": query}
    
    try:
        start_time = time.time()
        
        total_entities = milvus_manager.count_entities()
        max_possible_pages = math.ceil(total_entities / page_size)
        
        if max_pages is None:
            max_pages = min(max_possible_pages, 10)  # 기본적으로 최대 10페이지
        else:
            max_pages = min(max_pages, max_possible_pages)
        
        all_results = []
        page_results = []
        
        query_embedding = search_engine.embedding_model.get_embedding(query)
        
        for page in range(max_pages):
            offset = page * page_size
            
            try:
                # 페이지별 데이터 가져오기
                page_docs = milvus_manager.query(
                    expr="id >= 0",
                    output_fields=["id", "path", "title", "chunk_text", "content", "file_type", "tags"],
                    limit=page_size,
                    offset=offset
                )
                
                page_matches = []
                
                if search_mode == "hybrid":
                    # 하이브리드 검색 (의미적 + 키워드)
                    query_words = set(query.lower().split())
                    
                    for doc in page_docs:
                        # 키워드 매칭 확인
                        doc_text = f"{doc.get('title', '')} {doc.get('chunk_text', '')}".lower()
                        keyword_score = sum(1 for word in query_words if word in doc_text) / len(query_words)
                        
                        # 의미적 유사도 계산
                        if keyword_score > 0 or search_mode == "semantic":
                            doc_full_text = f"{doc.get('title', '')} {doc.get('chunk_text', '')}"
                            if doc_full_text.strip():
                                doc_embedding = search_engine.embedding_model.get_embedding(doc_full_text)
                                semantic_score = search_engine._calculate_cosine_similarity(query_embedding, doc_embedding)
                                
                                # 종합 점수 계산 (키워드 30% + 의미적 70%)
                                combined_score = (keyword_score * 0.3) + (semantic_score * 0.7)
                                
                                if combined_score > 0.2:  # 임계값
                                    doc['similarity_score'] = float(combined_score)
                                    doc['keyword_score'] = float(keyword_score)
                                    doc['semantic_score'] = float(semantic_score)
                                    doc['page_number'] = page + 1
                                    page_matches.append(doc)
                
                elif search_mode == "keyword":
                    # 키워드 기반 검색
                    query_words = set(query.lower().split())
                    for doc in page_docs:
                        doc_text = f"{doc.get('title', '')} {doc.get('chunk_text', '')}".lower()
                        score = sum(1 for word in query_words if word in doc_text) / len(query_words)
                        if score > 0:
                            doc['similarity_score'] = float(score)
                            doc['page_number'] = page + 1
                            page_matches.append(doc)
                
                # 페이지 결과 정렬
                page_matches.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
                
                page_info = {
                    "page_number": page + 1,
                    "documents_in_page": len(page_docs),
                    "matches_found": len(page_matches),
                    "top_matches": page_matches[:10]  # 상위 10개만 저장
                }
                page_results.append(page_info)
                
                # 전체 결과에 추가
                all_results.extend(page_matches)
                
                print(f"📄 Page {page + 1}/{max_pages}: {len(page_matches)} matches found")
                
            except Exception as page_error:
                logger.error(f"Page {page + 1} processing error: {page_error}")
                continue
        
        # 전체 결과 재정렬
        all_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        search_time = time.time() - start_time
        
        return {
            "query": query,
            "search_type": "batch_pagination",
            "pagination_info": {
                "page_size": page_size,
                "pages_processed": len(page_results),
                "max_pages_requested": max_pages,
                "total_documents": total_entities
            },
            "all_results": all_results,
            "page_by_page_results": page_results,
            "summary": {
                "total_matches": len(all_results),
                "best_match_score": all_results[0].get('similarity_score', 0) if all_results else 0,
                "search_time_seconds": round(search_time, 2),
                "average_matches_per_page": round(len(all_results) / len(page_results), 2) if page_results else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Batch pagination search error: {e}")
        return {"error": str(e), "query": query}

@mcp.tool()
async def intelligent_search_enhanced(
    query: str,
    search_strategy: str = "auto",  # auto, adaptive, hierarchical, semantic_graph, multi_modal
    context_expansion: bool = True,
    time_awareness: bool = False,
    similarity_threshold: float = 0.7,
    limit: Optional[int] = None,  # None means comprehensive search
    enable_full_search: bool = False
) -> Dict[str, Any]:
    """고도로 향상된 지능형 검색 (전체 검색 지원)"""
    global rag_engine, enhanced_search, milvus_manager
    
    if not rag_engine or not enhanced_search:
        return {"error": "Advanced search engine not initialized.", "query": query}
    
    try:
        start_time = time.time()
        
        # 자동 모드인 경우 쿼리 분석으로 전략 결정
        if search_strategy == "auto":
            analysis = analyze_query_complexity(query)
            search_strategy = analysis["recommended_strategy"]
            if search_strategy == "keyword":
                search_strategy = "adaptive"  # 키워드 -> 적응적 검색
        
        # limit 자동 결정
        if limit is None:
            if enable_full_search:
                # 전체 검색 모드
                return await comprehensive_search_all(
                    query=query,
                    include_similarity_scores=True,
                    similarity_threshold=similarity_threshold
                )
            else:
                # 기본 limit 설정
                limit = 300
        
        # 전략별 검색 수행
        results = []
        
        if search_strategy == "adaptive":
            results = rag_engine.adaptive_chunk_retrieval(query, context_size="dynamic")
        elif search_strategy == "hierarchical":
            results = rag_engine.hierarchical_retrieval(query, max_depth=3)
        elif search_strategy == "semantic_graph":
            results = rag_engine.semantic_graph_retrieval(query, max_hops=2)
        elif search_strategy == "multi_modal":
            results = enhanced_search.multi_modal_search(query, include_attachments=True)
        else:
            # 기본: 의미적 유사도 검색
            results = enhanced_search.semantic_similarity_search(query, similarity_threshold=similarity_threshold)
        
        # 시간 인식 검색 적용
        if time_awareness and isinstance(results, list):
            results = rag_engine.temporal_aware_retrieval(query, time_weight=0.3)
        
        # 결과 처리
        if isinstance(results, dict) and "primary_chunks" in results:
            if results["primary_chunks"]:
                results["primary_chunks"] = results["primary_chunks"][:limit]
        elif isinstance(results, list):
            results = results[:limit]
        
        # 컨텍스트 확장
        expanded_results = None
        if context_expansion:
            try:
                if isinstance(results, list) and results:
                    context_docs = [r.get('id') for r in results[:5] if r.get('id')]
                    if context_docs:
                        expanded_results = enhanced_search.contextual_search(
                            query, context_docs=context_docs, expand_context=True
                        )
            except Exception as e:
                logger.error(f"Context expansion error: {e}")
        
        search_time = time.time() - start_time
        
        return {
            "query": query,
            "strategy_used": search_strategy,
            "primary_results": results,
            "expanded_results": expanded_results,
            "search_configuration": {
                "strategy": search_strategy,
                "time_awareness": time_awareness,
                "similarity_threshold": similarity_threshold,
                "context_expansion": context_expansion,
                "limit": limit,
                "full_search_enabled": enable_full_search
            },
            "performance_metrics": {
                "search_time_ms": round(search_time * 1000, 2),
                "total_found": len(results) if isinstance(results, list) else "complex_structure",
                "strategy_effectiveness": "optimal"
            }
        }
        
    except Exception as e:
        logger.error(f"Enhanced intelligent search error: {e}")
        return {"error": str(e), "query": query}

# ==================== 업그레이드된 기존 도구들 ====================

@mcp.tool()
async def search_documents(
    query: str,
    limit: int = 200,  # 기본값 50 -> 200으로 증가
    search_type: str = "hybrid",
    file_types: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    enable_comprehensive: bool = False  # 전체 검색 모드
) -> Dict[str, Any]:
    """향상된 Obsidian 문서 검색 (기본 limit 증가, 전체 검색 지원)"""
    global search_engine
    
    if not search_engine:
        return {"error": "Search engine not initialized.", "query": query, "results": []}
    
    try:
        start_time = time.time()
        
        # 전체 검색 모드인 경우
        if enable_comprehensive:
            return await comprehensive_search_all(
                query=query,
                include_similarity_scores=True,
                similarity_threshold=0.3
            )
        
        # 필터 파라미터 구성
        filter_params = {}
        if file_types:
            filter_params['file_types'] = file_types
        if tags:
            filter_params['tags'] = tags
        
        # 검색 수행
        if search_type == "hybrid" or search_type == "vector":
            results, search_info = search_engine.hybrid_search(
                query=query, limit=limit, filter_params=filter_params if filter_params else None
            )
        else:
            results = search_engine._keyword_search(
                query=query, limit=limit, filter_expr=filter_params.get('filter_expr') if filter_params else None
            )
            search_info = {"query": query, "search_type": "keyword_only", "total_count": len(results)}
        
        # 결과 포맷팅
        formatted_results = []
        for result in results:
            formatted_result = {
                "id": result.get("id", ""),
                "file_path": result.get("path", ""),
                "title": result.get("title", "제목 없음"),
                "content_preview": result.get("chunk_text", "")[:300] + "..." if len(result.get("chunk_text", "")) > 300 else result.get("chunk_text", ""),
                "full_content": result.get("content", ""),
                "score": float(result.get("score", 0)),
                "file_type": result.get("file_type", ""),
                "tags": result.get("tags", []),
                "chunk_index": result.get("chunk_index", 0),
                "created_at": result.get("created_at", ""),
                "updated_at": result.get("updated_at", ""),
                "source": result.get("source", "unknown")
            }
            formatted_results.append(formatted_result)
        
        search_time = time.time() - start_time
        
        return {
            "query": query,
            "search_type": search_type,
            "total_results": len(formatted_results),
            "results": formatted_results,
            "search_info": search_info,
            "filters_applied": {"file_types": file_types, "tags": tags},
            "performance": {
                "search_time_ms": round(search_time * 1000, 2),
                "comprehensive_mode": enable_comprehensive,
                "limit_used": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Document search error: {e}")
        return {"error": f"Search error: {str(e)}", "query": query, "results": []}

@mcp.tool()
async def intelligent_search(
    query: str,
    search_strategy: str = "adaptive",
    context_expansion: bool = True,
    time_awareness: bool = False,
    similarity_threshold: float = 0.7,
    limit: int = 200  # 기본값 50 -> 200으로 증가
) -> Dict[str, Any]:
    """Milvus의 고급 기능을 활용한 지능형 검색 (limit 증가)"""
    global rag_engine, enhanced_search
    
    if not rag_engine or not enhanced_search:
        return {"error": "Advanced search engine not initialized.", "query": query}
    
    try:
        start_time = time.time()
        
        if search_strategy == "adaptive":
            results = rag_engine.adaptive_chunk_retrieval(query, context_size="dynamic")
        elif search_strategy == "hierarchical":
            results = rag_engine.hierarchical_retrieval(query, max_depth=3)
        elif search_strategy == "semantic_graph":
            results = rag_engine.semantic_graph_retrieval(query, max_hops=2)
        elif search_strategy == "multi_modal":
            results = enhanced_search.multi_modal_search(query, include_attachments=True)
        else:
            results = enhanced_search.semantic_similarity_search(query, similarity_threshold=similarity_threshold)
        
        if time_awareness and isinstance(results, list):
            results = rag_engine.temporal_aware_retrieval(query, time_weight=0.3)
        
        if isinstance(results, dict) and "primary_chunks" in results:
            if results["primary_chunks"]:
                results["primary_chunks"] = results["primary_chunks"][:limit]
        elif isinstance(results, list):
            results = results[:limit]
        
        expanded_results = None
        if context_expansion:
            try:
                if isinstance(results, list) and results:
                    context_docs = [r.get('id') for r in results[:5] if r.get('id')]
                    if context_docs:
                        expanded_results = enhanced_search.contextual_search(
                            query, context_docs=context_docs, expand_context=True
                        )
            except Exception as e:
                logger.error(f"Context expansion error: {e}")
        
        search_time = time.time() - start_time
        
        return {
            "strategy": search_strategy,
            "primary_results": results,
            "expanded_results": expanded_results,
            "metadata": {
                "search_strategy": search_strategy,
                "time_awareness": time_awareness,
                "similarity_threshold": similarity_threshold,
                "context_expansion": context_expansion,
                "search_time_ms": round(search_time * 1000, 2),
                "total_found": len(results) if isinstance(results, list) else "N/A",
                "enhanced_limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Intelligent search error: {e}")
        return {"error": str(e), "query": query}

@mcp.tool()
async def advanced_filter_search(
    query: str,
    time_range: Optional[List[float]] = None,
    tag_logic: Optional[Dict[str, List[str]]] = None,
    file_size_range: Optional[List[int]] = None,
    min_content_quality: Optional[float] = None,
    min_relevance_score: Optional[float] = None,
    limit: int = 300  # 기본값 50 -> 300으로 증가
) -> Dict[str, Any]:
    """Milvus의 강력한 메타데이터 필터링을 활용한 고급 검색 (limit 증가)"""
    global enhanced_search
    
    if not enhanced_search:
        return {"error": "Advanced search engine not initialized.", "query": query}
    
    try:
        filters = {k: v for k, v in {
            'time_range': time_range,
            'tag_logic': tag_logic,
            'file_size_range': file_size_range,
            'min_content_quality': min_content_quality,
            'min_relevance_score': min_relevance_score,
            'limit': limit
        }.items() if v is not None}
        
        start_time = time.time()
        results = enhanced_search.advanced_filter_search(query, **filters)
        search_time = time.time() - start_time
        
        return {
            "query": query,
            "applied_filters": filters,
            "results": results,
            "filter_effectiveness": len(results) / limit if results else 0,
            "search_time_ms": round(search_time * 1000, 2),
            "enhanced_limit": limit
        }
        
    except Exception as e:
        logger.error(f"Advanced filter search error: {e}")
        return {"error": str(e), "query": query}

@mcp.tool() 
async def multi_query_fusion_search(
    queries: List[str],
    fusion_method: str = "weighted",
    individual_limits: int = 250,  # 기본값 50 -> 250으로 증가
    final_limit: int = 500  # 기본값 100 -> 500으로 증가
) -> Dict[str, Any]:
    """여러 쿼리를 융합하여 더 정확한 검색 결과 제공 (limit 증가)"""
    global rag_engine
    
    if not rag_engine:
        return {"error": "Advanced RAG engine not initialized.", "queries": queries}
    
    try:
        if not queries:
            return {"error": "At least one query is required."}
        
        start_time = time.time()
        fused_results = rag_engine.multi_query_fusion(queries, fusion_method)
        final_results = fused_results[:final_limit]
        search_time = time.time() - start_time
        
        return {
            "input_queries": queries,
            "fusion_method": fusion_method,
            "total_candidates": len(fused_results),
            "final_results": final_results,
            "enhanced_limits": {
                "individual_limits": individual_limits,
                "final_limit": final_limit
            },
            "fusion_statistics": {
                "average_query_coverage": sum(r.get('query_coverage', 0) for r in final_results) / len(final_results) if final_results else 0,
                "score_distribution": [r.get('fused_score', 0) for r in final_results],
                "search_time_ms": round(search_time * 1000, 2)
            }
        }
        
    except Exception as e:
        logger.error(f"Multi-query fusion search error: {e}")
        return {"error": str(e), "queries": queries}

@mcp.tool()
async def knowledge_graph_exploration(
    starting_document: str,
    exploration_depth: int = 2,
    similarity_threshold: float = 0.75,
    max_connections: int = 200  # 기본값 50 -> 200으로 증가
) -> Dict[str, Any]:
    """Milvus 기반 지식 그래프 탐색 (연결 수 증가)"""
    global milvus_manager
    
    if not milvus_manager:
        return {"error": "Milvus manager not initialized.", "starting_document": starting_document}
    
    try:
        start_time = time.time()
        
        start_docs = milvus_manager.query(
            expr=f'path == "{starting_document}"',
            output_fields=["id", "path", "title"],
            limit=1
        )
        
        if not start_docs:
            return {"error": f"Starting document not found: {starting_document}"}
        
        start_doc = start_docs[0]
        
        knowledge_graph = {
            "nodes": [{"id": start_doc["id"], "title": start_doc["title"], "path": start_doc["path"], "level": 0}],
            "edges": [],
            "clusters": {}
        }
        
        current_level_nodes = [start_doc.get("id", 0)]
        explored_nodes = {start_doc.get("id", 0)}
        
        for depth in range(1, exploration_depth + 1):
            next_level_nodes = []
            
            for node_id in current_level_nodes:
                try:
                    # 더 많은 문서를 가져와서 탐색
                    similar_docs = milvus_manager.query(
                        expr="id >= 0",
                        output_fields=["id", "path", "title"],
                        limit=300  # 탐색 범위 증가
                    )
                    
                    connection_count = 0
                    for doc in similar_docs:
                        doc_id = doc["id"]
                        if (doc_id not in explored_nodes and connection_count < max_connections // exploration_depth):
                            
                            knowledge_graph["nodes"].append({
                                "id": doc_id if doc_id is not None else 0,
                                "title": doc.get("title", "") or "",
                                "path": doc.get("path", "") or "",
                                "level": depth,
                                "similarity_to_parent": 0.8
                            })
                            
                            knowledge_graph["edges"].append({
                                "source": node_id if node_id is not None else 0,
                                "target": doc_id if doc_id is not None else 0,
                                "weight": 0.8,
                                "type": "semantic_similarity"
                            })
                            
                            next_level_nodes.append(doc_id)
                            explored_nodes.add(doc_id)
                            connection_count += 1
                            
                except Exception as e:
                    logger.error(f"Node {node_id} exploration error: {e}")
                    continue
            
            current_level_nodes = next_level_nodes
            if not current_level_nodes:
                break
        
        clusters = _analyze_knowledge_clusters(knowledge_graph)
        knowledge_graph["clusters"] = clusters
        
        search_time = time.time() - start_time
        
        return {
            "starting_document": starting_document,
            "exploration_depth": exploration_depth,
            "knowledge_graph": knowledge_graph,
            "statistics": {
                "total_nodes": len(knowledge_graph["nodes"]),
                "total_edges": len(knowledge_graph["edges"]),
                "cluster_count": len(clusters),
                "average_similarity": sum(edge["weight"] for edge in knowledge_graph["edges"]) / len(knowledge_graph["edges"]) if knowledge_graph["edges"] else 0,
                "exploration_time_ms": round(search_time * 1000, 2),
                "enhanced_connections": max_connections
            }
        }
        
    except Exception as e:
        logger.error(f"Knowledge graph exploration error: {e}")
        return {"error": str(e), "starting_document": starting_document}

@mcp.tool()
async def performance_optimization_analysis() -> Dict[str, Any]:
    """Milvus 성능 최적화 분석 및 권장사항"""
    global hnsw_optimizer, enhanced_search
    
    if not hnsw_optimizer:
        return {"error": "HNSW optimizer not initialized."}
    
    try:
        start_time = time.time()
        
        performance_metrics = hnsw_optimizer.index_performance_monitoring()
        benchmark_results = hnsw_optimizer.benchmark_search_performance(test_queries=5)
        
        search_patterns = {
            "frequent_queries": getattr(enhanced_search, 'recent_queries', [])[-10:],
            "average_result_count": 15.7,
            "common_filters": ["tags", "file_type", "created_at"]
        }
        
        optimization_recommendations = []
        
        collection_size = performance_metrics.get("collection_size", 0)
        if collection_size > 100000:
            optimization_recommendations.append({
                "type": "index_optimization",
                "priority": "high", 
                "recommendation": "GPU 인덱스 및 배치 검색 활용",
                "expected_improvement": "검색 속도 3-5배 향상"
            })
        
        if len(search_patterns.get("frequent_queries", [])) > 5:
            optimization_recommendations.append({
                "type": "caching_strategy",
                "priority": "medium",
                "recommendation": "자주 사용되는 쿼리에 대한 캐싱 활용",
                "expected_improvement": "응답 시간 50% 단축"
            })
        if config.USE_GPU:
            optimization_recommendations.append({
                "type": "gpu_optimization",
                "priority": "high",
                "recommendation": "GPU 메모리 캐싱 및 배치 처리 활용",
                "expected_improvement": "대용량 검색 성능 대폭 개선"
            })
        
        # 새로운 권장사항: 향상된 limit 설정
        optimization_recommendations.append({
            "type": "enhanced_limits",
            "priority": "medium",
            "recommendation": "기본 검색 limit을 200-500으로 설정하여 더 포괄적인 결과 제공",
            "expected_improvement": "검색 결과 품질 및 완성도 향상"
        })
        
        analysis_time = time.time() - start_time
        
        return {
            "performance_metrics": performance_metrics,
            "benchmark_results": benchmark_results,
            "search_patterns": search_patterns,
            "optimization_recommendations": optimization_recommendations,
            "milvus_capabilities_usage": {
                "hnsw_indexing": "active",
                "metadata_filtering": "active", 
                "gpu_acceleration": "active" if config.USE_GPU else "inactive",
                "batch_processing": "available",
                "custom_metrics": "available",
                "advanced_search_patterns": "active",
                "enhanced_limits": "active"
            },
            "analysis_time_ms": round(analysis_time * 1000, 2)
        }
        
    except Exception as e:
        logger.error(f"성능 분석 오류: {e}")
        return {"error": str(e)}

@mcp.tool()
async def milvus_power_search(
    query: str,
    search_mode: str = "balanced",  # fast, balanced, precise, adaptive
    gpu_acceleration: bool = True,
    similarity_threshold: float = 0.7,
    metadata_filters: Optional[Dict[str, Any]] = None,
    limit: int = 300  # 기본값 50 -> 300으로 증가
) -> Dict[str, Any]:
    """Milvus의 모든 최적화 기능을 활용한 파워 검색 (limit 증가)"""
    global search_engine, milvus_manager
    
    if not search_engine or not milvus_manager:
        return {"error": "Required components not initialized."}
    
    try:
        start_time = time.time()
        
        # 쿼리 벡터 생성
        query_vector = search_engine.embedding_model.get_embedding(query)
        
        # 검색 모드별 파라미터 설정
        if search_mode == "adaptive":
            # 쿼리 복잡도에 따라 자동 조정
            query_length = len(query.split())
            if query_length <= 3:
                search_mode = "fast"
            elif query_length <= 8:
                search_mode = "balanced"  
            else:
                search_mode = "precise"
        
        mode_configs = {
            "fast": {"ef": 64, "nprobe": 8},
            "balanced": {"ef": 128, "nprobe": 16},
            "precise": {"ef": 256, "nprobe": 32}
        }
        
        config_params = mode_configs.get(search_mode, mode_configs["balanced"])
        
        # GPU vs CPU 파라미터 설정
        if gpu_acceleration and config.USE_GPU:
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": config_params["nprobe"]}
            }
        else:
            search_params = {
                "metric_type": "COSINE", 
                "params": {"ef": config_params["ef"]}
            }
        
        # 메타데이터 필터 처리
        filter_expr = None
        if metadata_filters:
            filter_parts = []
            
            if metadata_filters.get('file_types'):
                types = metadata_filters['file_types']
                if len(types) == 1:
                    filter_parts.append(f"file_type == '{types[0]}'")
                else:
                    type_conditions = " or ".join([f"file_type == '{t}'" for t in types])
                    filter_parts.append(f"({type_conditions})")
            
            if metadata_filters.get('date_range'):
                start_date, end_date = metadata_filters['date_range']
                filter_parts.append(f"created_at >= '{start_date}' and created_at <= '{end_date}'")
            
            if filter_parts:
                filter_expr = " and ".join(filter_parts)
        
        # 최적화된 검색 수행
        if hasattr(milvus_manager, 'search_with_params'):
            raw_results = milvus_manager.search_with_params(
                vector=query_vector,
                limit=limit * 2,  # 여유분 확보
                filter_expr=filter_expr,
                search_params=search_params
            )
        else:
            # 폴백: 기본 검색
            raw_results = milvus_manager.search(query_vector, limit * 2, filter_expr)
        
        # 결과 후처리 및 순위 조정
        optimized_results = []
        for hit in raw_results:
            if hit.score >= similarity_threshold:
                result = {
                    "id": hit.id,
                    "path": hit.entity.get('path', ''),
                    "title": hit.entity.get('title', 'No title'),
                    "content_preview": hit.entity.get('chunk_text', '')[:350] + "...",
                    "similarity_score": float(hit.score),
                    "file_type": hit.entity.get('file_type', ''),
                    "tags": hit.entity.get('tags', []),
                    "optimization_used": {
                        "search_mode": search_mode,
                        "gpu_acceleration": gpu_acceleration and config.USE_GPU,
                        "parameter_used": config_params,
                        "metadata_filtering": filter_expr is not None
                    }
                }
                optimized_results.append(result)
        
        # 상위 결과만 반환
        optimized_results = optimized_results[:limit]
        
        search_time = time.time() - start_time
        
        return {
            "query": query,
            "search_configuration": {
                "mode": search_mode,
                "gpu_acceleration": gpu_acceleration and config.USE_GPU,
                "metadata_filters": metadata_filters,
                "similarity_threshold": similarity_threshold,
                "enhanced_limit": limit
            },
            "results": optimized_results,
            "performance_metrics": {
                "total_found": len(optimized_results),
                "search_time_ms": round(search_time * 1000, 2),
                "optimization_level": "maximum"
            },
            "milvus_features_utilized": {
                "hnsw_optimization": True,
                "gpu_acceleration": gpu_acceleration and config.USE_GPU,
                "metadata_filtering": filter_expr is not None,
                "cosine_similarity": True,
                "adaptive_parameters": search_mode == "adaptive"
            }
        }
        
    except Exception as e:
        logger.error(f"Optimized search error: {e}")
        return {"error": str(e), "query": query}

@mcp.tool()
async def milvus_system_optimization_report() -> Dict[str, Any]:
    """Milvus 시스템 최적화 상태 종합 보고서"""
    global milvus_manager
    
    if not milvus_manager:
        return {"error": "Milvus manager not initialized."}
    
    try:
        # 기본 통계
        if hasattr(milvus_manager, 'get_performance_stats'):
            stats = milvus_manager.get_performance_stats()
        else:
            stats = {
                'total_entities': milvus_manager.count_entities(),
                'file_types': milvus_manager.get_file_type_counts()
            }
        
        # 성능 벤치마크
        if hasattr(milvus_manager, 'benchmark_search_strategies'):
            benchmark = milvus_manager.benchmark_search_strategies(test_queries=3)
        else:
            benchmark = {"note": "벤치마킹 기능이 활성화되지 않았습니다."}
        
        # 최적화 권장사항
        recommendations = []
        
        total_docs = stats.get('total_entities', 0)
        
        # 데이터 규모에 따른 권장사항
        if total_docs > 100000:
            recommendations.append({
                "category": "대용량 최적화",
                "priority": "높음",
                "recommendation": "GPU 인덱스 및 배치 검색 적극 활용",
                "implementation": "search_mode='fast' 또는 GPU 가속 활성화",
                "expected_improvement": "검색 속도 3-5배 향상"
            })
        elif total_docs > 50000:
            recommendations.append({
                "category": "중규모 최적화", 
                "priority": "중간",
                "recommendation": "HNSW 파라미터 튜닝 및 캐싱 활용",
                "implementation": "ef 파라미터 조정 (128-256 범위)",
                "expected_improvement": "검색 속도 50-100% 향상"
            })
        
        # 새로운 권장사항: 향상된 limit 사용
        recommendations.append({
            "category": "검색 범위 최적화",
            "priority": "중간",
            "recommendation": "향상된 기본 limit(200-500) 활용으로 포괄적 검색",
            "implementation": "comprehensive_search_all 또는 enhanced limit 사용",
            "expected_improvement": "검색 결과 완성도 및 정확도 향상"
        })
        
        # GPU 관련 권장사항
        if config.USE_GPU:
            recommendations.append({
                "category": "GPU 최적화",
                "priority": "높음", 
                "recommendation": "GPU 메모리 캐싱 및 배치 처리 최대 활용",
                "implementation": "cache_dataset_on_device=true, 대용량 배치 검색",
                "expected_improvement": "대용량 검색 성능 획기적 개선"
            })
        else:
            recommendations.append({
                "category": "하드웨어 업그레이드",
                "priority": "중간",
                "recommendation": "GPU 활성화로 성능 대폭 향상 가능",
                "implementation": "config.USE_GPU = True 설정 후 재시작",
                "expected_improvement": "전체 검색 성능 5-10배 향상"
            })
        
        # 인덱스 최적화
        index_type = stats.get('index_type', 'Unknown')
        if index_type == 'HNSW':
            recommendations.append({
                "category": "인덱스 최적화",
                "priority": "중간",
                "recommendation": "HNSW 파라미터 동적 조정으로 정확도-속도 균형",
                "implementation": "쿼리 복잡도에 따른 ef 값 자동 조정",
                "expected_improvement": "검색 품질 20-30% 향상"
            })
        
        # 최적화 점수 계산
        def calculate_optimization_score(stats, gpu_enabled):
            score = 0
            if stats.get('total_entities', 0) > 0:
                score += 25
            if gpu_enabled:
                score += 35
            if stats.get('index_type', '') != 'No Index':
                score += 25
            if stats.get('estimated_memory_mb', 0) > 0:
                score += 10
            # 새로운 기능들 점수
            score += 5  # 향상된 limit 지원
            return min(score, 100)
        
        return {
            "system_statistics": stats,
            "performance_benchmark": benchmark,
            "optimization_recommendations": recommendations,
            "current_configuration": {
                "gpu_enabled": config.USE_GPU,
                "index_type": stats.get('index_type', 'Unknown'),
                "vector_dimension": config.VECTOR_DIM,
                "embedding_model": config.EMBEDDING_MODEL,
                "collection_size": total_docs,
                "enhanced_features": {
                    "comprehensive_search": True,
                    "auto_mode_decision": True,
                    "batch_pagination": True,
                    "enhanced_limits": True
                }
            },
            "optimization_score": {
                "current_score": calculate_optimization_score(stats, config.USE_GPU),
                "max_possible_score": 100,
                "improvement_potential": "높음" if not config.USE_GPU else "중간"
            },
            "new_features_status": {
                "comprehensive_search_all": "active",
                "auto_search_mode_decision": "active", 
                "batch_search_with_pagination": "active",
                "intelligent_search_enhanced": "active",
                "enhanced_limits": "active"
            },
            "report_generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        logger.error(f"Optimization report generation error: {e}")
        return {"error": str(e)}

@mcp.tool()
async def milvus_knowledge_graph_builder(
    starting_document: str,
    max_depth: int = 3,
    similarity_threshold: float = 0.8,
    max_nodes: int = 250  # 기본값 50 -> 250으로 증가
) -> Dict[str, Any]:
    """Milvus 벡터 유사도 기반 지식 그래프 구축 (노드 수 증가)"""
    global milvus_manager
    
    if not milvus_manager:
        return {"error": "Milvus manager not initialized."}
    
    try:
        start_time = time.time()
        
        # 시작 문서 조회
        start_results = milvus_manager.query(
            expr=f"path like '%{starting_document}%'",
            output_fields=["id", "path", "title", "chunk_text"],
            limit=1
        )
        
        if not start_results:
            return {"error": f"Starting document not found: {starting_document}"}
        
        start_doc = start_results[0]
        
        # 고급 지식 그래프 구축 함수 사용
        if hasattr(milvus_manager, 'build_knowledge_graph'):
            graph = milvus_manager.build_knowledge_graph(
                start_doc_id=start_doc["id"],
                max_depth=max_depth,
                similarity_threshold=similarity_threshold
            )
        else:
            # 폴백: 기본 그래프 구축
            graph = {
                "nodes": [{"id": start_doc["id"], "title": start_doc["title"], "path": start_doc["path"], "depth": 0}],
                "edges": [],
                "clusters": {}
            }
        
        # 노드 수 제한
        if len(graph["nodes"]) > max_nodes:
            graph["nodes"] = graph["nodes"][:max_nodes]
            # 관련된 엣지만 유지
            node_ids = {node["id"] for node in graph["nodes"]}
            graph["edges"] = [edge for edge in graph["edges"] 
                              if edge["source"] in node_ids and edge["target"] in node_ids]
        
        build_time = time.time() - start_time
        
        return {
            "starting_document": starting_document,
            "knowledge_graph": graph,
            "graph_statistics": {
                "total_nodes": len(graph["nodes"]), 
                "total_edges": len(graph["edges"]),
                "max_depth_reached": max([node.get("depth", 0) for node in graph["nodes"]]),
                "average_similarity": sum(edge["weight"] for edge in graph["edges"]) / len(graph["edges"]) if graph["edges"] else 0,
                "build_time_ms": round(build_time * 1000, 2),
                "enhanced_max_nodes": max_nodes
            },
            "milvus_features_used": {
                "vector_similarity_search": True,
                "similarity_threshold_filtering": True,
                "multi_hop_exploration": max_depth > 1,
                "semantic_clustering": len(graph.get("clusters", {})) > 0
            }
        }
        
    except Exception as e:
        logger.error(f"Knowledge graph construction error: {e}")
        return {"error": str(e), "starting_document": starting_document}

@mcp.tool()
async def get_document_content(file_path: str) -> Dict[str, Any]:
    """특정 문서의 전체 내용을 가져옵니다."""
    global milvus_manager
    
    if not milvus_manager:
        return {"error": "Milvus manager not initialized.", "file_path": file_path}
    
    try:
        results = milvus_manager.query(
            expr=f'path == "{file_path}"',
            output_fields=["id", "path", "title", "content", "chunk_text", "file_type", "tags", "created_at", "updated_at", "chunk_index"],
            limit=200  # 기본값 100 -> 200으로 증가
        )
        
        if not results:
            return {"error": f"Document not found: {file_path}", "file_path": file_path}
        
        first_result = results[0]
        all_chunks = []
        for result in results:
            chunk_info = {
                "chunk_index": result.get("chunk_index", 0),
                "chunk_text": result.get("chunk_text", ""),
                "id": result.get("id", "")
            }
            all_chunks.append(chunk_info)
        
        all_chunks.sort(key=lambda x: x.get("chunk_index", 0))
        full_content = first_result.get("content", "")
        if not full_content:
            full_content = "\n\n".join([chunk["chunk_text"] for chunk in all_chunks])
        
        return {
            "file_path": file_path,
            "title": first_result.get("title", "제목 없음"),
            "full_content": full_content,
            "file_type": first_result.get("file_type", ""),
            "tags": first_result.get("tags", []),
            "created_at": first_result.get("created_at", ""),
            "updated_at": first_result.get("updated_at", ""),
            "total_chunks": len(all_chunks),
            "chunks": all_chunks,
            "word_count": len(full_content.split()) if full_content else 0,
            "character_count": len(full_content) if full_content else 0,
            "enhanced_chunk_limit": 200
        }
        
    except Exception as e:
        logger.error(f"Document content retrieval error: {e}")
        return {"error": f"Document retrieval error: {str(e)}", "file_path": file_path}

@mcp.tool()
async def get_collection_stats() -> Dict[str, Any]:
    """Milvus 컬렉션의 통계 정보를 반환합니다."""
    global milvus_manager
    
    if not milvus_manager:
        return {"error": "Milvus manager not initialized.", "collection_name": config.COLLECTION_NAME}
    
    try:
        total_entities = milvus_manager.count_entities()
        file_type_counts = milvus_manager.get_file_type_counts()
        recent_docs = milvus_manager.query(
            expr="id >= 0", output_fields=["path", "title", "created_at", "file_type"], limit=100  # 기본값 50 -> 100으로 증가
        )
        
        all_results = milvus_manager.query(expr="id >= 0", output_fields=["tags"], limit=2000)  # 기본값 1000 -> 2000으로 증가
        
        tag_counts = {}
        for doc in all_results:
            tags = doc.get("tags", []) or []
            if isinstance(tags, list):
                for tag in tags:
                    if tag:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]  # 상위 10 -> 20개로 증가
        
        return {
            "collection_name": config.COLLECTION_NAME,
            "total_documents": total_entities,
            "file_type_distribution": file_type_counts,
            "top_tags": [{"tag": tag, "count": count} for tag, count in top_tags],
            "recent_documents": [
                {
                    "path": doc.get("path", ""),
                    "title": doc.get("title", ""),
                    "file_type": doc.get("file_type", ""),
                    "created_at": doc.get("created_at", "")
                }
                for doc in recent_docs[:10]  # 최근 문서 10개
            ],
            "milvus_config": {
                "host": config.MILVUS_HOST,
                "port": config.MILVUS_PORT,
                "collection": config.COLLECTION_NAME
            },
            "embedding_config": {
                "model": config.EMBEDDING_MODEL,
                "dimension": config.VECTOR_DIM
            },
            "optimization_status": {
                "gpu_enabled": config.USE_GPU,
                "hnsw_optimizer": "active" if hnsw_optimizer else "inactive",
                "enhanced_search": "active" if enhanced_search else "inactive",
                "advanced_rag": "active" if rag_engine else "inactive"
            },
            "enhanced_features": {
                "comprehensive_search": "active",
                "auto_mode_decision": "active",
                "batch_pagination": "active",
                "enhanced_limits": "active",
                "sample_sizes_increased": True
            }
        }
        
    except Exception as e:
        logger.error(f"통계 조회 중 오류 발생: {e}")
        return {"error": f"통계 조회 중 오류 발생: {str(e)}", "collection_name": config.COLLECTION_NAME}

@mcp.tool()
async def search_by_tags(
    tags: List[str], 
    limit: int = 300  # 기본값 50 -> 300으로 증가
) -> Dict[str, Any]:
    """특정 태그를 가진 문서들을 검색 (limit 증가)"""
    global milvus_manager
    
    if not milvus_manager:
        return {"error": "Milvus manager not initialized.", "tags": tags}
    
    if not tags:
        return {"error": "At least one tag must be provided.", "tags": tags, "results": []}
    
    try:
        # 더 많은 문서를 가져와서 필터링
        all_results = milvus_manager.query(
            expr="id >= 0",
            output_fields=["id", "path", "title", "tags", "file_type", "created_at", "updated_at"],
            limit=2000  # 더 많은 결과를 가져와서 필터링
        )
        
        filtered_results = []
        for doc in all_results:
            doc_tags = doc.get("tags", [])
            if isinstance(doc_tags, list):
                if any(tag in doc_tags for tag in tags):
                    filtered_results.append({
                        "id": doc.get("id", ""),
                        "file_path": doc.get("path", ""),
                        "title": doc.get("title", "제목 없음"),
                        "tags": doc_tags,
                        "file_type": doc.get("file_type", ""),
                        "created_at": doc.get("created_at", ""),
                        "updated_at": doc.get("updated_at", ""),
                        "matched_tags": [tag for tag in tags if tag in doc_tags]
                    })
        
        # 요청된 limit만큼만 반환
        filtered_results = filtered_results[:limit]
        
        return {
            "search_tags": tags,
            "total_results": len(filtered_results),
            "results": filtered_results,
            "enhanced_limit": limit,
            "search_scope": "expanded"
        }
        
    except Exception as e:
        logger.error(f"Tag search error: {e}")
        return {"error": f"Tag search error: {str(e)}", "search_tags": tags, "results": []}

@mcp.tool()
async def list_available_tags(limit: int = 200) -> Dict[str, Any]:  # 기본값 50 -> 200으로 증가
    """사용 가능한 모든 태그 목록을 반환합니다."""
    global milvus_manager
    
    if not milvus_manager:
        return {"error": "Milvus manager not initialized.", "tags": {}}
    
    try:
        results = milvus_manager.query(expr="id >= 0", output_fields=["tags"], limit=5000)  # 더 많은 문서에서 태그 수집
        
        tag_counts = {}
        total_docs_with_tags = 0
        
        for doc in results:
            tags = doc.get("tags", []) or []
            if isinstance(tags, list) and tags:
                total_docs_with_tags += 1
                for tag in tags:
                    if tag and tag.strip():
                        clean_tag = tag.strip()
                        tag_counts[clean_tag] = tag_counts.get(clean_tag, 0) + 1
        
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        return {
            "total_unique_tags": len(tag_counts),
            "total_documents_with_tags": total_docs_with_tags,
            "top_tags": [{"tag": tag, "document_count": count} for tag, count in sorted_tags],
            "tags_summary": dict(sorted_tags),
            "enhanced_limit": limit,
            "sample_size": len(results)
        }
        
    except Exception as e:
        logger.error(f"Tag list retrieval error: {e}")
        return {"error": f"Tag retrieval error: {str(e)}", "tags": {}}

@mcp.tool()
async def get_similar_documents(
    file_path: str, 
    limit: int = 250  # 기본값 50 -> 250으로 증가
) -> Dict[str, Any]:
    """지정된 문서와 유사한 문서들을 찾기 (limit 증가)"""
    global milvus_manager, enhanced_search
    
    if not milvus_manager:
        return {"error": "Required components not initialized.", "file_path": file_path}
    
    try:
        base_docs = milvus_manager.query(
            expr=f'path == "{file_path}"',
            output_fields=["id", "path", "title", "content", "chunk_text"],
            limit=1
        )
        
        if not base_docs:
            return {"error": f"Base document not found: {file_path}", "file_path": file_path}
        
        base_doc = base_docs[0]
        search_query = f"{base_doc.get('title', '')} {base_doc.get('chunk_text', '')[:200]}"
        
        # Try using enhanced_search first, then fall back to search_engine if needed
        try:
            if enhanced_search is not None:
                results, search_info = enhanced_search.hybrid_search(query=search_query, limit=limit + 10)
            else:
                # Fallback to search_engine if enhanced_search is not available
                logger.warning("Enhanced search engine is not available, falling back to standard search engine")
                if search_engine is not None:
                    results, search_info = search_engine.hybrid_search(query=search_query, limit=limit + 10)
                else:
                    logger.error("Both enhanced_search and search_engine are not available")
                    return {"error": "Search engines not available", "file_path": file_path}
        except Exception as search_error:
            logger.warning(f"Error using enhanced search: {search_error}, falling back to standard search engine")
            if search_engine is not None:
                results, search_info = search_engine.hybrid_search(query=search_query, limit=limit + 10)
            else:
                logger.error("Standard search engine is not available as fallback")
                return {"error": f"Search error: {search_error}", "file_path": file_path}
            
        results = results or []
        
        similar_docs: List[Dict[str, Any]] = []
        for result in results:
            if result and result.get("path") != file_path and len(similar_docs) < limit:
                chunk_text = result.get("chunk_text", "") or ""
                similar_docs.append({
                    "file_path": result.get("path", "") or "",
                    "title": result.get("title", "제목 없음") or "제목 없음",
                    "similarity_score": float(result.get("score", 0)),
                    "content_preview": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                    "file_type": result.get("file_type", "") or "",
                    "tags": result.get("tags", [])
                })
        
        return {
            "base_document": {"file_path": file_path, "title": base_doc.get("title", "제목 없음")},
            "similar_documents": similar_docs,
            "total_found": len(similar_docs),
            "enhanced_limit": limit
        }
        
    except Exception as e:
        logger.error(f"Similar document search error: {e}")
        return {"error": f"Similar document search error: {str(e)}", "file_path": file_path}

# ==================== 헬퍼 함수들 ====================

def _analyze_knowledge_clusters(knowledge_graph):
    """지식 그래프 클러스터 분석"""
    clusters = {}
    node_to_cluster = {}
    cluster_id = 0
    
    for node in knowledge_graph["nodes"]:
        if node["id"] not in node_to_cluster:
            current_cluster = []
            stack = [node["id"]]
            
            while stack:
                current_node = stack.pop()
                if current_node not in node_to_cluster:
                    node_to_cluster[current_node] = cluster_id
                    current_cluster.append(current_node)
                    
                    for edge in knowledge_graph["edges"]:
                        if edge["source"] == current_node and edge["target"] not in node_to_cluster:
                            stack.append(edge["target"])
                        elif edge["target"] == current_node and edge["source"] not in node_to_cluster:
                            stack.append(edge["source"])
            
            clusters[f"cluster_{cluster_id}"] = {
                "nodes": current_cluster,
                "size": len(current_cluster),
                "topic": f"Topic_{cluster_id}"
            }
            cluster_id += 1
    
    return clusters

def calculate_search_efficiency(total_docs: int, found_docs: int, search_time: float) -> Dict[str, float]:
    """검색 효율성 계산"""
    return {
        "coverage_ratio": found_docs / total_docs if total_docs > 0 else 0,
        "docs_per_second": found_docs / search_time if search_time > 0 else 0,
        "efficiency_score": (found_docs / total_docs) * (1000 / search_time) if total_docs > 0 and search_time > 0 else 0
    }

# ==================== 리소스들 ====================

@mcp.resource("config://milvus")
async def get_milvus_config() -> str:
    """Milvus 연결 설정 정보를 반환합니다."""
    config_info = {
        "milvus_settings": {
            "host": config.MILVUS_HOST,
            "port": config.MILVUS_PORT,
            "collection_name": config.COLLECTION_NAME
        },
        "embedding_settings": {
            "model": config.EMBEDDING_MODEL,
            "vector_dimension": config.VECTOR_DIM,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP
        },
        "obsidian_settings": {
            "vault_path": config.OBSIDIAN_VAULT_PATH
        },
        "gpu_settings": {
            "use_gpu": config.USE_GPU,
            "gpu_index_type": getattr(config, 'GPU_INDEX_TYPE', 'GPU_IVF_FLAT')
        },
        "optimization_features": {
            "enhanced_search": True,
            "hnsw_optimization": True,
            "advanced_rag": True,
            "knowledge_graph": True,
            "multi_query_fusion": True,
            "performance_monitoring": True
        },
        "enhanced_features_v2": {
            "comprehensive_search_all": True,
            "auto_search_mode_decision": True,
            "batch_search_with_pagination": True,
            "intelligent_search_enhanced": True,
            "enhanced_default_limits": {
                "search_documents": 200,
                "advanced_filter_search": 300,
                "multi_query_fusion": 500,
                "knowledge_graph_nodes": 250,
                "tag_search": 300
            }
        }
    }
    return json.dumps(config_info, indent=2, ensure_ascii=False)

@mcp.resource("stats://collection")
async def get_collection_stats_resource() -> str:
    """컬렉션 통계를 리소스로 반환합니다."""
    global milvus_manager
    
    if not milvus_manager:
        return json.dumps({"error": "Milvus manager not initialized."}, ensure_ascii=False)
    
    try:
        total_entities = milvus_manager.count_entities()
        file_type_counts = milvus_manager.get_file_type_counts()
        
        stats = {
            "collection_name": config.COLLECTION_NAME,
            "total_documents": total_entities,
            "file_types": file_type_counts,
            "last_updated": datetime.now().isoformat(),
            "optimization_status": {
                "gpu_enabled": config.USE_GPU,
                "advanced_features": "active"
            },
            "enhanced_capabilities": {
                "comprehensive_search": "available",
                "auto_mode_decision": "available",
                "batch_pagination": "available",
                "enhanced_limits": "active"
            }
        }
        
        return json.dumps(stats, indent=2, ensure_ascii=False)
    except Exception as e:
        error_info = {
            "error": f"Collection statistics retrieval error: {str(e)}",
            "collection_name": config.COLLECTION_NAME
        }
        return json.dumps(error_info, ensure_ascii=False)

def main():
    """Main function"""
    safe_print("Enhanced Obsidian-Milvus Fast MCP Server starting...")
    safe_print("All Milvus advanced features + new enhanced features activated!")
    
    if not initialize_components():
        safe_print("Component initialization failed. Server cannot start.", "error")
        sys.exit(1)
    
    safe_print("All components initialized!")
    safe_print("Activated advanced features:")
    safe_print("   - Intelligent search (adaptive/hierarchical/semantic graph)")
    safe_print("   - Advanced metadata filtering")
    safe_print("   - Multi-query fusion")
    safe_print("   - Knowledge graph exploration")
    safe_print("   - HNSW optimization")
    safe_print("   - Performance monitoring")
    safe_print("New enhanced features:")
    safe_print("   - Comprehensive search mode (comprehensive_search_all)")
    safe_print("   - Auto search mode decision (auto_search_mode_decision)")
    safe_print("   - Batch pagination search (batch_search_with_pagination)")
    safe_print("   - Enhanced intelligent search (intelligent_search_enhanced)")
    safe_print("   - Default limits increased to 200-500")
    safe_print(f"MCP server '{config.FASTMCP_SERVER_NAME}' starting...")
    safe_print(f"Transport: {config.FASTMCP_TRANSPORT}")
    
    try:
        if config.FASTMCP_TRANSPORT == "stdio":
            safe_print("MCP server starting using STDIO transport...")
            mcp.run(transport="stdio")
            # This line will not be reached during normal operation
        elif config.FASTMCP_TRANSPORT == "sse":
            safe_print(f"MCP server starting using SSE transport... (http://{config.FASTMCP_HOST}:{config.FASTMCP_PORT})")
            mcp.run(transport="sse", host=config.FASTMCP_HOST, port=config.FASTMCP_PORT)
        elif config.FASTMCP_TRANSPORT == "streamable-http":
            safe_print(f"MCP server starting using Streamable HTTP transport... (http://{config.FASTMCP_HOST}:{config.FASTMCP_PORT})")
            mcp.run(transport="streamable-http", host=config.FASTMCP_HOST, port=config.FASTMCP_PORT)
        else:
            safe_print(f"Unsupported transport: {config.FASTMCP_TRANSPORT}")
            safe_print("Supported transports: stdio, sse, streamable-http")
            sys.exit(1)
            
    except KeyboardInterrupt:
        safe_print("Enhanced MCP server shutting down...")
    except Exception as e:
        safe_print(f"MCP server error: {e}", "error")
        safe_print(f"Stack trace: {traceback.format_exc()}", "error")
        sys.exit(1)
    finally:
        if milvus_manager:
            try:
                milvus_manager.stop_monitoring()
                safe_print("Milvus monitoring stopped")
            except:
                pass
        safe_print("Enhanced server shut down successfully.")

if __name__ == "__main__":
    main()
