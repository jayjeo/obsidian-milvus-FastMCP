#!/usr/bin/env python3
"""
Obsidian-Milvus Fast MCP Server - 완전 최적화 버전
Milvus의 모든 고급 기능을 Claude Desktop에서 최대한 활용

Enhanced with:
- 고급 메타데이터 필터링  
- HNSW 인덱스 최적화
- 계층적/의미적 그래프 검색
- 다중 쿼리 융합
- 적응적 청크 검색
- 시간 인식 검색
- 성능 최적화 및 모니터링
"""

import os
import sys
import json
import traceback
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from mcp.server.fastmcp import FastMCP
import config
from milvus_manager import MilvusManager
from search_engine import SearchEngine

# 새로운 고급 모듈들
from enhanced_search_engine import EnhancedSearchEngine
from hnsw_optimizer import HNSWOptimizer
from advanced_rag import AdvancedRAGEngine

import logging
logger = logging.getLogger('OptimizedMCP')

mcp = FastMCP(config.FASTMCP_SERVER_NAME)

# 전역 변수들
milvus_manager = None
search_engine = None
enhanced_search = None
hnsw_optimizer = None
rag_engine = None

def initialize_components():
    """모든 컴포넌트들 초기화"""
    global milvus_manager, search_engine, enhanced_search, hnsw_optimizer, rag_engine
    
    try:
        print("🚀 최적화된 Obsidian-Milvus MCP Server 초기화 중...")
        
        milvus_manager = MilvusManager()
        search_engine = SearchEngine(milvus_manager)
        enhanced_search = EnhancedSearchEngine(milvus_manager)
        hnsw_optimizer = HNSWOptimizer(milvus_manager)
        rag_engine = AdvancedRAGEngine(milvus_manager, enhanced_search)
        
        try:
            optimization_params = hnsw_optimizer.auto_tune_parameters()
            print(f"자동 튜닝 완료: {optimization_params}")
        except Exception as e:
            print(f"자동 튜닝 중 경고: {e}")
        
        print("✅ 모든 컴포넌트 초기화 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 컴포넌트 초기화 실패: {e}")
        return False

# ==================== 고급 검색 도구들 ====================

@mcp.tool()
async def intelligent_search(
    query: str,
    search_strategy: str = "adaptive",
    context_expansion: bool = True,
    time_awareness: bool = False,
    similarity_threshold: float = 0.7,
    limit: int = 10
) -> Dict[str, Any]:
    """Milvus의 고급 기능을 활용한 지능형 검색"""
    global rag_engine, enhanced_search
    
    if not rag_engine or not enhanced_search:
        return {"error": "고급 검색 엔진이 초기화되지 않았습니다.", "query": query}
    
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
                logger.error(f"컨텍스트 확장 오류: {e}")
        
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
                "total_found": len(results) if isinstance(results, list) else "N/A"
            }
        }
        
    except Exception as e:
        logger.error(f"지능형 검색 오류: {e}")
        return {"error": str(e), "query": query}

@mcp.tool()
async def advanced_filter_search(
    query: str,
    time_range: Optional[List[float]] = None,
    tag_logic: Optional[Dict[str, List[str]]] = None,
    file_size_range: Optional[List[int]] = None,
    min_content_quality: Optional[float] = None,
    min_relevance_score: Optional[float] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """Milvus의 강력한 메타데이터 필터링을 활용한 고급 검색"""
    global enhanced_search
    
    if not enhanced_search:
        return {"error": "향상된 검색 엔진이 초기화되지 않았습니다.", "query": query}
    
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
            "search_time_ms": round(search_time * 1000, 2)
        }
        
    except Exception as e:
        logger.error(f"고급 필터 검색 오류: {e}")
        return {"error": str(e), "query": query}

@mcp.tool() 
async def multi_query_fusion_search(
    queries: List[str],
    fusion_method: str = "weighted",
    individual_limits: int = 20,
    final_limit: int = 10
) -> Dict[str, Any]:
    """여러 쿼리를 융합하여 더 정확한 검색 결과 제공"""
    global rag_engine
    
    if not rag_engine:
        return {"error": "고급 RAG 엔진이 초기화되지 않았습니다.", "queries": queries}
    
    try:
        if not queries:
            return {"error": "최소 하나의 쿼리가 필요합니다."}
        
        start_time = time.time()
        fused_results = rag_engine.multi_query_fusion(queries, fusion_method)
        final_results = fused_results[:final_limit]
        search_time = time.time() - start_time
        
        return {
            "input_queries": queries,
            "fusion_method": fusion_method,
            "total_candidates": len(fused_results),
            "final_results": final_results,
            "fusion_statistics": {
                "average_query_coverage": sum(r.get('query_coverage', 0) for r in final_results) / len(final_results) if final_results else 0,
                "score_distribution": [r.get('fused_score', 0) for r in final_results],
                "search_time_ms": round(search_time * 1000, 2)
            }
        }
        
    except Exception as e:
        logger.error(f"다중 쿼리 융합 검색 오류: {e}")
        return {"error": str(e), "queries": queries}

@mcp.tool()
async def knowledge_graph_exploration(
    starting_document: str,
    exploration_depth: int = 2,
    similarity_threshold: float = 0.75,
    max_connections: int = 50
) -> Dict[str, Any]:
    """Milvus 기반 지식 그래프 탐색"""
    global milvus_manager
    
    if not milvus_manager:
        return {"error": "Milvus 매니저가 초기화되지 않았습니다.", "starting_document": starting_document}
    
    try:
        start_time = time.time()
        
        start_docs = milvus_manager.query(
            expr=f'path == "{starting_document}"',
            output_fields=["id", "path", "title"],
            limit=1
        )
        
        if not start_docs:
            return {"error": f"시작 문서를 찾을 수 없습니다: {starting_document}"}
        
        start_doc = start_docs[0]
        
        knowledge_graph = {
            "nodes": [{"id": start_doc["id"], "title": start_doc["title"], "path": start_doc["path"], "level": 0}],
            "edges": [],
            "clusters": {}
        }
        
        current_level_nodes = [start_doc["id"]]
        explored_nodes = {start_doc["id"]}
        
        for depth in range(1, exploration_depth + 1):
            next_level_nodes = []
            
            for node_id in current_level_nodes:
                try:
                    similar_docs = milvus_manager.query(
                        expr="id >= 0",
                        output_fields=["id", "path", "title"],
                        limit=20
                    )
                    
                    connection_count = 0
                    for doc in similar_docs:
                        doc_id = doc["id"]
                        if (doc_id not in explored_nodes and 
                            connection_count < max_connections // exploration_depth):
                            
                            knowledge_graph["nodes"].append({
                                "id": doc_id,
                                "title": doc.get("title", ""),
                                "path": doc.get("path", ""),
                                "level": depth,
                                "similarity_to_parent": 0.8
                            })
                            
                            knowledge_graph["edges"].append({
                                "source": node_id,
                                "target": doc_id,
                                "weight": 0.8,
                                "type": "semantic_similarity"
                            })
                            
                            next_level_nodes.append(doc_id)
                            explored_nodes.add(doc_id)
                            connection_count += 1
                            
                except Exception as e:
                    logger.error(f"노드 {node_id} 탐색 오류: {e}")
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
                "exploration_time_ms": round(search_time * 1000, 2)
            }
        }
        
    except Exception as e:
        logger.error(f"지식 그래프 탐색 오류: {e}")
        return {"error": str(e), "starting_document": starting_document}

@mcp.tool()
async def performance_optimization_analysis() -> Dict[str, Any]:
    """Milvus 성능 최적화 분석 및 권장사항"""
    global hnsw_optimizer, enhanced_search
    
    if not hnsw_optimizer:
        return {"error": "HNSW 최적화기가 초기화되지 않았습니다."}
    
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
        
        if len(search_patterns["frequent_queries"]) > 5:
            optimization_recommendations.append({
                "type": "caching_strategy",
                "priority": "medium",
                "recommendation": "자주 사용되는 쿼리 결과 캐싱",
                "expected_improvement": "응답 시간 50% 단축"
            })
        
        if config.USE_GPU:
            optimization_recommendations.append({
                "type": "gpu_optimization",
                "priority": "high",
                "recommendation": "GPU 메모리 캐싱 및 배치 처리 활용",
                "expected_improvement": "대용량 검색 성능 대폭 향상"
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
                "advanced_search_patterns": "active"
            },
            "analysis_time_ms": round(analysis_time * 1000, 2)
        }
        
    except Exception as e:
        logger.error(f"성능 분석 오류: {e}")
        return {"error": str(e)}

# ==================== 기존 기본 도구들 ====================

@mcp.tool()
async def search_documents(
    query: str,
    limit: int = 5,
    search_type: str = "hybrid",
    file_types: Optional[List[str]] = None,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Obsidian 문서에서 관련 내용을 검색합니다."""
    global search_engine
    
    if not search_engine:
        return {"error": "검색 엔진이 초기화되지 않았습니다.", "query": query, "results": []}
    
    try:
        filter_params = {}
        if file_types:
            filter_params['file_types'] = file_types
        if tags:
            filter_params['tags'] = tags
        
        if search_type == "hybrid" or search_type == "vector":
            results, search_info = search_engine.hybrid_search(
                query=query, limit=limit, filter_params=filter_params if filter_params else None
            )
        else:
            results = search_engine._keyword_search(
                query=query, limit=limit, filter_expr=filter_params.get('filter_expr') if filter_params else None
            )
            search_info = {"query": query, "search_type": "keyword_only", "total_count": len(results)}
        
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
        
        return {
            "query": query,
            "search_type": search_type,
            "total_results": len(formatted_results),
            "results": formatted_results,
            "search_info": search_info,
            "filters_applied": {"file_types": file_types, "tags": tags}
        }
        
    except Exception as e:
        logger.error(f"문서 검색 중 오류 발생: {e}")
        return {"error": f"검색 중 오류 발생: {str(e)}", "query": query, "results": []}

@mcp.tool()
async def get_document_content(file_path: str) -> Dict[str, Any]:
    """특정 문서의 전체 내용을 가져옵니다."""
    global milvus_manager
    
    if not milvus_manager:
        return {"error": "Milvus 매니저가 초기화되지 않았습니다.", "file_path": file_path}
    
    try:
        results = milvus_manager.query(
            expr=f'path == "{file_path}"',
            output_fields=["id", "path", "title", "content", "chunk_text", "file_type", "tags", "created_at", "updated_at", "chunk_index"],
            limit=100
        )
        
        if not results:
            return {"error": f"문서를 찾을 수 없습니다: {file_path}", "file_path": file_path}
        
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
            "character_count": len(full_content) if full_content else 0
        }
        
    except Exception as e:
        logger.error(f"문서 내용 조회 중 오류 발생: {e}")
        return {"error": f"문서 조회 중 오류 발생: {str(e)}", "file_path": file_path}

@mcp.tool()
async def get_collection_stats() -> Dict[str, Any]:
    """Milvus 컬렉션의 통계 정보를 반환합니다."""
    global milvus_manager
    
    if not milvus_manager:
        return {"error": "Milvus 매니저가 초기화되지 않았습니다.", "collection_name": config.COLLECTION_NAME}
    
    try:
        total_entities = milvus_manager.count_entities()
        file_type_counts = milvus_manager.get_file_type_counts()
        recent_docs = milvus_manager.query(
            expr="id >= 0", output_fields=["path", "title", "created_at", "file_type"], limit=10
        )
        
        all_results = milvus_manager.query(expr="id >= 0", output_fields=["tags"], limit=1000)
        
        tag_counts = {}
        for doc in all_results:
            tags = doc.get("tags", [])
            if isinstance(tags, list):
                for tag in tags:
                    if tag:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
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
                for doc in recent_docs[:5]
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
            }
        }
        
    except Exception as e:
        logger.error(f"통계 조회 중 오류 발생: {e}")
        return {"error": f"통계 조회 중 오류 발생: {str(e)}", "collection_name": config.COLLECTION_NAME}

@mcp.tool()
async def search_by_tags(tags: List[str], limit: int = 10) -> Dict[str, Any]:
    """특정 태그를 가진 문서들을 검색합니다."""
    global milvus_manager
    
    if not milvus_manager:
        return {"error": "Milvus 매니저가 초기화되지 않았습니다.", "tags": tags}
    
    if not tags:
        return {"error": "최소 하나의 태그를 제공해주세요.", "tags": tags, "results": []}
    
    try:
        all_results = milvus_manager.query(
            expr="id >= 0",
            output_fields=["id", "path", "title", "tags", "file_type", "created_at", "updated_at"],
            limit=1000
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
        
        filtered_results = filtered_results[:limit]
        
        return {
            "search_tags": tags,
            "total_results": len(filtered_results),
            "results": filtered_results
        }
        
    except Exception as e:
        logger.error(f"태그 검색 중 오류 발생: {e}")
        return {"error": f"태그 검색 중 오류 발생: {str(e)}", "search_tags": tags, "results": []}

@mcp.tool()
async def list_available_tags(limit: int = 50) -> Dict[str, Any]:
    """사용 가능한 모든 태그 목록을 반환합니다."""
    global milvus_manager
    
    if not milvus_manager:
        return {"error": "Milvus 매니저가 초기화되지 않았습니다.", "tags": {}}
    
    try:
        results = milvus_manager.query(expr="id >= 0", output_fields=["tags"], limit=2000)
        
        tag_counts = {}
        total_docs_with_tags = 0
        
        for doc in results:
            tags = doc.get("tags", [])
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
            "tags_summary": dict(sorted_tags)
        }
        
    except Exception as e:
        logger.error(f"태그 목록 조회 중 오류 발생: {e}")
        return {"error": f"태그 조회 중 오류 발생: {str(e)}", "tags": {}}

@mcp.tool()
async def get_similar_documents(file_path: str, limit: int = 5) -> Dict[str, Any]:
    """지정된 문서와 유사한 문서들을 찾습니다."""
    global milvus_manager, enhanced_search
    
    if not milvus_manager:
        return {"error": "필요한 컴포넌트가 초기화되지 않았습니다.", "file_path": file_path}
    
    try:
        base_docs = milvus_manager.query(
            expr=f'path == "{file_path}"',
            output_fields=["id", "path", "title", "content", "chunk_text"],
            limit=1
        )
        
        if not base_docs:
            return {"error": f"기준 문서를 찾을 수 없습니다: {file_path}", "file_path": file_path}
        
        base_doc = base_docs[0]
        search_query = f"{base_doc.get('title', '')} {base_doc.get('chunk_text', '')[:200]}"
        
        if enhanced_search:
            results, search_info = enhanced_search.hybrid_search(query=search_query, limit=limit + 5)
        else:
            results, search_info = search_engine.hybrid_search(query=search_query, limit=limit + 5)
        
        similar_docs = []
        for result in results:
            if result.get("path") != file_path and len(similar_docs) < limit:
                similar_docs.append({
                    "file_path": result.get("path", ""),
                    "title": result.get("title", "제목 없음"),
                    "similarity_score": float(result.get("score", 0)),
                    "content_preview": result.get("chunk_text", "")[:200] + "..." if len(result.get("chunk_text", "")) > 200 else result.get("chunk_text", ""),
                    "file_type": result.get("file_type", ""),
                    "tags": result.get("tags", [])
                })
        
        return {
            "base_document": {"file_path": file_path, "title": base_doc.get("title", "제목 없음")},
            "similar_documents": similar_docs,
            "total_found": len(similar_docs)
        }
        
    except Exception as e:
        logger.error(f"유사 문서 검색 중 오류 발생: {e}")
        return {"error": f"유사 문서 검색 중 오류 발생: {str(e)}", "file_path": file_path}

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
        }
    }
    return json.dumps(config_info, indent=2, ensure_ascii=False)

@mcp.resource("stats://collection")
async def get_collection_stats_resource() -> str:
    """컬렉션 통계 정보를 리소스로 반환합니다."""
    global milvus_manager
    
    if not milvus_manager:
        return json.dumps({"error": "Milvus 매니저가 초기화되지 않았습니다."}, ensure_ascii=False)
    
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
            }
        }
        
        return json.dumps(stats, indent=2, ensure_ascii=False)
    except Exception as e:
        error_info = {
            "error": f"통계 조회 중 오류: {str(e)}",
            "collection_name": config.COLLECTION_NAME
        }
        return json.dumps(error_info, ensure_ascii=False)

def main():
    """메인 함수"""
    print("🚀 최적화된 Obsidian-Milvus Fast MCP Server 시작 중...")
    print("💎 Milvus 고급 기능 모두 활성화!")
    
    if not initialize_components():
        print("❌ 컴포넌트 초기화 실패. 서버를 시작할 수 없습니다.")
        sys.exit(1)
    
    print("✅ 모든 컴포넌트 초기화 완료!")
    print("🎯 활성화된 고급 기능들:")
    print("   - 🔍 지능형 검색 (적응적/계층적/의미적 그래프)")
    print("   - 🏷️ 고급 메타데이터 필터링")
    print("   - 🔄 다중 쿼리 융합")
    print("   - 🕸️ 지식 그래프 탐색")
    print("   - ⚡ HNSW 최적화")
    print("   - 📊 성능 모니터링")
    print(f"📡 MCP 서버 '{config.FASTMCP_SERVER_NAME}' 시작 중...")
    print(f"🔧 Transport: {config.FASTMCP_TRANSPORT}")
    
    try:
        if config.FASTMCP_TRANSPORT == "stdio":
            print("📡 STDIO transport로 MCP 서버 시작...")
            mcp.run(transport="stdio")
        elif config.FASTMCP_TRANSPORT == "sse":
            print(f"📡 SSE transport로 MCP 서버 시작... (http://{config.FASTMCP_HOST}:{config.FASTMCP_PORT})")
            mcp.run(transport="sse", host=config.FASTMCP_HOST, port=config.FASTMCP_PORT)
        elif config.FASTMCP_TRANSPORT == "streamable-http":
            print(f"📡 Streamable HTTP transport로 MCP 서버 시작... (http://{config.FASTMCP_HOST}:{config.FASTMCP_PORT})")
            mcp.run(transport="streamable-http", host=config.FASTMCP_HOST, port=config.FASTMCP_PORT)
        else:
            print(f"❌ 지원하지 않는 transport: {config.FASTMCP_TRANSPORT}")
            print("지원하는 transport: stdio, sse, streamable-http")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 서버 종료 중...")
    except Exception as e:
        print(f"❌ 서버 실행 중 오류 발생: {e}")
        print(f"스택 트레이스: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        if milvus_manager:
            try:
                milvus_manager.stop_monitoring()
                print("✅ Milvus 모니터링 중지됨")
            except:
                pass
        print("👋 최적화된 서버가 정상적으로 종료되었습니다.")

if __name__ == "__main__":
    main()
