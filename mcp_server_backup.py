#!/usr/bin/env python3
"""
Obsidian-Milvus Fast MCP Server
임베딩된 Obsidian 문서들을 Claude Desktop에서 검색할 수 있게 해주는 MCP 서버

기존 OpenWebUI 통합을 Fast MCP로 변경한 버전
"""

import os
import sys
import asyncio
import json
import traceback
from typing import List, Dict, Any, Optional
from datetime import datetime

# Fast MCP 임포트
from mcp.server.fastmcp import FastMCP, Context

# 기존 모듈들 임포트
import config
from milvus_manager import MilvusManager
from search_engine import SearchEngine

# 로깅 설정
import logging
log_level_str = getattr(config, 'LOG_LEVEL', 'INFO')
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ObsidianMilvusMCP')

# MCP 서버 인스턴스 생성
mcp = FastMCP(config.FASTMCP_SERVER_NAME)

# 전역 변수들
milvus_manager = None
search_engine = None

def initialize_components():
    """핵심 컴포넌트들 초기화"""
    global milvus_manager, search_engine
    
    try:
        logger.info("🚀 Obsidian-Milvus MCP Server 초기화 중...")
        
        # Milvus 매니저 초기화
        logger.info("📊 Milvus 매니저 초기화 중...")
        milvus_manager = MilvusManager()
        
        # 검색 엔진 초기화
        logger.info("🔍 검색 엔진 초기화 중...")
        search_engine = SearchEngine(milvus_manager)
        
        logger.info("✅ 모든 컴포넌트 초기화 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 컴포넌트 초기화 실패: {e}")
        logger.error(f"스택 트레이스: {traceback.format_exc()}")
        return False


@mcp.tool()
async def search_documents(
    query: str,
    limit: int = 5,
    search_type: str = "hybrid",
    file_types: Optional[List[str]] = None,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Obsidian 문서에서 관련 내용을 검색합니다.
    
    Args:
        query: 검색할 질문이나 키워드
        limit: 반환할 최대 결과 수 (기본값: 5)
        search_type: 검색 유형 ("hybrid", "vector", "keyword")
        file_types: 필터링할 파일 타입 목록 (예: ["md", "pdf"])
        tags: 필터링할 태그 목록
    
    Returns:
        검색 결과와 메타데이터
    """
    global search_engine
    
    if not search_engine:
        return {
            "error": "검색 엔진이 초기화되지 않았습니다.",
            "query": query,
            "results": []
        }
    
    try:
        # 필터 파라미터 구성
        filter_params = {}
        if file_types:
            filter_params['file_types'] = file_types
        if tags:
            filter_params['tags'] = tags
        
        # 검색 유형에 따른 처리
        if search_type == "hybrid" or search_type == "vector":
            # 하이브리드 또는 벡터 검색
            results, search_info = search_engine.hybrid_search(
                query=query,
                limit=limit,
                filter_params=filter_params if filter_params else None
            )
        else:
            # 키워드 검색만
            results = search_engine._keyword_search(
                query=query,
                limit=limit,
                filter_expr=filter_params.get('filter_expr') if filter_params else None
            )
            search_info = {
                "query": query,
                "search_type": "keyword_only",
                "total_count": len(results)
            }
        
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
        
        return {
            "query": query,
            "search_type": search_type,
            "total_results": len(formatted_results),
            "results": formatted_results,
            "search_info": search_info,
            "filters_applied": {
                "file_types": file_types,
                "tags": tags
            }
        }
        
    except Exception as e:
        logger.error(f"문서 검색 중 오류 발생: {e}")
        return {
            "error": f"검색 중 오류 발생: {str(e)}",
            "query": query,
            "results": []
        }


@mcp.tool()
async def get_document_content(file_path: str) -> Dict[str, Any]:
    """
    특정 문서의 전체 내용을 가져옵니다.
    
    Args:
        file_path: 문서 파일 경로
    
    Returns:
        문서의 전체 내용과 메타데이터
    """
    global milvus_manager
    
    if not milvus_manager:
        return {
            "error": "Milvus 매니저가 초기화되지 않았습니다.",
            "file_path": file_path
        }
    
    try:
        # 파일 경로로 문서 검색
        results = milvus_manager.query(
            expr=f'path == "{file_path}"',
            output_fields=["id", "path", "title", "content", "chunk_text", "file_type", "tags", "created_at", "updated_at", "chunk_index"],
            limit=100  # 한 파일의 모든 청크 가져오기
        )
        
        if not results:
            return {
                "error": f"문서를 찾을 수 없습니다: {file_path}",
                "file_path": file_path
            }
        
        # 첫 번째 결과에서 메타데이터 추출
        first_result = results[0]
        
        # 모든 청크의 내용을 병합
        all_chunks = []
        for result in results:
            chunk_info = {
                "chunk_index": result.get("chunk_index", 0),
                "chunk_text": result.get("chunk_text", ""),
                "id": result.get("id", "")
            }
            all_chunks.append(chunk_info)
        
        # 청크 인덱스로 정렬
        all_chunks.sort(key=lambda x: x.get("chunk_index", 0))
        
        # 전체 내용 병합
        full_content = first_result.get("content", "")
        if not full_content:
            # content 필드가 없으면 chunk들을 병합
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
        return {
            "error": f"문서 조회 중 오류 발생: {str(e)}",
            "file_path": file_path
        }


@mcp.tool()
async def get_collection_stats() -> Dict[str, Any]:
    """
    Milvus 컬렉션의 통계 정보를 반환합니다.
    
    Returns:
        컬렉션 통계 정보
    """
    global milvus_manager
    
    if not milvus_manager:
        return {
            "error": "Milvus 매니저가 초기화되지 않았습니다.",
            "collection_name": config.COLLECTION_NAME
        }
    
    try:
        # 기본 통계
        total_entities = milvus_manager.count_entities()
        
        # 파일 타입별 통계
        file_type_counts = milvus_manager.get_file_type_counts()
        
        # 최근 추가된 문서들 (상위 10개)
        recent_docs = milvus_manager.query(
            expr="id >= 0",
            output_fields=["path", "title", "created_at", "file_type"],
            limit=10
        )
        
        # 태그 통계
        all_results = milvus_manager.query(
            expr="id >= 0",
            output_fields=["tags"],
            limit=1000  # 샘플링
        )
        
        tag_counts = {}
        for doc in all_results:
            tags = doc.get("tags", [])
            if isinstance(tags, list):
                for tag in tags:
                    if tag:  # 빈 태그 제외
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # 상위 태그들
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
            }
        }
        
    except Exception as e:
        logger.error(f"통계 조회 중 오류 발생: {e}")
        return {
            "error": f"통계 조회 중 오류 발생: {str(e)}",
            "collection_name": config.COLLECTION_NAME
        }


@mcp.tool()
async def search_by_tags(tags: List[str], limit: int = 10) -> Dict[str, Any]:
    """
    특정 태그를 가진 문서들을 검색합니다.
    
    Args:
        tags: 검색할 태그 목록
        limit: 반환할 최대 결과 수
    
    Returns:
        태그로 필터링된 문서 목록
    """
    global milvus_manager
    
    if not milvus_manager:
        return {
            "error": "Milvus 매니저가 초기화되지 않았습니다.",
            "tags": tags
        }
    
    if not tags:
        return {
            "error": "최소 하나의 태그를 제공해주세요.",
            "tags": tags,
            "results": []
        }
    
    try:
        # 모든 문서 가져오기 (태그 필터링을 위해)
        all_results = milvus_manager.query(
            expr="id >= 0",
            output_fields=["id", "path", "title", "tags", "file_type", "created_at", "updated_at"],
            limit=1000  # 충분한 수의 문서 가져오기
        )
        
        # 태그 필터링
        filtered_results = []
        for doc in all_results:
            doc_tags = doc.get("tags", [])
            if isinstance(doc_tags, list):
                # 요청된 태그 중 하나라도 포함되어 있으면 포함
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
        
        # 결과 제한
        filtered_results = filtered_results[:limit]
        
        return {
            "search_tags": tags,
            "total_results": len(filtered_results),
            "results": filtered_results
        }
        
    except Exception as e:
        logger.error(f"태그 검색 중 오류 발생: {e}")
        return {
            "error": f"태그 검색 중 오류 발생: {str(e)}",
            "search_tags": tags,
            "results": []
        }


@mcp.tool()
async def list_available_tags(limit: int = 50) -> Dict[str, Any]:
    """
    사용 가능한 모든 태그 목록을 반환합니다.
    
    Args:
        limit: 반환할 최대 태그 수
    
    Returns:
        태그 목록과 각 태그의 문서 수
    """
    global milvus_manager
    
    if not milvus_manager:
        return {
            "error": "Milvus 매니저가 초기화되지 않았습니다.",
            "tags": {}
        }
    
    try:
        # 모든 문서의 태그 조회
        results = milvus_manager.query(
            expr="id >= 0",
            output_fields=["tags"],
            limit=2000  # 충분한 수의 문서에서 태그 수집
        )
        
        # 태그 카운트
        tag_counts = {}
        total_docs_with_tags = 0
        
        for doc in results:
            tags = doc.get("tags", [])
            if isinstance(tags, list) and tags:
                total_docs_with_tags += 1
                for tag in tags:
                    if tag and tag.strip():  # 빈 태그 제외
                        clean_tag = tag.strip()
                        tag_counts[clean_tag] = tag_counts.get(clean_tag, 0) + 1
        
        # 상위 태그들만 반환
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        return {
            "total_unique_tags": len(tag_counts),
            "total_documents_with_tags": total_docs_with_tags,
            "top_tags": [
                {"tag": tag, "document_count": count}
                for tag, count in sorted_tags
            ],
            "tags_summary": dict(sorted_tags)
        }
        
    except Exception as e:
        logger.error(f"태그 목록 조회 중 오류 발생: {e}")
        return {
            "error": f"태그 조회 중 오류 발생: {str(e)}",
            "tags": {}
        }


@mcp.tool()
async def get_similar_documents(file_path: str, limit: int = 5) -> Dict[str, Any]:
    """
    지정된 문서와 유사한 문서들을 찾습니다.
    
    Args:
        file_path: 기준이 되는 문서의 파일 경로
        limit: 반환할 유사 문서 수
    
    Returns:
        유사한 문서들의 목록
    """
    global milvus_manager, search_engine
    
    if not milvus_manager or not search_engine:
        return {
            "error": "필요한 컴포넌트가 초기화되지 않았습니다.",
            "file_path": file_path
        }
    
    try:
        # 기준 문서 찾기
        base_docs = milvus_manager.query(
            expr=f'path == "{file_path}"',
            output_fields=["id", "path", "title", "content", "chunk_text"],
            limit=1
        )
        
        if not base_docs:
            return {
                "error": f"기준 문서를 찾을 수 없습니다: {file_path}",
                "file_path": file_path
            }
        
        base_doc = base_docs[0]
        
        # 문서의 주요 내용으로 검색 (제목 + 내용 일부)
        search_query = f"{base_doc.get('title', '')} {base_doc.get('chunk_text', '')[:200]}"
        
        # 유사 문서 검색
        results, search_info = search_engine.hybrid_search(
            query=search_query,
            limit=limit + 5  # 기준 문서 제외를 위해 더 많이 가져오기
        )
        
        # 기준 문서 제외
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
            "base_document": {
                "file_path": file_path,
                "title": base_doc.get("title", "제목 없음")
            },
            "similar_documents": similar_docs,
            "total_found": len(similar_docs)
        }
        
    except Exception as e:
        logger.error(f"유사 문서 검색 중 오류 발생: {e}")
        return {
            "error": f"유사 문서 검색 중 오류 발생: {str(e)}",
            "file_path": file_path
        }


# 리소스: 설정 정보 제공
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
            "last_updated": datetime.now().isoformat()
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
    print("🚀 Obsidian-Milvus Fast MCP Server 시작 중...")
    
    # 컴포넌트 초기화
    if not initialize_components():
        print("❌ 컴포넌트 초기화 실패. 서버를 시작할 수 없습니다.")
        sys.exit(1)
    
    print("✅ 컴포넌트 초기화 완료!")
    print(f"📡 MCP 서버 '{config.FASTMCP_SERVER_NAME}' 시작 중...")
    print(f"🔧 Transport: {config.FASTMCP_TRANSPORT}")
    
    # FastMCP 서버 실행
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
        # 정리 작업
        if milvus_manager:
            try:
                milvus_manager.stop_monitoring()
                print("✅ Milvus 모니터링 중지됨")
            except:
                pass
        print("👋 서버가 정상적으로 종료되었습니다.")


if __name__ == "__main__":
    main()
