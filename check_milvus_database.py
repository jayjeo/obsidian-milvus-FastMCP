"""
Milvus 데이터베이스 직접 확인 도구
특정 파일명이나 키워드로 Milvus 데이터베이스를 검색하고 결과를 보여줍니다.
"""
import os
import sys
import argparse
from pymilvus import Collection
import json
from colorama import Fore, Style, init

# 현재 디렉토리를 경로에 추가하여 모듈 import 가능하게 함
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 프로젝트 모듈 import
from milvus_manager import MilvusManager
import config

# colorama 초기화
init(autoreset=True)

def search_by_filename(query, limit=10, show_details=False):
    """
    파일명으로 Milvus 데이터베이스 검색
    
    Args:
        query (str): 검색할 파일명 또는 키워드
        limit (int): 반환할 최대 결과 수
        show_details (bool): 상세 정보 표시 여부
    """
    print(f"{Style.BRIGHT}파일명 '{query}'로 Milvus 데이터베이스 검색 중...{Style.RESET_ALL}")
    
    try:
        # MilvusManager 인스턴스 생성
        manager = MilvusManager()
        
        # 컬렉션 로드 확인
        if not manager.collection.is_loaded:
            print(f"{Fore.YELLOW}컬렉션을 로드합니다...{Fore.RESET}")
            manager.collection.load()
        
        # 파일명 필드로 검색 (exact match)
        expr = f"file_path like '%{query}%'"
        results = manager.collection.query(
            expr=expr,
            output_fields=["id", "file_path", "chunk_index", "title", "content"],
            limit=limit
        )
        
        if not results:
            print(f"{Fore.RED}검색 결과가 없습니다. 파일이 임베딩되지 않았거나 데이터베이스에 존재하지 않습니다.{Fore.RESET}")
            return
        
        print(f"{Fore.GREEN}검색 결과: {len(results)}개{Fore.RESET}")
        
        # 결과 표시
        for i, result in enumerate(results):
            print(f"\n{Style.BRIGHT}{Fore.CYAN}결과 #{i+1}{Style.RESET_ALL}")
            print(f"{Style.BRIGHT}파일 경로:{Style.RESET_ALL} {result['file_path']}")
            print(f"{Style.BRIGHT}청크 인덱스:{Style.RESET_ALL} {result['chunk_index']}")
            print(f"{Style.BRIGHT}제목:{Style.RESET_ALL} {result['title']}")
            
            if show_details:
                print(f"{Style.BRIGHT}내용:{Style.RESET_ALL}")
                print(f"{result['content'][:300]}..." if len(result['content']) > 300 else result['content'])
            
            print("-" * 80)
    
    except Exception as e:
        print(f"{Fore.RED}오류 발생: {str(e)}{Fore.RESET}")

def search_by_content(query, limit=10, show_details=False):
    """
    내용으로 Milvus 데이터베이스 검색 (벡터 검색)
    
    Args:
        query (str): 검색할 내용 키워드
        limit (int): 반환할 최대 결과 수
        show_details (bool): 상세 정보 표시 여부
    """
    print(f"{Style.BRIGHT}내용 '{query}'로 Milvus 데이터베이스 검색 중...{Style.RESET_ALL}")
    
    try:
        # MilvusManager 인스턴스 생성
        manager = MilvusManager()
        
        # 임베딩 모델 가져오기
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # 쿼리 텍스트를 벡터로 변환
        query_vector = model.encode(query).tolist()
        
        # 벡터 검색 수행
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        results = manager.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=limit,
            output_fields=["path", "chunk_index", "title", "content"]
        )
        
        # 결과 형식 변환
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "score": hit.score,
                    "file_path": hit.entity.get("path"),
                    "chunk_index": hit.entity.get("chunk_index"),
                    "title": hit.entity.get("title"),
                    "content": hit.entity.get("content")
                })
        
        if not formatted_results:
            print(f"{Fore.RED}검색 결과가 없습니다.{Fore.RESET}")
            return
        
        print(f"{Fore.GREEN}검색 결과: {len(formatted_results)}개{Fore.RESET}")
        
        # 결과 표시
        for i, result in enumerate(formatted_results):
            print(f"\n{Style.BRIGHT}{Fore.CYAN}결과 #{i+1} (유사도: {result['score']:.4f}){Style.RESET_ALL}")
            print(f"{Style.BRIGHT}파일 경로:{Style.RESET_ALL} {result['file_path']}")
            print(f"{Style.BRIGHT}청크 인덱스:{Style.RESET_ALL} {result['chunk_index']}")
            print(f"{Style.BRIGHT}제목:{Style.RESET_ALL} {result['title']}")
            
            if show_details:
                print(f"{Style.BRIGHT}내용:{Style.RESET_ALL}")
                print(f"{result['content'][:300]}..." if len(result['content']) > 300 else result['content'])
            
            print("-" * 80)
    
    except Exception as e:
        print(f"{Fore.RED}오류 발생: {str(e)}{Fore.RESET}")

def list_all_files(limit=100):
    """
    Milvus 데이터베이스에 저장된 모든 파일 목록 표시
    
    Args:
        limit (int): 반환할 최대 결과 수
    """
    print(f"{Style.BRIGHT}Milvus 데이터베이스에 저장된 모든 파일 목록 조회 중...{Style.RESET_ALL}")
    
    try:
        # MilvusManager 인스턴스 생성
        manager = MilvusManager()
        
        # 컬렉션 로드 확인
        if not manager.collection.is_loaded:
            print(f"{Fore.YELLOW}컬렉션을 로드합니다...{Fore.RESET}")
            manager.collection.load()
        
        # 모든 파일 경로 조회 (중복 제거)
        results = manager.collection.query(
            expr="chunk_index >= 0",  # 모든 문서 조회
            output_fields=["file_path"],
            limit=10000  # 충분히 큰 값
        )
        
        if not results:
            print(f"{Fore.RED}데이터베이스에 저장된 파일이 없습니다.{Fore.RESET}")
            return
        
        # 파일 경로 중복 제거
        unique_files = set()
        for result in results:
            unique_files.add(result['file_path'])
        
        # 정렬 및 제한
        file_list = sorted(list(unique_files))[:limit]
        
        print(f"{Fore.GREEN}총 파일 수: {len(file_list)}개{Fore.RESET}")
        
        # 결과 표시
        for i, file_path in enumerate(file_list):
            print(f"{i+1}. {file_path}")
    
    except Exception as e:
        print(f"{Fore.RED}오류 발생: {str(e)}{Fore.RESET}")

def show_collection_stats():
    """
    Milvus 컬렉션 통계 정보 표시
    """
    print(f"{Style.BRIGHT}Milvus 컬렉션 통계 정보 조회 중...{Style.RESET_ALL}")
    
    try:
        # MilvusManager 인스턴스 생성
        manager = MilvusManager()
        
        # 컬렉션 로드 확인
        if not manager.collection.is_loaded:
            print(f"{Fore.YELLOW}컬렉션을 로드합니다...{Fore.RESET}")
            manager.collection.load()
        
        # 컬렉션 통계 정보
        stats = manager.collection.get_stats()
        row_count = stats["row_count"]
        
        print(f"{Style.BRIGHT}컬렉션 이름:{Style.RESET_ALL} {manager.collection.name}")
        print(f"{Style.BRIGHT}총 문서 수:{Style.RESET_ALL} {row_count}")
        print(f"{Style.BRIGHT}벡터 차원:{Style.RESET_ALL} {config.VECTOR_DIM}")
        print(f"{Style.BRIGHT}임베딩 모델:{Style.RESET_ALL} {config.EMBEDDING_MODEL}")
        
        # 청크 인덱스 분포 (선택적)
        try:
            results = manager.collection.query(
                expr="chunk_index >= 0",
                output_fields=["chunk_index"],
                limit=10000
            )
            
            if results:
                max_chunk = max(result["chunk_index"] for result in results)
                print(f"{Style.BRIGHT}최대 청크 인덱스:{Style.RESET_ALL} {max_chunk}")
        except:
            pass
        
    except Exception as e:
        print(f"{Fore.RED}오류 발생: {str(e)}{Fore.RESET}")

def main():
    parser = argparse.ArgumentParser(description="Milvus 데이터베이스 검색 도구")
    
    # 검색 모드 그룹
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("-f", "--filename", help="파일명으로 검색")
    mode_group.add_argument("-c", "--content", help="내용으로 검색 (벡터 검색)")
    mode_group.add_argument("-l", "--list", action="store_true", help="모든 파일 목록 표시")
    mode_group.add_argument("-s", "--stats", action="store_true", help="컬렉션 통계 정보 표시")
    
    # 추가 옵션
    parser.add_argument("--limit", type=int, default=10, help="반환할 최대 결과 수 (기본값: 10)")
    parser.add_argument("--details", action="store_true", help="상세 내용 표시")
    
    args = parser.parse_args()
    
    try:
        if args.filename:
            search_by_filename(args.filename, args.limit, args.details)
        elif args.content:
            search_by_content(args.content, args.limit, args.details)
        elif args.list:
            list_all_files(args.limit)
        elif args.stats:
            show_collection_stats()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}사용자에 의해 중단되었습니다.{Fore.RESET}")
    except Exception as e:
        print(f"\n{Fore.RED}오류 발생: {str(e)}{Fore.RESET}")

if __name__ == "__main__":
    main()
