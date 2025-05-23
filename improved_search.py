import sys
import logging
import argparse
from milvus_manager import MilvusManager
from embeddings import EmbeddingModel

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ImprovedSearch")

def main():
    # 명령줄 인수 파싱
    parser = argparse.ArgumentParser(description='Milvus 개선된 검색 테스트')
    parser.add_argument('query', type=str, help='검색할 쿼리')
    parser.add_argument('--limit', type=int, default=10, help='결과 제한 수 (기본값: 10)')
    args = parser.parse_args()
    
    query = args.query
    limit = args.limit
    
    logger.info(f"검색 쿼리: '{query}', 결과 제한: {limit}")
    logger.info("MilvusManager 인스턴스 생성 중...")
    mm = MilvusManager()
    
    # 1. 벡터 검색
    logger.info("벡터 검색 수행 중...")
    embedding_model = EmbeddingModel()
    query_vector = embedding_model.get_embedding(query)
    vector_results = mm.search(query_vector, limit=limit*2)
    
    # 2. 키워드 검색 (모든 문서 가져와서 메모리에서 필터링)
    logger.info("키워드 검색 수행 중...")
    all_docs = mm.query("id >= 0", limit=1000)
    
    # 검색어로 필터링
    import re
    keyword_results = []
    query_terms = re.findall(r'[\w가-힣]+', query.lower())
    
    for doc in all_docs:
        path = doc.get('path', '').lower()
        title = doc.get('title', '').lower()
        content = doc.get('chunk_text', '').lower()
        
        # 점수 계산
        score = 0
        for term in query_terms:
            # 경로에 검색어가 있으면 높은 점수
            if term in path:
                score += 5
            # 제목에 검색어가 있으면 중간 점수
            if term in title:
                score += 3
            # 내용에 검색어가 있으면 낮은 점수
            if term in content:
                score += 1
        
        if score > 0:
            doc['score'] = score
            keyword_results.append(doc)
    
    # 점수 기준으로 정렬
    keyword_results.sort(key=lambda x: x.get('score', 0), reverse=True)
    keyword_results = keyword_results[:limit]
    
    # 3. 결과 병합 및 중복 제거
    logger.info("결과 병합 및 중복 제거 중...")
    all_results = []
    seen_paths = set()
    
    # 벡터 검색 결과 추가
    for hit in vector_results:
        path = hit.entity.get('path')
        if path and path not in seen_paths:
            seen_paths.add(path)
            all_results.append({
                "id": hit.id,
                "path": path,
                "title": hit.entity.get('title', '제목 없음'),
                "content": hit.entity.get('chunk_text', ''),
                "score": hit.score,
                "source": "vector"
            })
    
    # 키워드 검색 결과 추가
    for doc in keyword_results:
        path = doc.get('path')
        if path and path not in seen_paths:
            seen_paths.add(path)
            all_results.append({
                "id": doc.get('id'),
                "path": path,
                "title": doc.get('title', '제목 없음'),
                "content": doc.get('chunk_text', ''),
                "score": doc.get('score', 0),
                "source": "keyword"
            })
    
    # 결과 출력
    print(f"\n===== '{query}'에 대한 검색 결과 =====")
    print(f"총 {len(all_results)}개의 결과를 찾았습니다.\n")
    
    for i, result in enumerate(all_results[:limit], 1):
        print(f"{i}. 파일: {result['path']}")
        print(f"   제목: {result['title']}")
        print(f"   점수: {result['score']:.4f} (출처: {result['source']})")
        print(f"   내용 일부: {result['content'][:150]}...\n")

if __name__ == "__main__":
    main()
