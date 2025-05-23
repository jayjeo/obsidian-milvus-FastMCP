import logging
from milvus_manager import MilvusManager

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SimpleSearch")

def main():
    """간단한 검색 테스트"""
    logger.info("MilvusManager 인스턴스 생성 중...")
    mm = MilvusManager()
    
    # 모든 문서 가져오기
    logger.info("모든 문서 조회 중...")
    all_docs = mm.query("id >= 0", limit=1000)
    
    # 파일 경로 기준으로 중복 제거
    unique_paths = {}
    for doc in all_docs:
        path = doc.get('path', '')
        if path:
            unique_paths[path] = doc
    
    # 결과 출력
    logger.info(f"컬렉션 통계:")
    logger.info(f"총 문서 수: {len(all_docs)}")
    logger.info(f"고유 파일 수: {len(unique_paths)}")
    
    print("\n===== 인덱싱된 모든 파일 목록 =====")
    for i, (path, doc) in enumerate(sorted(unique_paths.items()), 1):
        print(f"\n{i}. 파일: {path}")
        print(f"   제목: {doc.get('title', '제목 없음')}")
        print(f"   ID: {doc.get('id')}")
    
    print(f"\n총 {len(unique_paths)}개의 고유 파일이 인덱싱되어 있습니다.")
    
    # 특정 파일명 검색
    search_term = "2023년"
    print(f"\n\n===== '{search_term}'가 포함된 파일 검색 =====")
    found = False
    
    for path, doc in unique_paths.items():
        if search_term in path:
            found = True
            print(f"\n파일: {path}")
            print(f"제목: {doc.get('title', '제목 없음')}")
            print(f"ID: {doc.get('id')}")
    
    if not found:
        print(f"'{search_term}'가 포함된 파일을 찾을 수 없습니다.")

if __name__ == "__main__":
    main()
