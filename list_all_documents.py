import sys
import logging
from milvus_manager import MilvusManager

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ListAllDocuments")

def main():
    """Milvus 데이터베이스의 모든 문서 목록을 출력"""
    logger.info("MilvusManager 인스턴스 생성 중...")
    mm = MilvusManager()
    
    # 모든 문서 가져오기 (id >= 0 쿼리 사용)
    logger.info("모든 문서 조회 중...")
    results = mm.query("id >= 0", limit=1000)
    
    # 파일 경로 기준으로 중복 제거
    unique_paths = set()
    unique_documents = []
    
    for result in results:
        path = result.get('path', '')
        if path and path not in unique_paths:
            unique_paths.add(path)
            unique_documents.append(result)
    
    # 결과 출력
    logger.info(f"컬렉션 통계:")
    logger.info(f"총 문서 수: {len(results)}")
    logger.info(f"고유 파일 수: {len(unique_documents)}")
    
    print("\n===== 인덱싱된 모든 파일 목록 =====")
    for i, doc in enumerate(unique_documents, 1):
        print(f"\n{i}. 파일: {doc.get('path', '경로 없음')}")
        print(f"   제목: {doc.get('title', '제목 없음')}")
        print(f"   ID: {doc.get('id')}")
    
    print(f"\n총 {len(unique_documents)}개의 고유 파일이 인덱싱되어 있습니다.")

if __name__ == "__main__":
    main()
