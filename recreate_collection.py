import sys
import os
import logging
from pymilvus import connections, utility
from milvus_manager import MilvusManager

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RecreateCollection')

def main():
    """Milvus 컬렉션을 재생성합니다."""
    try:
        logger.info("Milvus 컬렉션 재생성을 시작합니다...")
        
        # MilvusManager 인스턴스 생성
        manager = MilvusManager()
        
        # 컬렉션 이름 확인
        collection_name = manager.collection_name
        logger.info(f"컬렉션 이름: {collection_name}")
        
        # 컬렉션 존재 여부 확인
        if utility.has_collection(collection_name):
            logger.info(f"기존 컬렉션 '{collection_name}'을 삭제합니다...")
            utility.drop_collection(collection_name)
            logger.info(f"컬렉션 '{collection_name}'이 삭제되었습니다.")
        
        # 컬렉션 재생성
        logger.info(f"컬렉션 '{collection_name}'을 재생성합니다...")
        manager.create_collection()
        logger.info(f"컬렉션 '{collection_name}'이 성공적으로 재생성되었습니다.")
        
        # 컬렉션 로드
        logger.info(f"컬렉션 '{collection_name}'을 로드합니다...")
        manager.collection.load()
        logger.info(f"컬렉션 '{collection_name}'이 로드되었습니다.")
        
        # 컬렉션 정보 출력
        logger.info("컬렉션 정보:")
        schema = manager.collection.schema
        for field in schema.fields:
            logger.info(f"  - 필드: {field.name}, 타입: {field.dtype}, 기타 속성: {field.__dict__}")
        
        logger.info("Milvus 컬렉션 재생성이 완료되었습니다.")
        return True
        
    except Exception as e:
        logger.error(f"Milvus 컬렉션 재생성 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
