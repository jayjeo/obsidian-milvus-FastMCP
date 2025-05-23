import sys
import logging
import os
from milvus_manager import MilvusManager
from collections import defaultdict

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ShowEmbeddedFiles")

def main():
    """Milvus 데이터베이스에 임베딩된 모든 Obsidian 파일 목록을 출력"""
    logger.info("MilvusManager 인스턴스 생성 중...")
    mm = MilvusManager()
    
    # 모든 문서 가져오기 (id >= 0 쿼리 사용)
    logger.info("모든 문서 조회 중...")
    results = mm.query("id >= 0", limit=10000)
    
    # 파일 경로 기준으로 중복 제거 및 그룹화
    file_groups = defaultdict(list)
    unique_paths = set()
    
    for result in results:
        path = result.get('path', '')
        if path:
            file_groups[path].append(result)
            unique_paths.add(path)
    
    # 결과 출력
    logger.info(f"컬렉션 통계:")
    logger.info(f"총 청크 수: {len(results)}")
    logger.info(f"고유 파일 수: {len(unique_paths)}")
    
    # 파일 확장자별 통계
    extensions = defaultdict(int)
    for path in unique_paths:
        ext = os.path.splitext(path)[1].lower()
        extensions[ext] += 1
    
    print("\n===== 파일 확장자별 통계 =====")
    for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True):
        print(f"{ext or '(확장자 없음)'}: {count}개")
    
    print("\n===== 임베딩된 모든 파일 목록 =====")
    for i, path in enumerate(sorted(unique_paths), 1):
        chunks = file_groups[path]
        first_chunk = chunks[0]
        title = first_chunk.get('title', '제목 없음')
        
        print(f"\n{i}. 파일: {path}")
        print(f"   제목: {title}")
        print(f"   청크 수: {len(chunks)}")
        print(f"   ID: {first_chunk.get('id')}")
        
        # 파일 생성/수정 시간 정보 (있는 경우)
        if 'created_at' in first_chunk:
            print(f"   생성 시간: {first_chunk.get('created_at')}")
        if 'updated_at' in first_chunk:
            print(f"   수정 시간: {first_chunk.get('updated_at')}")
    
    print(f"\n총 {len(unique_paths)}개의 고유 파일이 임베딩되어 있습니다.")
    print(f"총 {len(results)}개의 청크로 분할되어 저장되어 있습니다.")

if __name__ == "__main__":
    main()
