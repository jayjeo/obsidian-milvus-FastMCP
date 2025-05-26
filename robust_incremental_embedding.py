# 개선된 증분 임베딩 로직
import os
import math
from tqdm import tqdm

# 기존: processor = ObsidianProcessor(milvus_manager)
# 필요한 경우 외부에서 제공됨

def estimate_chunk_count(file_size, chunk_size=1000, chunk_overlap=100, min_chunk_size=100):
    if file_size < min_chunk_size:
        return 0
    stride = chunk_size - chunk_overlap
    return max(1, math.ceil((file_size - chunk_overlap) / stride))

def process_incremental_embedding(processor):
    vault_path = processor.vault_path
    milvus = processor.milvus_manager
    embedding_model = processor.embedding_model

    # 1. Milvus에서 기존 파일 정보 (path → updated_at, chunk_count) 수집
    existing_files_info = {}   # path → updated_at (float)
    chunk_counts = {}          # path → chunk 개수

    try:
        # expr 인자 추가 - 모든 문서 조회를 위한 "id >= 0" 표현식 사용
        # milvus.collection이 아닌 milvus.collection에 접근
        results = milvus.collection.query(
            expr="id >= 0",
            output_fields=["path", "updated_at"], 
            limit=20000
        )
        
        for r in results:
            path = r["path"]
            ts = processor._normalize_timestamp(r.get("updated_at"))
            existing_files_info[path] = ts
            chunk_counts[path] = chunk_counts.get(path, 0) + 1
            
    except Exception as e:
        print(f"\n⚠️ 기존 파일 정보 조회 중 오류 발생: {e}")
        print("💡 이 오류는 Milvus 컬렉션이 비어있거나 데이터베이스 연결 문제일 수 있습니다.")
        print("💡 처리를 계속 진행합니다...")
        # 오류가 발생해도 계속 진행

    # 2. 파일 시스템 스캔 및 판단
    fs_paths = set()
    files_to_process = []
    skipped = []
    to_delete = []

    for root, _, files in os.walk(vault_path):
        for fname in files:
            if not fname.endswith((".md", ".pdf")) or fname.startswith("."):
                continue

            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, vault_path)
            fs_paths.add(rel_path)

            file_size = os.path.getsize(full_path)
            file_mtime = os.path.getmtime(full_path)
            est_chunks = estimate_chunk_count(file_size)

            db_ts = existing_files_info.get(rel_path, None)
            db_chunks = chunk_counts.get(rel_path, 0)

            if db_ts is None:
                files_to_process.append((full_path, rel_path))
                continue

            time_diff = abs(file_mtime - db_ts)

            if time_diff > 2.0:
                to_delete.append(rel_path)
                files_to_process.append((full_path, rel_path))
            elif time_diff < 0.1:
                if db_chunks >= est_chunks:
                    skipped.append(rel_path)
                else:
                    to_delete.append(rel_path)
                    files_to_process.append((full_path, rel_path))
            else:
                # VERIFY 단계: 의심 파일은 확인 후 재처리
                if db_chunks < est_chunks:
                    to_delete.append(rel_path)
                    files_to_process.append((full_path, rel_path))
                else:
                    skipped.append(rel_path)

    # 3. 삭제 대상 처리
    print(f"\n📄 수정된 파일 처리: {len(to_delete)}개")
    try:
        for rel_path in to_delete:
            milvus.mark_for_deletion(rel_path)
    except Exception as e:
        print(f"\n⚠️ 수정된 파일 삭제 중 오류: {e}")

    # 4. 삭제된 파일 처리
    db_paths = set(existing_files_info.keys())
    deleted_files = db_paths - fs_paths
    print(f"\n🚮 삭제된 파일 처리: {len(deleted_files)}개")
    try:
        for rel_path in deleted_files:
            milvus.mark_for_deletion(rel_path)
    except Exception as e:
        print(f"\n⚠️ 삭제된 파일 삭제 중 오류: {e}")

    # 5. 실질 처리 시작
    print(f"✅ 새로 처리할 파일: {len(files_to_process)}개")
    processed_count = 0
    failed_count = 0
    failed_files = []
    
    for full_path, rel_path in tqdm(files_to_process):
        try:
            processor.process_file(full_path)
            processed_count += 1
        except Exception as e:
            failed_count += 1
            failed_files.append(rel_path)
            print(f"\n⚠️ 파일 처리 오류 ({rel_path}): {e}")
    
    # 6. 삭제 적용
    try:
        if milvus.pending_deletions:
            print(f"\n🗑️ 배치 삭제 적용 중 ({len(milvus.pending_deletions)}개 파일)...")
            milvus.execute_pending_deletions()
        else:
            print("\nℹ️ 삭제할 파일이 없습니다.")
    except Exception as e:
        print(f"\n⚠️ 삭제 적용 중 오류: {e}")
    
    # 7. 결과 요약 출력
    print("\n------- 결과 요약 -------")
    print(f"🔁 변경 없음: {len(skipped)}개 파일")
    print(f"✅ 성공적 처리: {processed_count}개 파일")
    print(f"🗑️ 삭제된 파일: {len(deleted_files)}개")
    
    if failed_count > 0:
        print(f"\n⚠️ 오류 발생: {failed_count}개 파일")
        for f in failed_files[:5]:  # 처음 5개만 표시
            print(f"  - {f}")
        if len(failed_files) > 5:
            print(f"  - ... 그리고 {len(failed_files) - 5}개 더")

    return processed_count
