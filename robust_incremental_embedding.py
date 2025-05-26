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

    results = milvus.query(output_fields=["path", "updated_at"], limit=20000)
    for r in results:
        path = r["path"]
        ts = processor._normalize_timestamp(r.get("updated_at"))
        existing_files_info[path] = ts
        chunk_counts[path] = chunk_counts.get(path, 0) + 1

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
    for rel_path in to_delete:
        milvus.mark_for_deletion(rel_path)

    # 4. 삭제된 파일 처리
    db_paths = set(existing_files_info.keys())
    deleted_files = db_paths - fs_paths
    for rel_path in deleted_files:
        milvus.mark_for_deletion(rel_path)

    # 5. 실질 처리 시작
    print(f"✅ 새로 처리할 파일: {len(files_to_process)}개")
    for full_path, rel_path in tqdm(files_to_process):
        processor.process_file(full_path)

    # 6. 삭제 적용
    milvus.execute_pending_deletions()

    print(f"🔁 SKIP된 파일: {len(skipped)}개")
    print(f"🗑️ 삭제된 파일: {len(deleted_files)}개")

    return len(files_to_process)
