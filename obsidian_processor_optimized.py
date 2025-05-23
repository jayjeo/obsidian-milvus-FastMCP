"""
기존 obsidian_processor.py의 최적화된 버전
주요 개선사항: 배치 처리, 메모리 효율성, 속도 최적화
"""

import os
import re
import PyPDF2
import json
import yaml
import psutil
import colorama
from colorama import Fore, Style
import time
import threading
import gc
import sys
from datetime import datetime
import config
from embeddings import EmbeddingModel
from tqdm import tqdm

colorama.init()

class OptimizedObsidianProcessor:
    """최적화된 Obsidian 파일 처리기"""
    
    def __init__(self, milvus_manager):
        self.vault_path = config.OBSIDIAN_VAULT_PATH
        self.milvus_manager = milvus_manager
        self.embedding_model = EmbeddingModel()
        self.next_id = self._get_next_id()
        
        # 배치 처리를 위한 설정
        self.batch_embedding_size = 2000  # 대폭 증가
        self.chunk_accumulator = []
        self.metadata_accumulator = []
        
        print(f"✓ 최적화된 프로세서 초기화 완료 (배치 크기: {self.batch_embedding_size})")
        
    def _get_next_id(self):
        """다음 ID 값 가져오기"""
        results = self.milvus_manager.query("id >= 0", output_fields=["id"], limit=1)
        if not results:
            return 1
        return max([r['id'] for r in results]) + 1
    
    def process_all_files_optimized(self):
        """모든 파일을 최적화된 방식으로 처리"""
        print(f"\n{Fore.CYAN}최적화된 전체 파일 처리 시작{Style.RESET_ALL}")
        print(f"볼트 경로: {self.vault_path}")
        
        if not os.path.exists(self.vault_path):
            print(f"{Fore.RED}오류: 볼트 경로를 찾을 수 없습니다: {self.vault_path}{Style.RESET_ALL}")
            return 0
        
        # 1단계: 모든 파일 수집
        print("1단계: 처리할 파일 수집 중...")
        files_to_process = []
        total_size = 0
        
        for root, _, files in os.walk(self.vault_path):
            if os.path.basename(root).startswith(('.', '_')):
                continue
                
            for file in files:
                if file.endswith(('.md', '.pdf')) and not file.startswith('.'):
                    full_path = os.path.join(root, file)
                    file_size = os.path.getsize(full_path)
                    files_to_process.append(full_path)
                    total_size += file_size
        
        print(f"✓ 발견된 파일: {len(files_to_process)}개 ({total_size/(1024*1024):.1f} MB)")
        
        if not files_to_process:
            print("처리할 파일이 없습니다.")
            return 0
        
        # 2단계: 모든 청크 수집 (임베딩 없이)
        print("2단계: 모든 파일에서 청크 추출 중...")
        all_chunks = []
        all_metadata = []
        
        for i, file_path in enumerate(tqdm(files_to_process, desc="청크 추출")):
            try:
                chunks, metadata = self._extract_chunks_from_file(file_path)
                if chunks and metadata:
                    all_chunks.extend(chunks)
                    # 각 청크에 대해 메타데이터 복사
                    for chunk_idx, chunk in enumerate(chunks):
                        metadata_copy = metadata.copy()
                        metadata_copy['chunk_index'] = chunk_idx
                        metadata_copy['chunk_text'] = chunk
                        all_metadata.append(metadata_copy)
                        
            except Exception as e:
                print(f"\n파일 처리 오류 {file_path}: {e}")
                continue
        
        print(f"✓ 추출된 청크: {len(all_chunks)}개")
        
        if not all_chunks:
            print("추출된 청크가 없습니다.")
            return 0
        
        # 3단계: 대용량 배치 임베딩
        print("3단계: 대용량 배치 임베딩 처리...")
        start_time = time.time()
        
        vectors = self._process_chunks_in_batches(all_chunks)
        
        embedding_time = time.time() - start_time
        chunks_per_second = len(all_chunks) / embedding_time if embedding_time > 0 else 0
        
        print(f"✓ 임베딩 완료: {len(vectors)}개 벡터")
        print(f"  처리 시간: {embedding_time:.1f}초")
        print(f"  처리 속도: {chunks_per_second:.1f} 청크/초")
        
        # 4단계: Milvus에 저장
        print("4단계: 벡터 데이터베이스에 저장...")
        save_start = time.time()
        
        success_count = self._save_vectors_batch(vectors, all_chunks, all_metadata)
        
        save_time = time.time() - save_start
        
        print(f"✓ 저장 완료: {success_count}개 벡터")
        print(f"  저장 시간: {save_time:.1f}초")
        
        total_time = time.time() - start_time
        print(f"\n{Fore.GREEN}전체 처리 완료!{Style.RESET_ALL}")
        print(f"총 시간: {total_time:.1f}초")
        print(f"평균 속도: {len(all_chunks)/total_time:.1f} 청크/초")
        
        return success_count
    
    def _process_chunks_in_batches(self, chunks):
        """청크를 대용량 배치로 처리"""
        vectors = []
        batch_size = self.batch_embedding_size
        
        print(f"배치 크기: {batch_size}")
        
        # 메모리 정리
        gc.collect()
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 배치별 처리
        for i in tqdm(range(0, len(chunks), batch_size), desc="임베딩 배치 처리"):
            batch_chunks = chunks[i:i+batch_size]
            
            try:
                # 임베딩 모델의 배치 처리 메서드 직접 호출
                batch_vectors = self.embedding_model.get_embeddings_batch(batch_chunks)
                vectors.extend(batch_vectors)
                
                # 주기적 메모리 정리 (5배치마다)
                if (i // batch_size) % 5 == 0:
                    gc.collect()
                    if hasattr(torch, 'cuda') and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                print(f"\n배치 {i//batch_size + 1} 처리 오류: {e}")
                # 오류 발생 시 개별 처리로 폴백
                for chunk in batch_chunks:
                    try:
                        vector = self.embedding_model.get_embedding(chunk)
                        vectors.append(vector)
                    except:
                        vectors.append([0] * config.VECTOR_DIM)
        
        return vectors
    
    def _extract_chunks_from_file(self, file_path):
        """파일에서 청크와 메타데이터 추출 (최적화된 버전)"""
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            return None, None
            
        try:
            rel_path = os.path.relpath(file_path, self.vault_path)
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()[1:]
            
            if file_ext not in ['pdf', 'md']:
                return None, None
            
            # 파일 시간 정보
            file_stats = os.stat(file_path)
            created_at = str(file_stats.st_ctime)
            updated_at = str(file_stats.st_mtime)
            
            # 파일 타입별 처리
            if file_ext == 'pdf':
                content, title, tags = self._extract_pdf_fast(file_path)
            else:  # md
                content, title, tags = self._extract_markdown_fast(file_path)
            
            if not content or not content.strip():
                return None, None
            
            # 청크 분할
            chunks = self._split_into_chunks_fast(content)
            if not chunks:
                return None, None
            
            # 메타데이터 준비
            metadata = {
                "rel_path": rel_path,
                "title": title,
                "content": content if len(content) < 10000 else content[:10000],  # 길이 제한
                "file_ext": file_ext,
                "tags": tags,
                "created_at": created_at,
                "updated_at": updated_at
            }
            
            return chunks, metadata
            
        except Exception as e:
            print(f"\n파일 처리 오류 {file_path}: {e}")
            return None, None
    
    def _extract_markdown_fast(self, file_path):
        """마크다운 빠른 추출"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # 제목 추출
            title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            title = title_match.group(1) if title_match else os.path.basename(file_path).replace('.md', '')
            
            # 간단한 태그 추출
            tags = []
            
            # YAML 프론트매터에서 태그 추출 (단순화)
            yaml_match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
            if yaml_match:
                try:
                    frontmatter = yaml.safe_load(yaml_match.group(1))
                    if isinstance(frontmatter, dict) and 'tags' in frontmatter:
                        tags_data = frontmatter['tags']
                        if isinstance(tags_data, list):
                            tags = [str(tag).strip() for tag in tags_data if tag]
                        elif isinstance(tags_data, str):
                            tags = [tags_data.strip()]
                except:
                    pass
            
            # 인라인 태그 추출
            inline_tags = re.findall(r'#([a-zA-Z0-9_-]+)', content)
            tags.extend(inline_tags)
            
            # 중복 제거
            tags = list(set(tags))
            
            # 내용 정리
            content = re.sub(r'\$.*?\$', ' ', content)  # 수식 제거
            content = re.sub(r'\n{3,}', '\n\n', content)  # 과도한 줄바꿈 제거
            
            return content.strip(), title, tags
            
        except Exception as e:
            return "", os.path.basename(file_path).replace('.md', ''), []
    
    def _extract_pdf_fast(self, file_path):
        """PDF 빠른 추출"""
        title = os.path.basename(file_path).replace('.pdf', '')
        content = ""
        
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # 메타데이터에서 제목 추출
                if reader.metadata and '/Title' in reader.metadata:
                    title = reader.metadata['/Title']
                
                # 페이지별 내용 추출 (최대 50페이지만)
                for i, page in enumerate(reader.pages[:50]):  # 페이지 제한
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            content += page_text + "\n\n"
                    except:
                        continue
                
        except Exception as e:
            print(f"PDF 처리 오류 {file_path}: {e}")
            return None, None, None
        
        if not content.strip():
            return None, None, None
        
        return content.strip(), title, []
    
    def _split_into_chunks_fast(self, text):
        """빠른 청크 분할"""
        if not text:
            return []
        
        # 기본 설정
        chunk_size = config.CHUNK_SIZE
        chunk_overlap = config.CHUNK_OVERLAP
        chunk_min_size = config.CHUNK_MIN_SIZE
        
        # 텍스트가 청크 크기보다 작으면 그대로 반환
        if len(text) < chunk_size:
            return [text] if len(text) >= chunk_min_size else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # 문장 경계에서 분할 시도
            if end < len(text):
                # 문단 경계 찾기
                paragraph_end = text.find('\n\n', start, end)
                if paragraph_end != -1:
                    end = paragraph_end + 2
                else:
                    # 문장 경계 찾기
                    sentence_end = max(
                        text.rfind('. ', start, end),
                        text.rfind('.\n', start, end)
                    )
                    if sentence_end != -1:
                        end = sentence_end + 2
            
            # 청크 추출
            chunk = text[start:end].strip()
            
            if len(chunk) >= chunk_min_size:
                chunks.append(chunk)
            
            # 다음 시작 위치 (오버랩 적용)
            start = max(start + 1, end - chunk_overlap)
            
            # 진행 상황 확인
            if start >= len(text):
                break
        
        # 최대 청크 수 제한
        max_chunks = 200
        if len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]
        
        return chunks
    
    def _save_vectors_batch(self, vectors, chunks, metadata_list):
        """벡터를 배치로 저장"""
        if not vectors or len(vectors) != len(chunks):
            return 0
        
        success_count = 0
        batch_size = 100  # Milvus 저장 배치 크기
        
        for i in tqdm(range(0, len(vectors), batch_size), desc="데이터베이스 저장"):
            batch_vectors = vectors[i:i+batch_size]
            batch_chunks = chunks[i:i+batch_size]
            batch_metadata = metadata_list[i:i+batch_size]
            
            try:
                # 배치 데이터 준비
                batch_data = []
                for vector, chunk, metadata in zip(batch_vectors, batch_chunks, batch_metadata):
                    # 안전한 문자열 처리
                    def safe_string(s, max_len=1000):
                        if not isinstance(s, str):
                            return ""
                        return s[:max_len] if len(s) > max_len else s
                    
                    tags_json = json.dumps(metadata.get("tags", []))
                    
                    data_item = {
                        "id": self.next_id,
                        "path": safe_string(metadata["rel_path"]),
                        "title": safe_string(metadata["title"]),
                        "content": safe_string(metadata.get("content", "")),
                        "chunk_text": safe_string(chunk),
                        "chunk_index": metadata.get("chunk_index", 0),
                        "file_type": metadata["file_ext"],
                        "tags": safe_string(tags_json),
                        "created_at": metadata["created_at"],
                        "updated_at": metadata["updated_at"],
                        "vector": vector
                    }
                    batch_data.append(data_item)
                    self.next_id += 1
                
                # 배치 삽입
                for data_item in batch_data:
                    self.milvus_manager.insert_data(data_item)
                    success_count += 1
                
                # 주기적 flush
                if i % (batch_size * 5) == 0:
                    self.milvus_manager.collection.flush()
                    
            except Exception as e:
                print(f"\n배치 저장 오류: {e}")
        
        # 최종 flush
        self.milvus_manager.collection.flush()
        
        return success_count


def test_optimization():
    """최적화 테스트"""
    print("\n" + "="*60)
    print("최적화된 프로세서 테스트")
    print("="*60)
    
    try:
        from milvus_manager import MilvusManager
        
        # 매니저 초기화
        milvus_manager = MilvusManager()
        processor = OptimizedObsidianProcessor(milvus_manager)
        
        print(f"볼트 경로: {processor.vault_path}")
        print(f"배치 크기: {processor.batch_embedding_size}")
        
        # 파일 수 확인
        md_files = 0
        pdf_files = 0
        for root, _, files in os.walk(processor.vault_path):
            for file in files:
                if file.endswith('.md'):
                    md_files += 1
                elif file.endswith('.pdf'):
                    pdf_files += 1
        
        print(f"처리 대상 파일: MD {md_files}개, PDF {pdf_files}개")
        
        if md_files + pdf_files == 0:
            print("처리할 파일이 없습니다.")
            return
        
        print("\n테스트를 시작하시겠습니까? (y/n)")
        choice = input().strip().lower()
        
        if choice == 'y':
            result = processor.process_all_files_optimized()
            print(f"\n테스트 완료: {result}개 항목 처리됨")
        else:
            print("테스트를 취소했습니다.")
            
    except Exception as e:
        print(f"테스트 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_optimization()
