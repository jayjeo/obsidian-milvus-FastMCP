import os
import re
import PyPDF2
import markdown
import json
import yaml  # Added PyYAML for better frontmatter parsing
import psutil
import colorama
from colorama import Fore, Style
import time
import threading
import gc
import sys
from datetime import datetime
from bs4 import BeautifulSoup
import config
from embeddings import EmbeddingModel
from tqdm import tqdm
from functools import lru_cache
from progress_monitor_cmd import ProgressMonitor

# Windows에서 색상 표시를 위한 colorama 초기화
colorama.init()

class ObsidianProcessor:
    def __init__(self, milvus_manager):
        self.vault_path = config.OBSIDIAN_VAULT_PATH
        self.milvus_manager = milvus_manager
        self.embedding_model = EmbeddingModel()
        self.next_id = self._get_next_id()
        
        # GPU 사용 설정
        self.use_gpu = config.USE_GPU
        self.device_idx = config.GPU_DEVICE_ID if hasattr(config, 'GPU_DEVICE_ID') else 0
        
        # 임베딩 진행 상태 추적을 위한 변수
        self.embedding_in_progress = False
        self.embedding_progress = {
            "total_files": 0,
            "processed_files": 0,
            "total_size": 0,
            "processed_size": 0,
            "start_time": None,
            "current_file": "",
            "estimated_time_remaining": "",
            "percentage": 0,
            "is_full_reindex": False,
            "cpu_percent": 0,
            "memory_percent": 0,
            "gpu_percent": 0,
            "gpu_memory_used": 0,
            "gpu_memory_total": 0,
            "current_batch_size": 0
        }
        
        # 시스템 리소스 사용량 제한을 위한 설정
        self.max_cpu_percent = 85
        self.max_memory_percent = 80
        self.resource_check_interval = 2
        self.last_resource_check = 0
        self.dynamic_batch_size = config.BATCH_SIZE * 2
        self.min_batch_size = max(1, config.BATCH_SIZE // 2)
        self.max_batch_size = config.BATCH_SIZE * 4
        
        # 진행률 및 리소스 모니터링 관리자 생성
        self.monitor = ProgressMonitor(self)
        
        # 전역 처리 타임아웃 추가
        self.processing_timeout = 600  # 10분 (초 단위)
        
    def _get_next_id(self):
        """다음 ID 값 가져오기"""
        results = self.milvus_manager.query("id >= 0", output_fields=["id"], limit=1)
        if not results:
            return 1
        return max([r['id'] for r in results]) + 1
        
    def _create_ascii_bar(self, percent, width=20):
        """퍼센트 값을 받아 ASCII 그래프 바 생성"""
        # 값이 유효한 범위인지 확인
        if not isinstance(percent, (int, float)) or percent < 0:
            percent = 0
        elif percent > 100:
            percent = 100
            
        # 채워질 길이 계산 (반올림하여 최소 1칸은 표시)
        filled_length = max(1, int(width * percent / 100)) if percent > 0 else 0
        
        # 사용량에 따라 다른 문자 사용
        if percent > 90:
            bar_char = '#'  # 매우 높음
        elif percent > 70:
            bar_char = '='  # 높음
        elif percent > 50:
            bar_char = '-'  # 중간
        elif percent > 30:
            bar_char = '.'  # 낮음
        elif percent > 0:
            bar_char = '·'  # 매우 낮음
        else:
            bar_char = ' '  # 0%
        
        # 그래프 바 생성 (최대 길이 제한)
        filled_length = min(filled_length, width)
        bar = bar_char * filled_length + ' ' * (width - filled_length)
        return bar
        
    def _check_system_resources(self):
        """시스템 리소스 사용량 확인 및 배치 크기 조절"""
        current_time = time.time()
        
        if current_time - self.last_resource_check < self.resource_check_interval:
            return self.dynamic_batch_size
            
        self.last_resource_check = current_time
        
        # ProgressMonitor의 _update_system_resources 메서드 호출
        self.monitor._update_system_resources()
        
        return self.dynamic_batch_size
        
    def _update_progress_stats(self):
        """임베딩 진행률과 예상 남은 시간을 계산하는 메소드 (파일 크기 기반)"""
        if not self.embedding_in_progress:
            return
            
        # 진행률 계산 (파일 크기 기반)
        total_size = self.embedding_progress["total_size"]
        processed_size = self.embedding_progress["processed_size"]
        
        # 파일 개수도 함께 표시하기 위해 유지
        total_files = self.embedding_progress["total_files"]
        processed_files = self.embedding_progress["processed_files"]
        
        if total_size <= 0:
            self.embedding_progress["percentage"] = 0
            self.embedding_progress["estimated_time_remaining"] = "계산 중..."
            return
            
        # 진행도는 파일 크기 기준으로 계산
        if total_size > 0:
            percentage = min(99, int((processed_size / total_size) * 100))  # 100%는 완전히 완료되었을 때만 표시
            self.embedding_progress["percentage"] = percentage
        
        # 예상 남은 시간 계산 (파일 크기 기반)
        if processed_size > 0 and self.embedding_progress["start_time"] is not None:
            elapsed_time = time.time() - self.embedding_progress["start_time"]
            bytes_per_second = processed_size / elapsed_time if elapsed_time > 0 else 0
            
            if bytes_per_second > 0:
                remaining_size = total_size - processed_size
                remaining_seconds = remaining_size / bytes_per_second
                
                # 예상 시간 포맷팅
                if remaining_seconds < 60:
                    time_str = f"{int(remaining_seconds)}초"
                elif remaining_seconds < 3600:
                    minutes = int(remaining_seconds / 60)
                    seconds = int(remaining_seconds % 60)
                    time_str = f"{minutes}분 {seconds}초"
                else:
                    hours = int(remaining_seconds / 3600)
                    minutes = int((remaining_seconds % 3600) / 60)
                    time_str = f"{hours}시간 {minutes}분"
                    
                self.embedding_progress["estimated_time_remaining"] = time_str
            else:
                self.embedding_progress["estimated_time_remaining"] = "계산 중..."
        else:
            self.embedding_progress["estimated_time_remaining"] = "계산 중..."
        
        # 시스템 리소스 사용량 업데이트
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            
            self.embedding_progress["cpu_percent"] = cpu_percent
            self.embedding_progress["memory_percent"] = memory_percent
        except Exception as e:
            # 예외 발생 시 무시
            pass
        
    def start_monitoring(self):
        """모든 모니터링 시작"""
        self.monitor.start()
        
    def stop_monitoring(self):
        """모든 모니터링 중지"""
        self.monitor.stop()
    
    def process_file(self, file_path):
        """단일 파일 처리 및 색인 (최적화 및 안전장치 추가)"""
        if not os.path.exists(file_path):
            error_msg = f"Error: File not found: {file_path}"
            print(error_msg)
            if hasattr(self, 'monitor') and hasattr(self.monitor, 'add_error_log'):
                self.monitor.add_error_log(error_msg)
            return False
            
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        
        # 출력 줄이기 - 진행률 표시에서 표시할 것임
        
        # 임베딩 진행 상태 업데이트
        self.embedding_in_progress = True
        
        # 현재 파일 정보만 업데이트하고 기존 총 파일 수/크기 유지
        current_progress = self.embedding_progress.copy()
        
        # 현재 파일 처리를 위한 진행 정보 초기화 (전체 진행률은 유지)
        self.embedding_progress = {
            "total_files": current_progress.get("total_files", 1), # 기존 총 파일 수 유지
            "processed_files": current_progress.get("processed_files", 0), # 기존 처리된 파일 수 유지
            "total_size": current_progress.get("total_size", file_size), # 기존 총 크기 유지
            "processed_size": current_progress.get("processed_size", 0), # 기존 처리된 크기 유지
            "start_time": current_progress.get("start_time", time.time()),
            "current_file": file_name,
            "estimated_time_remaining": current_progress.get("estimated_time_remaining", "Calculating..."),
            "percentage": current_progress.get("percentage", 0),
            "is_full_reindex": current_progress.get("is_full_reindex", False),
            "cpu_percent": current_progress.get("cpu_percent", 0),
            "memory_percent": current_progress.get("memory_percent", 0),
            "current_batch_size": self.dynamic_batch_size
        }
        
        # 전역 타임아웃 설정
        processing_completed = threading.Event()
        processing_result = {"success": False}
        
        def process_with_timeout():
            try:
                # 리소스 모니터링 시작
                self.start_monitoring()
                
                try:
                    # 파일에서 청크 추출
                    chunks, metadata = self._extract_chunks_from_file(file_path)
                    if not chunks or not metadata:
                        # 진행률 표시에서 표시할 것이므로 출력 줄임
                        processing_result["success"] = False
                        self.monitor.last_processed_status = f"{Fore.RED}Fail{Fore.RESET}"
                        return
                    
                    # 청크 추출 성공 메시지 제거 - 진행률 표시에서 표시할 것임
                    
                    # 임베딩 진행 정보 업데이트
                    self.embedding_progress["current_file"] = file_name
                    
                    # 메모리 사용량 확인
                    self._check_memory_usage("Before embedding generation")
                    
                    # 청크에 대한 임베딩 생성
                    vectors = []
                    
                    # 모든 청크를 개별적으로 처리
                    for i, chunk in enumerate(chunks):
                        try:
                            # 메모리 사용량 체크 및 조절
                            if i > 0 and i % 5 == 0:
                                self._check_memory_usage(f"Processing chunk {i}/{len(chunks)}")
                                
                            # 벡터 임베딩 생성
                            vector = self.embedding_model.get_embedding(chunk)
                            vectors.append(vector)
                            
                            # 진행 상황 업데이트 메시지 출력 제거
                            # if i > 0 and i % 5 == 0:
                            #     print(f"Processed {i}/{len(chunks)} chunks")
                                
                        except Exception as e:
                            print(f"Error embedding chunk {i}: {e}")
                            # 오류 발생 시 빈 벡터 추가 (처리 계속 진행)
                            vectors.append([0] * config.VECTOR_DIM)
                    
                    # 메타데이터 매핑 준비
                    chunk_file_map = [metadata] * len(chunks)
                    
                    # 메모리 사용량 확인
                    self._check_memory_usage("Before saving to Milvus")
                    
                    # 벡터 저장
                    success = self._save_vectors_to_milvus(vectors, chunks, chunk_file_map)
                    
                    # 메모리 효율성을 위한 명시적 변수 해제
                    del chunks
                    del vectors
                    del chunk_file_map
                    del metadata
                    
                    # 메모리 정리
                    gc.collect()
                    if 'torch' in sys.modules:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # 처리 결과 저장 및 상태 표시 업데이트
                    processing_result["success"] = success
                    
                    # 성공/실패 상태 업데이트
                    if success:
                        self.monitor.last_processed_status = f"{Fore.GREEN}Success{Fore.RESET}"
                    else:
                        self.monitor.last_processed_status = f"{Fore.RED}Fail{Fore.RESET}"
                    
                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")
                    processing_result["success"] = False
                    # 모니터링은 finally 블록에서 중지됨
                
            finally:
                # 리소스 모니터링 중지
                self.stop_monitoring()

                # 임베딩 진행 상태 완료
                # 현재 파일의 크기를 추가하여 업데이트
                if "total_size" in self.embedding_progress and file_size > 0:
                    # 이미 추가되지 않았다면 처리된 크기 추가
                    if not hasattr(self, '_processed_this_file') or not self._processed_this_file:
                        self.embedding_progress["processed_size"] += file_size
                        self._processed_this_file = True
                    
                    # 진행률 업데이트
                    if self.embedding_progress["total_size"] > 0:
                        percentage = min(99, int((self.embedding_progress["processed_size"] / self.embedding_progress["total_size"]) * 100))
                        self.embedding_progress["percentage"] = percentage
                
                self.embedding_in_progress = False
                
                # 이벤트 설정하여 타임아웃 처리 알림
                processing_completed.set()
        
        # 별도 스레드에서 처리 실행
        processing_thread = threading.Thread(target=process_with_timeout)
        processing_thread.daemon = True
        processing_thread.start()
        
        # 임시 속성 초기화 (파일간 중복 처리 방지)
        self._processed_this_file = False
        
        # 타임아웃 적용
        completed = processing_completed.wait(timeout=self.processing_timeout)
        
        if not completed:
            print(f"Error: Processing timed out after {self.processing_timeout} seconds")
            # 리소스 모니터링 중지 (타임아웃 발생 시)
            self.stop_monitoring()
            self.embedding_in_progress = False
            return False
        
        return processing_result["success"]
    
    def _check_memory_usage(self, stage=""):
        """메모리 사용량 확인 및 필요시 강제 정리"""
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent
        
        # 메모리 사용량 메시지 출력 제거
        # print(f"Memory usage at {stage}: {memory_percent:.1f}%")
        
        # 메모리 사용량이 임계값을 넘으면 강제 정리
        if memory_percent > 90:
            # print(f"Warning: High memory usage detected ({memory_percent:.1f}%), forcing cleanup")
            gc.collect()
            if 'torch' in sys.modules:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 2차 메모리 확인
            memory_info = psutil.virtual_memory()
            # print(f"Memory usage after cleanup: {memory_info.percent:.1f}%")
    
    def _extract_chunks_from_file(self, file_path):
        """파일에서 청크와 메타데이터를 추출하는 최적화된 메소드"""
        # 파일 경로 검증
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            return None, None
            
        try:
            rel_path = os.path.relpath(file_path, self.vault_path)
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()[1:] if '.' in file_path else ''
            
            # 파일명 검증 - 비어있거나 특수 문자만 있는 경우 처리
            if not file_name or file_name.startswith('.'):
                return None, None
            
            # 마크다운과 PDF만 처리 (다른 파일은 벡터 임베딩 제외)
            if file_ext.lower() not in ['pdf', 'md']:
                print(f"Skipping non-supported file type: {file_ext} - {file_path}")
                return None, None
                
            # 파일 생성/수정 시간 가져오기
            file_stats = os.stat(file_path)
            created_at = str(file_stats.st_ctime)
            updated_at = str(file_stats.st_mtime)
            
            # 파일 타입에 따라 텍스트 추출
            try:
                if file_ext == 'pdf':
                    content, title, tags = self._extract_pdf(file_path)
                elif file_ext == 'md':
                    content, title, tags = self._extract_markdown(file_path)
                else:
                    return None, None
            except Exception as e:
                print(f"Error extracting content from {file_path}: {e}")
                return None, None
            
            # 내용이 비어있는지 확인
            if not content or not content.strip():
                return None, None
            
            # 청크로 분할
            chunks = self._split_into_chunks(content)
            if not chunks:
                return None, None
                
            # 파일 메타데이터 준비 - content는 첫 번째 청크에만 저장
            metadata = {
                "rel_path": rel_path,
                "title": title,
                "content": content,  # 청크 처리 후 메모리에서 제거됨
                "file_ext": file_ext,
                "is_pdf": file_ext.lower() == 'pdf',
                "tags": tags,
                "created_at": created_at,
                "updated_at": updated_at
            }
            
            return chunks, metadata
            
        except Exception as e:
            error_msg = f"Error processing file {file_path}: {e}"
            print(error_msg)
            if hasattr(self, 'monitor') and hasattr(self.monitor, 'add_error_log'):
                self.monitor.add_error_log(error_msg)
            return None, None
    
    def _extract_markdown(self, file_path):
        """마크다운 파일에서 텍스트 및 메타데이터 추출 (최적화)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # 제목 추출 (첫 번째 # 헤딩 또는 파일명)
            title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            title = title_match.group(1) if title_match else os.path.basename(file_path).replace('.md', '')
            
            # YAML 프론트매터 및 태그 추출 (개선된 방식)
            tags = []
            yaml_match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
            
            if yaml_match:
                try:
                    # 안전한 YAML 파싱
                    frontmatter_text = yaml_match.group(1)
                    # 특수 문자 및 악성 문자열 제거 (security)
                    frontmatter_text = re.sub(r'[^\w\s\-\[\]:#\'",._{}]+', ' ', frontmatter_text)
                    
                    try:
                        # YAML 파싱 시도
                        frontmatter = yaml.safe_load(frontmatter_text)
                        if isinstance(frontmatter, dict):
                            # 태그 추출
                            if 'tags' in frontmatter:
                                tags_data = frontmatter['tags']
                                if isinstance(tags_data, list):
                                    tags = [str(tag).strip() for tag in tags_data if tag]
                                elif isinstance(tags_data, str):
                                    tags = [tags_data.strip()]
                    except Exception as yaml_err:
                        error_msg = f"YAML parsing error: {yaml_err}, falling back to regex for {os.path.basename(file_path)}"
                        print(error_msg)
                        if hasattr(self, 'monitor') and hasattr(self.monitor, 'add_error_log'):
                            self.monitor.add_error_log(error_msg)
                        # YAML 파싱 실패 시 정규식으로 폴백
                        tag_match = re.search(r'tags:\s*\[(.*?)\]', frontmatter_text, re.DOTALL)
                        if tag_match:
                            tags_str = tag_match.group(1)
                            tags = [tag.strip().strip("'\"") for tag in tags_str.split(',') if tag.strip()]
                        else:
                            tag_lines = re.findall(r'tags:\s*\n((?:\s*-\s*.+\n)+)', frontmatter_text)
                            if tag_lines:
                                for line in tag_lines[0].split('\n'):
                                    tag_match = re.match(r'\s*-\s*(.+)', line)
                                    if tag_match:
                                        tags.append(tag_match.group(1).strip().strip("'\""))
                except Exception as e:
                    print(f"Error processing frontmatter: {e}")
            
            # 인라인 태그 추출 (#태그)
            inline_tags = re.findall(r'#([a-zA-Z0-9_-]+)', content)
            tags.extend(inline_tags)
            
            # 중복 태그 제거
            tags = list(set([tag for tag in tags if tag and isinstance(tag, str)]))
            
            # 특수 문자 처리 (LaTeX 수식 등 처리)
            # $~$ 같은 수식 기호를 일반 텍스트로 변환
            content = re.sub(r'\$~\$', ' ', content)
            content = re.sub(r'\${2}.*?\${2}', ' ', content, flags=re.DOTALL)  # 블록 수식 처리
            content = re.sub(r'\$.*?\$', ' ', content)  # 인라인 수식 처리
            
            # 불필요한 여러 줄 공백 제거
            content = re.sub(r'\n{3,}', '\n\n', content)
            
            # 후행 공백 제거
            content = content.rstrip()
            
            return content, title, tags
            
        except Exception as e:
            print(f"Error in _extract_markdown: {e}")
            # 오류 발생 시 기본값 반환
            return "", os.path.basename(file_path).replace('.md', ''), []
    
    def _extract_pdf(self, file_path):
        """PDF 파일에서 텍스트 추출 (메모리 효율성 개선)"""
        title = os.path.basename(file_path).replace('.pdf', '')
        content = ""
        
        try:
            with open(file_path, 'rb') as file:
                # 안전하게 PDF 읽기 시도
                try:
                    reader = PyPDF2.PdfReader(file)
                    
                    # PDF 메타데이터에서 정보 추출
                    metadata = reader.metadata
                    if metadata and '/Title' in metadata and metadata['/Title']:
                        title = metadata['/Title']
                    
                    # 페이지 별로 내용 추출 (메모리 효율성 개선)
                    for i, page in enumerate(reader.pages):
                        # 메모리 관리를 위해 10페이지마다 정리
                        if i > 0 and i % 10 == 0:
                            gc.collect()
                            
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                content += page_text + "\n\n"
                        except Exception as e:
                            print(f"Error extracting text from page {i}: {e}")
                
                except Exception as e:
                    print(f"Error reading PDF: {e}")
        
        except Exception as e:
            print(f"Error opening PDF file: {e}")
        
        # 빈 내용인 경우 확인
        if not content.strip():
            error_msg = f"Warning: No content extracted from PDF {file_path} - likely a scanned document"
            print(f"\n{Fore.YELLOW}{error_msg}{Style.RESET_ALL}")
            if hasattr(self, 'monitor') and hasattr(self.monitor, 'add_error_log'):
                self.monitor.add_error_log(error_msg)
            # 스캔본으로 판단되는 PDF는 처리하지 않음
            return None, None, None
        
        # 텍스트 품질 확인 (스캔본 PDF 감지)
        if len(content) > 0:
            # 텍스트 품질 확인 - 의미 있는 텍스트의 비율
            # 스캔본 PDF는 종종 의미 없는 문자나 기호가 많이 포함됨
            meaningful_chars = sum(1 for c in content if c.isalnum() or c.isspace())
            total_chars = len(content)
            quality_ratio = meaningful_chars / total_chars if total_chars > 0 else 0
            
            # 품질이 너무 낮은 경우 (스캔본 PDF로 간주)
            if quality_ratio < 0.5 and total_chars > 100:
                error_msg = f"Warning: Low quality text extracted from PDF {file_path} (quality: {quality_ratio:.2f}) - likely a scanned document"
                print(f"\n{Fore.YELLOW}{error_msg}{Style.RESET_ALL}")
                if hasattr(self, 'monitor') and hasattr(self.monitor, 'add_error_log'):
                    self.monitor.add_error_log(error_msg)
                return None, None, None
        
        return content, title, []  # PDF는 기본적으로 태그가 없음
    
    def _split_into_chunks(self, text):
        """텍스트를 청크로 분할 (성능 및 메모리 효율성 최적화)"""
        if not text:
            return []
        
        # 텍스트 전처리 - 특수 문자 및 반복되는 공백 정리
        # 이 단계에서 특수 기호나 문제를 일으킬 수 있는 패턴을 처리
        text = re.sub(r'\s+', ' ', text)  # 여러 공백을 하나로 통합
        text = re.sub(r'\.{3,}', '...', text)  # 여러 점(...)을 하나로 통합
        
        # 안전을 위한 텍스트 길이 제한
        max_safe_length = 100000  # 최대 10만 자
        if len(text) > max_safe_length:
            print(f"Warning: Text too long ({len(text)} chars), truncating")
            text = text[:max_safe_length]
            
        # 사전 검사로 메모리 효율성 개선
        text_length = len(text)
        chunk_size = config.CHUNK_SIZE
        chunk_overlap = config.CHUNK_OVERLAP
        chunk_min_size = config.CHUNK_MIN_SIZE
            
        # 너무 짧은 텍스트는 그대로 반환
        if text_length < chunk_size:
            return [text] if text_length >= chunk_min_size else []
        
        # 청크 분할
        chunks = []
        start = 0
        
        # 성능 개선을 위한 로컬 변수 캐싱
        text_find = text.find
        text_rfind = text.rfind
        
        # 무한 루프 방지를 위한 안전 장치
        max_iterations = text_length * 2  # 극단적인 경우에도 안전하게 종료되도록
        iteration_count = 0
        
        while start < text_length and iteration_count < max_iterations:
            iteration_count += 1
            
            # 청크 크기 계산 (최소 크기 보장)
            end = min(start + chunk_size, text_length)
            
            # 의미 단위(문장, 문단) 경계에서 분할
            if end < text_length:
                # 문단 경계 찾기
                paragraph_end = text_find('\n\n', start, end)
                if paragraph_end != -1:
                    end = paragraph_end + 2
                else:
                    # 문장 경계 찾기
                    sentence_end = max(
                        text_rfind('. ', start, end),
                        text_rfind('? ', start, end),
                        text_rfind('! ', start, end),
                        text_rfind('.\n', start, end),
                        text_rfind('?\n', start, end),
                        text_rfind('!\n', start, end)
                    )
                    
                    if sentence_end != -1:
                        end = sentence_end + 2
                    else:
                        # 단어 경계 찾기
                        space_pos = text_rfind(' ', start, end)
                        if space_pos != -1:
                            end = space_pos + 1
                        # 공백을 찾지 못하면 그냥 청크 크기 사용
            
            # 청크 추출
            chunk = text[start:end].strip()
            
            # 유효한 청크만 추가
            if len(chunk) >= chunk_min_size:
                chunks.append(chunk)
            
            # 진행 무한 루프 방지
            if start == end:
                print(f"Warning: Chunking stuck at position {start}, breaking")
                break
                
            # 다음 청크로 이동 (오버랩 적용)
            start = max(start + 1, end - chunk_overlap)  # 최소 1자 이상 진행 보장
            
            # 시작 위치가 텍스트 끝을 넘어가는 경우 중지
            if start >= text_length:
                break
        
        # 무한 루프 검사
        if iteration_count >= max_iterations:
            print("Warning: Max iterations reached in chunking, potential infinite loop avoided")
        
        # 중복 청크 제거
        unique_chunks = []
        seen = set()
        for chunk in chunks:
            # 짧은 chunk는 그대로 추가
            if len(chunk) < 100:
                unique_chunks.append(chunk)
            else:
                # 긴 청크는 해시를 통해 중복 검사
                chunk_hash = hash(chunk)
                if chunk_hash not in seen:
                    seen.add(chunk_hash)
                    unique_chunks.append(chunk)
        
        # 청크 개수 제한
        max_chunks = 100  # 최대 청크 수 제한
        if len(unique_chunks) > max_chunks:
            print(f"Warning: Too many chunks ({len(unique_chunks)}), limiting to {max_chunks}")
            unique_chunks = unique_chunks[:max_chunks]
            
        return unique_chunks
    
    def _save_vectors_to_milvus(self, vectors, chunks, chunk_file_map):
        """벡터와 청크 데이터를 Milvus에 저장하는 최적화된 메소드"""
        if not vectors or not chunks or not chunk_file_map or len(vectors) != len(chunks):
            return False
            
        try:
            # 각 청크와 벡터를 개별적으로 처리하여 Milvus에 삽입
            # 파일별 청크 인덱스 추적
            file_chunk_indices = {}
            
            # 성공적으로 삽입된 항목 수 추적
            success_count = 0
            
            # 각 청크와 벡터 처리
            for i, (vector, chunk, metadata) in enumerate(zip(vectors, chunks, chunk_file_map)):
                # 메모리 모니터링
                if i > 0 and i % 20 == 0:
                    self._check_memory_usage(f"Milvus insertion {i}/{len(chunks)}")
                
                rel_path = metadata["rel_path"]
                
                # 파일별 청크 인덱스 추적
                if rel_path not in file_chunk_indices:
                    file_chunk_indices[rel_path] = 0
                chunk_index = file_chunk_indices[rel_path]
                file_chunk_indices[rel_path] += 1
                
                # 태그 JSON 변환 (안전한 형식으로)
                try:
                    tags_json = json.dumps(metadata["tags"]) if metadata["tags"] else "[]"
                except:
                    tags_json = "[]"
                
                # 최대 문자열 길이 (Milvus 제한보다 안전하게 설정)
                MAX_STRING_LENGTH = 65000  # Milvus 최대 한계: 65535
                
                # 문자열 안전하게 자르기 위한 함수
                def safe_truncate(text, max_len=MAX_STRING_LENGTH):
                    if not isinstance(text, str):
                        return text
                    return text[:max_len] if text and len(text) > max_len else text
                
                # 각 항목을 개별적으로 삽입
                single_data = {
                    "id": self.next_id,
                    "path": safe_truncate(rel_path, 500),
                    "title": safe_truncate(metadata["title"], 500) if metadata["title"] else "",
                    # 첫 번째 청크일 때만 전체 내용 저장, 나머지는 빈 문자열
                    "content": safe_truncate(metadata["content"], MAX_STRING_LENGTH) if chunk_index == 0 else "",
                    "chunk_text": safe_truncate(chunk, MAX_STRING_LENGTH),
                    "chunk_index": chunk_index,
                    "file_type": safe_truncate(metadata["file_ext"], 10),
                    "tags": safe_truncate(tags_json, 1000),
                    "created_at": safe_truncate(metadata["created_at"], 30),
                    "updated_at": safe_truncate(metadata["updated_at"], 30),
                    "vector": vector
                }
                
                # 추가 데이터 유효성 검사 (안전 장치)
                valid_data = True
                for key, value in single_data.items():
                    if key != "vector" and isinstance(value, str) and len(value) > MAX_STRING_LENGTH:
                        print(f"Warning: Field {key} still too long ({len(value)} chars) after truncation, forcing truncation")
                        single_data[key] = value[:MAX_STRING_LENGTH]  # 강제 제한
                
                # 단일 항목 삽입
                try:
                    if valid_data:
                        self.milvus_manager.insert_data(single_data)
                        success_count += 1
                        # 10개 항목마다 flush - 메모리 관리
                        if success_count % 10 == 0:
                            self.milvus_manager.collection.flush()
                except Exception as e:
                    print(f"Error inserting item {self.next_id}: {e}")
                
                self.next_id += 1
            
            # 최종 flush
            self.milvus_manager.collection.flush()
            
            print(f"Successfully inserted {success_count} out of {len(chunks)} items")
            return success_count > 0
            
        except Exception as e:
            print(f"Error saving vectors to Milvus: {e}")
            return False
    
    # DEPRECATED: This method is no longer used for performance reasons
    # Embedding validation is now done during initial data loading
    def _verify_file_has_valid_embeddings_DEPRECATED(self, file_path):
        """DEPRECATED: Verify that a file has valid embedding data in the database.
        
        This method was causing performance issues because it was called for every
        file during scanning. Now we pre-load all embedding validation data.
        """
        pass
    
    def process_updated_files(self):
        """볼트의 새로운 파일 또는 수정된 파일만 처리 + 삭제된 파일 정리 (증분 임베딩) - ENHANCED VERSION"""
        print(f"\n{Fore.CYAN}[DEBUG] Starting process_updated_files with deleted file cleanup (ENHANCED VERSION){Style.RESET_ALL}")
        print(f"Processing new/modified files AND cleaning up deleted files in {self.vault_path}")
        
        # 경로 존재 확인
        if not os.path.exists(self.vault_path):
            error_msg = f"Error: Obsidian vault path not found: {self.vault_path}"
            print(f"\n{Fore.RED}{error_msg}{Style.RESET_ALL}")
            if hasattr(self, 'monitor') and hasattr(self.monitor, 'add_error_log'):
                self.monitor.add_error_log(error_msg)
            return 0
        
        print(f"\n{Fore.CYAN}[DEBUG] Vault path exists: {self.vault_path}{Style.RESET_ALL}")
        
        # 파일 시스템 접근 확인
        try:
            test_file = os.path.join(self.vault_path, "test_access.txt")
            with open(test_file, 'w') as f:
                f.write("test access")
            os.remove(test_file)
            print(f"{Fore.CYAN}[DEBUG] File system access test passed{Style.RESET_ALL}")
        except Exception as e:
            error_msg = f"Error: Cannot write to vault directory: {e}"
            print(f"\n{Fore.RED}{error_msg}{Style.RESET_ALL}")
            import traceback
            print(f"\n{Fore.RED}Stack trace:\n{traceback.format_exc()}{Style.RESET_ALL}")
            if hasattr(self, 'monitor') and hasattr(self.monitor, 'add_error_log'):
                self.monitor.add_error_log(error_msg)
            self.embedding_in_progress = False
            return 0
        
        # 임베딩 진행 상태 초기화
        self.embedding_in_progress = True
        self.embedding_progress = {
            "total_files": 0,
            "processed_files": 0,
            "total_size": 0,
            "processed_size": 0,
            "start_time": time.time(),
            "current_file": "",
            "estimated_time_remaining": "Calculating...",
            "percentage": 0,
            "is_full_reindex": False,  # 증분 임베딩은 전체 재인덱싱이 아님
            "cpu_percent": 0,
            "memory_percent": 0,
            "current_batch_size": self.dynamic_batch_size
        }
        
        # 진행도 계산을 위한 변수 초기화
        self.embedding_progress["processed_size"] = 0
        self.embedding_progress["processed_files"] = 0
        
        # 전체 파일 수와 크기 미리 계산 전 Milvus 연결 테스트
        try:
            print(f"\n{Fore.CYAN}[DEBUG] Checking Milvus connection...{Style.RESET_ALL}")
            test_query = self.milvus_manager.query("id >= 0", limit=1)
            print(f"\n{Fore.CYAN}[DEBUG] Milvus connection successful. Query result: {test_query}{Style.RESET_ALL}")
        except Exception as e:
            error_msg = f"Error connecting to Milvus: {e}"
            print(f"\n{Fore.RED}{error_msg}{Style.RESET_ALL}")
            import traceback
            print(f"\n{Fore.RED}Stack trace:\n{traceback.format_exc()}{Style.RESET_ALL}")
            if hasattr(self, 'monitor') and hasattr(self.monitor, 'add_error_log'):
                self.monitor.add_error_log(error_msg)
            self.embedding_in_progress = False  # 반드시 상태를 False로 설정
            return 0
        
        # 임베딩 모델 테스트
        try:
            print(f"\n{Fore.CYAN}[DEBUG] Testing embedding model...{Style.RESET_ALL}")
            test_vector = self.embedding_model.get_embedding("Test embedding model")
            print(f"\n{Fore.CYAN}[DEBUG] Embedding model OK, vector dimension: {len(test_vector)}{Style.RESET_ALL}")
        except Exception as e:
            error_msg = f"Error with embedding model: {e}"
            print(f"\n{Fore.RED}{error_msg}{Style.RESET_ALL}")
            import traceback
            print(f"\n{Fore.RED}Stack trace:\n{traceback.format_exc()}{Style.RESET_ALL}")
            if hasattr(self, 'monitor') and hasattr(self.monitor, 'add_error_log'):
                self.monitor.add_error_log(error_msg)
            self.embedding_in_progress = False  # 반드시 상태를 False로 설정
            return 0
        
        # 기존 파일 정보 가져오기 - PERFORMANCE OPTIMIZED VERSION
        existing_files_info = {}
        files_with_valid_embeddings = set()
        
        try:
            print(f"{Fore.CYAN}[DEBUG] Querying existing files from Milvus (optimized)...{Style.RESET_ALL}")
            max_limit = 16000
            offset = 0
            
            while True:
                results = self.milvus_manager.query(
                    output_fields=["path", "updated_at", "chunk_text"],
                    limit=max_limit,
                    offset=offset,
                    expr="id >= 0"
                )
                
                if not results:
                    break
                    
                for doc in results:
                    path = doc.get("path")
                    updated_at = doc.get('updated_at')
                    chunk_text = doc.get('chunk_text', "")
                    
                    if path:
                        # Take the latest updated_at for each path (in case of multiple chunks)
                        if path not in existing_files_info:
                            existing_files_info[path] = updated_at
                        else:
                            # Compare and keep the latest timestamp
                            try:
                                current_time = float(existing_files_info[path]) if existing_files_info[path] else 0
                                new_time = float(updated_at) if updated_at else 0
                                if new_time > current_time:
                                    existing_files_info[path] = updated_at
                            except (ValueError, TypeError):
                                # If conversion fails, use the new value
                                existing_files_info[path] = updated_at
                        
                        # Track files with valid embedding data
                        if chunk_text and len(chunk_text.strip()) > 0:
                            files_with_valid_embeddings.add(path)
                
                offset += max_limit
                if len(results) < max_limit:
                    break
                
                # 메모리 관리
                gc.collect()
                
            print(f"{Fore.CYAN}[DEBUG] Found {len(existing_files_info)} files in DB, {len(files_with_valid_embeddings)} with valid embeddings{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"Warning: Error fetching existing files: {e}")
        
        # 파일 목록 수집 및 삭제된 파일 탐지
        print("Scanning files for changes and detecting deleted files...")
        files_to_process = []
        skipped_count = 0
        total_files_count = 0
        total_files_size = 0
        new_or_modified_count = 0
        new_or_modified_size = 0
        
        # 현재 파일 시스템의 파일들 수집
        fs_files = set()
        
        print(f"{Fore.CYAN}[DEBUG] Walking through directory: {self.vault_path}{Style.RESET_ALL}")
        
        for root, _, files in os.walk(self.vault_path):
            # 숨겨진 폴더 건너뛰기
            if os.path.basename(root).startswith(('.', '_')):
                print(f"{Fore.CYAN}[DEBUG] Skipping hidden directory: {root}{Style.RESET_ALL}")
                continue
                
            for file in files:
                # 마크다운과 PDF만 처리
                if file.endswith(('.md', '.pdf')) and not file.startswith('.'):
                    total_files_count += 1
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.vault_path)
                    
                    # 파일 시스템 파일 목록에 추가
                    fs_files.add(rel_path)
                    
                    try:
                        # 파일 크기 및 수정 시간 가져오기
                        file_size = os.path.getsize(full_path)
                        total_files_size += file_size
                        file_mtime = os.path.getmtime(full_path)
                        
                        # 새 파일이거나 수정된 파일인지 확인 - IMPROVED LOGIC
                        is_new_or_modified = True
                        skip_reason = ""
                        
                        if rel_path in existing_files_info:
                            # 기존 파일이 있는 경우 수정 시간 비교
                            existing_mtime = existing_files_info.get(rel_path, 0)
                            
                            # 상세 디버깅 정보 출력
                            print(f"{Fore.CYAN}[DEBUG] File: {rel_path}{Style.RESET_ALL}")
                            print(f"{Fore.CYAN}[DEBUG] - Existing mtime in DB: {existing_mtime}{Style.RESET_ALL}")
                            print(f"{Fore.CYAN}[DEBUG] - Current file mtime: {file_mtime}{Style.RESET_ALL}")
                            print(f"{Fore.CYAN}[DEBUG] - Existing mtime type: {type(existing_mtime)}{Style.RESET_ALL}")
                            print(f"{Fore.CYAN}[DEBUG] - Current mtime type: {type(file_mtime)}{Style.RESET_ALL}")
                            
                            # 전체 재처리 모드인지 확인
                            is_full_reindex = self.embedding_progress.get("is_full_reindex", False)
                            
                            # 전체 재처리 모드인 경우 모든 파일 처리
                            if is_full_reindex:
                                print(f"{Fore.CYAN}[DEBUG] - Result: FULL REINDEX MODE (will process) - File: {rel_path}{Style.RESET_ALL}")
                            else:
                                # 증분 처리 모드에서는 수정 시간 비교 AND 임베딩 데이터 존재 확인
                                try:
                                    if isinstance(existing_mtime, str):
                                        existing_mtime = float(existing_mtime)
                                except (ValueError, TypeError):
                                    print(f"{Fore.YELLOW}[WARNING] Could not convert existing_mtime to float: {existing_mtime}{Style.RESET_ALL}")
                                    # 변환 실패 시 파일을 새로 처리하도록 설정
                                    existing_mtime = 0
                                
                                # PERFORMANCE FIX: Check embeddings from pre-loaded memory data
                                has_valid_embeddings = rel_path in files_with_valid_embeddings
                                
                                # 수정 시간 비교: 파일의 수정 시간이 DB에 저장된 시간보다 더 최신인 경우에만 처리
                                # BUT ALSO: Only skip if valid embeddings actually exist
                                if existing_mtime and file_mtime <= existing_mtime and has_valid_embeddings:
                                    # 파일이 변경되지 않음 AND 유효한 임베딩이 존재함
                                    is_new_or_modified = False
                                    skipped_count += 1
                                    skip_reason = f"UNCHANGED with valid embeddings"
                                    print(f"{Fore.CYAN}[DEBUG] - Result: {skip_reason} (skipping) - File time: {file_mtime}, DB time: {existing_mtime}{Style.RESET_ALL}")
                                else:
                                    # 파일이 수정되었거나 임베딩이 유효하지 않음
                                    if not has_valid_embeddings:
                                        skip_reason = f"INVALID/MISSING embeddings (will reprocess)"
                                        print(f"{Fore.YELLOW}[DEBUG] - Result: {skip_reason} - File time: {file_mtime}, DB time: {existing_mtime}{Style.RESET_ALL}")
                                    else:
                                        skip_reason = f"MODIFIED (will process)"
                                        print(f"{Fore.CYAN}[DEBUG] - Result: {skip_reason} - File time: {file_mtime}, DB time: {existing_mtime}{Style.RESET_ALL}")
                        else:
                            print(f"{Fore.CYAN}[DEBUG] File: {rel_path} - NEW FILE (not in database){Style.RESET_ALL}")
                        
                        if is_new_or_modified:
                            # 새 파일이거나 수정된 파일
                            new_or_modified_count += 1
                            new_or_modified_size += file_size
                            
                            # 수정된 경우 이전 데이터 삭제
                            if rel_path in existing_files_info:
                                print(f"{Fore.YELLOW}File will be reprocessed: {rel_path} (Reason: {skip_reason}){Style.RESET_ALL}")
                                self.milvus_manager.mark_for_deletion(rel_path)
                            else:
                                print(f"{Fore.GREEN}New file found: {rel_path}{Style.RESET_ALL}")
                            
                            # 처리할 파일 목록에 추가
                            files_to_process.append((full_path, file_size))
                    except Exception as e:
                        print(f"Warning: Error checking file {rel_path}: {e}")
        
        # ENHANCED: 삭제된 파일 탐지 및 정리
        print(f"\n{Fore.MAGENTA}[DELETED FILES CLEANUP] Detecting deleted files...{Style.RESET_ALL}")
        deleted_files = set(existing_files_info.keys()) - fs_files
        
        if deleted_files:
            print(f"{Fore.YELLOW}Found {len(deleted_files)} deleted files:{Style.RESET_ALL}")
            
            # 처음 5개 파일만 표시
            display_count = min(5, len(deleted_files))
            for i, file_path in enumerate(list(deleted_files)[:display_count]):
                print(f"{Fore.YELLOW}  {i+1}. {file_path}{Style.RESET_ALL}")
            
            if len(deleted_files) > display_count:
                print(f"{Fore.YELLOW}  ... and {len(deleted_files) - display_count} more files{Style.RESET_ALL}")
            
            print(f"\n{Fore.CYAN}Starting automatic cleanup of deleted files...{Style.RESET_ALL}")
            
            try:
                # 삭제된 파일들의 임베딩 정리
                cleanup_count = self.cleanup_deleted_embeddings(list(deleted_files))
                
                if cleanup_count > 0:
                    print(f"{Fore.GREEN}✅ Successfully cleaned up {cleanup_count} deleted files{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}⚠️ No deleted files were cleaned up{Style.RESET_ALL}")
                    
            except Exception as e:
                print(f"{Fore.RED}Error during deleted files cleanup: {e}{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}✅ No deleted files found - database is in sync{Style.RESET_ALL}")
        
        # 삭제 표시된 파일들 일괄 삭제 (수정된 파일들의 이전 버전)
        self.milvus_manager.execute_pending_deletions()
        
        # 전체 파일 수와 크기 업데이트 - 이 값들이 전체 진행도의 분모가 됨
        self.embedding_progress["total_files"] = new_or_modified_count
        self.embedding_progress["total_size"] = new_or_modified_size
        
        # 파일 처리 완료 후 디버깅 메시지 출력
        print(f"\n{Fore.CYAN}[SUMMARY] File processing summary:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[DEBUG] Total files scanned: {total_files_count}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[DEBUG] New or modified files: {new_or_modified_count}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[DEBUG] Skipped files (unchanged): {skipped_count}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[DEBUG] Deleted files cleaned up: {len(deleted_files)}{Style.RESET_ALL}")
        
        # 전체 파일 수와 크기 정보 출력
        print(f"Total files to process: {new_or_modified_count} files ({new_or_modified_size/(1024*1024):.2f} MB)")
        
        # 처리할 파일이 없는 경우
        if not files_to_process:
            print(f"{Fore.GREEN}[INFO] No new or modified files found. Nothing to process.{Style.RESET_ALL}")
            if len(deleted_files) > 0:
                print(f"{Fore.GREEN}[INFO] However, {len(deleted_files)} deleted files were cleaned up.{Style.RESET_ALL}")
            self.embedding_in_progress = False
            return len(deleted_files)  # Return count of cleaned up files
        
        # 리소스 모니터링 및 진행률 업데이트 스레드 시작
        self.start_monitoring()
        
        try:
            # 최적의 배치 크기 계산
            current_batch_size = self._check_system_resources()
            
            # 파일 배치 처리
            processed_count = self._process_file_batch(files_to_process, current_batch_size)
            
            # 임베딩 완료 표시
            self.embedding_in_progress = False
            print(f"\n{Fore.GREEN}[SUCCESS] Incremental embedding & deleted cleanup completed successfully!{Style.RESET_ALL}")
            print(f"{Fore.GREEN}✅ Processed {processed_count} new/modified files{Style.RESET_ALL}")
            print(f"{Fore.GREEN}✅ Cleaned up {len(deleted_files)} deleted files{Style.RESET_ALL}")
            
            # 메모리 정리
            gc.collect()
            if 'torch' in sys.modules:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return processed_count + len(deleted_files)
            
        except Exception as e:
            error_msg = f"Error in process_updated_files: {e}"
            print(f"\n{Fore.RED}{error_msg}{Style.RESET_ALL}")
            import traceback
            print(f"\n{Fore.RED}Stack trace:\n{traceback.format_exc()}{Style.RESET_ALL}")
            if hasattr(self, 'monitor') and hasattr(self.monitor, 'add_error_log'):
                self.monitor.add_error_log(error_msg)
            
            # 오류 발생 시 임베딩 중지
            self.embedding_in_progress = False
            
            # 메모리 정리
            gc.collect()
            if 'torch' in sys.modules:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return 0
        finally:
            # 항상 모니터링 중지
            self.stop_monitoring()
            print(f"\n{Fore.CYAN}[DEBUG] Exiting process_updated_files{Style.RESET_ALL}")
    
    def process_all_files(self):
        """볼트의 모든 파일 처리 (메모리 사용량 최적화 및 성능 개선)
        전체 임베딩 - 모든 파일을 다시 처리하는 방식"""
        print(f"\n{Fore.CYAN}[DEBUG] Starting process_all_files (FULL REINDEXING){Style.RESET_ALL}")
        print(f"Processing all files in {self.vault_path} - All files will be reprocessed")
        
        # 경로 존재 확인
        if not os.path.exists(self.vault_path):
            error_msg = f"Error: Obsidian vault path not found: {self.vault_path}"
            print(f"\n{Fore.RED}{error_msg}{Style.RESET_ALL}")
            if hasattr(self, 'monitor') and hasattr(self.monitor, 'add_error_log'):
                self.monitor.add_error_log(error_msg)
            return 0
        
        print(f"\n{Fore.CYAN}[DEBUG] Vault path exists: {self.vault_path}{Style.RESET_ALL}")
        
        # 파일 시스템 접근 확인
        try:
            test_file = os.path.join(self.vault_path, "test_access.txt")
            with open(test_file, 'w') as f:
                f.write("test access")
            os.remove(test_file)
            print(f"{Fore.CYAN}[DEBUG] File system access test passed{Style.RESET_ALL}")
        except Exception as e:
            error_msg = f"Error: Cannot write to vault directory: {e}"
            print(f"\n{Fore.RED}{error_msg}{Style.RESET_ALL}")
            import traceback
            print(f"\n{Fore.RED}Stack trace:\n{traceback.format_exc()}{Style.RESET_ALL}")
            if hasattr(self, 'monitor') and hasattr(self.monitor, 'add_error_log'):
                self.monitor.add_error_log(error_msg)
            self.embedding_in_progress = False
            return 0
        
        # 임베딩 진행 상태 초기화
        self.embedding_in_progress = True
        self.embedding_progress = {
            "total_files": 0,
            "processed_files": 0,
            "total_size": 0,
            "processed_size": 0,
            "start_time": time.time(),
            "current_file": "",
            "estimated_time_remaining": "Calculating...",
            "percentage": 0,
            "is_full_reindex": True,
            "cpu_percent": 0,
            "memory_percent": 0,
            "current_batch_size": self.dynamic_batch_size
        }
        
        # 진행도 계산을 위한 변수 초기화
        self.embedding_progress["processed_size"] = 0
        self.embedding_progress["processed_files"] = 0
        
        # 전체 파일 수와 크기 미리 계산 전 Milvus 연결 테스트
        try:
            print(f"\n{Fore.CYAN}[DEBUG] Checking Milvus connection...{Style.RESET_ALL}")
            test_query = self.milvus_manager.query("id >= 0", limit=1)
            print(f"\n{Fore.CYAN}[DEBUG] Milvus connection successful. Query result: {test_query}{Style.RESET_ALL}")
        except Exception as e:
            error_msg = f"Error connecting to Milvus: {e}"
            print(f"\n{Fore.RED}{error_msg}{Style.RESET_ALL}")
            import traceback
            print(f"\n{Fore.RED}Stack trace:\n{traceback.format_exc()}{Style.RESET_ALL}")
            if hasattr(self, 'monitor') and hasattr(self.monitor, 'add_error_log'):
                self.monitor.add_error_log(error_msg)
            self.embedding_in_progress = False  # 반드시 상태를 False로 설정
            return 0
        
        # 임베딩 모델 테스트
        try:
            print(f"\n{Fore.CYAN}[DEBUG] Testing embedding model...{Style.RESET_ALL}")
            test_vector = self.embedding_model.get_embedding("Test embedding model")
            print(f"\n{Fore.CYAN}[DEBUG] Embedding model OK, vector dimension: {len(test_vector)}{Style.RESET_ALL}")
        except Exception as e:
            error_msg = f"Error with embedding model: {e}"
            print(f"\n{Fore.RED}{error_msg}{Style.RESET_ALL}")
            import traceback
            print(f"\n{Fore.RED}Stack trace:\n{traceback.format_exc()}{Style.RESET_ALL}")
            if hasattr(self, 'monitor') and hasattr(self.monitor, 'add_error_log'):
                self.monitor.add_error_log(error_msg)
            self.embedding_in_progress = False  # 반드시 상태를 False로 설정
            return 0
        
        # 전체 파일 수와 크기 미리 계산
        print("Calculating total files and size...")
        total_files_count = 0
        total_files_size = 0
        
        # 전체 파일 수와 크기 계산
        for root, _, files in os.walk(self.vault_path):
            # 숨겨진 폴더 건너뛰기
            if os.path.basename(root).startswith(('.', '_')):
                continue
                
            for file in files:
                # 마크다운과 PDF만 처리
                if file.endswith(('.md', '.pdf')) and not file.startswith('.'):
                    total_files_count += 1
                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        total_files_size += file_size
                    except Exception as e:
                        print(f"Error getting file size for {file_path}: {e}")
        
        # 전체 파일 수와 크기 업데이트 - 이 값들이 전체 진행도의 분모가 됨
        self.embedding_progress["total_files"] = total_files_count
        self.embedding_progress["total_size"] = total_files_size
        print(f"Found {total_files_count} files with total size of {total_files_size/(1024*1024):.2f} MB")
        
        # 리소스 모니터링 및 진행률 업데이트 스레드 시작
        self.start_monitoring()
        
        try:
            # 처리할 파일 목록 수집
            print(f"\n{Fore.CYAN}[DEBUG] Collecting files to process...{Style.RESET_ALL}")
            
            # 기존 파일 정보 가져오기
            existing_files_info = {}
        except Exception as e:
            error_msg = f"Error initializing file processing: {e}"
            print(f"\n{Fore.RED}{error_msg}{Style.RESET_ALL}")
            import traceback
            print(f"\n{Fore.RED}Stack trace:\n{traceback.format_exc()}{Style.RESET_ALL}")
            return 0
            
        try:
            try:
                print(f"{Fore.CYAN}[DEBUG] Querying existing files from Milvus...{Style.RESET_ALL}")
                max_limit = 16000
                offset = 0
                
                while True:
                    results = self.milvus_manager.query(
                        output_fields=["path", "updated_at"],
                        limit=max_limit,
                        offset=offset,
                        expr="id >= 0"
                    )
                    
                    if not results:
                        break
                        
                    for doc in results:
                        path = doc.get("path")
                        if path and path not in existing_files_info:
                            existing_files_info[path] = doc.get('updated_at')
                    
                    offset += max_limit
                    if len(results) < max_limit:
                        break
                    
                    # 메모리 관리
                    gc.collect()
            except Exception as e:
                print(f"Warning: Error fetching existing files: {e}")
            
            # 파일 목록 수집
            print("Scanning files...")
            files_to_process = []
            skipped_count = 0
            
            # 이미 미리 계산한 전체 파일 수와 크기를 사용
            # 중복 계산을 피하기 위해 여기서는 추가로 계산하지 않음
            total_files = self.embedding_progress["total_files"]
            total_size = self.embedding_progress["total_size"]
            
            print(f"{Fore.CYAN}[DEBUG] Walking through directory: {self.vault_path}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}[DEBUG] Total files to process: {total_files} files ({total_size/(1024*1024):.2f} MB){Style.RESET_ALL}")
            file_count = 0
            
            for root, _, files in os.walk(self.vault_path):
                # 디렉토리별 로그 추가
                print(f"{Fore.CYAN}[DEBUG] Scanning directory: {root}{Style.RESET_ALL}")
                
                # 숨겨진 폴더 건너뛰기
                if os.path.basename(root).startswith(('.', '_')):
                    print(f"{Fore.CYAN}[DEBUG] Skipping hidden directory: {root}{Style.RESET_ALL}")
                    continue
                    
                for file in files:
                    # 마크다운과 PDF만 처리
                    if file.endswith(('.md', '.pdf')) and not file.startswith('.'):
                        file_count += 1
                        # 파일 경로
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, self.vault_path)
                        print(f"{Fore.CYAN}[DEBUG] Found file {file_count}: {rel_path}{Style.RESET_ALL}")
                        
                        # 문제가 있는 특정 파일 처리
                        if "(shorter version(2)).md" in full_path:
                            print(f"Found problematic file: {full_path}")
                            # 이 파일 특별 처리
                            try:
                                # 파일 내용 확인
                                with open(full_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                
                                # 파일 내용이 비정상적으로 긴 경우 또는 특수 문자가 많은 경우
                                if content.count('$~$') > 10:
                                    print("File contains many math placeholders, cleaning...")
                                    # 파일 정리 및 저장
                                    content = re.sub(r'\$~\$\n+', '\n', content)
                                    content = re.sub(r'\n{3,}', '\n\n', content)
                                    content = content.rstrip()
                                    
                                    # 정리된 파일 저장
                                    with open(full_path + ".cleaned", 'w', encoding='utf-8') as f:
                                        f.write(content)
                                    print(f"Cleaned file saved to {full_path}.cleaned")
                            except Exception as e:
                                print(f"Error analyzing problematic file: {e}")
                        
                        # 전체 임베딩에서는 모든 파일을 처리 (수정 시간 비교 없음)
                        try:
                            # 임베딩 진행 정보에 전체 재처리 모드 표시 (이미 main.py에서 설정됨)
                            
                            # PDF 파일 처리 여부 확인 (필요에 따라 건너뛰 수 있음)
                            if file.lower().endswith('.pdf') and getattr(config, 'SKIP_PDF_IN_FULL_EMBEDDING', False):
                                print(f"Skipping PDF in full embedding: {rel_path}")
                                skipped_count += 1
                                continue
                                
                            # 기존 파일이 있는 경우 삭제 표시
                            if rel_path in existing_files_info:
                                print(f"Reprocessing existing file: {rel_path}")
                                self.milvus_manager.mark_for_deletion(rel_path)
                        except Exception as e:
                            print(f"Warning: Error checking file {rel_path}: {e}")
                        
                        # 파일 크기 가져오기
                        file_size = os.path.getsize(full_path)
                        
                        # 처리할 파일 목록에 추가
                        files_to_process.append((full_path, file_size))
                        
                        # 파일 목록이 너무 커지면 중간에 처리
                        if len(files_to_process) >= 10000:
                            print(f"Reached 10,000 files, processing batch...")
                            current_batch_size = self._check_system_resources()
                            self._process_file_batch(files_to_process, current_batch_size)
                            files_to_process = []
                            
                            # 메모리 정리
                            gc.collect()
                            self.embedding_model.clear_cache()
            
            # 삭제 표시된 파일들만 일괄 삭제
            # 전체 재색인 모드에서는 이미 main.py에서 recreate_choice에 따라 컴렉션을 재생성했으므로 여기서는 추가 삭제 작업을 하지 않음
            self.milvus_manager.execute_pending_deletions()
            
            # 파일 처리 완료 후 디버깅 메시지 출력
            print(f"{Fore.CYAN}[DEBUG] Files found in this scan: {file_count}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}[DEBUG] Files to process: {len(files_to_process)}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}[DEBUG] Skipped files: {skipped_count}{Style.RESET_ALL}")
            
            # 전체 파일 수와 크기 정보 출력
            print(f"Total files to process: {total_files} files ({total_size/(1024*1024):.2f} MB)")
            
            # 처리할 파일이 없는 경우 테스트 파일 생성 (테스트 모드)
            if not files_to_process:
                print(f"{Fore.YELLOW}[WARNING] No files found to process. Creating test file for demonstration.{Style.RESET_ALL}")
                # 테스트 파일 생성
                test_dir = os.path.join(self.vault_path, "test")
                os.makedirs(test_dir, exist_ok=True)
                test_file = os.path.join(test_dir, "test_file.md")
                
                # 테스트 파일 작성
                with open(test_file, "w", encoding="utf-8") as f:
                    f.write("# Test File\n\nThis is a test file for embedding process demonstration.\n\n## Section 1\n\nSome content here.\n\n## Section 2\n\nMore content here.")
                
                # 파일 크기 가져오기
                file_size = os.path.getsize(test_file)
                
                # 처리할 파일 목록에 추가
                files_to_process.append((test_file, file_size))
                
                # 전체 파일 수와 크기 업데이트
                self.embedding_progress["total_files"] = 1
                self.embedding_progress["total_size"] = file_size
                
                print(f"Created test file: {test_file} ({file_size} bytes)")
            
            # 파일 배치 목록 출력 (처음 5개만)
            if files_to_process:
                first_five = files_to_process[:5]
                print(f"{Fore.CYAN}[DEBUG] First 5 files in batch: {[os.path.basename(fp) for fp, _ in first_five]}{Style.RESET_ALL}")
            
            # 남은 파일 처리
            if files_to_process:
                print(f"{Fore.CYAN}[DEBUG] Processing files...{Style.RESET_ALL}")
                current_batch_size = self._check_system_resources()
                processed_count = self._process_file_batch(files_to_process, current_batch_size)
                print(f"{Fore.CYAN}[DEBUG] Batch processing completed. Processed {processed_count} files.{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}[WARNING] No files to process after filtering!{Style.RESET_ALL}")
                # 파일이 없어도 임베딩 진행 상태를 유지하여 진행바가 표시되도록 함
                time.sleep(5)  # 5초 대기
            
            # 최종 메모리 정리
            gc.collect()
            self.embedding_model.clear_cache()
            
            print(f"Successfully processed files, skipped {skipped_count} unchanged files")
            
            # 명시적으로 성공 코드 반환
            return 1
            
        except Exception as e:
            error_msg = f"Error in process_all_files: {e}"
            print(f"\n{Fore.RED}{error_msg}{Style.RESET_ALL}")
            import traceback
            print(f"\n{Fore.RED}Stack trace:\n{traceback.format_exc()}{Style.RESET_ALL}")
            if hasattr(self, 'monitor') and hasattr(self.monitor, 'add_error_log'):
                self.monitor.add_error_log(error_msg)
            return 0
            
        finally:
            # 리소스 모니터링 중지
            self.stop_monitoring()
            
            # 임베딩 진행 상태 완료
            self.embedding_in_progress = False
            self.embedding_progress["percentage"] = 100
            print(f"\n{Fore.CYAN}[DEBUG] Exiting process_all_files with status: {0 if 'error_msg' in locals() else 1}{Style.RESET_ALL}")
    
    def _process_file_batch(self, files_to_process, batch_size):
        """파일 배치 처리 로직"""
        if not files_to_process:
            print(f"{Fore.YELLOW}[DEBUG] No files to process in batch.{Style.RESET_ALL}")
            return 0
            
        processed_count = 0
        total_files = len(files_to_process)
        total_size = sum(file_size for _, file_size in files_to_process)
        
        # 배치 처리 정보 업데이트
        print(f"{Fore.CYAN}[DEBUG] Processing batch: {total_files} files in this batch ({total_size/(1024*1024):.2f} MB){Style.RESET_ALL}")
        
        # 임베딩 진행 상태 확인
        self.embedding_in_progress = True
        
        # 파일 목록 출력 (처음 5개만)
        print(f"{Fore.CYAN}[DEBUG] First 5 files in batch: {[os.path.basename(fp) for fp, _ in files_to_process[:5]]}{Style.RESET_ALL}")
        
        # 배치 처리
        total_size_mb = total_size / (1024 * 1024)
        with tqdm(total=total_size_mb, desc="Indexing files", unit="MB", ncols=100) as pbar:
            # 메모리 효율을 위한 배치 처리
            for i in range(0, total_files, batch_size):
                batch = files_to_process[i:i+batch_size]
                
                # 메모리 사용량 확인
                self._check_memory_usage(f"Before processing batch {i//batch_size+1}/{(total_files+batch_size-1)//batch_size}")
                
                # 배치 처리
                for file_item in batch:
                    try:
                        # 파일 경로와 크기 분리
                        file_path, file_size = file_item
                        
                        # 현재 처리중인 파일 표시
                        rel_path = os.path.relpath(file_path, self.vault_path)
                        self.embedding_progress["current_file"] = rel_path
                        self.embedding_progress["processed_files"] += 1
                        
                        # 파일 크기를 추가하고 processed_size는 그대로 유지
                        # 중요: process_file가 이미 processed_size를 업데이트하므로 여기서는 추가하지 않음
                        # 대신 이미 추가되어 있는지 표시하는 플래그 설정
                        self._processed_this_file = True
                        
                        # 단일 파일 처리
                        success = self.process_file(file_path)
                        if success:
                            processed_count += 1
                            
                        # 진행률 업데이트
                        file_size_mb = file_size / (1024 * 1024)
                        pbar.update(file_size_mb)
                        
                    except Exception as e:
                        print(f"Error processing file in batch: {e}")
                
                # 메모리 정리
                gc.collect()
                self.embedding_model.clear_cache()
        
        return processed_count
    
    def detect_deleted_files(self):
        """삭제된 파일 탐지 (메모리 효율적)"""
        from colorama import Fore, Style
        
        print(f"{Fore.CYAN}Scanning Milvus database for file paths...{Style.RESET_ALL}")
        
        # 1. Milvus에서 모든 파일 경로 조회 (페이지네이션)
        db_files = set()
        offset = 0
        max_limit = 16000
        total_db_files = 0
        
        try:
            while True:
                results = self.milvus_manager.query(
                    expr="id >= 0",
                    output_fields=["path"],
                    limit=max_limit,
                    offset=offset
                )
                
                if not results:
                    break
                    
                for doc in results:
                    path = doc.get("path")
                    if path and path not in db_files:
                        db_files.add(path)
                        total_db_files += 1
                
                offset += max_limit
                if len(results) < max_limit:
                    break
                    
                # 진행상황 표시
                if total_db_files % 1000 == 0 and total_db_files > 0:
                    print(f"{Fore.CYAN}Found {total_db_files} files in database so far...{Style.RESET_ALL}")
                
                # 메모리 관리
                gc.collect()
                
        except Exception as e:
            print(f"{Fore.RED}Error querying Milvus database: {e}{Style.RESET_ALL}")
            return []
        
        print(f"{Fore.GREEN}Found {len(db_files)} unique files in Milvus database{Style.RESET_ALL}")
        
        # 2. 현재 파일 시스템 스캔
        print(f"{Fore.CYAN}Scanning file system...{Style.RESET_ALL}")
        fs_files = set()
        total_fs_files = 0
        
        try:
            for root, _, files in os.walk(self.vault_path):
                # 숨겨진 폴더 건너뛰기
                if os.path.basename(root).startswith(('.', '_')):
                    continue
                    
                for file in files:
                    # 마크다운과 PDF만 처리
                    if file.endswith(('.md', '.pdf')) and not file.startswith('.'):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, self.vault_path)
                        fs_files.add(rel_path)
                        total_fs_files += 1
                        
                        # 진행상황 표시
                        if total_fs_files % 1000 == 0:
                            print(f"{Fore.CYAN}Scanned {total_fs_files} files in file system...{Style.RESET_ALL}")
        
        except Exception as e:
            print(f"{Fore.RED}Error scanning file system: {e}{Style.RESET_ALL}")
            return []
        
        print(f"{Fore.GREEN}Found {len(fs_files)} files in file system{Style.RESET_ALL}")
        
        # 3. 삭제된 파일 찾기
        deleted_files = db_files - fs_files
        
        if deleted_files:
            print(f"{Fore.YELLOW}Found {len(deleted_files)} deleted files{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}No deleted files found{Style.RESET_ALL}")
        
        return list(deleted_files)
    
    def cleanup_deleted_embeddings(self, deleted_files):
        """삭제된 파일들의 embedding 제거"""
        from colorama import Fore, Style
        
        if not deleted_files:
            print(f"{Fore.GREEN}No files to clean up{Style.RESET_ALL}")
            return 0
        
        print(f"{Fore.CYAN}Starting cleanup of {len(deleted_files)} deleted files...{Style.RESET_ALL}")
        
        success_count = 0
        error_count = 0
        
        try:
            # 배치 삭제를 위해 pending_deletions에 추가
            for file_path in deleted_files:
                self.milvus_manager.mark_for_deletion(file_path)
            
            print(f"{Fore.CYAN}Executing batch deletion...{Style.RESET_ALL}")
            
            # 배치 삭제 실행
            self.milvus_manager.execute_pending_deletions()
            
            # 삭제 결과 확인
            print(f"{Fore.CYAN}Verifying deletion results...{Style.RESET_ALL}")
            
            # 삭제 후 검증
            remaining_files = []
            for file_path in deleted_files:
                try:
                    # 파일이 여전히 DB에 있는지 확인
                    results = self.milvus_manager.query(
                        expr=f"path == '{file_path}'",
                        output_fields=["path"],
                        limit=1
                    )
                    
                    if results:
                        remaining_files.append(file_path)
                        error_count += 1
                    else:
                        success_count += 1
                        
                except Exception as e:
                    print(f"{Fore.YELLOW}Warning: Could not verify deletion of {file_path}: {e}{Style.RESET_ALL}")
                    error_count += 1
            
            # 결과 보고
            print(f"\n{Fore.GREEN}Cleanup Results:{Style.RESET_ALL}")
            print(f"{Fore.GREEN}✅ Successfully removed: {success_count} files{Style.RESET_ALL}")
            
            if error_count > 0:
                print(f"{Fore.YELLOW}⚠️ Failed to remove: {error_count} files{Style.RESET_ALL}")
                if remaining_files:
                    print(f"{Fore.YELLOW}Files that could not be deleted:{Style.RESET_ALL}")
                    for file_path in remaining_files[:5]:  # 최대 5개만 표시
                        print(f"{Fore.YELLOW}  - {file_path}{Style.RESET_ALL}")
                    if len(remaining_files) > 5:
                        print(f"{Fore.YELLOW}  ... and {len(remaining_files) - 5} more{Style.RESET_ALL}")
            
            # 메모리 정리
            gc.collect()
            
            return success_count
            
        except Exception as e:
            print(f"{Fore.RED}Error during cleanup: {e}{Style.RESET_ALL}")
            import traceback
            print(f"{Fore.RED}Stack trace: {traceback.format_exc()}{Style.RESET_ALL}")
            return 0