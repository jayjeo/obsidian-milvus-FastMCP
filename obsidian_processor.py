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

# Import centralized logger
from logger import get_logger

# Initialize module logger
logger = get_logger(__name__)

# Windows에서 색상 표시를 위한 colorama 초기화
colorama.init()

class ObsidianProcessor:
    def __init__(self, milvus_manager):
        logger.info("Initializing ObsidianProcessor")
        
        self.vault_path = config.OBSIDIAN_VAULT_PATH
        logger.info(f"Using Obsidian vault path: {self.vault_path}")
        
        self.milvus_manager = milvus_manager
        self.embedding_model = EmbeddingModel()
        self.next_id = self._get_next_id()
        logger.debug(f"Next ID initialized to: {self.next_id}")
        
        # GPU 사용 설정
        self.use_gpu = config.USE_GPU
        self.device_idx = config.GPU_DEVICE_ID if hasattr(config, 'GPU_DEVICE_ID') else 0
        logger.info(f"GPU settings - Use GPU: {self.use_gpu}, Device index: {self.device_idx}")
        
        # 처리 타임아웃 설정 (초 단위)
        self.processing_timeout = 300  # 기본값: 5분
        logger.debug(f"Processing timeout set to: {self.processing_timeout} seconds")
        
        # 임베딩 진행 상태 추적을 위한 변수
        self.embedding_in_progress = False
        logger.info("ObsidianProcessor initialization complete")
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
        
        # ENHANCED: 시스템 리소스 사용량 제한 및 동적 최적화 설정
        self.max_cpu_percent = 85
        self.max_memory_percent = 80
        self.resource_check_interval = 2
        self.last_resource_check = 0
        
        # 동적 배치 크기: embedding_model에서 최적값 가져오기
        try:
            if hasattr(self.embedding_model, 'batch_optimizer'):
                self.dynamic_batch_size = self.embedding_model.batch_optimizer.current_batch_size
                self.min_batch_size = self.embedding_model.batch_optimizer.min_batch_size
                self.max_batch_size = self.embedding_model.batch_optimizer.max_batch_size
                print(f"🚀 Using optimized batch sizes from embedding model: {self.dynamic_batch_size} (range: {self.min_batch_size}-{self.max_batch_size})")
            else:
                # 폴백: 기본 설정
                self.dynamic_batch_size = getattr(config, 'BATCH_SIZE', 32)
                self.min_batch_size = max(1, self.dynamic_batch_size // 2)
                self.max_batch_size = self.dynamic_batch_size * 4
                print(f"⚠️ Using fallback batch sizes: {self.dynamic_batch_size} (range: {self.min_batch_size}-{self.max_batch_size})")
        except Exception as e:
            print(f"Error initializing dynamic batch sizes: {e}")
            # 안전 폴백
            self.dynamic_batch_size = 32
            self.min_batch_size = 8
            self.max_batch_size = 128
        
        # 진행률 및 리소스 모니터링 관리자 생성
        self.monitor = ProgressMonitor(self)
        
        # OPTIMIZATION: Session cache for verification results
        self.verification_cache = {}
        
        # OPTIMIZATION: Performance thresholds for smart decision making
        self.FAST_SKIP_THRESHOLD = 0.1  # Files with time diff < 0.1s are very likely unchanged
        self.FAST_PROCESS_THRESHOLD = 2.0  # Files with time diff > 2.0s are definitely changed
        
    def _get_next_id(self):
        """다음 ID 값 가져오기 (강화된 오류 처리)"""
        try:
            # 쿼리 결과 가져오기
            results = self.milvus_manager.query("id >= 0", output_fields=["id"], limit=1)
            
            # 결과가 없거나 비어 있으면 1로 시작
            if not results or len(results) == 0:
                logger.debug("No existing IDs found, starting with ID 1")
                return 1
            
            # 결과에서 ID 추출 (안전하게)
            valid_ids = []
            for r in results:
                try:
                    # ID가 실제 정수인지 확인
                    if 'id' in r and r['id'] is not None and isinstance(r['id'], (int, float)):
                        valid_ids.append(int(r['id']))
                    else:
                        logger.warning(f"Skipping invalid ID format: {r}")
                except Exception as id_err:
                    logger.warning(f"Error processing ID entry: {r}, error: {id_err}")
            
            # 유효한 ID가 있으면 최대값 + 1 반환
            if valid_ids:
                next_id = max(valid_ids) + 1
                logger.debug(f"Found valid IDs, next ID will be: {next_id}")
                return next_id
            else:
                logger.warning("No valid IDs found, starting with ID 1")
                return 1
                
        except Exception as e:
            # 모든 예외 처리하고 안전하게 1 반환
            logger.error(f"Error getting next ID: {e}, using default ID 1")
            return 1
        
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
        """시스템 리소스 사용량 확인 및 동적 배치 크기 조절 (ENHANCED)"""
        current_time = time.time()
        
        if current_time - self.last_resource_check < self.resource_check_interval:
            return self.dynamic_batch_size
            
        self.last_resource_check = current_time
        
        try:
            # ENHANCED: embedding_model의 동적 배치 최적화 사용
            if hasattr(self.embedding_model, 'system_monitor'):
                system_status = self.embedding_model.system_monitor.get_system_status()
                memory_percent = system_status.get('memory_percent', 50)
                cpu_percent = system_status.get('cpu_percent', 50)
                gpu_percent = system_status.get('gpu_percent', 0)
                
                # embedding_model의 batch_optimizer로 최적 배치 크기 결정
                if hasattr(self.embedding_model, 'batch_optimizer'):
                    optimal_batch = self.embedding_model.batch_optimizer.adjust_batch_size({
                        'memory_percent': memory_percent,
                        'cpu_percent': cpu_percent,
                        'gpu_percent': gpu_percent,
                        'processing_time': 1.0
                    })
                    
                    # 동적 배치 크기 업데이트
                    self.dynamic_batch_size = optimal_batch
                    
                    # 진행률 정보 업데이트
                    self.embedding_progress["current_batch_size"] = self.dynamic_batch_size
                    
                    print(f"📈 Dynamic batch size adjusted to: {self.dynamic_batch_size} (Memory: {memory_percent:.1f}%, GPU: {gpu_percent:.1f}%)")
            
            # ProgressMonitor 업데이트
            if hasattr(self.monitor, '_update_system_resources'):
                self.monitor._update_system_resources()
                
        except Exception as e:
            print(f"Error in enhanced system resource check: {e}")
        
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
                    
                    # ENHANCED: 청크에 대한 배치 임베딩 생성 (속도 대폭 개선!)
                    logger.info(f"Processing {len(chunks)} chunks with batch embedding for file: {file_name}")
                    print(f"🚀 Processing {len(chunks)} chunks with FORCED batch embedding...")
                    
                    # Check for special characters in file path that might need careful handling
                    has_special_chars = any(c in file_path for c in "'\"()[]{},;")
                    if has_special_chars:
                        logger.debug(f"File path contains special characters: {file_path}")
                    
                    # STEP 1: 배치 크기 확인 및 최적화
                    optimal_batch_size = self._check_system_resources()
                    if hasattr(self.embedding_model, 'batch_optimizer'):
                        current_batch_size = self.embedding_model.batch_optimizer.current_batch_size
                        logger.debug(f"Current optimal batch size: {current_batch_size} for {len(chunks)} chunks")
                        print(f"📦 Current optimal batch size: {current_batch_size}")
                    
                    vectors = []
                    batch_success = False
                    
                    try:
                        # STEP 2: 강제 배치 처리 (폴백 없이)
                        logger.debug(f"Starting batch processing for {len(chunks)} chunks")
                        print(f"🔥 FORCING batch processing for {len(chunks)} chunks...")
                        start_time = time.time()
                        
                        # 배치 처리 강제 실행
                        vectors = self.embedding_model.get_embeddings_batch_adaptive(chunks)
                        
                        batch_time = time.time() - start_time
                        logger.debug(f"Batch processing completed in {batch_time:.2f} seconds")
                        
                        # 결과 검증
                        if vectors and len(vectors) == len(chunks):
                            batch_success = True
                            logger.info(f"Batch processing succeeded: {len(chunks)} chunks in {batch_time:.2f}s ({len(chunks)/batch_time:.1f} chunks/sec)")
                            print(f"✅ BATCH SUCCESS: {len(chunks)} chunks in {batch_time:.2f}s ({len(chunks)/batch_time:.1f} chunks/sec)")
                            print(f"🎥 GPU utilization should be HIGH during this process")
                        else:
                            logger.warning(f"Batch processing failed: Expected {len(chunks)} vectors, got {len(vectors) if vectors else 0}")
                            print(f"❌ BATCH FAILED: Expected {len(chunks)} vectors, got {len(vectors) if vectors else 0}")
                            
                    except Exception as e:
                        # Check if timeout-related error (handling the processing_timeout attribute)
                        if "timeout" in str(e).lower():
                            logger.error(f"Batch processing timed out after {self.processing_timeout} seconds: {e}", exc_info=True)
                        else:
                            logger.error(f"Batch processing error: {e}", exc_info=True)
                            
                        print(f"❌ BATCH PROCESSING ERROR: {e}")
                        import traceback
                        print(f"📍 Error details: {traceback.format_exc()}")
                    
                    # STEP 3: 배치가 실패한 경우에만 개별 처리
                    if not batch_success:
                        logger.warning(f"Falling back to individual processing for {len(chunks)} chunks")
                        print(f"⚠️ Falling back to individual processing (this should be rare)...")
                        vectors = []
                        individual_start = time.time()
                        
                        successful_chunks = 0
                        failed_chunks = 0
                        
                        for i, chunk in enumerate(chunks):
                            try:
                                vector = self.embedding_model.get_embedding(chunk)
                                vectors.append(vector)
                                successful_chunks += 1
                            except Exception as e:
                                logger.error(f"Error embedding chunk {i} in individual processing: {e}")
                                print(f"Error embedding chunk {i}: {e}")
                                # Use zero vector as fallback
                                vectors.append([0] * config.VECTOR_DIM)
                                failed_chunks += 1
                        
                        individual_time = time.time() - individual_start
                        logger.info(f"Individual processing completed: {successful_chunks} succeeded, {failed_chunks} failed, took {individual_time:.2f}s")
                        print(f"🐌 Individual processing completed in {individual_time:.2f}s ({len(chunks)/individual_time:.1f} chunks/sec)")
                    
                    # STEP 4: 성능 통계 출력
                    if batch_success:
                        logger.info(f"Performance: Batch processing achieved {len(chunks)/batch_time:.1f} chunks/second")
                        print(f"🏆 PERFORMANCE: Batch processing achieved {len(chunks)/batch_time:.1f} chunks/second")
                        print(f"💪 Expected GPU usage: HIGH during batch processing")
                    else:
                        logger.warning("Batch processing failed - review logs for details")
                        print(f"🔨 WARNING: Batch processing failed - investigating...")
                    
                    # Check for special characters in file path before Milvus operations
                    if has_special_chars:
                        logger.info(f"Preparing to insert file with special characters into Milvus: {file_path}")
                    
                    # 메타데이터 매핑 준비
                    chunk_file_map = [metadata] * len(chunks)
                    logger.debug(f"Prepared {len(chunks)} chunk-file mappings with metadata")
                    
                    # 메모리 사용량 확인
                    self._check_memory_usage("Before saving to Milvus")
                    
                    # 벡터 저장
                    logger.info(f"Saving {len(vectors)} vectors to Milvus for file: {file_name}")
                    try:
                        success = self._save_vectors_to_milvus(vectors, chunks, chunk_file_map)
                        if success:
                            logger.info(f"Successfully saved vectors to Milvus for file: {file_name}")
                        else:
                            logger.error(f"Failed to save vectors to Milvus for file: {file_name}")
                            # 추가 진단 정보 로깅
                            logger.error(f"  File details - Size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'} bytes")
                            logger.error(f"  Path characteristics - Has special chars: {any(c in file_path for c in '[](){}#$%^&*')}")
                            logger.error(f"  Path starts with number: {bool(re.match(r'^\d', os.path.basename(file_path)))}")
                            logger.error(f"  Current memory usage: {psutil.Process().memory_info().rss / (1024 * 1024):.1f} MB")
                    except Exception as e:
                        error_type = type(e).__name__
                        error_msg = str(e)
                        logger.error(f"Error saving vectors to Milvus for file: {file_name}")
                        logger.error(f"  Error type: {error_type}, Message: {error_msg}")
                        
                        # 상세한 오류 정보 추가
                        logger.error(f"  File details - Path: {file_path}")
                        logger.error(f"  Chunks: {len(chunks) if chunks else 0}, Vectors: {len(vectors) if vectors else 0}")
                        
                        # 문제가 발생할 수 있는 특별한 조건 검사
                        if "DataNotMatchException" in error_type or "schema" in error_msg.lower():
                            logger.error("  Possible schema mismatch issue - Check collection fields")
                            
                            # 상세 진단 - 숫자로 시작하는 파일명 문제 확인
                            base_name = os.path.basename(file_path)
                            if re.match(r'^\d', base_name):
                                logger.error(f"  CRITICAL: File '{base_name}' starts with a number - this is likely causing the schema issue")
                                logger.error("  SOLUTION: Add 'file_' prefix to filenames and 'Title_' prefix to titles starting with numbers")
                                
                            # 'id' 필드 관련 문제 확인
                            if "id" in error_msg.lower():
                                logger.error("  'id' field issue detected in error message - verify schema compatibility")
                                # 첫번째 청크 데이터 구조 확인
                                if chunk_file_map and len(chunk_file_map) > 0:
                                    first_chunk = chunk_file_map[0] if chunk_file_map else None
                                    if first_chunk:
                                        logger.error(f"  First chunk metadata keys: {list(first_chunk.keys() if first_chunk else [])}")
                        elif "timeout" in error_msg.lower():
                            logger.error("  Possible timeout issue - Check network or increase processing_timeout")
                        elif "memory" in error_msg.lower():
                            logger.error("  Possible memory issue - Check available system resources")
                            
                        logger.error(f"  Stack trace:", exc_info=True)
                        success = False
                    
                    # 메모리 효율성을 위한 명시적 변수 해제
                    logger.debug("Explicitly releasing memory for large variables")
                    del chunks
                    del vectors
                    del chunk_file_map
                    del metadata
                    
                    # 메모리 정리
                    logger.debug("Running garbage collection and clearing GPU cache")
                    gc.collect()
                    if 'torch' in sys.modules:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # 처리 결과 저장 및 상태 표시 업데이트
                    processing_result["success"] = success
                    
                    # 성공/실패 상태 업데이트
                    if success:
                        logger.info(f"Successfully processed file: {file_path}")
                        self.monitor.last_processed_status = f"{Fore.GREEN}Success{Fore.RESET}"
                    else:
                        logger.warning(f"Failed to process file: {file_path}")
                        self.monitor.last_processed_status = f"{Fore.RED}Fail{Fore.RESET}"
                    
                except Exception as e:
                    # Check if it's a timeout issue
                    if "timeout" in str(e).lower():
                        logger.error(f"Processing timed out for file: {file_path} after {self.processing_timeout} seconds", exc_info=True)
                    # Check if it's related to special characters in the path
                    elif any(c in file_path for c in "'\"()[]{},;"):
                        logger.error(f"Error processing file with special characters: {file_path}: {e}", exc_info=True)
                    else:
                        logger.error(f"Error processing file {file_name}: {e}", exc_info=True)
                        
                    print(f"Error processing file {file_name}: {e}")
                    processing_result["success"] = False
                    # 모니터링은 finally 블록에서 중지됨
                
            finally:
                # 리소스 모니터링 중지
                logger.debug("Stopping resource monitoring")
                self.stop_monitoring()

                # 임베딩 진행 상태 완료
                # 현재 파일의 크기를 추가하여 업데이트
                if "total_size" in self.embedding_progress and file_size > 0:
                    # 이미 추가되지 않았다면 처리된 크기 추가
                    if not hasattr(self, '_processed_this_file') or not self._processed_this_file:
                        logger.debug(f"Updating processed size: +{file_size} bytes")
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
        logger.debug(f"Waiting for processing to complete with timeout of {self.processing_timeout} seconds")
        completed = processing_completed.wait(timeout=self.processing_timeout)
        
        if not completed:
            # Check if this file has special characters in its path
            has_special_chars = any(c in file_path for c in "'\"()[]{},;")
            if has_special_chars:
                logger.error(f"Processing timed out for file with special characters: {file_path} after {self.processing_timeout} seconds")
            else:
                logger.error(f"Processing timed out after {self.processing_timeout} seconds for file: {file_path}")
                
            print(f"Error: Processing timed out after {self.processing_timeout} seconds")
            # 리소스 모니터링 중지 (타임아웃 발생 시)
            logger.debug("Stopping resource monitoring due to timeout")
            self.stop_monitoring()
            self.embedding_in_progress = False
            return False
        
        logger.debug(f"Processing completed successfully within timeout period ({self.processing_timeout}s)")
        
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
            logger.warning(f"File does not exist or is not a file: {file_path}")
            return None, None
            
        try:
            rel_path = os.path.relpath(file_path, self.vault_path)
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()[1:] if '.' in file_path else ''
            
            # Check for special characters in file path that might need careful handling
            has_special_chars = any(c in file_path for c in "'\"()[]{},;")
            if has_special_chars:
                logger.debug(f"Extracting chunks from file with special characters in path: {rel_path}")
            
            logger.debug(f"Extracting chunks from file: {rel_path} (extension: {file_ext})")
            
            # 파일명 검증 - 비어있거나 특수 문자만 있는 경우 처리
            if not file_name or file_name.startswith('.'):
                logger.warning(f"Invalid filename detected: {file_name}")
                return None, None
            
            # 마크다운과 PDF만 처리 (다른 파일은 벡터 임베딩 제외)
            if file_ext.lower() not in ['pdf', 'md']:
                logger.info(f"Skipping non-supported file type: {file_ext} - {rel_path}")
                print(f"Skipping non-supported file type: {file_ext} - {file_path}")
                return None, None
                
            # 파일 생성/수정 시간 가져오기
            file_stats = os.stat(file_path)
            created_at = str(file_stats.st_ctime)
            updated_at = str(file_stats.st_mtime)
            logger.debug(f"File stats: created={created_at}, updated={updated_at} for {rel_path}")
            
            # 파일 타입에 따라 텍스트 추출
            try:
                # Log special attention for files with special characters
                if has_special_chars:
                    logger.info(f"Attempting to extract content from file with special characters: {rel_path}")
                    
                if file_ext == 'pdf':
                    logger.debug(f"Extracting content from PDF file: {rel_path}")
                    content, title, tags = self._extract_pdf(file_path)
                elif file_ext == 'md':
                    logger.debug(f"Extracting content from Markdown file: {rel_path}")
                    content, title, tags = self._extract_markdown(file_path)
                else:
                    return None, None
                    
                # Log successful extraction
                logger.debug(f"Successfully extracted content from {rel_path}, title: '{title}', tags: {tags}")
                
            except Exception as e:
                # Check if it's an Excalidraw file (known to have special characters)
                if "excalidraw" in file_path.lower():
                    logger.error(f"Error extracting content from Excalidraw file: {rel_path}: {e}", exc_info=True)
                elif has_special_chars:
                    logger.error(f"Error extracting content from file with special characters: {rel_path}: {e}", exc_info=True)
                else:
                    logger.error(f"Error extracting content from {rel_path}: {e}", exc_info=True)
                    
                print(f"Error extracting content from {file_path}: {e}")
                return None, None
            
            # 내용이 비어있는지 확인
            if not content or not content.strip():
                logger.warning(f"Empty content extracted from {rel_path}")
                return None, None
            
            # 청크로 분할
            logger.debug(f"Splitting content into chunks for {rel_path}")
            try:
                chunks = self._split_into_chunks(content)
                if not chunks:
                    logger.warning(f"No chunks generated from {rel_path}")
                    return None, None
                    
                logger.info(f"Successfully generated {len(chunks)} chunks from {rel_path}")
                
            except Exception as e:
                logger.error(f"Error splitting content into chunks for {rel_path}: {e}", exc_info=True)
                return None, None
                
            # 파일 메타데이터 준비 - content는 첫 번째 청크에만 저장
            try:
                # Check if we need to handle special characters in paths for Milvus
                path_for_milvus = rel_path
                
                # Add logging for special character detection in path that might affect Milvus
                if has_special_chars:
                    logger.debug(f"Preparing metadata for file with special characters: {rel_path}")
                
                metadata = {
                    "rel_path": path_for_milvus,  # Use the potentially sanitized path
                    "title": title,
                    "content": content,  # 청크 처리 후 메모리에서 제거됨
                    "file_ext": file_ext,
                    "is_pdf": file_ext.lower() == 'pdf',
                    "tags": tags,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "path": rel_path  # Store the original path as well
                }
                
                logger.debug(f"Created metadata for {rel_path} with {len(chunks)} chunks")
                
                # 2차 처리용 임시저장 제거
                metadata.pop('content', None)
                
                return chunks, metadata
                
            except Exception as e:
                logger.error(f"Error preparing metadata for {rel_path}: {e}", exc_info=True)
                return None, None
            
        except Exception as e:
            # Check if it's an Excalidraw file or has special characters
            if "excalidraw" in file_path.lower():
                logger.error(f"Error extracting chunks from Excalidraw file: {file_path}: {e}", exc_info=True)
            elif has_special_chars and 'has_special_chars' in locals():
                logger.error(f"Error extracting chunks from file with special characters: {file_path}: {e}", exc_info=True)
            else:
                logger.error(f"Error extracting chunks from {file_path}: {e}", exc_info=True)
                
            print(f"Error extracting chunks from {file_path}: {e}")
            return None, None
    
    def _extract_markdown(self, file_path):
        """마크다운 파일에서 텍스트 및 메타데이터 추출 (최적화)"""
        # Check for special characters in the file path
        rel_path = os.path.relpath(file_path, self.vault_path) if hasattr(self, 'vault_path') else file_path
        has_special_chars = any(c in file_path for c in "'\"()[]{},;")
        is_excalidraw = "excalidraw" in file_path.lower()
        
        if has_special_chars:
            logger.debug(f"Extracting markdown from file with special characters: {rel_path}")
        if is_excalidraw:
            logger.debug(f"Processing Excalidraw file: {rel_path}")
            
        try:
            # Log file open operation for tracking potential file access issues
            logger.debug(f"Opening markdown file: {rel_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decode error for {rel_path}, trying with alternative encodings")
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
                    
            # 제목 추출 (첫 번째 # 헤딩 또는 파일명)
            title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            title = title_match.group(1) if title_match else os.path.basename(file_path).replace('.md', '')
            logger.debug(f"Extracted title: '{title}' from {rel_path}")
            
            # YAML 프론트매터 및 태그 추출 (개선된 방식)
            tags = []
            
            # 처리 전 원본 콘텐츠 보존
            original_content = content
            
            # 콘텐츠에서 프론트매터 구문 추출
            yaml_match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
            
            if yaml_match:
                # 원래 프론트매터 텍스트 추출
                original_frontmatter = yaml_match.group(1)
                
                # 프론트매터 전처리 (YAML 구문 문제 수정)
                lines = original_frontmatter.split('\n')
                processed_lines = []
                
                for line in lines:
                    if not line.strip() or line.startswith('#'):
                        processed_lines.append(line)
                        continue
                        
                    # 1. 키:값 형태인지 확인
                    if ':' in line:
                        # 키와 값 분리
                        key_value = line.split(':', 1)
                        key = key_value[0].strip()
                        value = key_value[1].strip() if len(key_value) > 1 else ''
                        
                        # 2. title 필드 특별 처리
                        if key.lower() == 'title':
                            # 값에 콜론이 있거나 특수 문자가 있는지 확인
                            if ':' in value or any(c in value for c in '&@#%'):
                                # 이미 따옴표로 감싸져 있지 않으면 따옴표로 감싸기
                                if not (value.startswith('"') and value.endswith('"')) and not (value.startswith('\'') and value.endswith('\'')):
                                    value = f'"{value}"'
                                line = f"{key}: {value}"
                                logger.debug(f"Quoted title with special chars: {value}")
                        
                        # 3. 태그 필드 특별 처리
                        elif key.lower() == 'tags':
                            # 배열 형태로 표현되지 않은 경우 처리
                            if not value.startswith('[') and not value.startswith('-'):
                                line = f"{key}: [{value}]"
                        
                        # 4. 일반 필드에 콜론이 포함된 경우
                        elif ':' in value and not (value.startswith('"') or value.startswith('\'')):
                            value = f'"{value}"'
                            line = f"{key}: {value}"
                            
                        # 5. 숫자로 시작하는 값 처리
                        elif value and value[0].isdigit() and not (value.startswith('"') or value.startswith('\'')):
                            value = f'"{value}"'
                            line = f"{key}: {value}"
                    
                    processed_lines.append(line)
                
                # 최종 처리된 프론트매터 텍스트
                frontmatter_text = '\n'.join(processed_lines)
                
                # 안전을 위한 추가 필터링 (URL 안전 문자 유지하면서 악성 문자열만 제거)
                # URL에 사용되는 문자 +/# 등을 유지
                frontmatter_text = re.sub(r'[^\w\s\-\[\]:#\'\",._{}@&%/\\(\)\+=]+', ' ', frontmatter_text)
                
                try:
                    # 숫자로 시작하는 값과 콜론이 포함된 값을 처리하기 위한 YAML 수정
                    lines = frontmatter_text.split('\n')
                    fixed_lines = []
                    
                    for line in lines:
                        if not line.strip() or line.strip().startswith('#'):
                            fixed_lines.append(line)
                            continue
                            
                        # 키-값 라인인지 확인
                        if ':' in line:
                            parts = line.split(':', 1)  # 첫 번째 콜론에서만 분리
                            key_part = parts[0].strip()
                            value_part = parts[1].strip() if len(parts) > 1 else ''
                            
                            # 값에 콜론이 포함되어 있는지 확인
                            if ':' in value_part and not (value_part.startswith('"') or value_part.startswith('\'')):
                                # title 필드에 콜론이 있는 경우 (예: title: Robots and jobs: Evidence from US labor markets)
                                value_part = f'"{value_part}"'
                                line = f'{key_part}: {value_part}'
                                logger.debug(f"Added quotes to value with colon: {value_part}")
                            
                            # PDF 파일 참조 처리 (예: [PDF] 2008.pdf)
                            elif '[PDF]' in value_part and re.search(r'\d+\.pdf', value_part):
                                value_part = f'"{value_part}"'
                                line = f'{key_part}: {value_part}'
                            
                            # 숫자로 시작하는 값에 따옴표 추가
                            elif value_part and value_part[0].isdigit() and not (value_part.startswith('"') or value_part.startswith('\'')):
                                value_part = f'"{value_part}"'
                                line = f'{key_part}: {value_part}'
                                
                            # URL 형식 문제 사전 처리 (share_link 포함)
                            elif any(url_field in key_part.lower() for url_field in ['url', 'link', 'share_link', 'source']):
                                # https: 문제 수정 (슬래시가 빠진 URL)
                                if 'https:' in value_part and not 'https://' in value_part:
                                    value_part = value_part.replace('https:', 'https://')
                                # http: 문제도 같이 수정
                                if 'http:' in value_part and not 'http://' in value_part:
                                    value_part = value_part.replace('http:', 'http://')
                                    
                                # URL에 공백이 있으면 따옴표로 묶기
                                if not (value_part.startswith('"') or value_part.startswith('\'')):
                                    if ' ' in value_part or '+' in value_part or '#' in value_part:
                                        value_part = f'"{value_part}"'
                                line = f'{key_part}: {value_part}'
                                logger.debug(f"Special handling for URL field {key_part}: {value_part[:30]}...")
                                
                            # title 필드의 경우 특별 처리 (일반적으로 콜론이나 특수문자를 포함할 가능성이 높음)
                            elif key_part.lower() == 'title' and not (value_part.startswith('"') or value_part.startswith('\'')):
                                # 이미 따옴표로 감싸져 있지 않으면 무조건 따옴표 추가
                                value_part = f'"{value_part}"'
                                line = f'{key_part}: {value_part}'
                                
                        fixed_lines.append(line)
                    
                    frontmatter_fixed = '\n'.join(fixed_lines)
                    logger.debug(f"Fixed potential formatting issues in frontmatter for {rel_path}")
                    
                    # YAML 파싱 실패 시 정규식으로 폴백 (향상된 버전)
                    frontmatter = {}
                    
                    # 특수 URL 필드를 미리 추출 (일반 정규식으로는 URL 처리가 어려움)
                    url_fields = ['url', 'link', 'share_link', 'source']
                    url_pattern = re.compile(r'^((?:' + '|'.join(url_fields) + ')(?:_[\w]+)?)\s*:\s*(.+)$', re.IGNORECASE)
                    
                    # 먼저 URL 필드 추출 시도
                    for line in frontmatter_text.split('\n'):
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                            
                        url_match = url_pattern.match(line)
                        if url_match:
                            key = url_match.group(1).strip()
                            value = url_match.group(2).strip()
                            # URL 값에서 따옴표 제거
                            if (value.startswith('"') and value.endswith('"')) or (value.startswith('\'') and value.endswith('\'')):
                                value = value[1:-1]
                            frontmatter[key] = value
                            logger.debug(f"Extracted URL field with regex: {key}: {value[:30]}...")
                    
                    # 나머지 일반 키-값 추출을 위한 정규식
                    pattern = re.compile(r'^([\w\-]+)\s*:\s*(.+)$')
                    
                    # 나머지 일반 키-값 추출
                    for line in frontmatter_text.split('\n'):
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                            
                        match = pattern.match(line)
                        if match:
                            key = match.group(1).strip()
                            value = match.group(2).strip()
                            # 태그 필드 처리는 별도로 진행
                            if key.lower() == 'tags':
                                if value.startswith('[') and value.endswith(']'):
                                    tag_list = value[1:-1].split(',')
                                    frontmatter['tags'] = [tag.strip().strip('"\'\'') for tag in tag_list if tag.strip()]
                                else:
                                    frontmatter['tags'] = [value] if value else []
                            else:
                                frontmatter[key] = value
                    
                    # 태그 추출
                    if 'tags' in frontmatter:
                        tags_data = frontmatter['tags']
                        if isinstance(tags_data, list):
                            tags = [str(tag).strip() for tag in tags_data if tag]
                        elif isinstance(tags_data, str):
                            tags = [tags_data.strip()]
                        logger.debug(f"Extracted {len(tags)} tags from frontmatter for {rel_path}")
                        
                    # URL 필드가 있는 경우 추가 처리
                    if 'url' in frontmatter and isinstance(frontmatter['url'], str):
                        url_value = frontmatter['url']
                        if not url_value.startswith(('http://', 'https://')):
                            # URL 형식 자동 수정
                            if url_value.startswith('www.'):
                                frontmatter['url'] = f"https://{url_value}"
                                logger.debug(f"Fixed URL format in frontmatter: {frontmatter['url']}")
                            elif ' ' in url_value and not url_value.startswith('"'):
                                # 따옴표로 감싸서 URL의 공백 문제 방지
                                frontmatter['url'] = f"\"{url_value}\""
                except yaml.YAMLError as yaml_err:
                    logger.error(f"YAML parsing error in {rel_path}: {yaml_err}")
                except Exception as e:
                    # Special handling for files with special characters
                    if has_special_chars:
                        logger.warning(f"Special character handling for frontmatter in {rel_path}: {e}")
                        logger.warning(f"YAML parsing error in file with special characters: {rel_path}")
                    elif is_excalidraw:
                        logger.warning(f"YAML parsing error in Excalidraw file: {rel_path}")
                    else:
                        logger.error(f"Error processing frontmatter in {rel_path}: {e}")
                        logger.warning(f"YAML parsing error: falling back to regex for {rel_path}")
                        
                    error_msg = f"YAML parsing error: falling back to regex for {os.path.basename(file_path)}"
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
                            # YAML 형식의 태그 처리 (리스트 형식)
                            for line in tag_lines[0].split('\n'):
                                tag_item = re.match(r'\s*-\s*(.+)', line)
                                if tag_item:
                                    tags.append(tag_item.group(1).strip().strip("'\""))
                
                # $~$ 같은 수식 기호를 일반 텍스트로 변환
                content = re.sub(r'\$~\$', ' ', content)
                content = re.sub(r'\${2}.*?\${2}', ' ', content, flags=re.DOTALL)  # 블록 수식 처리
                content = re.sub(r'\$.*?\$', ' ', content)  # 인라인 수식 처리
            
            # 불필요한 여러 줄 공백 제거
            content = re.sub(r'\n{3,}', '\n\n', content)
            
            # 후행 공백 제거
            content = content.rstrip()
        except Exception as e:
            logger.warning(f"Error during content cleanup for {rel_path}: {e}")
        
        logger.info(f"Successfully extracted markdown from {rel_path}: {len(content)} chars, {len(tags)} tags")
        return content, title, tags
        
    def _extract_pdf(self, file_path):
        """PDF 파일에서 텍스트 및 메타데이터 추출 (손상된 PDF 처리 개선)"""
        rel_path = os.path.relpath(file_path, self.vault_path) if hasattr(self, 'vault_path') else file_path
        has_special_chars = any(c in file_path for c in "'\"()[]{},;")
        
        if has_special_chars:
            logger.debug(f"Extracting content from PDF file with special characters: {rel_path}")
            
        try:
            # Log file open operation for tracking potential file access issues
            logger.debug(f"Opening PDF file: {rel_path}")
            
            # Extract text from PDF using PyPDF2
            content = ""
            with open(file_path, 'rb') as file:
                try:
                    # 안전한 PDF 읽기 시도
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                    # '/Root' KeyError 같은 문제를 감지하기 위한 안전 처리
                    try:
                        num_pages = len(pdf_reader.pages)
                    except (KeyError, AttributeError, TypeError) as struct_err:
                        # PDF 구조 문제 (손상되거나 암호화된 PDF)
                        logger.error(f"PDF structure error in {rel_path}: {struct_err} - likely corrupted or encrypted PDF")
                        return None, None, None  # 손상된 PDF는 None 반환하여 건너뛀
                    
                    # 페이지가 없는 경우 처리
                    if num_pages == 0:
                        logger.warning(f"PDF file {rel_path} has 0 pages")
                        return None, None, None
                    
                    # Extract text from each page
                    for page_num in range(num_pages):
                        try:
                            page = pdf_reader.pages[page_num]
                            page_text = page.extract_text()
                            if page_text:
                                content += page_text + "\n\n"
                        except Exception as page_err:
                            # 특정 페이지 처리 오류는 무시하고 계속 진행
                            logger.warning(f"Error extracting text from page {page_num} in {rel_path}: {page_err}")
                            continue
                    
                    # 내용이 없는 경우 처리
                    if not content.strip():
                        logger.warning(f"Extracted empty content from PDF {rel_path} - might be a scanned document")
                        return None, None, None
                        
                    # Clean up the content
                    content = content.strip()
                    # Remove excessive newlines
                    content = re.sub(r'\n{3,}', '\n\n', content)
                    
                except KeyError as key_err:
                    # 특정 키가 없는 문제 ('/Root' 등)
                    logger.error(f"PyPDF2 KeyError processing {rel_path}: {key_err} - PDF may be corrupted")
                    return None, None, None  # 손상된 PDF는 None 반환하여 건너뛀
                except Exception as pdf_err:
                    logger.error(f"PyPDF2 error processing {rel_path}: {pdf_err}")
                    return None, None, None  # 기타 오류도 None 반환하여 건너뛀
            
            # Use filename as title (remove extension)
            title = os.path.basename(file_path)
            if title.lower().endswith('.pdf'):
                title = title[:-4]
                
            # PDFs don't have tags in our system, so return empty list
            tags = []
            
            logger.info(f"Successfully extracted content from PDF {rel_path}: {len(content)} chars")
            return content, title, tags
            
        except Exception as e:
            logger.error(f"Error processing PDF file {os.path.basename(file_path)}: {e}")
            # 오류 발생 시에도 예외를 전파하지 않고 None 반환
            return None, None, None
        
    def _extract_pdf_content(self, file_path):
        """PDF 파일에서 내용 추출"""
        content = ""
        title = os.path.basename(file_path).replace('.pdf', '')
        rel_path = os.path.relpath(file_path, self.vault_path)
        
        try:
            with open(file_path, 'rb') as file:
                # 안전하게 PDF 읽기 시도
                try:
                    reader = PyPDF2.PdfReader(file)
                    
                    # PDF 메타데이터에서 정보 추출
                    metadata = reader.metadata
                    if metadata and '/Title' in metadata and metadata['/Title']:
                        title = metadata['/Title']
                        logger.debug(f"Extracted title '{title}' from PDF metadata for {rel_path}")
                    
                    # 페이지 별로 내용 추출 (메모리 효율성 개선)
                    logger.debug(f"Extracting text from {len(reader.pages)} pages in {rel_path}")
                    for i, page in enumerate(reader.pages):
                        # 메모리 관리를 위해 10페이지마다 정리
                        if i > 0 and i % 10 == 0:
                            gc.collect()
                            logger.debug(f"Garbage collection performed after processing {i} pages")
                            
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                content += page_text + "\n\n"
                        except Exception as e:
                            logger.warning(f"Error extracting text from page {i} in {rel_path}: {e}")
                
                except Exception as e:
                    logger.error(f"Error reading PDF {rel_path}: {e}", exc_info=True)
        
        except Exception as e:
            logger.error(f"Error opening PDF file {rel_path}: {e}", exc_info=True)
        
        # 빈 내용인 경우 확인
        if not content.strip():
            error_msg = f"Warning: No content extracted from PDF {rel_path} - likely a scanned document"
            logger.warning(error_msg)
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
        
        # 🔧 ENHANCED: 안전을 위한 텍스트 길이 제한 (속도와 안정성 균형)
        # Use safe document length from config - no truncation, just warning
        max_document_length = getattr(config, 'MAX_DOCUMENT_LENGTH', 2000000)  # 2M chars
        if len(text) > max_document_length:
            print(f"Warning: Document very long ({len(text)} chars), processing may take longer")
            # Don't truncate - let chunking handle large documents
            
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
        
        # 🔧 ENHANCED: GPU/CPU 성능에 따른 동적 청크 개수 제한
        if hasattr(self, 'embedding_model') and hasattr(self.embedding_model, 'hardware_profiler'):
            profile = self.embedding_model.hardware_profiler.performance_profile
            
            # GPU 성능에 따른 청크 수 결정
            if 'professional_gpu' in profile:
                max_chunks = 200  # Tesla, A100, H100 등
            elif 'flagship_gpu' in profile:
                max_chunks = 150  # RTX 5090, RX 7900 XTX 등
            elif 'ultra_high_end_gpu' in profile:
                max_chunks = 120  # RTX 5080, RTX 4080 등
            elif 'high_end_gpu' in profile:
                max_chunks = 100  # RTX 5070, RTX 4070 등
            elif 'mid_range_gpu' in profile:
                max_chunks = 80   # RTX 3060, RTX 2080 등
            elif 'low_mid_gpu' in profile:
                max_chunks = 60   # RTX 2060, GTX 1660 Ti 등
            elif 'low_end_gpu' in profile:
                max_chunks = 50   # GTX 1650, RX 580 등
            elif 'very_low_end_gpu' in profile:
                max_chunks = 40   # GTX 1050 등
            elif 'high_end_cpu' in profile:
                max_chunks = 60   # 고성능 CPU
            elif 'mid_range_cpu' in profile:
                max_chunks = 40   # 중급 CPU
            else:
                max_chunks = 30   # 저성능 CPU
            
            # 하드웨어 성능이 좋더라도 최대 100개로 제한 (안정성)
            # Use config max chunks per file - no arbitrary limits
            max_chunks_per_file = getattr(config, 'MAX_CHUNKS_PER_FILE', 1000)  # 1000
            
            print(f"Dynamic chunk processing based on {profile}: up to {max_chunks_per_file} chunks per file")
        else:
            # Fallback: use config value
            max_chunks_per_file = getattr(config, 'MAX_CHUNKS_PER_FILE', 1000)  # 1000
            print(f"Using config chunk limit: {max_chunks_per_file} chunks per file")
        
        # Process all chunks - no truncation for complete coverage
        if len(unique_chunks) > max_chunks_per_file:
            print(f"Large file detected: {len(unique_chunks)} chunks (will process all chunks)")
            # Don't truncate - process all chunks for complete coverage
        
        # Split long chunks instead of truncating to preserve all content
        safe_chunks = []
        max_chunk_length = getattr(config, 'MAX_CHUNK_LENGTH', 50000)  # 50K chars from config
        
        for chunk in unique_chunks:
            if len(chunk) > max_chunk_length:
                print(f"Long chunk detected: {len(chunk)} chars, splitting into smaller chunks")
                # Split long chunk instead of truncating to preserve content
                chunk_parts = [chunk[i:i+max_chunk_length] for i in range(0, len(chunk), max_chunk_length)]
                safe_chunks.extend(chunk_parts)
                print(f"Split into {len(chunk_parts)} parts to preserve all content")
            else:
                safe_chunks.append(chunk)
        
        unique_chunks = safe_chunks
            
        return unique_chunks
    
    def _save_vectors_to_milvus(self, vectors, chunks, chunk_file_map):
        """벡터와 청크 데이터를 Milvus에 저장하는 최적화된 메소드 (문자열 길이 제한 강화)
        개별 청크 삽입 실패 시에도 계속 진행하며, 일정 수준의 성공만으로도 전체 처리를 성공으로 간주합니다.
        """
        if not vectors or not chunks or not chunk_file_map or len(vectors) != len(chunks):
            return False
            
        # 총 항목 수와 성공/실패 카운트 추적
        total_items = len(vectors)
        success_count = 0
        failed_count = 0
        file_chunk_indices = {}  # 파일별 청크 인덱스 추적
        
        # 처리 시작 로깅
        logger.info(f"Starting to save {total_items} vectors to Milvus")
        
        # 각 청크와 벡터 처리
        for i, (vector, chunk, metadata) in enumerate(zip(vectors, chunks, chunk_file_map)):
            # 메모리 모니터링 (더 자주 체크)
            if i > 0 and i % 10 == 0:
                self._check_memory_usage(f"Milvus insertion {i}/{total_items}")
                logger.info(f"Progress: {i}/{total_items} items processed. Success: {success_count}, Failed: {failed_count}")
            
            try:
                rel_path = metadata["rel_path"]
                
                # 파일별 청크 인덱스 추적
                if rel_path not in file_chunk_indices:
                    file_chunk_indices[rel_path] = 0
                chunk_index = file_chunk_indices[rel_path]
                file_chunk_indices[rel_path] += 1
                
                # 태그 JSON 변환 (안전한 형식으로)
                try:
                    tags_json = json.dumps(metadata["tags"]) if metadata["tags"] else "[]"
                except Exception as json_error:
                    logger.warning(f"Error converting tags to JSON: {json_error}, using empty array")
                    tags_json = "[]"
                
                # 최대 문자열 길이 (더 안전한 마진)
                MAX_STRING_LENGTH = 32000  # Milvus 제한 65535보다 충분히 안전하게 설정
                MAX_CONTENT_LENGTH = 16000  # content 필드는 더 짧게
                MAX_CHUNK_LENGTH = 16000    # chunk_text 필드도 더 짧게
                
                # 강화된 문자열 안전 자르기 함수
                def safe_truncate(text, max_len=MAX_STRING_LENGTH):
                    if not isinstance(text, str):
                        return str(text) if text is not None else ""
                    if not text:
                        return ""
                    # UTF-8 바이트 기준으로도 확인
                    try:
                        text_bytes = text.encode('utf-8', errors='ignore')[:max_len//2]
                        truncated = text_bytes.decode('utf-8', errors='ignore')
                        # 최종적으로 문자 길이도 확인
                        return truncated[:max_len] if len(truncated) > max_len else truncated
                    except Exception as enc_error:
                        logger.warning(f"Encoding error in safe_truncate: {enc_error}, returning empty string")
                        return ""
                
                # 각 항목을 개별적으로 삽입 (안전한 딕셔너리 접근 방식 사용)
                # 숫자로 시작하는 파일명 처리
                original_path = rel_path  # 원본 경로 저장
                
                # 안전한 파일 경로 생성 (한글 및 특수 문자 처리 강화)
                file_dir = os.path.dirname(rel_path)
                file_name = os.path.basename(rel_path)
                
                # 1. 특수 문자 및 공백을 안전하게 변환
                # 숫자로 시작하면 접두사 추가, 특수 문자는 ASCII로 변환 시도
                try:
                    # 숫자로 시작하는지 확인
                    if re.match(r'^\d', file_name):
                        # 접두사 추가
                        safe_file_name = f"file_{file_name}"
                    else:
                        safe_file_name = file_name
                    
                    # 특수 문자가 있는지 확인
                    if re.search(r'[^\w\-\. ]', safe_file_name):
                        # URL 인코딩과 유사한 방식으로 특수 문자 처리
                        # 한글은 그대로 유지하되 위험한 특수 문자만 처리
                        safe_file_name = safe_file_name.replace('\\', '_').replace('/', '_').replace(':', '_')\
                                                    .replace('*', '_').replace('?', '_').replace('"', '_')\
                                                    .replace('<', '_').replace('>', '_').replace('|', '_')\
                                                    .replace('\t', '_').replace('\n', '_')
                    
                    # 최종 안전 경로 생성
                    safe_path = os.path.join(file_dir, safe_file_name)
                    logger.debug(f"Created safe path: {safe_path} from original: {rel_path}")
                except Exception as path_error:
                    # 경로 처리 중 오류 발생 시 원본 경로 사용
                    logger.warning(f"Error creating safe path: {path_error}, using original path")
                    safe_path = rel_path
                
                # 안전한 제목 생성 (파일 경로와 동일한 방식으로 처리)
                original_title = metadata.get("title", "")
                
                try:
                    # 숫자로 시작하는지 확인
                    if original_title and re.match(r'^\d', original_title):
                        safe_title = f"Title_{original_title}"
                    else:
                        safe_title = original_title
                    
                    # 특수 문자가 있는지 확인
                    if safe_title and re.search(r'[^\w\-\. ]', safe_title):
                        # URL 인코딩과 유사한 방식으로 특수 문자 처리
                        # 한글은 그대로 유지하되 위험한 특수 문자만 처리
                        safe_title = safe_title.replace('\\', '_').replace('/', '_').replace(':', '_')\
                                               .replace('*', '_').replace('?', '_').replace('"', '_')\
                                               .replace('<', '_').replace('>', '_').replace('|', '_')\
                                               .replace('\t', '_').replace('\n', '_')
                    
                    logger.debug(f"Created safe title: {safe_title} from original: {original_title}")
                except Exception as title_error:
                    # 제목 처리 중 오류 발생 시 원본 제목 사용
                    logger.warning(f"Error creating safe title: {title_error}, using original title")
                    safe_title = original_title
                
                # 현재 파일 정보 설정 (로깅용)
                current_file = os.path.basename(rel_path)
                
                # 경로 정보를 path 필드에 저장
                # 숫자로 시작하는 파일명은 safe_path를 사용하여 해결
                # 중요: original_path 필드도 포함 (스키마 요구사항)
                single_data = {
                    # "id" 필드는 제거됨 - Milvus에서 자동 생성됨
                    "path": safe_truncate(safe_path, 500),  # 안전한 경로 사용
                    "original_path": safe_truncate(original_path, 500),  # 원본 경로 추가 - 스키마 요구사항
                    "title": safe_truncate(safe_title, 500),  # 안전한 제목 사용
                    # 첫 번째 청크일 때만 전체 내용 저장, 나머지는 빈 문자열
                    # 안전한 방식으로 content 키에 접근 (기본값 빈 문자열 사용)
                    "content": safe_truncate(metadata.get("content", ""), MAX_CONTENT_LENGTH) if chunk_index == 0 else "",
                    "chunk_text": safe_truncate(chunk, MAX_CHUNK_LENGTH),  # chunk_text 길이 제한 강화
                    "chunk_index": chunk_index,
                    "file_type": safe_truncate(metadata.get("file_ext", ""), 10),
                    "tags": safe_truncate(tags_json, 1000),
                    "created_at": safe_truncate(metadata.get("created_at", ""), 30),
                    "updated_at": safe_truncate(metadata.get("updated_at", ""), 30),
                    "vector": vector
                }
                
                # 강화된 데이터 유효성 검사 (안전 장치)
                valid_data = True
                for key, value in single_data.items():
                    if key != "vector" and isinstance(value, str):
                        # 모든 문자열 필드에 대해 강제 길이 제한
                        if key == "content":
                            max_field_len = MAX_CONTENT_LENGTH
                        elif key == "chunk_text":
                            max_field_len = MAX_CHUNK_LENGTH
                        else:
                            max_field_len = MAX_STRING_LENGTH
                        
                        if len(value) > max_field_len:
                            logger.warning(f"Field {key} too long ({len(value)} chars), forcing truncation to {max_field_len}")
                            single_data[key] = value[:max_field_len]
                
                # FINAL SAFETY: 모든 문자열이 안전한 길이인지 최종 확인
                for key, value in single_data.items():
                    if key != "vector" and isinstance(value, str) and len(value) > 16000:
                        logger.warning(f"EMERGENCY: Field {key} still too long after all checks ({len(value)} chars), emergency truncation")
                        single_data[key] = value[:10000]  # 응급 처치 - 매우 보수적으로 10K로 제한
                
                # 특수 문자 처리 개선 (콤마, 괴호, 인용부호 등)
                sanitized_data = {}
                for key, value in single_data.items():
                    if key == "vector":
                        sanitized_data[key] = value
                    elif isinstance(value, str):
                        # 문자열 필드의 경우 특수 문자 처리
                        if key in ["path", "title", "original_path"]:  # original_path 필드 추가
                            # 경로와 제목은 중요하므로 인코딩 문제 확인
                            try:
                                # Milvus에서 사용하는 표현식에 중요한 특수 문자 이스케이핑
                                escaped_value = value.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
                                sanitized_data[key] = escaped_value
                            except Exception as esc_error:
                                logger.warning(f"Error escaping special chars in {key}: {esc_error}, using original value")
                                sanitized_data[key] = value
                        else:
                            # 다른 문자열 필드는 기본 처리
                            sanitized_data[key] = value
                    else:
                        sanitized_data[key] = value
                
                # 추가 가능성 검사 - original_path 필드가 없는 경우 추가
                if "original_path" not in sanitized_data and "path" in sanitized_data:
                    # 반드시 original_path가 포함되도록 보장 (스키마 요구사항)
                    sanitized_data["original_path"] = sanitized_data.get("path", "")  # path 값을 기본값으로 사용
                    logger.debug(f"Added missing original_path field (schema requirement)")
                    
                # 확인용 로깅
                logger.debug(f"Final fields ready for insertion: {list(sanitized_data.keys())}")
                if "original_path" not in sanitized_data:
                    logger.warning(f"WARNING: original_path field is still missing after all fixes")
                else:
                    logger.debug(f"original_path field is present with value: '{sanitized_data['original_path'][:30]}...'")
                    
                # 중요: id 필드가 있으면 제거 (Milvus에서 자동 관리됨)
                if 'id' in sanitized_data:
                    logger.debug(f"Removing 'id' field from sanitized_data to prevent DataNotMatchException")
                    del sanitized_data['id']
                
                # 이미지 참조 탐지 및 제거 (![[...]] 형태 처리)
                if 'chunk_text' in sanitized_data and isinstance(sanitized_data['chunk_text'], str):
                    # 이미지 참조 제거 및 공백으로 대체
                    image_pattern = re.compile(r'!\[\[Pasted image [^\]]+\]\]')
                    sanitized_data['chunk_text'] = image_pattern.sub(' [IMAGE] ', sanitized_data['chunk_text'])
                    
                    # 만약 content도 있다면 동일하게 처리
                    if 'content' in sanitized_data and isinstance(sanitized_data['content'], str):
                        sanitized_data['content'] = image_pattern.sub(' [IMAGE] ', sanitized_data['content'])
                
                # 한글 및 특수 문자 처리
                for key in ['path', 'title', 'original_path', 'chunk_text', 'content']:
                    if key in sanitized_data and isinstance(sanitized_data[key], str):
                        # 한글이 포함된 경우 추가 로깅
                        if any('\u3131' <= c <= '\ud7a3' for c in sanitized_data[key]):
                            logger.debug(f"Field {key} contains Korean characters")
                            
                            # 길이 제한 적용 - 한글은 일반적으로 더 많은 바이트를 차지함
                            max_length = min(len(sanitized_data[key]), 800 if key in ['chunk_text', 'content'] else 500)
                            if len(sanitized_data[key]) > max_length:
                                sanitized_data[key] = sanitized_data[key][:max_length]
                                logger.debug(f"Truncated {key} with Korean content to {max_length} characters")
                
                # 특수 문자를 포함하는 경우 추가 로깅
                has_special_chars = False
                for key, value in sanitized_data.items():
                    if key != "vector" and isinstance(value, str) and any(c in value for c in "'\"()[]{},;"):
                        has_special_chars = True
                        logger.debug(f"Field {key} contains special characters that might need careful handling")
                
                        try:
                            encoded_bytes = value.encode('utf-8')
                            byte_len = len(encoded_bytes)
                            # 특정 범위의 바이트 값 출력 (오류 발생 가능성이 높은 위치 확인)
                            if byte_len > 100:
                                sample_bytes = encoded_bytes[:50] + b'...' + encoded_bytes[-50:]
                                logger.debug(f"  Field {key} encoding (byte length: {byte_len}): {sample_bytes}")
                        except Exception as enc_error:
                            logger.warning(f"  Field {key} has encoding issues: {enc_error}")

                # 4. 삽입 시도 전 Milvus 문서 확인
                try:
                    # 콜렉션 스키마 정보 요청
                    schema = self.milvus_manager.collection.schema
                    field_names = [field.name for field in schema.fields]
                    logger.debug(f"  Milvus schema fields: {field_names}")

                    # 스키마에 없는 필드 찾기
                    extra_fields = [key for key in sanitized_data.keys() if key not in field_names]
                    if extra_fields:
                        logger.warning(f"  Fields not in schema: {extra_fields} - these might cause errors")

                        # 불필요한 필드 제거
                        for field in extra_fields:
                            if field in sanitized_data:
                                del sanitized_data[field]
                                logger.debug(f"Removed extra field '{field}' not in schema")

                        # 스키마에 있지만 데이터에 없는 필드 찾기
                        missing_fields = [name for name in field_names if name not in sanitized_data and name != 'vector']
                        if missing_fields:
                            logger.warning(f"  Missing fields from schema: {missing_fields}")

                            # 필수 필드 추가 (빈 문자열 사용)
                            for field in missing_fields:
                                if field != "vector":  # 벡터 필드는 처리하지 않음
                                    sanitized_data[field] = ""
                                    logger.debug(f"Added missing field '{field}' required by schema")
                except Exception as schema_error:
                    logger.warning(f"  Could not verify schema compatibility: {schema_error}")

                # 5. 삽입 시도 (강화된 오류 처리 및 재시도 로직)
                logger.debug(f"  Attempting to insert data for {current_file}...")

                # 재시도 로직 추가 - 최대 5회로 증가
                max_retries = 5
                retry_count = 0
                last_error = None

                # 데이터 전처리: 경로 및 파일 이름 특별 처리
                # 숫자로 시작하는 파일명에 접두사 추가
                if 'path' in sanitized_data and isinstance(sanitized_data['path'], str):
                    # 경로에서 파일명 추출
                    filename = os.path.basename(sanitized_data['path'])
                    if filename and filename[0].isdigit():
                        # 원래 경로 보존
                        if 'original_path' not in sanitized_data:
                            sanitized_data['original_path'] = sanitized_data['path']
                        # 숫자로 시작하는 파일명에 접두사 추가
                        sanitized_data['path'] = os.path.join(
                            os.path.dirname(sanitized_data['path']),
                            f"file_{filename}"
                        )
                        logger.debug(f"Added prefix to numeric filename: {filename} -> file_{filename}")
                
                # 제목이 숫자로 시작하는 경우 처리
                if 'title' in sanitized_data and isinstance(sanitized_data['title'], str):
                    if sanitized_data['title'] and sanitized_data['title'][0].isdigit():
                        sanitized_data['title'] = f"Title_{sanitized_data['title']}"
                        logger.debug(f"Added prefix to numeric title: {sanitized_data['title']}")
                
                # 이미지 참조 패턴 감지 및 정리 (![[Pasted image...]])
                image_pattern = re.compile(r'!\[\[(Pasted image[^\]]+)\]\]')
                for key, value in list(sanitized_data.items()):
                    if key != "vector" and isinstance(value, str) and '![[' in value:
                        sanitized_data[key] = image_pattern.sub(r'Image: \1', value)
                        logger.debug(f"Sanitized image references in field '{key}'")
                
                # 한국어 텍스트 및 특수 문자 인코딩 문제 처리 - 모든 문자열 필드에 대해 추가 처리
                for key, value in list(sanitized_data.items()):
                    if key != "vector" and isinstance(value, str):
                        # 한국어 특수 처리: 길이 제한 적용 (바이트 기준)
                        if any(ord(c) > 127 for c in value):
                            # 한글이 포함된 경우 바이트 길이 계산 및 제한
                            try:
                                byte_length = len(value.encode('utf-8'))
                                max_bytes = 2000  # Milvus 권장 최대 바이트 수
                                
                                if byte_length > max_bytes:
                                    # 바이트 기준으로 안전하게 자르기
                                    truncated = ''
                                    current_bytes = 0
                                    for char in value:
                                        char_bytes = len(char.encode('utf-8'))
                                        if current_bytes + char_bytes <= max_bytes - 3:  # 여유 공간 확보
                                            truncated += char
                                            current_bytes += char_bytes
                                        else:
                                            break
                                    sanitized_data[key] = truncated + '...'
                                    logger.debug(f"Korean text in '{key}' truncated from {byte_length} to {current_bytes} bytes")
                            except UnicodeEncodeError as enc_err:
                                logger.warning(f"Korean encoding issue with {key}, applying special handling: {enc_err}")
                                # 인코딩 오류 시 안전하게 처리
                                sanitized_data[key] = ''.join(c if ord(c) < 128 else '?' for c in value[:200]) + '...'
                        
                        # 일반 인코딩 테스트 및 문제 처리
                        try:
                            # 텍스트 인코딩 테스트
                            encoded = value.encode('utf-8')
                        except UnicodeEncodeError as enc_err:
                            # 인코딩 문제가 있는 경우 ascii로 필터링
                            logger.warning(f"Encoding issue with {key}, sanitizing: {enc_err}")
                            sanitized_data[key] = value.encode('ascii', 'ignore').decode('ascii')

                while retry_count < max_retries:
                    try:
                        start_time = time.time()
                        
                        # 제일 묘하고 개선된 방법으로 시도
                        if retry_count == 0:
                            # 첫 번째 시도: 정제된 데이터 그대로 시도
                            result = self.milvus_manager.insert_data(sanitized_data)
                        # 두 번째 시도: 일부 필드 간소화 및 YAML 문제 필드 특별 처리
                        elif retry_count == 1:
                            # 중요하지 않은 필드 제거 후 재시도
                            minimal_data = dict(sanitized_data)
                            for field in ['tags', 'created_at', 'updated_at']:
                                if field in minimal_data:
                                    del minimal_data[field]
                            
                            # YAML 프론트매터 관련 특수 문자 문제 처리
                            if 'title' in minimal_data and isinstance(minimal_data['title'], str):
                                # 콜론이 포함된 제목 처리
                                if ':' in minimal_data['title']:
                                    minimal_data['title'] = minimal_data['title'].replace(':', ' - ')
                                    logger.debug(f"Replaced colons in title with hyphens")
                                # 따옴표 처리
                                if '"' in minimal_data['title'] or "'" in minimal_data['title']:
                                    minimal_data['title'] = minimal_data['title'].replace('"', '').replace("'", '')
                                    logger.debug(f"Removed quotes from title")
                            
                            result = self.milvus_manager.insert_data(minimal_data)
                        # 세 번째 시도: 노이즈가 있는 필드 표준화 및 Excalidraw 파일 특별 처리
                        elif retry_count == 2:
                            # 모든 문자열 필드 더 강력하게 정제
                            ultra_safe_data = dict(sanitized_data)
                            for key, value in ultra_safe_data.items():
                                if key != "vector" and isinstance(value, str):
                                    # 안전한 문자만 유지 (더 관대한 버전으로 수정)
                                    if 'excalidraw' in value.lower():
                                        # Excalidraw 파일 특별 처리
                                        logger.debug(f"Applying special handling for Excalidraw content in {key}")
                                        ultra_safe_data[key] = f"Excalidraw drawing {self.next_id}"
                                    else:                                  
                                        # 한글과 영어 및 기본 문장 부호 유지, 나머지 특수문자 치환
                                        ultra_safe_data[key] = re.sub(r'[^\w\-\. ,;:\(\)\[\]가-힣]', '_', value)[:200]
                            result = self.milvus_manager.insert_data(ultra_safe_data)
                        # 네 번째 시도: 색인 원본 파일을 최소한 필드로만 구성
                        elif retry_count == 3:
                            # 필수 필드만을 사용하여 기본 삽입 시도 - 한글 지원 강화
                            # 원본 경로 보존
                            safe_path = f"safe_path_{self.next_id}"
                            
                            # 원본 경로가 있으면 사용, 없으면 안전한 값 생성
                            if original_path:
                                safe_original = original_path[:100]
                            else:
                                # 현재 파일 경로에서 추출 시도
                                if current_file and isinstance(current_file, str):
                                    safe_original = current_file[:100]
                                else:
                                    safe_original = f"fallback_path_{self.next_id}"
                            
                            # 청크 텍스트 안전하게 처리
                            if isinstance(chunk, str):
                                # 한글이 포함된 경우 특별 처리
                                if any(ord(c) > 127 for c in chunk):
                                    safe_chunk = ''.join(c for c in chunk[:50] if ord(c) < 1000) + '...'
                                else:
                                    safe_chunk = chunk[:100]
                            else:
                                safe_chunk = f"Safe chunk text {self.next_id}"
                            
                            fallback_data = {
                                "path": safe_path,
                                "original_path": safe_original,
                                "title": f"Safe Title {self.next_id}",
                                "chunk_text": safe_chunk,
                                "chunk_index": chunk_index,
                                "vector": vector
                            }
                            result = self.milvus_manager.insert_data(fallback_data)
                        # 마지막 시도: 고정 값 사용
                        else:
                            # 고정 값을 사용한 가장 안전한 삽입 시도
                            emergency_data = {
                                "path": f"emergency_path_{self.next_id}",
                                "original_path": f"emergency_original_path_{self.next_id}",
                                "title": f"Emergency Title {self.next_id}",
                                "chunk_text": f"Emergency chunk {self.next_id}",
                                "chunk_index": 0,
                                "vector": vector
                            }
                            result = self.milvus_manager.insert_data(emergency_data)
                            
                        end_time = time.time()

                        # 성공 시 추가 정보 로깅
                        success_count += 1
                        if retry_count > 0:
                            logger.info(f"Successfully inserted data for file: {current_file} after {retry_count+1} attempts (took {end_time - start_time:.2f}s)")
                        else:
                            logger.info(f"Successfully inserted data for file: {current_file} (took {end_time - start_time:.2f}s)")

                        logger.debug(f"  Insert result: {result}")
                        break  # 성공하면 루프 비활성화
                    except Exception as insert_error:
                        retry_count += 1
                        last_error = insert_error

                        # 오류 발생 시 로깅 개선
                        error_type = type(insert_error).__name__
                        error_message = str(insert_error)

                        logger.warning(f"Insert attempt {retry_count}/{max_retries} failed: {error_type} - {error_message[:100]}...")

                        # 재시도 전 추가 조치 (오류 유형에 따라 다른 전략 적용)
                        if "schema" in error_message.lower() or "DataNotMatchException" in error_type:
                            # 스키마 문제인 경우 기록 및 다음 시도에 대비
                            logger.debug(f"Schema issue detected, will try alternative approach in next retry")
                            # 아무 처리도 하지 않음 - 다음 시도에서 다른 전략 사용
                        elif "timeout" in error_message.lower() or "connection" in error_message.lower():
                            # 연결 문제인 경우 잠시 대기 후 재시도
                            time.sleep(1.0)  # 1초 대기 후 재시도
                            logger.debug(f"Connection issue, waiting before retry {retry_count}")
                        else:
                            # 기타 오류에 대한 로깅
                            logger.debug(f"General error in retry {retry_count}, will use more aggressive sanitization in next attempt")

                        # 마지막 시도에서도 실패하면 오류 처리
                        if retry_count >= max_retries:
                            failed_count += 1
                            logger.error(f"Failed to insert data for {current_file} after {max_retries} attempts")
                            # 오류 세부 정보 추가 로깅
                            logger.error(f"Final error: {error_type} - {error_message}")

                # 6. 일정 개수마다 flush - 메모리 관리
                if success_count % 10 == 0:
                    try:
                        flush_start = time.time()
                        self.milvus_manager.collection.flush()
                        flush_end = time.time()
                        logger.debug(f"Successfully flushed after {success_count} insertions (took {flush_end - flush_start:.2f}s)")
                    except Exception as flush_error:
                        logger.warning(f"Non-critical flush error (continuing): {flush_error}")
                    
                # 유효하지 않은 데이터인 경우
                else:  # valid_data가 False인 경우
                    failed_count += 1
                    logger.warning(f"Skipping invalid data for item {self.next_id} (data validation failed)")
            
            except Exception as overall_error:  # 전체 처리 중 발생한 예외
                # 삽입 오류 발생 시 이 항목은 건너뛰고 계속 진행
                failed_count += 1
                error_type = type(overall_error).__name__
                error_message = str(overall_error)
                
                # current_file은 이미 위에서 정의됨
                    
                # 1. 기본 오류 정보 로깅
                logger.error(f"Failed to insert data for file: {current_file}")
                logger.error(f"Error type: {error_type}, Message: {error_message}")
                
                # 2. 상세 오류 정보와 스택 트레이스 로깅
                import traceback
                stack_trace = traceback.format_exc()
                logger.error(f"Exception stack trace:\n{stack_trace}")
                
                # 3. 오류 분석
                if "DataNotMatchException" in error_type or "schema" in error_message.lower():
                    logger.error("This appears to be a schema mismatch error. Check if field definitions match Milvus schema.")
                    # 필드 정보 출력
                    logger.error("Field values that might be causing the error:")
                    for key, value in sanitized_data.items():
                        if key != "vector":
                            value_type = type(value).__name__
                            value_preview = str(value)[:50] + "..." if isinstance(value, str) and len(str(value)) > 50 else value
                            logger.error(f"  Field '{key}' ({value_type}): {value_preview}")
                
                elif "timeout" in error_message.lower() or "connection" in error_message.lower():
                    logger.error("This appears to be a connection or timeout issue with Milvus.")
                    # 연결 정보 로깅
                    logger.error(f"Milvus connection info: {self.milvus_manager.host}:{self.milvus_manager.port}")
                
                elif "quota" in error_message.lower() or "limit" in error_message.lower():
                    logger.error("This appears to be a quota or limit exceeded error in Milvus.")
                    # 일부 데이터 크기 출력
                    for key, value in sanitized_data.items():
                        if key != "vector" and isinstance(value, str):
                            logger.error(f"  Field '{key}' length: {len(value)} chars")
                    
                    # 4. 전체 데이터 정보 디버깅 로깅
                    logger.error("Complete data details for debugging:")
                    for field_name, field_value in sanitized_data.items():
                        if field_name != "vector":
                            value_type = type(field_value).__name__
                            value_preview = str(field_value)[:100] + "..." if isinstance(field_value, str) and len(str(field_value)) > 100 else field_value
                            logger.error(f"  Field '{field_name}' ({value_type}): {value_preview}")
                    
                    # 5. 벡터 정보 로깅
                    vector = sanitized_data.get("vector", [])
                    logger.error(f"  Vector dimension: {len(vector)}")
                    
                    # 6. 문자열 필드의 바이트 크기 확인 (인코딩 문제 진단)
                    for key, value in sanitized_data.items():
                        if key != "vector" and isinstance(value, str):
                            try:
                                encoded_bytes = value.encode('utf-8')
                                logger.error(f"  Field '{key}' byte length: {len(encoded_bytes)}")
                                
                                # 위험한 문자 찾기
                                for i, char in enumerate(value[:100]):
                                    if ord(char) > 127 or char in '\\\'"`<>{}[]':
                                        logger.error(f"  Field '{key}' contains potentially problematic character '{char}' (ord={ord(char)}) at position {i}")
                            except Exception as enc_error:
                                logger.error(f"  Field '{key}' encoding error: {enc_error}")
                    
                    # 7. 오류 정보 자세히 기록하지만 전체 프로세스는 계속 진행
                    logger.debug(f"Continuing with next item despite insertion error for item {self.next_id}")
                
                # ID 증가 (항상 증가해야 중복 ID 방지)
                self.next_id += 1
                
            except Exception as item_error:
                # 항목 자체 처리 중 오류 발생해도 다음 항목으로 계속 진행
                failed_count += 1
                logger.error(f"Error processing item {i}/{total_items}: {item_error}", exc_info=True)
                # ID는 항상 증가 (안전장치)
                self.next_id += 1
        
        # 최종 flush 시도
        try:
            self.milvus_manager.collection.flush()
            logger.info("Final flush completed successfully")
        except Exception as final_flush_error:
            logger.warning(f"Error during final flush (non-critical): {final_flush_error}")
        
        # 최종 결과 로깅
        success_rate = (success_count / total_items) * 100 if total_items > 0 else 0
        logger.info(f"Vector insertion complete. Total: {total_items}, Success: {success_count}, Failed: {failed_count}, Success Rate: {success_rate:.1f}%")
        
        # 성공률 50% 이상이면 성공으로 간주
        # 또는 적어도 하나의 항목이 성공했고 실패가 적으면 성공으로 간주
        success_threshold = 0.5  # 50% 성공률 임계값
        min_success_count = 1    # 최소 성공 항목 수
        
        if (total_items > 0 and success_count / total_items >= success_threshold) or \
           (success_count >= min_success_count and success_count > failed_count):
            logger.info(f"Vector insertion considered successful with {success_rate:.1f}% success rate")
            return True
        else:
            logger.warning(f"Vector insertion considered failed with only {success_rate:.1f}% success rate")
            return False
    
    def _fast_decision_engine(self, file_path, file_mtime, existing_mtime, file_size):
        """OPTIMIZATION: 3-Tier fast decision making for file processing"""
        rel_path = os.path.relpath(file_path, self.vault_path)
        
        # Cache key for this file
        cache_key = f"{rel_path}:{file_mtime}:{file_size}"
        
        # Check cache first
        if cache_key in self.verification_cache:
            cached_result = self.verification_cache[cache_key]
            print(f"{Fore.BLUE}[CACHED] {rel_path}: {cached_result['decision']} ({cached_result['reason']}){Style.RESET_ALL}")
            return cached_result['decision'], cached_result['reason']
        
        # Calculate time difference
        time_diff = abs(file_mtime - existing_mtime) if existing_mtime > 0 else float('inf')
        
        decision = None
        reason = ""
        
        # TIER 1: Lightning Fast Decisions (90%+ of files)
        if time_diff > self.FAST_PROCESS_THRESHOLD:
            # File definitely modified - immediate process
            decision = "PROCESS"
            reason = f"definitely modified (time_diff: {time_diff:.2f}s)"
            print(f"{Fore.GREEN}[FAST-PROCESS] {rel_path}: {reason}{Style.RESET_ALL}")
            
        elif time_diff < self.FAST_SKIP_THRESHOLD:
            # File very likely unchanged - immediate skip (low risk)
            decision = "SKIP"
            reason = f"very likely unchanged (time_diff: {time_diff:.2f}s)"
            print(f"{Fore.CYAN}[FAST-SKIP] {rel_path}: {reason}{Style.RESET_ALL}")
            
        else:
            # TIER 2: Smart Batch Check for ambiguous cases
            decision = "VERIFY"
            reason = f"ambiguous timestamp (time_diff: {time_diff:.2f}s) - needs verification"
            print(f"{Fore.YELLOW}[NEED-VERIFY] {rel_path}: {reason}{Style.RESET_ALL}")
        
        # Cache the result
        result = {"decision": decision, "reason": reason}
        self.verification_cache[cache_key] = result
        
        return decision, reason

    def _batch_existence_check(self, suspect_files):
        """OPTIMIZATION: Batch check multiple files at once"""
        if not suspect_files:
            return {}
            
        try:
            # Build batch query for multiple files
            paths = [os.path.relpath(fp, self.vault_path) for fp, _ in suspect_files]
            path_conditions = " or ".join([f"path == '{path}'" for path in paths])
            
            # Single query to check all suspect files
            results = self.milvus_manager.query(
                expr=f"({path_conditions})",
                output_fields=["path", "id"],
                limit=len(paths) * 10  # Assume max 10 chunks per file
            )
            
            # Count chunks per file
            file_chunk_counts = {}
            for result in results:
                path = result.get("path")
                if path:
                    file_chunk_counts[path] = file_chunk_counts.get(path, 0) + 1
            
            print(f"{Fore.CYAN}[BATCH-CHECK] Verified {len(suspect_files)} files in single query{Style.RESET_ALL}")
            
            return file_chunk_counts
            
        except Exception as e:
            print(f"{Fore.YELLOW}[WARNING] Batch check failed: {e}{Style.RESET_ALL}")
            return {}
    
    def _normalize_timestamp(self, timestamp_value):
        """Normalize timestamp to float for reliable comparison"""
        try:
            if timestamp_value is None:
                return 0.0
            
            if isinstance(timestamp_value, (int, float)):
                return float(timestamp_value)
            
            if isinstance(timestamp_value, str):
                # Handle empty strings
                if not timestamp_value.strip():
                    return 0.0
                
                # Try direct float conversion
                try:
                    return float(timestamp_value)
                except ValueError:
                    # Try parsing as datetime string if needed
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                        return dt.timestamp()
                    except:
                        print(f"{Fore.YELLOW}[WARNING] Could not parse timestamp: {timestamp_value}{Style.RESET_ALL}")
                        return 0.0
            
            print(f"{Fore.YELLOW}[WARNING] Unknown timestamp type: {type(timestamp_value)}{Style.RESET_ALL}")
            return 0.0
            
        except Exception as e:
            print(f"{Fore.YELLOW}[WARNING] Error normalizing timestamp {timestamp_value}: {e}{Style.RESET_ALL}")
            return 0.0
    
    def process_updated_files(self):
        """볼트의 새로운 파일 또는 수정된 파일만 처리 + 삭제된 파일 정리 (증분 임베딩) - ENHANCED VERSION"""
        print(f"\n{Fore.CYAN}[DEBUG] Starting FIXED process_updated_files with deleted file cleanup{Style.RESET_ALL}")
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
        
        # Get existing file information - IMPROVED VERSION (more robust)
        existing_files_info = {}
        
        try:
            print(f"{Fore.CYAN}[DEBUG] Querying existing files from Milvus (intelligent batch sizing)...{Style.RESET_ALL}")
            # Use MilvusManager's intelligent batch sizing
            max_limit = self.milvus_manager._get_optimal_query_limit()
            offset = 0
            
            while True:
                # Get only path and updated_at for timestamp comparison
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
                    updated_at = doc.get('updated_at')
                    
                    if path and path not in existing_files_info:
                        # Store normalized timestamp
                        normalized_timestamp = self._normalize_timestamp(updated_at)
                        existing_files_info[path] = normalized_timestamp
                
                offset += max_limit
                if len(results) < max_limit:
                    break
                
                # Memory management
                gc.collect()
                
            print(f"{Fore.CYAN}[DEBUG] Found {len(existing_files_info)} unique files in DB{Style.RESET_ALL}")
            
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
                        
                        # OPTIMIZED: Use fast decision engine
                        file_mtime = os.path.getmtime(full_path)
                        existing_mtime = self._normalize_timestamp(existing_files_info.get(rel_path, 0))
                        
                        decision, reason = self._fast_decision_engine(full_path, file_mtime, existing_mtime, file_size)
                        
                        if decision == "PROCESS":
                            new_or_modified_count += 1
                            new_or_modified_size += file_size
                            
                            if rel_path in existing_files_info:
                                self.milvus_manager.mark_for_deletion(rel_path)
                            
                            files_to_process.append((full_path, file_size))
                            
                        elif decision == "SKIP":
                            skipped_count += 1
                            
                        elif decision == "VERIFY":
                            # Will be handled in batch verification below
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
        
        # OPTIMIZATION: Performance summary
        total_scanned = total_files_count
        if total_scanned > 0:
            print(f"\n{Fore.CYAN}[PERFORMANCE SUMMARY]{Style.RESET_ALL}")
            print(f"Files scanned: {total_scanned}")
            print(f"Processing decisions: {len(files_to_process)}/{total_scanned} ({len(files_to_process)/total_scanned*100:.1f}%)")
            print(f"Skipped decisions: {skipped_count}/{total_scanned} ({skipped_count/total_scanned*100:.1f}%)")
            print(f"Cache entries: {len(self.verification_cache)}")
        
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
            try:
                print(f"{Fore.CYAN}[DEBUG] Querying existing files from Milvus (intelligent batch sizing)...{Style.RESET_ALL}")
                # Use MilvusManager's intelligent batch sizing
                max_limit = self.milvus_manager._get_optimal_query_limit()
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
                                if content.count('$~') > 10:
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
        first_5_files = [os.path.basename(fp) for fp, _ in files_to_process[:5]]
        logger.debug(f"First 5 files in batch: {first_5_files}")
        print(f"{Fore.CYAN}[DEBUG] First 5 files in batch: {first_5_files}{Style.RESET_ALL}")
        
        # 배치 처리
        total_size_mb = total_size / (1024 * 1024)
        logger.info(f"Starting batch processing of {total_files} files (total size: {total_size_mb:.2f} MB)")
        with tqdm(total=total_size_mb, desc="Indexing files", unit="MB", ncols=100) as pbar:
            # 메모리 효율을 위한 배치 처리
            for i in range(0, total_files, batch_size):
                batch = files_to_process[i:i+batch_size]
                batch_number = i//batch_size+1
                total_batches = (total_files+batch_size-1)//batch_size
                
                logger.debug(f"Processing batch {batch_number}/{total_batches} with {len(batch)} files")
                
                # 메모리 사용량 확인
                self._check_memory_usage(f"Before processing batch {batch_number}/{total_batches}")
                
                # 배치 처리
                for file_item in batch:
                    try:
                        # 파일 경로와 크기 분리
                        file_path, file_size = file_item
                        
                        # 현재 처리중인 파일 표시
                        rel_path = os.path.relpath(file_path, self.vault_path)
                        logger.debug(f"Processing file: {rel_path}")
                        self.embedding_progress["current_file"] = rel_path
                        self.embedding_progress["processed_files"] += 1
                        
                        # 파일 크기를 추가하고 processed_size는 그대로 유지
                        # 중요: process_file가 이미 processed_size를 업데이트하므로 여기서는 추가하지 않음
                        # 대신 이미 추가되어 있는지 표시하는 플래그 설정
                        self._processed_this_file = True
                        
                        # 단일 파일 처리
                        success = self.process_file(file_path)
                        if success:
                            logger.debug(f"Successfully processed file: {rel_path}")
                            processed_count += 1
                        else:
                            logger.warning(f"Failed to process file: {rel_path}")
                            
                        # 진행률 업데이트
                        file_size_mb = file_size / (1024 * 1024)
                        pbar.update(file_size_mb)
                        
                    except Exception as e:
                        logger.error(f"Error processing file in batch: {rel_path}: {e}", exc_info=True)
                        print(f"Error processing file in batch: {e}")
                
                # 메모리 정리
                logger.debug("Running garbage collection and clearing model cache")
                gc.collect()
                self.embedding_model.clear_cache()
        
        return processed_count
    
    def detect_deleted_files(self):
        """삭제된 파일 탐지 (메모리 효율적)"""
        from colorama import Fore, Style
        
        logger.info("Starting deleted files detection with intelligent batch sizing")
        print(f"{Fore.CYAN}Scanning Milvus database for file paths (intelligent batch sizing)...{Style.RESET_ALL}")
        
        # 1. Milvus에서 모든 파일 경로 조회 (페이지네이션)
        db_files = set()
        # Use MilvusManager's intelligent batch sizing
        max_limit = self.milvus_manager._get_optimal_query_limit()
        logger.debug(f"Using optimal query limit of {max_limit} for batch queries")
        offset = 0
        total_db_files = 0
        
        try:
            logger.info("Starting pagination query of Milvus database for file paths")
            while True:
                logger.debug(f"Querying batch with offset {offset}, limit {max_limit}")
                results = self.milvus_manager.query(
                    expr="id >= 0",
                    output_fields=["path"],
                    limit=max_limit,
                    offset=offset
                )
                
                if not results:
                    logger.debug("No more results returned from query")
                    break
                    
                # Check for paths with special characters
                paths_with_special_chars = 0
                
                for doc in results:
                    path = doc.get("path")
                    if path and path not in db_files:
                        # Check for special characters that might cause issues
                        has_special_chars = any(c in path for c in "'\"()[]{},;")
                        if has_special_chars:
                            logger.debug(f"Found path with special characters: {path}")
                            paths_with_special_chars += 1
                            
                        db_files.add(path)
                        total_db_files += 1
                
                if paths_with_special_chars > 0:
                    logger.info(f"Batch contains {paths_with_special_chars} paths with special characters")
                        
                offset += max_limit
                if len(results) < max_limit:
                    logger.debug(f"Received {len(results)} results, which is less than limit {max_limit}. Pagination complete.")
                    break
                    
                # 진행상황 표시
                if total_db_files % 1000 == 0 and total_db_files > 0:
                    logger.info(f"Found {total_db_files} files in database so far")
                    print(f"{Fore.CYAN}Found {total_db_files} files in database so far...{Style.RESET_ALL}")
                
                # 메모리 관리
                if total_db_files % 5000 == 0:
                    logger.debug("Running garbage collection for memory management")
                    gc.collect()
                
        except Exception as e:
            logger.error(f"Error querying Milvus database: {e}", exc_info=True)
            print(f"{Fore.RED}Error querying Milvus database: {e}{Style.RESET_ALL}")
            return []
            
        logger.info(f"Found {len(db_files)} unique files in Milvus database")
        print(f"{Fore.GREEN}Found {len(db_files)} unique files in Milvus database{Style.RESET_ALL}")
        
        # 2. 현재 파일 시스템 스캔
        logger.info("Starting file system scan for comparison with database")
        print(f"{Fore.CYAN}Scanning file system...{Style.RESET_ALL}")
        fs_files = set()
        total_fs_files = 0
        special_char_files = 0
        
        try:
            for root, _, files in os.walk(self.vault_path):
                # 숨겨진 폴더 건너뛰기
                if os.path.basename(root).startswith(('.', '_')):
                    logger.debug(f"Skipping hidden directory: {root}")
                    continue
                    
                for file in files:
                    # 마크다운과 PDF만 처리
                    if file.endswith(('.md', '.pdf')) and not file.startswith('.'):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, self.vault_path)
                        
                        # Check for special characters that might cause issues
                        has_special_chars = any(c in rel_path for c in "'\"()[]{},;")
                        if has_special_chars:
                            logger.debug(f"Found file system path with special characters: {rel_path}")
                            special_char_files += 1
                            
                        fs_files.add(rel_path)
                        total_fs_files += 1
                        
                        # 진행상황 표시
                        if total_fs_files % 1000 == 0:
                            logger.info(f"Found {total_fs_files} files in file system so far")
                            print(f"{Fore.CYAN}Scanned {total_fs_files} files in file system...{Style.RESET_ALL}")
                            
                # Occasional garbage collection
                if total_fs_files % 10000 == 0 and total_fs_files > 0:
                    logger.debug("Running garbage collection during file system scan")
                    gc.collect()
                            
        except Exception as e:
            logger.error(f"Error scanning file system: {e}", exc_info=True)
            print(f"{Fore.RED}Error scanning file system: {e}{Style.RESET_ALL}")
            return []
            
        if special_char_files > 0:
            logger.info(f"Found {special_char_files} files with special characters in file system")
            
        logger.info(f"File system scan complete - found {total_fs_files} files")
        print(f"{Fore.GREEN}Found {len(fs_files)} files in file system{Style.RESET_ALL}")
        
        # 3. 삭제된 파일 찾기
        logger.info("Comparing database files with file system to identify deleted files")
        deleted_files = db_files - fs_files
        
        if deleted_files:
            logger.info(f"Found {len(deleted_files)} files that exist in database but not in file system")
            
            # Check for special characters in deleted files paths
            special_chars_in_deleted = [p for p in deleted_files if any(c in p for c in "'\"()[]{},;")]
            if special_chars_in_deleted:
                logger.info(f"Deleted files include {len(special_chars_in_deleted)} paths with special characters")
                logger.debug(f"Sample of deleted files with special characters: {special_chars_in_deleted[:5]}")
                
            print(f"{Fore.YELLOW}Found {len(deleted_files)} deleted files{Style.RESET_ALL}")
        else:
            logger.info("No deleted files found")
            print(f"{Fore.GREEN}No deleted files found{Style.RESET_ALL}")
        
        return list(deleted_files)
    
    def cleanup_deleted_embeddings(self, deleted_files):
        """삭제된 파일들의 embedding 제거"""
        from colorama import Fore, Style
        
        if not deleted_files:
            logger.info("No files to clean up")
            print(f"{Fore.GREEN}No files to clean up{Style.RESET_ALL}")
            return 0
        
        # Check for files with special characters that might need careful handling
        special_char_files = [p for p in deleted_files if any(c in p for c in "'\"()[]{},;")]
        if special_char_files:
            logger.info(f"Cleanup includes {len(special_char_files)} files with special characters")
            logger.debug(f"Sample of files with special characters: {special_char_files[:5]}")
            
        logger.info(f"Starting cleanup of {len(deleted_files)} deleted files")
        print(f"{Fore.CYAN}Starting cleanup of {len(deleted_files)} deleted files...{Style.RESET_ALL}")
        
        success_count = 0
        error_count = 0
        
        try:
            logger.debug("Marking files for deletion")
            # 배치 삭제를 위해 pending_deletions에 추가
            for file_path in deleted_files:
                try:
                    self.milvus_manager.mark_for_deletion(file_path)
                except Exception as e:
                    logger.warning(f"Error marking file for deletion: {file_path}: {e}")
            
            logger.info("Executing batch deletion of marked files")
            print(f"{Fore.CYAN}Executing batch deletion...{Style.RESET_ALL}")
            
            # 배치 삭제 실행
            self.milvus_manager.execute_pending_deletions()
            
            # 삭제 결과 확인
            logger.info("Verifying deletion results")
            print(f"{Fore.CYAN}Verifying deletion results...{Style.RESET_ALL}")
            
            # 삭제 후 검증
            remaining_files = []
            verification_batch_size = 100  # Smaller batch size for verification to prevent query issues
            
            # Process verification in smaller batches to avoid query issues with special characters
            for i in range(0, len(deleted_files), verification_batch_size):
                batch = deleted_files[i:i+verification_batch_size]
                logger.debug(f"Verifying deletion batch {i//verification_batch_size + 1}/{(len(deleted_files)+verification_batch_size-1)//verification_batch_size}")
                
                for file_path in batch:
                    try:
                        # Check for special characters that might cause issues with query expressions
                        has_special_chars = any(c in file_path for c in "'\"()[]{},;")
                        if has_special_chars:
                            logger.debug(f"Using safe query for file with special characters: {file_path}")
                            # Use a safer query approach for files with special characters
                            expr = self.milvus_manager._sanitize_query_expr(f"path == '{file_path}'")
                        else:
                            expr = f"path == '{file_path}'"
                            
                        # 파일이 여전히 DB에 있는지 확인
                        results = self.milvus_manager.query(
                            expr=expr,
                            output_fields=["path"],
                            limit=1
                        )
                        
                        if results:
                            logger.warning(f"File still exists after deletion attempt: {file_path}")
                            remaining_files.append(file_path)
                            error_count += 1
                        else:
                            logger.debug(f"Successfully deleted: {file_path}")
                            success_count += 1
                    except Exception as e:
                        logger.error(f"Error verifying deletion of {file_path}: {e}", exc_info=True)
                        print(f"{Fore.YELLOW}Warning: Could not verify deletion of {file_path}: {e}{Style.RESET_ALL}")
                        error_count += 1
            
            # 결과 보고
            logger.info(f"Cleanup results: {success_count} files successfully removed, {error_count} files failed")
            print(f"\n{Fore.GREEN}Cleanup Results:{Style.RESET_ALL}")
            print(f"{Fore.GREEN}✅ Successfully removed: {success_count} files{Style.RESET_ALL}")
            
            if error_count > 0:
                logger.warning(f"Failed to remove {error_count} files")
                print(f"{Fore.YELLOW}⚠️ Failed to remove: {error_count} files{Style.RESET_ALL}")
                
                if remaining_files:
                    # Log all remaining files at debug level, but show only a sample to the user
                    logger.debug(f"Files that could not be deleted: {remaining_files}")
                    
                    # Check for special characters in remaining files
                    special_chars_in_remaining = [p for p in remaining_files if any(c in p for c in "'\"()[]{},;")]
                    if special_chars_in_remaining:
                        logger.warning(f"{len(special_chars_in_remaining)} of the failed files contain special characters")
                    
                    print(f"{Fore.YELLOW}Files that could not be deleted:{Style.RESET_ALL}")
                    for file_path in remaining_files[:5]:  # 최대 5개만 표시
                        print(f"{Fore.YELLOW}  - {file_path}{Style.RESET_ALL}")
                    if len(remaining_files) > 5:
                        logger.debug(f"Additional failed files: {remaining_files[5:]}")
                        print(f"{Fore.YELLOW}  ... and {len(remaining_files) - 5} more{Style.RESET_ALL}")
            
            # 메모리 정리
            logger.debug("Running garbage collection after cleanup operation")
            gc.collect()
            
            logger.info(f"Cleanup operation completed: {success_count} files successfully removed")
            return success_count
            
        except Exception as e:
            logger.error(f"Critical error during cleanup operation: {e}", exc_info=True)
            print(f"{Fore.RED}Error during cleanup: {e}{Style.RESET_ALL}")
            import traceback
            print(f"{Fore.RED}Stack trace: {traceback.format_exc()}{Style.RESET_ALL}")
            return 0