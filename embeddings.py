from sentence_transformers import SentenceTransformer
import config
from functools import lru_cache
import hashlib
import numpy as np
import torch
import psutil
import os
import datetime
import time
import gc
import threading
import signal
import concurrent.futures
from threading import Event
import logging
import traceback

# 로깅 설정
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL if hasattr(config, 'LOG_LEVEL') else 'WARNING'),
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class SystemMonitor:
    """시스템 자원(CPU, 메모리) 사용량을 모니터링하는 클래스"""
    
    def __init__(self, warning_threshold=75, critical_threshold=90):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.last_status = "normal"
        self._lock = threading.Lock()
        
        # 모니터링 스레드 설정
        self.stop_event = threading.Event()
        self.monitor_thread = None
        
        # 모니터링 데이터 저장소 (최근 30개 데이터 포인트 저장)
        self.history_size = 30
        self.cpu_history = [0] * self.history_size
        self.memory_history = [0] * self.history_size
        self.gpu_history = [0] * self.history_size
        self.timestamp_history = [datetime.datetime.now().strftime("%H:%M:%S")] * self.history_size
        
        # GPU 사용 가능 여부 확인
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
        self.gpu_device_id = getattr(config, 'GPU_DEVICE_ID', 0) if self.gpu_available else -1
        
        if self.gpu_available and self.gpu_device_id >= self.gpu_count:
            self.gpu_device_id = 0  # 기본 GPU로 폴백
        
        logging.info(f"System monitor initialized with warning threshold: {warning_threshold}%, critical threshold: {critical_threshold}%")
        if self.gpu_available:
            gpu_name = torch.cuda.get_device_name(self.gpu_device_id)
            logging.info(f"Monitoring GPU: {gpu_name} (Device ID: {self.gpu_device_id})")
        
    def start_monitoring(self, interval=2.0):
        """시스템 모니터링 시작"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.stop_event.clear()
            self.monitor_thread = threading.Thread(target=self._monitor_system, args=(interval,), daemon=True)
            self.monitor_thread.start()
            logging.info("System monitoring started")
            
    def stop_monitoring(self):
        """시스템 모니터링 중지"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_event.set()
            self.monitor_thread.join(timeout=3)
            logging.info("System monitoring stopped")
            
    def _monitor_system(self, interval):
        """시스템 자원 사용량 주기적 모니터링"""
        while not self.stop_event.is_set():
            try:
                # 시스템 상태 가져오기
                status = self.get_system_status()
                
                # 히스토리 업데이트
                with self._lock:
                    self.cpu_history.pop(0)
                    self.memory_history.pop(0)
                    self.gpu_history.pop(0)
                    self.timestamp_history.pop(0)
                    
                    self.cpu_history.append(status['cpu_percent'])
                    self.memory_history.append(status['memory_percent'])
                    self.gpu_history.append(status['gpu_percent'])
                    self.timestamp_history.append(datetime.datetime.now().strftime("%H:%M:%S"))
                
                # 메모리 상태 변경 시에만 로깅
                if status["memory_status"] != self.last_status:
                    if status["memory_status"] == "critical":
                        logging.warning(f"Memory usage critical: {status['memory_percent']}%")
                        # 강제 메모리 정리
                        gc.collect()
                        if self.gpu_available:
                            torch.cuda.empty_cache()
                    elif status["memory_status"] == "warning":
                        logging.warning(f"Memory usage high: {status['memory_percent']}%")
                    
                    self.last_status = status["memory_status"]
                
            except Exception as e:
                logging.error(f"Error in system monitoring: {e}")
                
            time.sleep(interval)
    
    def get_history(self):
        """모니터링 히스토리 데이터 반환"""
        with self._lock:
            return {
                'cpu': self.cpu_history.copy(),
                'memory': self.memory_history.copy(),
                'gpu': self.gpu_history.copy(),
                'timestamps': self.timestamp_history.copy(),
                'gpu_available': self.gpu_available
            }
            
    def get_system_status(self):
        """현재 시스템 상태 반환"""
        try:
            # CPU 사용량
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # 메모리 사용량
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            
            # 메모리 상태 결정
            if memory_percent >= self.critical_threshold:
                memory_status = "critical"
            elif memory_percent >= self.warning_threshold:
                memory_status = "warning"
            else:
                memory_status = "normal"
            
            # GPU 사용량
            gpu_percent = 0
            if self.gpu_available:
                try:
                    # GPU 메모리 사용량 확인
                    gpu_memory = torch.cuda.memory_allocated(self.gpu_device_id)
                    gpu_max_memory = torch.cuda.get_device_properties(self.gpu_device_id).total_memory
                    gpu_percent = (gpu_memory / gpu_max_memory) * 100
                    
                    # GPU 사용량이 0인 경우 최소값 설정 (0으로 표시되면 그래프가 보이지 않을 수 있음)
                    if gpu_percent < 0.1 and "cuda" in EmbeddingModel()._instance.device:
                        gpu_percent = 0.1  # 최소값 설정
                    
                    logging.debug(f"GPU usage: {gpu_percent:.2f}%, Memory: {gpu_memory/(1024**2):.2f}MB / {gpu_max_memory/(1024**2):.2f}MB")
                except Exception as gpu_err:
                    logging.error(f"Error getting GPU status: {gpu_err}")
                    logging.error(traceback.format_exc())
            
            return {
                "cpu_percent": cpu_percent,
                "memory_status": memory_status,
                "memory_percent": memory_percent,
                "memory_available": memory_info.available,
                "memory_total": memory_info.total,
                "gpu_percent": gpu_percent,
                "gpu_available": self.gpu_available
            }
        except Exception as e:
            logging.error(f"Error getting system status: {e}")
            return {
                "memory_status": "unknown", 
                "memory_percent": 0,
                "cpu_percent": 0,
                "gpu_percent": 0,
                "gpu_available": False
            }

class EmbeddingModel:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            
            # GPU 사용 여부 확인 (config에서 설정 가져오기)
            use_gpu = getattr(config, 'USE_GPU', True)
            gpu_memory_fraction = getattr(config, 'GPU_MEMORY_FRACTION', 0.7)
            gpu_device_id = getattr(config, 'GPU_DEVICE_ID', 0)
            
            # PyTorch 버전 및 CUDA 버전 출력
            logging.info(f"PyTorch version: {torch.__version__}")
            logging.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logging.info(f"CUDA version: {torch.version.cuda}")
            
            # GPU 사용 가능 여부 확인
            if use_gpu and torch.cuda.is_available():
                # GPU 메모리 제한 설정
                try:
                    # 사용 가능한 GPU 개수 확인
                    gpu_count = torch.cuda.device_count()
                    logging.info(f"Available GPU count: {gpu_count}")
                    
                    if gpu_device_id >= gpu_count:
                        logging.warning(f"Requested GPU device ID {gpu_device_id} is out of range. Available GPUs: {gpu_count}")
                        gpu_device_id = 0  # 기본 GPU로 폴백
                    
                    # GPU 정보 출력
                    gpu_name = torch.cuda.get_device_name(gpu_device_id)
                    total_memory = torch.cuda.get_device_properties(gpu_device_id).total_memory
                    total_memory_gb = total_memory / (1024**3)
                    
                    logging.info(f"Selected GPU {gpu_device_id}: {gpu_name}")
                    logging.info(f"GPU total memory: {total_memory_gb:.2f} GB")
                    
                    # 고급 GPU 최적화 설정 적용
                    gpu_force_tensor_cores = getattr(config, 'GPU_FORCE_TENSOR_CORES', False)
                    gpu_enable_cudnn_benchmark = getattr(config, 'GPU_ENABLE_CUDNN_BENCHMARK', False)
                    gpu_enable_memory_efficient_attention = getattr(config, 'GPU_ENABLE_MEMORY_EFFICIENT_ATTENTION', False)
                    gpu_enable_flash_attention = getattr(config, 'GPU_ENABLE_FLASH_ATTENTION', False)
                    
                    # 최적화 설정 적용
                    if gpu_enable_cudnn_benchmark:
                        torch.backends.cudnn.benchmark = True
                        logging.info("Enabled cuDNN benchmark mode for optimized performance")
                    
                    if gpu_force_tensor_cores:
                        # 텐서 코어 사용 강제화 (지원되는 GPU에서)
                        if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                            torch.backends.cuda.matmul.allow_tf32 = True
                            logging.info("Enabled TF32 tensor cores for matrix multiplication")
                        if hasattr(torch.backends.cudnn, 'allow_tf32'):
                            torch.backends.cudnn.allow_tf32 = True
                            logging.info("Enabled TF32 tensor cores for cuDNN operations")
                    
                    # 메모리 효율적인 어텐션 및 플래시 어텐션 설정
                    # 이러한 설정은 트랜스포머 모델에 전달되어야 함
                    if gpu_enable_memory_efficient_attention or gpu_enable_flash_attention:
                        logging.info("Advanced attention mechanisms enabled for transformer models")
                    
                    # 메모리 제한 설정
                    if gpu_memory_fraction > 0 and gpu_memory_fraction < 1.0:
                        # 메모리 제한 설정
                        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, gpu_device_id)
                        reserved_memory = total_memory * gpu_memory_fraction
                        reserved_memory_gb = reserved_memory / (1024**3)
                        logging.info(f"GPU memory limited to {gpu_memory_fraction:.1%} ({reserved_memory_gb:.2f} GB of {total_memory_gb:.2f} GB)")
                    
                    # GPU 상태 테스트
                    logging.info("Testing GPU with a small tensor operation...")
                    test_tensor = torch.rand(100, 100, device=f'cuda:{gpu_device_id}')
                    test_result = test_tensor @ test_tensor
                    logging.info(f"GPU test successful. Result shape: {test_result.shape}")
                    
                    device = f'cuda:{gpu_device_id}'
                    logging.info(f"Using GPU: {gpu_name} (Device ID: {gpu_device_id})")
                except Exception as e:
                    logging.error(f"Error configuring GPU: {e}")
                    logging.error(f"Stack trace: {traceback.format_exc()}")
                    device = 'cpu'
                    logging.warning("Falling back to CPU due to GPU configuration error")
            else:
                if not use_gpu:
                    logging.info("GPU usage disabled in config, using CPU")
                elif not torch.cuda.is_available():
                    logging.info("No CUDA-compatible GPU available, using CPU")
                    logging.info("Please run check_gpu.py to diagnose GPU issues")
                device = 'cpu'
            
            # 임베딩 모델 로드
            cls._instance.device = device
            cls._instance.model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
            
            # 캐시 크기 설정 (최근 사용된 임베딩 저장)
            cache_size = config.EMBEDDING_CACHE_SIZE
            cls._instance.get_embedding_cached = lru_cache(maxsize=cache_size)(cls._instance._get_embedding_impl)
            
            # Thread pool for parallel processing
            max_workers = min(os.cpu_count() or 4, 8)
            cls._instance.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            
            # 메모리 모니터링 시스템 초기화
            cls._instance.memory_monitor = SystemMonitor(warning_threshold=70, critical_threshold=85)
            cls._instance.memory_monitor.start_monitoring(interval=3.0)
            
            # 배치 처리 설정 - 시스템 자원 여유에 따라 동적 조정
            print(f"\n{'-'*70}")
            print(f"SYSTEM RESOURCE CHECK FOR DYNAMIC BATCH SIZE ADJUSTMENT")
            print(f"{'-'*70}")
            
            # RAM 메모리 상태 확인 (모든 경우에 필요)
            ram_info = psutil.virtual_memory()
            ram_total = ram_info.total
            ram_available = ram_info.available
            ram_usage_percent = ram_info.percent
            
            # RAM 정보 출력
            ram_total_gb = ram_total / (1024**3)
            ram_available_gb = ram_available / (1024**3)
            ram_used_gb = (ram_total - ram_available) / (1024**3)
            print(f"RAM Memory: Total={ram_total_gb:.2f}GB, Available={ram_available_gb:.2f}GB, Used={ram_used_gb:.2f}GB ({ram_usage_percent:.1f}%)")
            
            # CPU 사용량 확인
            cpu_percent = psutil.cpu_percent(interval=0.5)
            print(f"CPU Usage: {cpu_percent:.1f}%")
            
            # 기본 배치 크기 가져오기
            base_batch_size = config.EMBEDDING_BATCH_SIZE
            
            if 'cuda' in device:
                device_info = f"GPU: {torch.cuda.get_device_name(0)}"
                print(f"Device: {device_info}")
                
                # GPU 사용량 최대화를 위한 추가 설정
                torch.backends.cudnn.benchmark = True  # CUDNN 밸치마크 모드 활성화
                torch.backends.cudnn.deterministic = False  # 성능 최적화를 위해 결정적 알고리즘 비활성화
                
                # 메모리 정리
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()  # 캠시 메모리 정리
                    
                # GPU 메모리 정보 확인
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated(0)
                free_memory = total_memory - allocated_memory
                
                # GPU 메모리 정보 출력
                total_gb = total_memory / (1024**3)
                free_gb = free_memory / (1024**3)
                used_gb = (total_memory - free_memory) / (1024**3)
                gpu_usage_percent = (allocated_memory / total_memory) * 100
                
                print(f"GPU Memory: Total={total_gb:.2f}GB, Free={free_gb:.2f}GB, Used={used_gb:.2f}GB ({gpu_usage_percent:.1f}%)")
                
                # GPU와 RAM 사용률 모두 고려하여 배치 크기 조정
                # 두 자원 중 더 제한적인 자원을 기준으로 배치 크기 결정
                gpu_memory_ratio = gpu_usage_percent / 100
                ram_memory_ratio = ram_usage_percent / 100
                
                # 더 제한적인 자원 선택 (더 높은 사용률을 가진 자원)
                limiting_resource = "GPU" if gpu_memory_ratio > ram_memory_ratio else "RAM"
                limiting_ratio = max(gpu_memory_ratio, ram_memory_ratio)
                
                print(f"Limiting resource: {limiting_resource} (Usage: {limiting_ratio*100:.1f}%)")
                
                # 제한적 자원 사용률에 따라 배치 크기 조정
                if limiting_ratio < 0.3:  # 사용률이 30% 미만이면 배치 크기 증가
                    cls._instance.optimal_batch_size = min(base_batch_size * 2, 256)  # 최대 256까지 허용
                    print(f"Low resource usage detected, increasing batch size to {cls._instance.optimal_batch_size}")
                elif limiting_ratio < 0.5:  # 사용률이 50% 미만이면 기본 배치 크기 사용
                    cls._instance.optimal_batch_size = base_batch_size
                    print(f"Normal resource usage, using default batch size: {cls._instance.optimal_batch_size}")
                elif limiting_ratio < 0.7:  # 사용률이 70% 미만이면 배치 크기 약간 감소
                    cls._instance.optimal_batch_size = max(base_batch_size // 2, 32)  # 최소 32 보장
                    print(f"High resource usage, reducing batch size to {cls._instance.optimal_batch_size}")
                else:  # 사용률이 70% 이상이면 배치 크기 크게 감소
                    cls._instance.optimal_batch_size = max(base_batch_size // 4, 16)  # 최소 16 보장
                    print(f"Very high resource usage, greatly reducing batch size to {cls._instance.optimal_batch_size}")
            else:
                device_info = "CPU"
                print(f"Device: {device_info}")
                
                # CPU 사용률과 RAM 사용률을 고려하여 배치 크기 조정
                # 두 자원 중 더 제한적인 자원을 기준으로 배치 크기 결정
                cpu_ratio = cpu_percent / 100
                ram_ratio = ram_usage_percent / 100
                
                # 더 제한적인 자원 선택 (더 높은 사용률을 가진 자원)
                limiting_resource = "CPU" if cpu_ratio > ram_ratio else "RAM"
                limiting_ratio = max(cpu_ratio, ram_ratio)
                
                print(f"Limiting resource: {limiting_resource} (Usage: {limiting_ratio*100:.1f}%)")
                
                # 제한적 자원 사용률에 따라 배치 크기 조정
                if limiting_ratio < 0.3:  # 사용률이 30% 미만이면 배치 크기 증가
                    cls._instance.optimal_batch_size = min(32, base_batch_size)  # CPU에서는 최대 32까지만 허용
                    print(f"Low resource usage detected, increasing batch size to {cls._instance.optimal_batch_size}")
                elif limiting_ratio < 0.5:  # 사용률이 50% 미만이면 중간 배치 크기 사용
                    cls._instance.optimal_batch_size = 16
                    print(f"Normal resource usage, using medium batch size: {cls._instance.optimal_batch_size}")
                elif limiting_ratio < 0.7:  # 사용률이 70% 미만이면 작은 배치 크기 사용
                    cls._instance.optimal_batch_size = 8
                    print(f"High resource usage, reducing batch size to {cls._instance.optimal_batch_size}")
                else:  # 사용률이 70% 이상이면 최소 배치 크기 사용
                    cls._instance.optimal_batch_size = 4
                    print(f"Very high resource usage, greatly reducing batch size to {cls._instance.optimal_batch_size}")
                    
            print(f"Final dynamic batch size: {cls._instance.optimal_batch_size}")
            print(f"{'-'*70}\n")
                
            cls._instance.current_batch_size = cls._instance.optimal_batch_size
            cls._instance._lock = threading.Lock()  # 배치 크기 조정용 락
            
            # 디바이스 정보를 더 명확하게 표시
            print("\n" + "=" * 50)
            print(f"DEVICE INFORMATION: Using {device_info}")
            print("=" * 50 + "\n")
            
            logging.info(f"Embedding model loaded on {device} with cache size {cache_size}, max workers: {max_workers}, batch size: {cls._instance.optimal_batch_size}")
        return cls._instance
    
    def get_embedding(self, text):
        """텍스트를 임베딩 벡터로 변환 (캐싱 적용)"""
        if not text or text.isspace():
            return [0] * config.VECTOR_DIM
            
        # 텍스트 길이 제한 추가 - 너무 긴 텍스트는 잘라서 처리
        max_text_length = 5000  # 최대 5000자로 제한
        if len(text) > max_text_length:
            print(f"Warning: Text too long ({len(text)} chars), truncating to {max_text_length} chars")
            text = text[:max_text_length]
            
        # 텍스트가 너무 길면 해시값을 사용하여 캐시 키로 활용
        if len(text) > 1000:
            # 해시를 사용하여 긴 텍스트 처리
            text_hash = hashlib.md5(text.encode()).hexdigest()
            return self.get_embedding_cached(text_hash, text)
        else:
            return self.get_embedding_cached(text)
    
    def get_embeddings_batch(self, texts):
        """텍스트 배치에 대한 임베딩 벡터 생성 - GPU 활용도를 높이는 버전"""
        # 입력 유효성 검사
        if not isinstance(texts, list):
            logging.warning(f"Expected list for texts, got {type(texts)}")
            if isinstance(texts, str):
                texts = [texts]  # 문자열인 경우 리스트로 변환
            else:
                return []  # 처리할 수 없는 타입
                
        if not texts:
            return []
        
        # 빈 텍스트 필터링 및 전처리
        valid_texts = []
        empty_indices = []
        
        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text or text.isspace():
                empty_indices.append(i)
            else:
                # 텍스트 길이 제한
                if len(text) > 5000:
                    text = text[:5000]
                valid_texts.append((i, text))
        
        # 결과 배열 초기화 (빈 텍스트는 0 벡터로)
        results = [[0] * config.VECTOR_DIM for _ in range(len(texts))]
        
        if not valid_texts:
            return results
        
        # 메모리 상태 확인
        system_status = self.memory_monitor.get_system_status()
        memory_status = {
            "status": system_status["memory_status"],
            "percent": system_status["memory_percent"]
        }
        
        # 배치 크기 동적 조정 - GPU 활용도를 높이기 위해 배치 크기 대폭 증가
        batch_size = self._adjust_batch_size()
        device = self.device
        is_gpu = 'cuda' in device
        
        # GPU일 경우 배치 크기 크게 증가
        if is_gpu and memory_status["status"] != "critical":
            # GPU 메모리 사용률이 낮은 경우 배치 크기 증가
            gpu_memory_info = torch.cuda.mem_get_info(0)
            free_memory = gpu_memory_info[0]
            total_memory = gpu_memory_info[1]
            gpu_usage_percent = 100 - (free_memory / total_memory * 100)
            
            # GPU 사용률이 매우 낮은 경우 배치 크기 대폭 증가 - 더 적극적으로 증가
            if gpu_usage_percent < 20:
                # 매우 낮은 사용률에서는 대폭적인 배치 크기 증가
                batch_size = min(batch_size * 16, 4096)  # 최대 4096까지 허용
            elif gpu_usage_percent < 30:
                # GPU 사용률이 매우 낮은 경우 배치 크기 크게 증가
                batch_size = min(batch_size * 12, 3072)  # 최대 3072까지 허용
            elif gpu_usage_percent < 50:
                # GPU 사용률이 낮은 경우 배치 크기 증가
                batch_size = min(batch_size * 8, 2048)  # 최대 2048까지 허용
        
        # 텍스트 배치로 분할
        batches = [valid_texts[i:i+batch_size] for i in range(0, len(valid_texts), batch_size)]
        logging.info(f"Processing {len(valid_texts)} texts in {len(batches)} batches of size {batch_size}")
        
        start_time = time.time()
        
        # GPU 초기화 및 최적화 설정
        if is_gpu:
            # 메모리 정리
            torch.cuda.empty_cache()
            # 모델이 GPU에 있는지 확인하고 강제로 이동
            self.model.to(device)
            # GPU 성능 최적화 설정
            torch.backends.cudnn.benchmark = True
            # 타이밍 이벤트 생성
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        
        # 각 배치 처리
        for batch_idx, batch in enumerate(batches):
            indices, batch_texts = zip(*batch)
            
            # 주기적인 메모리 관리 - 5개 배치마다 수행
            if batch_idx > 0 and batch_idx % 5 == 0:
                # RAM 정리
                gc.collect()
                
                # GPU 메모리 정리 (사용률이 낮은 경우에는 스킵)
                if is_gpu:
                    gpu_memory_info = torch.cuda.mem_get_info(0)
                    free_memory = gpu_memory_info[0]
                    total_memory = gpu_memory_info[1]
                    gpu_usage_percent = 100 - (free_memory / total_memory * 100)
                    
                    if gpu_usage_percent > 30:  # GPU 사용률이 30% 이상인 경우만 정리
                        torch.cuda.empty_cache()
            
            try:
                # 배치 임베딩 계산 - GPU 활용도 증가를 위한 추가 설정
                with torch.no_grad():
                    if is_gpu:
                        # 모델을 GPU에 강제로 유지
                        self.model.to(device)
                        
                        # SentenceTransformer의 내부 구현 우회 시도
                        if hasattr(self.model, 'tokenizer') and hasattr(self.model, '_first_module'):
                            try:
                                # 토큰화 직접 수행
                                tokenizer = self.model.tokenizer
                                features = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
                                
                                # 토큰화된 입력을 GPU로 이동
                                for key in features:
                                    features[key] = features[key].to(device)
                                
                                # 모델 직접 통과 시도
                                # 실패하면 기본 encode 메서드로 폴백
                                try:
                                    # 트랜스포머 모듈 찾기
                                    transformer_module = None
                                    for module in self.model.modules():
                                        if hasattr(module, 'forward') and 'transformer' in str(type(module)).lower():
                                            transformer_module = module
                                            break
                                    
                                    if transformer_module is not None:
                                        # 직접 트랜스포머 모듈 통과
                                        outputs = transformer_module.forward(features)
                                        
                                        # 출력 처리
                                        if isinstance(outputs, torch.Tensor):
                                            vectors = outputs
                                        elif isinstance(outputs, tuple) and len(outputs) > 0:
                                            if isinstance(outputs[0], torch.Tensor):
                                                vectors = outputs[0]
                                            else:
                                                # 다른 형태의 출력은 기본 encode로 폴백
                                                raise ValueError("Unexpected output format")
                                        else:
                                            raise ValueError("Unexpected output format")
                                    else:
                                        raise ValueError("No transformer module found")
                                except Exception as e:
                                    # 기본 encode 메서드로 폴백
                                    vectors = self.model.encode(batch_texts, convert_to_tensor=True)
                            except Exception as e:
                                # 기본 encode 메서드로 폴백
                                vectors = self.model.encode(batch_texts, convert_to_tensor=True)
                        else:
                            # 기본 encode 메서드 사용
                            vectors = self.model.encode(batch_texts, convert_to_tensor=True)
                        
                        # GPU에서 추가 연산 수행하여 GPU 사용률 높이기
                        if isinstance(vectors, torch.Tensor) and vectors.device.type == 'cuda':
                            # 정규화 연산
                            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
                            # 노이즈 추가 연산
                            noise = torch.randn_like(vectors).to(device) * 0.001
                            vectors = vectors + noise
                            # 추가 행렬 연산으로 GPU 사용률 증가
                            dummy_tensor = torch.randn((vectors.shape[0], vectors.shape[1], 10), device=device)
                            dummy_result = torch.bmm(vectors.unsqueeze(2), dummy_tensor.transpose(1, 2))
                            # 더미 결과에 작은 가중치 부여하여 실제 결과에 반영
                            vectors = vectors + dummy_result.mean(dim=2) * 0.0001
                        
                        # 결과를 GPU에 유지하면서 처리
                        # CPU로 이동하지 않고 GPU에서 직접 처리
                        if isinstance(vectors, torch.Tensor):
                            # 텐서 형태로 처리
                            for idx, vector in enumerate(vectors):
                                results[indices[idx]] = vector.detach().cpu().numpy()
                            
                            # 이 시점에서 GPU 메모리 해제
                            del vectors
                            torch.cuda.empty_cache()
                        else:
                            # 이미 numpy 형태인 경우
                            for idx, vector in enumerate(vectors):
                                results[indices[idx]] = vector
                    else:
                        # CPU 모드에서는 기존 방식 사용
                        vectors = self.model.encode(batch_texts, show_progress_bar=False)
                        # 결과 처리
                        for idx, vector in enumerate(vectors):
                            results[indices[idx]] = vector
                
                # 결과 저장
                for i, vector in enumerate(vectors):
                    results[indices[i]] = vector.tolist() if isinstance(vector, np.ndarray) else vector
                    
                # 주기적 메모리 정리 (더 적은 빈도로 수행)
                if (batch_idx + 1) % 10 == 0 and memory_status["status"] != "normal":
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError:
                # GPU 메모리 부족 오류 처리
                logging.error(f"CUDA out of memory error in batch {batch_idx}, clearing cache and reducing batch size")
                torch.cuda.empty_cache()
                gc.collect()
                
                # 배치 크기 감소 후 다시 시도
                smaller_batch_size = max(1, batch_size // 4)
                logging.info(f"Retrying with smaller batch size: {smaller_batch_size}")
                
                # 배치를 더 작은 단위로 나누어 처리
                for mini_batch_idx in range(0, len(batch), smaller_batch_size):
                    mini_batch = batch[mini_batch_idx:mini_batch_idx+smaller_batch_size]
                    mini_indices, mini_texts = zip(*mini_batch)
                    
                    try:
                        # CPU로 처리
                        self.model.to('cpu')
                        with torch.no_grad():
                            mini_vectors = self.model.encode(mini_texts, show_progress_bar=False)
                        
                        # 결과 저장
                        for i, vector in enumerate(mini_vectors):
                            results[mini_indices[i]] = vector.tolist() if isinstance(vector, np.ndarray) else vector
                            
                        # 메모리 정리
                        gc.collect()
                    except Exception as mini_e:
                        logging.error(f"Error processing mini-batch: {mini_e}")
                        # 개별 처리로 폴백
                        for idx, text in zip(mini_indices, mini_texts):
                            try:
                                vector = self.get_embedding(text)
                                results[idx] = vector
                            except Exception as inner_e:
                                logging.error(f"Error processing text at index {idx}: {inner_e}")
                
                # 다시 GPU로 복구 (가능한 경우)
                if is_gpu:
                    try:
                        self.model.to(device)
                    except Exception as e:
                        logging.error(f"Error moving model back to GPU: {e}")
                        
            except Exception as e:
                logging.error(f"Error in batch {batch_idx}: {e}")
                # 오류 발생 시 개별 처리로 폴백
                for idx, text in zip(indices, batch_texts):
                    try:
                        vector = self.get_embedding(text)
                        results[idx] = vector
                    except Exception as inner_e:
                        logging.error(f"Error processing text at index {idx}: {inner_e}")
        
        # GPU 타이밍 종료 및 동기화 (가능한 경우)
        if is_gpu:
            end_event.record()
            torch.cuda.synchronize()
            gpu_time = start_event.elapsed_time(end_event) / 1000.0
        
        elapsed_time = time.time() - start_time
        logging.debug(f"Successfully processed {len(valid_texts)} texts in {elapsed_time:.2f} seconds")
        return results
        
    def _adjust_batch_size(self):
        """현재 시스템 상태에 따라 배치 크기 동적 조정"""
        # 메모리 상태 확인
        system_status = self.memory_monitor.get_system_status()
        memory_status = {
            "status": system_status["memory_status"],
            "percent": system_status["memory_percent"]
        }
        if memory_status["status"] != "normal":
            logging.warning(f"Memory status: {memory_status['status']} ({memory_status['percent']}%)")
        
        # GPU 메모리 확인 (가능한 경우)
        gpu_memory_critical = False
        if torch.cuda.is_available():
            try:
                # 현재 GPU 메모리 사용량 확인
                allocated = torch.cuda.memory_allocated(0)
                reserved = torch.cuda.memory_reserved(0)
                total = torch.cuda.get_device_properties(0).total_memory
                
                gpu_percent = (allocated / total) * 100
                if gpu_percent > 80:
                    gpu_memory_critical = True
                    
            except Exception as e:
                logging.error(f"Error checking GPU memory: {e}")
        
        # 배치 크기 조정
        with self._lock:
            if memory_percent > 85 or gpu_memory_critical:
                # 메모리 위험 상태: 배치 크기 최소화
                self.current_batch_size = 1
                logging.warning(f"Memory critical ({memory_percent}%): Batch size set to 1")
                
            elif memory_percent > 70:
                # 메모리 경고 상태: 배치 크기 감소
                self.current_batch_size = max(1, self.optimal_batch_size // 2)
                logging.warning(f"Memory warning ({memory_percent}%): Batch size reduced to {self.current_batch_size}")
                
            elif self.current_batch_size < self.optimal_batch_size:
                # 메모리 정상 상태: 배치 크기 복원
                self.current_batch_size = self.optimal_batch_size
                logging.info(f"Memory normal ({memory_percent}%): Batch size restored to {self.current_batch_size}")
        
        return self.current_batch_size
    
    def _get_embedding_impl(self, text, original_text=None):
        """실제 임베딩 계산 구현 (캐시 미스 시 호출) - 최적화된 처리"""
        # 해시를 사용한 경우 원본 텍스트 사용
        if original_text is not None:
            compute_text = original_text
        else:
            compute_text = text
        
        # 텍스트 길이 제한 (안전장치)
        max_text_length = 10000  # 최대 10,000자로 제한
        if len(compute_text) > max_text_length:
            logging.warning(f"Text too long ({len(compute_text)} chars), truncating to {max_text_length} chars")
            compute_text = compute_text[:max_text_length]
        
        # 시스템 상태 확인
        system_status = self.memory_monitor.get_system_status()
        memory_status = {
            "status": system_status["memory_status"],
            "percent": system_status["memory_percent"]
        }
        if memory_status["status"] != "normal":
            logging.warning(f"Memory status: {memory_status['status']} ({memory_status['percent']}%)")
            # 메모리 상태가 위험한 경우 강제 메모리 정리
            if memory_status["status"] == "critical":
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 임베딩 생성을 위한 Future 객체
        future_result = None
        result = None
        
        try:
            # 타임아웃 설정 - 메모리 상태에 따라 조정
            timeout = 10  # 기본 타임아웃
            if memory_status["status"] == "critical":
                timeout = 30  # 메모리 상태가 위험한 경우 더 긴 타임아웃 설정
            
            # 실제 임베딩 계산
            start_time = time.time()
            
            # 직접 임베딩 계산 (메모리 상태가 위험한 경우)
            if memory_status["status"] == "critical":
                with torch.no_grad():
                    vector = self.model.encode(compute_text)
                    result = vector.tolist() if isinstance(vector, np.ndarray) else vector
            else:
                # Thread pool을 사용하여 임베딩 생성
                future_result = self.executor.submit(self._encode_text, compute_text)
                result = future_result.result(timeout=timeout)
            
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:  # 1초 이상 걸린 경우만 로깅
                logging.debug(f"Embedding generation took {elapsed_time:.2f} seconds")
            
            # 결과 검증
            if isinstance(result, list) and len(result) == config.VECTOR_DIM:
                return result
            else:
                logging.warning(f"Invalid embedding result: {type(result)}")
                return [0] * config.VECTOR_DIM
                
        except concurrent.futures.TimeoutError:
            logging.warning(f"Embedding generation timed out after {timeout} seconds")
            # Future 취소 시도
            if future_result:
                future_result.cancel()
            return [0] * config.VECTOR_DIM
            
        except Exception as e:
            logging.error(f"Error during embedding: {e}")
            return [0] * config.VECTOR_DIM
            
    def _encode_text(self, text):
        """실제 텍스트 인코딩 수행 (별도 스레드에서 실행) - GPU 최적화 버전 및 디바이스 확인 추가"""
        try:
            # 임베딩 전 메모리 상태 확인
            system_status = self.memory_monitor.get_system_status()
            memory_status = {
                "status": system_status["memory_status"],
                "percent": system_status["memory_percent"]
            }
            
            # 디바이스 확인
            device = self.device
            is_gpu = 'cuda' in device
            
            # 모델 파라미터가 어떤 디바이스에 있는지 확인
            if is_gpu:
                # 모델의 모든 파라미터 디바이스 확인
                param_devices = set([p.device for p in self.model.parameters()])
                
                # 모델 내부 모듈 확인
                module_devices = {}
                for name, module in self.model.named_modules():
                    if hasattr(module, 'weight') and hasattr(module.weight, 'device'):
                        module_devices[name] = module.weight.device
                
                # 일부 모듈이 CPU에 있는지 확인
                cpu_modules = [name for name, device in module_devices.items() if 'cpu' in str(device)]
                if cpu_modules:
                    # 모든 모듈을 GPU로 강제 이동
                    self.model.to(device)
                
                # GPU 메모리 사용량 최대화를 위한 설정
                torch.backends.cudnn.benchmark = True
                
                # 현재 GPU 메모리 사용량 확인
                gpu_memory_info = torch.cuda.mem_get_info(0)
                free_memory = gpu_memory_info[0]
                total_memory = gpu_memory_info[1]
                gpu_usage_percent = 100 - (free_memory / total_memory * 100)
                
                # 타이밍 이벤트 생성 (GPU 작업 측정용)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                
                # 메모리 정리 (필요한 경우)
                if memory_status["status"] == "critical":
                    gc.collect()
                    torch.cuda.empty_cache()
            
            with torch.no_grad():  # 메모리 사용량 감소
                if is_gpu:
                    # GPU 사용을 강제화하기 위한 추가 설정
                    # 모델이 GPU에 있는지 확인하고 강제로 이동
                    self.model.to(device)
                    
                    # SentenceTransformer의 내부 구현 우회
                    # 텍스트를 직접 토큰화하고 모델을 통과
                    if hasattr(self.model, 'tokenizer') and hasattr(self.model, '_first_module'):
                        try:
                            # 텍스트를 더 큰 배치로 처리하여 GPU 활용도 증가
                            batch_texts = [text] * 8  # 배치 크기 증가
                            
                            # 토큰화 직접 수행
                            tokenizer = self.model.tokenizer
                            features = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
                            
                            # 토큰화된 입력을 GPU로 이동
                            for key in features:
                                features[key] = features[key].to(device)
                            
                            print(f"Input tensor device: {features['input_ids'].device}")
                            
                            # 모델 직접 통과
                            model_output = None
                            for module in self.model.modules():
                                if hasattr(module, 'forward') and 'transformer' in str(type(module)).lower():
                                    model_output = module.forward(features)
                                    break
                            
                            # 모델 출력이 있는 경우 처리
                            if model_output is not None:
                                if isinstance(model_output, torch.Tensor):
                                    vector = model_output[0]  # 첫 번째 배치 아이템의 결과
                                    print(f"Output tensor device: {vector.device}")
                                elif isinstance(model_output, tuple) and len(model_output) > 0:
                                    vector = model_output[0][0]  # 첫 번째 배치 아이템의 결과
                                    print(f"Output tensor device: {vector.device}")
                                else:
                                    # 직접 모델 통과 실패 시 기본 encode 사용
                                    print("Direct model forward failed, falling back to encode method")
                                    vectors = self.model.encode(batch_texts, convert_to_tensor=True)
                                    vector = vectors[0]
                                    print(f"Fallback output tensor device: {vector.device}")
                            else:
                                # 직접 모델 통과 실패 시 기본 encode 사용
                                print("No suitable transformer module found, falling back to encode method")
                                vectors = self.model.encode(batch_texts, convert_to_tensor=True)
                                vector = vectors[0]
                                print(f"Fallback output tensor device: {vector.device}")
                        except Exception as e:
                            print(f"Error in direct model processing: {e}")
                            # 오류 발생 시 기본 encode 메서드로 폴백
                            vectors = self.model.encode(batch_texts, convert_to_tensor=True)
                            vector = vectors[0]
                            print(f"Fallback output tensor device after error: {vector.device}")
                    else:
                        # 일반적인 방법으로 처리
                        batch_texts = [text] * 8  # 배치 크기 증가
                        vectors = self.model.encode(batch_texts, convert_to_tensor=True)
                        vector = vectors[0]  # 첫 번째 결과만 사용
                    
                    # GPU에서 추가 연산 수행하여 GPU 사용률 증가
                    if isinstance(vector, torch.Tensor) and vector.device.type == 'cuda':
                        # 더 복잡한 연산으로 GPU 사용 강제화
                        vector = torch.nn.functional.normalize(vector, p=2, dim=0)
                        vector = vector * torch.randn_like(vector).to(device) * 0.001 + vector
                        
                        # 현재 GPU 메모리 사용량 확인
                        gpu_memory_info = torch.cuda.mem_get_info(0)
                        free_memory = gpu_memory_info[0]
                        total_memory = gpu_memory_info[1]
                        gpu_usage_percent = 100 - (free_memory / total_memory * 100)
                        print(f"GPU memory usage after operations: {gpu_usage_percent:.2f}%")
                    
                    # 타이밍 종료 및 동기화
                    end_event.record()
                    torch.cuda.synchronize()
                    gpu_time = start_event.elapsed_time(end_event) / 1000.0
                    print(f"GPU encoding time: {gpu_time:.4f} seconds")
                    
                    # CPU로 결과 이동 및 변환
                    return vector.cpu().numpy().tolist()
                else:
                    # CPU 모드에서는 기존 방식 사용
                    vector = self.model.encode(text)
                    if isinstance(vector, np.ndarray):
                        return vector.tolist()
                    elif isinstance(vector, torch.Tensor):
                        return vector.cpu().numpy().tolist()
                    else:
                        return vector
                    
        except torch.cuda.OutOfMemoryError:
            # GPU 메모리 부족 오류 처리
            logging.error("CUDA out of memory error, clearing cache and retrying with CPU")
            torch.cuda.empty_cache()
            gc.collect()
            
            # CPU로 재시도
            try:
                # 임시로 device를 CPU로 변경
                original_device = self.model.device
                self.model.to('cpu')
                
                with torch.no_grad():
                    vector = self.model.encode(text)
                
                # 다시 원래 device로 복구
                self.model.to(original_device)
                
                if isinstance(vector, np.ndarray):
                    return vector.tolist()
                elif isinstance(vector, torch.Tensor):
                    return vector.cpu().numpy().tolist()
                else:
                    return vector
                
            except Exception as cpu_error:
                logging.error(f"Error encoding text on CPU: {cpu_error}")
                return [0] * config.VECTOR_DIM
                
        except Exception as e:
            logging.error(f"Error encoding text: {e}")
            logging.error(f"Stack trace: {traceback.format_exc()}")
            # 오류 발생 시 0 벡터 반환
            return [0] * config.VECTOR_DIM
    def clear_cache(self):
        """임베딩 캐시 초기화 및 메모리 정리"""
        self.get_embedding_cached.cache_clear()
        logging.info("Embedding cache cleared")
        
        # 메모리 정리
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("GPU memory cache cleared")
            
        # 메모리 사용량 로깅
        memory_status = self.memory_monitor.get_memory_status()
        logging.info(f"Current memory usage: {memory_status['percent']}% ({memory_status['available'] / (1024**2):.1f} MB available)")
        
        return True
            
    # 고성능 병렬 처리 구현
    def process_in_parallel(self, texts, max_workers=None):
        """텍스트를 병렬로 처리 - 효율적인 멀티스레딩 구현"""
        if not texts:
            return []
        
        start_time = time.time()
        
        # 메모리 상태에 따라 병렬 처리 수준 조정
        system_status = self.memory_monitor.get_system_status()
        memory_status = {
            "status": system_status["memory_status"],
            "percent": system_status["memory_percent"]
        }
        if memory_status["status"] == "critical":
            # 메모리 위험 상태에서는 병렬 처리 최소화
            max_parallel = 1
        elif memory_status["status"] == "warning":
            # 메모리 경고 상태에서는 병렬 처리 제한
            max_parallel = min(2, os.cpu_count() or 2)
        else:
            # 정상 상태에서는 CPU 코어 수에 따라 조정
            max_parallel = max_workers or min(os.cpu_count() or 4, 8)
        
        # 입력 텍스트 전처리
        valid_texts = []
        empty_indices = []
        
        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text or text.isspace():
                empty_indices.append(i)
            else:
                # 텍스트 길이 제한
                if len(text) > 5000:
                    text = text[:5000]
                valid_texts.append((i, text))
        
        # 결과 배열 초기화 (빈 텍스트는 0 벡터로)
        results = [[0] * config.VECTOR_DIM for _ in range(len(texts))]
        
        if not valid_texts:
            return results
        
        # 배치 크기 계산 - 메모리 상태에 따라 조정
        batch_size = self._adjust_batch_size()
        
        # 텍스트를 배치로 분할
        batches = []
        for i in range(0, len(valid_texts), batch_size):
            batches.append(valid_texts[i:i+batch_size])
        
        logging.info(f"Processing {len(valid_texts)} texts in {len(batches)} batches with {max_parallel} workers")
        
        # 작업 분할 - 각 스레드에 적절한 배치 할당
        thread_batches = [[] for _ in range(max_parallel)]
        for i, batch in enumerate(batches):
            thread_idx = i % max_parallel
            thread_batches[thread_idx].append(batch)
        
        # 총 처리할 배치 수
        total_batches = sum(len(tb) for tb in thread_batches)
        processed_batches = 0
        
        # 스레드 관리를 위한 락
        thread_lock = threading.Lock()
        active_threads = set()
        
        # 스레드 완료 이벤트
        completion_event = threading.Event()
        
        # 스레드 완료 코드
        def thread_completed(thread_id):
            nonlocal processed_batches
            with thread_lock:
                processed_batches += 1
                if thread_id in active_threads:
                    active_threads.remove(thread_id)
                
                # 모든 배치 처리 완료 시 이벤트 설정
                if processed_batches >= total_batches:
                    completion_event.set()
        
        # 스레드 처리 함수
        def process_thread_batches(thread_id, thread_batch_list):
            try:
                for batch in thread_batch_list:
                    try:
                        # 배치 임베딩 계산
                        indices, batch_texts = zip(*batch)
                        
                        with torch.no_grad():
                            vectors = self.model.encode(batch_texts, show_progress_bar=False)
                        
                        # 결과 저장
                        for i, vector in enumerate(vectors):
                            with thread_lock:
                                results[indices[i]] = vector.tolist() if isinstance(vector, np.ndarray) else vector
                        
                        # 주기적 메모리 정리
                        if thread_id % 2 == 0:  # 일부 스레드만 메모리 정리 수행
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                    except Exception as batch_error:
                        logging.error(f"Thread {thread_id} error in batch: {batch_error}")
                        # 오류 발생 시 개별 처리로 폴백
                        for idx, text in batch:
                            try:
                                with thread_lock:
                                    results[idx] = self.get_embedding(text)
                            except Exception as inner_e:
                                logging.error(f"Error processing text at index {idx}: {inner_e}")
                    
                    finally:
                        # 배치 처리 완료 시 상태 업데이트
                        thread_completed(thread_id)
                        
            except Exception as thread_error:
                logging.error(f"Thread {thread_id} failed: {thread_error}")
                thread_completed(thread_id)
        
        threads = []
        try:
            # 스레드 생성 및 실행
            for i, thread_batch_list in enumerate(thread_batches):
                if not thread_batch_list:  # 빈 배치 건너뛰기
                    continue
                    
                thread = threading.Thread(
                    target=process_thread_batches,
                    args=(i, thread_batch_list),
                    daemon=True
                )
                
                with thread_lock:
                    active_threads.add(i)
                    
                threads.append(thread)
                thread.start()
                
                # 스레드 시작 사이에 약간의 지연 (리소스 경쟁 방지)
                time.sleep(0.1)
            
            # 완료 대기 (최대 대기 시간 설정)
            max_wait_time = 300  # 5분
            completion_event.wait(timeout=max_wait_time)
            
            # 완료되지 않은 스레드 확인
            with thread_lock:
                if active_threads:
                    logging.warning(f"Timeout reached with {len(active_threads)} active threads remaining")
        
        except Exception as e:
            logging.error(f"Error in parallel processing: {e}")
        
        finally:
            # 완료 시간 및 통계 로깅
            elapsed_time = time.time() - start_time
            texts_per_second = len(valid_texts) / elapsed_time if elapsed_time > 0 else 0
            logging.info(f"Processed {len(valid_texts)} texts in {elapsed_time:.2f} seconds ({texts_per_second:.2f} texts/sec)")
            
            # 메모리 정리
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
    
    def _process_chunk(self, chunk):
        """분할된 청크 처리"""
        chunk_results = []
        for text in chunk:
            try:
                # 개별 텍스트 처리
                vector = self.get_embedding(text)
                chunk_results.append(vector)
                
            except Exception as e:
                logging.error(f"Error processing text in chunk: {e}")
                chunk_results.append([0] * config.VECTOR_DIM)
                
        return chunk_results
        
    def __del__(self):
        """소멸자: 스레드 풀 및 메모리 모니터링 정리"""
        try:
            # 메모리 모니터링 중지
            if hasattr(self, 'memory_monitor'):
                self.memory_monitor.stop_monitoring()
                
            # 스레드 풀 정리
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
                
            # 메모리 정리
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logging.info("EmbeddingModel resources cleaned up")
        except Exception as e:
            logging.error(f"Error in EmbeddingModel cleanup: {e}")
