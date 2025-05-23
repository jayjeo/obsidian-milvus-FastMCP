from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, utility
import config
import subprocess
import time
import os
import socket
import json
import threading
import logging
import shutil

# 로깅 설정
log_level_str = getattr(config, 'LOG_LEVEL', 'INFO')
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MilvusManager')

class MilvusManager:
    def __init__(self):
        self.host = config.MILVUS_HOST
        self.port = config.MILVUS_PORT
        self.collection_name = config.COLLECTION_NAME
        self.dimension = getattr(config, 'VECTOR_DIM', 768)  # 벡터 차원 추가 (768로 기본값 변경)
        self.milvus_containers = {
            'standalone': 'milvus-standalone',
            'etcd': 'milvus-etcd',
            'minio': 'milvus-minio'
        }
        # 컨테이너 상태 모니터링 관련 설정
        self.monitoring_interval = 60  # 60초마다 체크 (필요에 따라 조정 가능)
        self.monitoring_active = True  # 모니터링 활성화 상태
        self.monitoring_thread = None  # 모니터링 스레드
        self.connection_lock = threading.Lock()  # 스레드 안전한 연결 관리를 위한 락
        
        # 초기 설정
        self.ensure_milvus_running()
        self.connect()
        self.ensure_collection()
        # 삭제 작업 배치 처리를 위한 추가 필드
        self.pending_deletions = set()
        
        # 모니터링 스레드 시작
        self.start_monitoring()
    
    def is_port_in_use(self, port):
        """포트가 사용 중인지 확인"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    def get_container_runtime_path(self):
        """Podman 실행 파일 경로 찾기"""
        # Podman 경로 확인
        if hasattr(config, 'PODMAN_PATH') and config.PODMAN_PATH:
            logger.debug(f"Using Podman path from config: {config.PODMAN_PATH}")
            # 경로가 유효한지 테스트
            try:
                result = subprocess.run(
                    [config.PODMAN_PATH, "--version"],
                    check=False,
                    text=True,
                    capture_output=True
                )
                if result.returncode == 0:
                    return config.PODMAN_PATH
                else:
                    logger.warning(f"Podman path in config is invalid: {config.PODMAN_PATH}")
            except Exception as e:
                logger.warning(f"Error testing Podman path from config: {e}")
        
        # 일반적인 Podman 설치 경로 목록
        possible_paths = [
            "podman",  # PATH에 있는 경우
            r"C:\Program Files\RedHat\Podman\podman.exe",
            r"C:\Users\%USERNAME%\AppData\Local\Programs\RedHat\Podman\podman.exe"
        ]
        
        # PATH에서 Podman 찾기
        podman_path = shutil.which("podman")
        if podman_path:
            logger.info(f"Found Podman in PATH: {podman_path}")
            return podman_path
            
        # 각 경로 확인
        for path in possible_paths:
            try:
                # %USERNAME% 환경 변수 처리
                if "%USERNAME%" in path:
                    path = path.replace("%USERNAME%", os.environ.get("USERNAME", ""))
                    
                # 테스트 명령 실행
                result = subprocess.run(
                    [path, "--version"],
                    check=False,
                    text=True,
                    capture_output=True
                )
                if result.returncode == 0:
                    logger.info(f"Found Podman at: {path}")
                    return path
            except FileNotFoundError:
                continue
        
        # Podman을 찾지 못한 경우
        logger.error("Podman not found in any of the expected locations")
        logger.error("Please install Podman or add it to your PATH")
        logger.error("Or specify the Podman path in config.py (PODMAN_PATH)")
        raise FileNotFoundError("Podman executable not found")
    
    def get_container_status(self):
        """모든 Milvus 관련 컨테이너의 상태를 확인"""
        container_status = {}
        
        try:
            # 컨테이너 런타임 경로 가져오기
            runtime_path = self.get_container_runtime_path()
            
            # 컨테이너 목록 가져오기
            result = subprocess.run(
                [runtime_path, "ps", "-a", "--format", "{{.Names}}:{{.Status}}"],
                check=True,
                text=True,
                capture_output=True
            )
            
            # 결과 파싱
            containers = result.stdout.strip().split('\n')
            for container in containers:
                if not container:
                    continue
                parts = container.split(':', 1)
                if len(parts) == 2:
                    name, status = parts
                    # 우리가 관심있는 컨테이너만 필터링
                    if name in self.milvus_containers.values():
                        # Podman의 상태 문자열 확인 ('up' 또는 'running')
                        container_status[name] = status.lower().startswith('up') or 'running' in status.lower()
            
            return container_status
        except subprocess.CalledProcessError as e:
            logger.error(f"Error checking container status: {e}")
            logger.error(f"Command error: {e.stderr}")
            return {}
    
    def start_container(self, container_name):
        """특정 컨테이너 시작"""
        try:
            # 컨테이너 런타임 경로 가져오기
            runtime_path = self.get_container_runtime_path()
            
            logger.info(f"Starting container: {container_name}")
            subprocess.run(
                [runtime_path, "start", container_name],
                check=True,
                text=True,
                capture_output=True
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error starting container {container_name}: {e}")
            logger.error(f"Command error: {e.stderr}")
            return False
    
    def ensure_external_storage_directories(self):
        """외부 저장소 디렉토리를 확인하고 필요한 경우 생성"""
        # 외부 저장소 경로 설정
        external_storage_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EmbeddingResult")
        logger.info(f"외부 저장소 경로: {external_storage_path}")
        
        # 디렉토리가 존재하지 않으면 생성
        if not os.path.exists(external_storage_path):
            logger.info(f"외부 저장소 디렉토리 생성: {external_storage_path}")
            os.makedirs(external_storage_path, exist_ok=True)
        
        # 하위 디렉토리 확인 및 생성
        subdirs = ["minio", "etcd"]
        for subdir in subdirs:
            subdir_path = os.path.join(external_storage_path, subdir)
            if not os.path.exists(subdir_path):
                logger.info(f"하위 디렉토리 생성: {subdir_path}")
                os.makedirs(subdir_path, exist_ok=True)
        
        return external_storage_path
    
    def ensure_milvus_running(self):
        """Milvus 서버와 모든 관련 컨테이너가 실행 중인지 확인하고, 실행 중이 아니면 시작"""
        # 외부 저장소 디렉토리 확인 및 생성
        self.ensure_external_storage_directories()
        
        # Podman 경로 가져오기
        runtime_path = self.get_container_runtime_path()
        runtime_type = "Podman"
        logger.info(f"Using {runtime_type} at: {runtime_path}")
        
        # 먼저 포트 확인으로 빠른 체크
        if self.is_port_in_use(self.port):
            logger.info(f"Milvus server is already running on port {self.port}")
            return True
            
        logger.info(f"Milvus server is not running on port {self.port}. Checking containers...")
        
        # 컨테이너 상태 확인
        container_status = self.get_container_status()
        all_running = True
        
        # 각 컨테이너 상태 확인 및 필요시 시작
        for container_type, container_name in self.milvus_containers.items():
            if container_name not in container_status or not container_status[container_name]:
                logger.info(f"{container_type} container ({container_name}) is not running")
                all_running = False
                
                # 개별 컨테이너 시작 시도
                if container_name in container_status:  # 컨테이너가 존재하지만 실행 중이 아님
                    if not self.start_container(container_name):
                        # 개별 시작에 실패하면 podman-compose로 전체 시작
                        all_running = False
                        break
        
        # 개별 시작이 실패했거나 일부 컨테이너가 없는 경우 podman-compose로 전체 시작
        if not all_running:
            logger.info("Some containers are missing or failed to start individually. Starting all with podman-compose...")
            
            # 프로젝트 루트 디렉토리 찾기
            project_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Podman compose 파일 선택
            compose_file = os.path.join(project_dir, 'milvus-podman-compose.yml')
            
            try:
                # Podman compose 설정
                compose_path = None
                # Podman Compose 경로 확인
                if hasattr(config, 'PODMAN_COMPOSE_PATH') and config.PODMAN_COMPOSE_PATH:
                    logger.info(f"Using Podman Compose path from config: {config.PODMAN_COMPOSE_PATH}")
                    try:
                        result = subprocess.run(
                            [config.PODMAN_COMPOSE_PATH, "--version"],
                            check=False,
                            text=True,
                            capture_output=True
                        )
                        if result.returncode == 0:
                            compose_path = config.PODMAN_COMPOSE_PATH
                        else:
                            logger.warning(f"Podman Compose path in config is invalid: {config.PODMAN_COMPOSE_PATH}")
                    except Exception as e:
                        logger.warning(f"Error testing Podman Compose path from config: {e}")
                
                # PATH에서 podman-compose 찾기
                if not compose_path:
                    podman_compose_path = shutil.which("podman-compose")
                    if podman_compose_path:
                        logger.info(f"Found podman-compose in PATH: {podman_compose_path}")
                        compose_path = podman_compose_path
                
                # podman-compose가 없으면 podman play kube 또는 podman-compose 명령 시도
                if not compose_path:
                    logger.info("podman-compose not found, using podman play kube or podman compose...")
                    # Podman의 경우 직접 podman 명령으로 compose 기능 사용
                    subprocess.run(
                        [runtime_path, "compose", "-f", compose_file, "up", "-d"],
                        check=True,
                        text=True,
                        capture_output=True
                    )
                else:
                    # podman-compose 명령 실행
                    subprocess.run(
                        [compose_path, "-f", compose_file, "up", "-d"],
                        check=True,
                        text=True,
                        capture_output=True
                    )
            except subprocess.CalledProcessError as e:
                logger.error(f"Error starting Milvus containers with Podman: {e}")
                logger.error(f"Command output: {e.stdout if hasattr(e, 'stdout') else ''}")
                logger.error(f"Command error: {e.stderr if hasattr(e, 'stderr') else ''}")
                raise RuntimeError(f"Failed to start Milvus containers. Please check Podman installation.")
        else:
            logger.info("All required containers are now running")
        
        # Milvus 서버가 완전히 시작될 때까지 대기
        max_retries = 30
        retry_interval = 2
        retries = 0
        
        while retries < max_retries:
            if self.is_port_in_use(self.port):
                logger.info(f"Milvus server is now running on port {self.port}")
                # 추가 대기 시간 (서비스가 완전히 초기화될 때까지)
                time.sleep(5)
                return True
                
            logger.info(f"Waiting for Milvus to start... (attempt {retries+1}/{max_retries})")
            time.sleep(retry_interval)
            retries += 1
            
        error_msg = "Timed out waiting for Milvus server to start"
        logger.error(error_msg)
        raise TimeoutError(error_msg)
    
    def connect(self):
        """Milvus 서버에 연결 (스레드 안전)"""
        with self.connection_lock:
            try:
                # Podman 경로 확인 - 없으면 예외 발생
                runtime_path = self.get_container_runtime_path()
                
                # 이미 연결되어 있는지 확인
                try:
                    if connections.has_connection():
                        logger.info("Already connected to Milvus server")
                        return
                except:
                    pass  # has_connection이 실패하면 연결이 없는 것으로 간주
                
                # 포트 확인
                if not self.is_port_in_use(self.port):
                    logger.error(f"Milvus server is not running on port {self.port}")
                    logger.error(f"Please make sure Podman is running and Milvus containers are started")
                    raise ConnectionError(f"Cannot connect to Milvus server: port {self.port} is not available")
                
                # 연결 시도
                connections.connect(host=self.host, port=self.port)
                logger.info(f"Connected to Milvus server at {self.host}:{self.port}")
            except Exception as e:
                logger.error(f"Error connecting to Milvus: {e}")
                raise  # 예외를 다시 발생시켜 Podman이 필수적으로 필요하도록 함
    
    def ensure_collection(self):
        """컨렉션이 없으면 생성"""
        # 컨테이너 런타임 경로 확인 - 없으면 예외 발생
        runtime_path = self.get_container_runtime_path()
        
        # 포트 확인
        if not self.is_port_in_use(self.port):
            logger.error(f"Milvus server is not running on port {self.port}")
            logger.error(f"Please make sure Podman is running and Milvus containers are started")
            raise ConnectionError(f"Cannot connect to Milvus server: port {self.port} is not available")
        
        # 컨렉션 확인 및 생성
        if not utility.has_collection(self.collection_name):
            self.create_collection()
        else:
            self.collection = Collection(self.collection_name)
            self.collection.load()
        
        logger.info(f"Collection '{self.collection_name}' is ready")
        
    def count_entities(self):
        """컨렉션에 있는 엔티티 수 가져오기"""
        try:
            # 컨렉션이 있는지 확인
            if not utility.has_collection(self.collection_name):
                return 0
                
            # 새로운 방법: num_entities 속성 사용
            if hasattr(self.collection, 'num_entities'):
                return self.collection.num_entities
            
            # 대안 방법: 간단한 쿼리로 카운트
            try:
                results = self.collection.query(
                    expr="id >= 0",
                    output_fields=["id"],
                    limit=16384  # Milvus 최대 제한
                )
                return len(results)
            except:
                # 최후의 수단: 컬렉션이 있으면 최소 0 반환
                return 0
                
        except Exception as e:
            logger.error(f"Error getting entity count: {e}")
            return 0
            
    def get_file_type_counts(self):
        """파일 타입별 문서 수 가져오기"""
        try:
            # 컨렉션이 있는지 확인
            if not utility.has_collection(self.collection_name):
                return {}
                
            # 새로운 방법: 전체 데이터를 가져와서 계산
            try:
                all_results = self.collection.query(
                    expr="id >= 0",
                    output_fields=["file_type"],
                    limit=16384
                )
                
                md_count = sum(1 for r in all_results if r.get('file_type', '').startswith('md'))
                pdf_count = sum(1 for r in all_results if r.get('file_type', '').startswith('pdf'))
                other_count = len(all_results) - md_count - pdf_count
                
                return {
                    "md": md_count,
                    "pdf": pdf_count,
                    "other": other_count,
                    "total": len(all_results)
                }
            except Exception as e:
                logger.warning(f"Could not get detailed file type counts: {e}")
                # 기본값 반환
                return {"md": 0, "pdf": 0, "other": 0, "total": 0}
        except Exception as e:
            logger.error(f"Error getting file type counts: {e}")
            return {"md": 0, "pdf": 0, "other": 0, "total": 0}
    
    def ensure_external_storage_directories(self):
        """외부 저장소 디렉토리가 존재하는지 확인하고 없으면 생성"""
        # 외부 저장소 경로 설정 (Podman 볼륨 대신 로컬 디렉토리 사용)
        external_storage_path = "G:/JJ Dropbox/J J/PythonWorks/milvus/obsidian-milvus-FastMCP/EmbeddingResult"
        
        # 필요한 하위 디렉토리 목록
        subdirs = ['etcd', 'minio', 'milvus']
        
        # 기본 디렉토리 확인 및 생성
        if not os.path.exists(external_storage_path):
            logger.info(f"Creating external storage directory: {external_storage_path}")
            os.makedirs(external_storage_path, exist_ok=True)
        
        # 하위 디렉토리 확인 및 생성
        for subdir in subdirs:
            subdir_path = os.path.join(external_storage_path, subdir)
            if not os.path.exists(subdir_path):
                logger.info(f"Creating subdirectory: {subdir_path}")
                os.makedirs(subdir_path, exist_ok=True)
        
        logger.info(f"External storage directories ready at: {external_storage_path}")
    
    def start_monitoring(self):
        """스레드 시작"""
        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            logger.info("Monitoring thread is already running")
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_containers, daemon=True)
        self.monitoring_thread.start()
        logger.info(f"Started Milvus container monitoring thread (interval: {self.monitoring_interval}s)")
        
    def _is_gpu_available(self):
        """시스템에 GPU가 사용 가능한지 확인"""
        try:
            # PyTorch를 통해 GPU 사용 가능 여부 확인
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_device_id = getattr(config, 'GPU_DEVICE_ID', 0)
                
                if gpu_device_id < gpu_count:
                    gpu_name = torch.cuda.get_device_name(gpu_device_id)
                    logger.info(f"GPU available: {gpu_name} (Device ID: {gpu_device_id})")
                    return True
                else:
                    logger.warning(f"Requested GPU device ID {gpu_device_id} is out of range. Available GPUs: {gpu_count}")
                    return False
            else:
                logger.info("No CUDA-compatible GPU available")
                return False
        except ImportError:
            logger.warning("PyTorch not installed, cannot check GPU availability")
            return False
        except Exception as e:
            logger.warning(f"Error checking GPU availability: {e}")
            return False
    
    def stop_monitoring(self):
        """Milvus 컨테이너 상태 모니터링 중지"""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
            logger.info("Stopped Milvus container monitoring")
    
    def _monitor_containers(self):
        """백그라운드에서 주기적으로 Milvus 컨테이너 상태 확인 및 복구"""
        logger.info("Milvus container monitoring started")
        
        while self.monitoring_active:
            try:
                # 서비스가 실행 중인지 확인
                if not self.is_port_in_use(self.port):
                    logger.warning(f"Milvus service on port {self.port} is not responding. Attempting to recover...")
                    
                    # 컨테이너 상태 확인 및 복구 시도
                    if self.ensure_milvus_running():
                        # 서비스가 복구되면 재연결 시도
                        try:
                            self.connect()
                            logger.info("Successfully reconnected to Milvus after recovery")
                        except Exception as e:
                            logger.error(f"Failed to reconnect to Milvus after recovery: {e}")
                else:
                    # 서비스는 실행 중이지만 컨테이너 상태 확인
                    container_status = self.get_container_status()
                    all_running = all(container_status.get(container, False) for container in self.milvus_containers.values())
                    
                    if not all_running:
                        missing_containers = [name for name, running in container_status.items() if not running]
                        logger.warning(f"Some Milvus containers are not running: {missing_containers}")
                        
                        # 포트는 열려있지만 일부 컨테이너가 실행되지 않는 경우 - 모니터링만 하고 자동 복구는 하지 않음
                        # 이 부분은 필요에 따라 자동 복구 로직을 추가할 수 있음
            except Exception as e:
                logger.error(f"Error in Milvus container monitoring: {e}")
            
            # 다음 체크까지 대기
            time.sleep(self.monitoring_interval)
        
        logger.info("Milvus container monitoring stopped")
    
    def create_collection(self, collection_name=None, dimension=None):
        """컬렉션 생성 (없는 경우)"""
        if collection_name is None:
            collection_name = self.collection_name
        if dimension is None:
            dimension = self.dimension
            
        # 컬렉션이 이미 존재하는지 확인
        if utility.has_collection(collection_name):
            print(f"Collection '{collection_name}' already exists.")
            return
        
        # 필드 정의
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="file_type", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="updated_at", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension)
        ]
        
        # 스키마 생성
        schema = CollectionSchema(fields=fields, description="Obsidian notes collection")
        
        # 컬렉션 생성
        collection = Collection(name=collection_name, schema=schema)
        self.collection = collection
        
        # GPU 인덱스 설정
        if config.USE_GPU:
            gpu_index_type = getattr(config, 'GPU_INDEX_TYPE', "GPU_IVF_FLAT")
            
            # GPU 인덱스 타입에 따른 파라미터 설정
            if gpu_index_type == "GPU_IVF_FLAT":
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "GPU_IVF_FLAT",
                    "params": {"nlist": 1024}
                }
            elif gpu_index_type == "GPU_IVF_PQ":
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "GPU_IVF_PQ",
                    "params": {"nlist": 1024, "m": 8, "nbits": 8}
                }
            elif gpu_index_type == "GPU_CAGRA":
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "GPU_CAGRA",
                    "params": {"search_width": 32, "build_algo": "IVF_PQ"}
                }
            else:  # 기본값: GPU_IVF_FLAT
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "GPU_IVF_FLAT",
                    "params": {"nlist": 1024}
                }
        else:  # CPU 인덱스 설정
            if not self._is_gpu_available():
                logger.warning("GPU not available, falling back to CPU index")
                
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64}
            }
        
        # 인덱스 생성
        collection.create_index(field_name="vector", index_params=index_params)
        
        # 컬렉션 로드
        collection.load()
        
        print(f"Collection '{collection_name}' created and loaded successfully.")
        
    def recreate_collection(self, collection_name=None, dimension=None):
        """컬렉션을 완전히 삭제하고 다시 생성"""
        if collection_name is None:
            collection_name = self.collection_name
        if dimension is None:
            dimension = self.dimension
            
        # 컬렉션이 존재하는지 확인
        if utility.has_collection(collection_name):
            # 컬렉션 삭제
            utility.drop_collection(collection_name)
            print(f"Collection '{collection_name}' has been dropped.")
        
        # 새 컬렉션 생성
        self.create_collection(collection_name, dimension)
        print(f"Collection '{collection_name}' has been recreated.")
        return True
            
    def _load_collection(self, use_gpu=True, index_params=None):
        """컬렉션을 로드하는 내부 메서드"""
        try:
            load_options = {}
            if use_gpu and self._is_gpu_available() and index_params and "GPU_" in index_params["index_type"]:
                gpu_device_id = getattr(config, 'GPU_DEVICE_ID', 0)
                load_options = {"device_id": gpu_device_id}
                logger.info(f"Loading collection with GPU device ID: {gpu_device_id}")
                
            self.collection.load(**load_options)
            logger.info(f"Collection '{self.collection_name}' loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            # 오류 발생 시 CPU 인덱스로 폴백
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64}
            }
            self.collection.create_index("vector", index_params)
            self.collection.load()
            logger.info(f"Fallback: Collection '{self.collection_name}' created with CPU index")
    
    def insert_data(self, data):
        """데이터를 Milvus에 삽입 (배치 처리 지원)"""
        try:
            # 데이터 형식 검증
            if not data or not isinstance(data, dict):
                print("Warning: Invalid data format for insert")
                return None
                
            # auto_id=True로 설정되어 있으므로 'id' 필드 제거
            if 'id' in data:
                data_copy = data.copy()
                del data_copy['id']
            else:
                data_copy = data
                
            # 대용량 데이터 처리를 위한 성능 최적화
            result = self.collection.insert(data_copy)
            # flush는 호출자가 관리 (배치마다 한 번만 호출)
            return result
        except Exception as e:
            print(f"Error inserting data: {e}")
            return None
    
    # 삭제 작업을 위한 메모리 효율적인 작업 관리
        
    def mark_for_deletion(self, file_path):
        """파일을 삭제 대기열에 추가 (메모리 효율성 개선)"""
        if file_path is None or not isinstance(file_path, str):
            return
        self.pending_deletions.add(file_path)
    
    def delete_all_data(self):
        """컬렉션의 모든 데이터 삭제 (전체 임베딩 시 사용)"""
        try:
            logger.info(f"Deleting all data from collection '{self.collection_name}'...")
            
            # 컬렉션이 존재하는지 확인
            if not utility.has_collection(self.collection_name):
                logger.warning(f"Collection '{self.collection_name}' does not exist. Nothing to delete.")
                return False
            
            # 컬렉션 로드 확인
            if not self.collection.is_loaded:
                self.collection.load()
            
            # 전체 데이터 삭제 (모든 엔티티 삭제)
            expr = "id >= 0"  # 모든 ID 선택
            delete_result = self.collection.delete(expr)
            
            # 삭제 결과 확인
            deleted_count = delete_result.delete_count
            logger.info(f"Successfully deleted {deleted_count} entities from collection '{self.collection_name}'")
            
            # 컬렉션 플러시하여 변경사항 적용
            self.collection.flush()
            logger.info(f"Collection '{self.collection_name}' flushed after deletion")
            
            return True
        except Exception as e:
            logger.error(f"Error deleting all data: {e}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return False
    
    def execute_pending_deletions(self):
        """삭제 대기열에 있는 모든 파일 삭제 (배치 처리)"""
        if not self.pending_deletions:
            return
            
        print(f"Executing batch deletion for {len(self.pending_deletions)} files...")
        
        try:
            # 수정: 페이지네이션을 사용하여 대용량 데이터 처리
            max_limit = 16000  # Milvus의 최대 한계보다 조금 작게 설정
            offset = 0
            all_results = []
            
            # 페이지네이션을 통한 데이터 가져오기
            while True:
                results = self.collection.query(
                    output_fields=["id", "path"],
                    limit=max_limit,
                    offset=offset,
                    expr="id >= 0"  # 모든 문서 조회
                )
                
                if not results:  # 결과가 없으면 중단
                    break
                    
                all_results.extend(results)
                offset += max_limit
                
                # 만약 결과가 한계보다 적으면 더 이상 없는 것으로 간주
                if len(results) < max_limit:
                    break
            
            # 삭제할 ID 목록 수집
            ids_to_delete = []
            deleted_files = set()
            
            for doc in all_results:
                path = doc.get("path")
                doc_id = doc.get("id")
                
                if path in self.pending_deletions and doc_id is not None:
                    ids_to_delete.append(doc_id)
                    deleted_files.add(path)
            
            # 삭제 실행
            if ids_to_delete:
                # 대용량 삭제 시 배치 사이즈 제한
                batch_size = 1000
                for i in range(0, len(ids_to_delete), batch_size):
                    batch = ids_to_delete[i:i+batch_size]
                    self.collection.delete(f"id in {batch}")
                
                print(f"Deleted {len(ids_to_delete)} chunks from {len(deleted_files)} files")
            
            # 삭제 대기열 초기화
            self.pending_deletions.clear()
            
        except Exception as e:
            print(f"Warning: Error in batch deletion: {e}")
    
    def delete_by_path(self, file_path):
        """파일 경로로 데이터 삭제 (레거시 지원)"""
        # 파일 경로가 None이면 바로 리턴
        if file_path is None:
            print("Warning: Attempted to delete with a None file path")
            return
            
        try:
            # 메모리 효율성 개선을 위해 필터링 최적화
            # path에 대한 직접 필터링 시도
            expr = f"path == '{file_path}'"
            count = self.collection.query(expr=expr, output_fields=["count(*)"]).get("count")
            
            if count and count > 0:
                self.collection.delete(expr)
                print(f"Deleted {count} chunks for file {file_path}")
                return
                
            # 직접 필터링이 실패한 경우 백업 방법 사용
            results = self.collection.query(
                output_fields=["id", "path"],
                limit=1000,
                expr="id >= 0"
            )
            
            ids_to_delete = [doc.get("id") for doc in results if doc.get("path") == file_path and doc.get("id") is not None]
            
            if ids_to_delete:
                self.collection.delete(f"id in {ids_to_delete}")
                print(f"Deleted {len(ids_to_delete)} chunks for file {file_path}")
            else:
                print(f"No documents found for path: {file_path}")
                
        except Exception as e:
            print(f"Warning: Error deleting file {file_path}: {e}")
            # 오류가 발생해도 계속 진행
    
    def search(self, vector, limit=5, filter_expr=None):
        """벡터 유사도 검색 수행 (GPU 검색 추가)"""
        # GPU 사용 여부 확인
        use_gpu = getattr(config, 'USE_GPU', False)
        gpu_index_type = getattr(config, 'GPU_INDEX_TYPE', 'GPU_IVF_FLAT')
        
        # 검색 파라미터 설정 - GPU 추가
        if use_gpu and self._is_gpu_available() and hasattr(self.collection, '_index_info') and self.collection._index_info:
            # 인덱스 정보에서 현재 인덱스 타입 확인
            index_info = self.collection._index_info.get('vector', {})
            index_type = index_info.get('index_type', '')
            
            if "GPU_" in index_type:
                logger.debug(f"Using GPU search parameters for index type: {index_type}")
                if index_type == "GPU_IVF_FLAT":
                    search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
                elif index_type == "GPU_CAGRA":
                    search_params = {"metric_type": "COSINE", "params": {"search_width": 64}}
                elif index_type == "GPU_BRUTE_FORCE":
                    search_params = {"metric_type": "COSINE", "params": {}}
                else:
                    # 기본 검색 파라미터
                    search_params = {"metric_type": "COSINE", "params": {"ef": 32}}
            else:
                # CPU 인덱스의 경우
                search_params = {"metric_type": "COSINE", "params": {"ef": 32}}
        else:
            # 기본 검색 파라미터
            search_params = {"metric_type": "COSINE", "params": {"ef": 32}}
        
        # 검색 인자 설정
        search_args = {
            "data": [vector],
            "anns_field": "vector",
            "param": search_params,
            "limit": limit,
            "output_fields": ["id", "path", "title", "content", "chunk_text", "tags", "file_type", "chunk_index"]
        }
        
        # 필터 표현식 추가 (있는 경우에만)
        if filter_expr is not None:
            search_args["expr"] = filter_expr
        
        # GPU 사용 여부에 따라 추가 옵션 설정
        if use_gpu and self._is_gpu_available():
            gpu_device_id = getattr(config, 'GPU_DEVICE_ID', 0)
            search_args["search_options"] = {"device_id": gpu_device_id}
            
        # 검색 수행
        results = self.collection.search(**search_args)
        return results[0]  # 첫 번째 쿼리의 결과
    
    def _sanitize_query_expr(self, expr):
        """쿼리 표현식에서 한글과 특수문자를 안전하게 처리"""
        import re
        
        # 이미 안전한 쿼리인 경우 그대로 반환
        if expr is None or expr == "id >= 0":
            return expr
            
        # 'path = 'value'' 또는 'path like 'value%'' 패턴 감지
        path_equals_pattern = re.compile(r"(path\s*=\s*['\"])(.*?)(['\"])")
        path_like_pattern = re.compile(r"(path\s+like\s+['\"])(.*?)(['\"])")
        title_equals_pattern = re.compile(r"(title\s*=\s*['\"])(.*?)(['\"])")
        title_like_pattern = re.compile(r"(title\s+like\s+['\"])(.*?)(['\"])")
        
        # 한글이나 특수문자가 포함된 경우 처리
        if re.search(r'[가-힣\(\)\s]', expr):
            # 'path = 'value'' 패턴 처리
            if path_equals_pattern.search(expr):
                logger.debug(f"한글/특수문자 포함된 path = 쿼리 감지: {expr}")
                # id >= 0으로 대체하고 후처리에서 필터링
                return "id >= 0"
                
            # 'path like 'value%'' 패턴 처리
            elif path_like_pattern.search(expr):
                match = path_like_pattern.search(expr)
                if match:
                    prefix = match.group(2).rstrip('%')
                    # 접두사에 한글이 포함된 경우
                    if re.search(r'[가-힣]', prefix):
                        logger.debug(f"한글 포함된 path like 쿼리 감지: {expr}")
                        # 접두사 검색은 지원하지만 한글이 포함된 경우 주의 필요
                        if prefix.endswith('%'):
                            # 와일드카드가 접두사 내에 있으면 id >= 0으로 대체
                            return "id >= 0"
                        else:
                            # 접두사 검색은 유지 (path like 'prefix%')
                            return expr
            
            # title 관련 쿼리도 유사하게 처리
            elif title_equals_pattern.search(expr) or title_like_pattern.search(expr):
                logger.debug(f"한글/특수문자 포함된 title 쿼리 감지: {expr}")
                return "id >= 0"  # id >= 0으로 대체하고 후처리에서 필터링
        
        # 변경이 필요 없는 경우 원래 표현식 반환
        return expr
    
    def _post_filter_results(self, results, original_expr):
        """쿼리 결과를 원래 표현식에 맞게 후처리"""
        import re
        
        # 원래 표현식이 없거나 결과가 없으면 그대로 반환
        if not original_expr or not results or original_expr == "id >= 0":
            return results
            
        # 'path = 'value'' 패턴 감지
        path_equals_pattern = re.compile(r"path\s*=\s*['\"](.+?)['\"]")
        path_match = path_equals_pattern.search(original_expr)
        
        if path_match:
            path_value = path_match.group(1)
            # 정확한 경로 일치 필터링
            filtered_results = [r for r in results if r.get('path') == path_value]
            logger.debug(f"경로 정확 일치 필터링: {len(filtered_results)}/{len(results)} 결과 남음")
            return filtered_results
            
        # 'title = 'value'' 패턴 감지
        title_equals_pattern = re.compile(r"title\s*=\s*['\"](.+?)['\"]")
        title_match = title_equals_pattern.search(original_expr)
        
        if title_match:
            title_value = title_match.group(1)
            # 정확한 제목 일치 필터링
            filtered_results = [r for r in results if r.get('title') == title_value]
            logger.debug(f"제목 정확 일치 필터링: {len(filtered_results)}/{len(results)} 결과 남음")
            return filtered_results
            
        # 'path like 'value%'' 패턴 감지
        path_like_pattern = re.compile(r"path\s+like\s+['\"](.+?)['\"]")
        path_like_match = path_like_pattern.search(original_expr)
        
        if path_like_match:
            like_value = path_like_match.group(1)
            if '%' in like_value:
                # 와일드카드 패턴 처리
                if like_value.endswith('%') and not like_value.startswith('%'):
                    # 접두사 패턴 (prefix%)
                    prefix = like_value.rstrip('%')
                    filtered_results = [r for r in results if r.get('path', '').startswith(prefix)]
                    logger.debug(f"경로 접두사 필터링: {len(filtered_results)}/{len(results)} 결과 남음")
                    return filtered_results
                elif like_value.startswith('%') and like_value.endswith('%'):
                    # 포함 패턴 (%substring%)
                    substring = like_value.strip('%')
                    filtered_results = [r for r in results if substring in r.get('path', '')]
                    logger.debug(f"경로 포함 필터링: {len(filtered_results)}/{len(results)} 결과 남음")
                    return filtered_results
        
        # 'title like 'value%'' 패턴 감지
        title_like_pattern = re.compile(r"title\s+like\s+['\"](.+?)['\"]")
        title_like_match = title_like_pattern.search(original_expr)
        
        if title_like_match:
            like_value = title_like_match.group(1)
            if '%' in like_value:
                # 와일드카드 패턴 처리
                if like_value.endswith('%') and not like_value.startswith('%'):
                    # 접두사 패턴 (prefix%)
                    prefix = like_value.rstrip('%')
                    filtered_results = [r for r in results if r.get('title', '').startswith(prefix)]
                    logger.debug(f"제목 접두사 필터링: {len(filtered_results)}/{len(results)} 결과 남음")
                    return filtered_results
                elif like_value.startswith('%') and like_value.endswith('%'):
                    # 포함 패턴 (%substring%)
                    substring = like_value.strip('%')
                    filtered_results = [r for r in results if substring in r.get('title', '')]
                    logger.debug(f"제목 포함 필터링: {len(filtered_results)}/{len(results)} 결과 남음")
                    return filtered_results
        
        # 다른 패턴은 원래 결과 반환
        return results
    
    def query(self, expr, output_fields=None, limit=100, offset=0):
        """Milvus에서 쿼리 실행 (페이지네이션 지원)
        한글과 특수문자가 포함된 쿼리를 안전하게 처리합니다.
        """
        if output_fields is None:
            # 스키마에 존재하는 필드만 요청
            output_fields = ["id", "path", "title", "content", "chunk_text", "chunk_index", "file_type", "tags", "created_at", "updated_at"]
        
        # Handle None expr case
        if expr is None:
            expr = "id >= 0"  # Default expression to match all documents
        
        # 원래 표현식 저장
        original_expr = expr
        
        # 한글과 특수문자 처리
        sanitized_expr = self._sanitize_query_expr(expr)
        logger.debug(f"원래 쿼리: {expr} -> 처리된 쿼리: {sanitized_expr}")
        
        # 쿼리 실행
        try:
            # 페이지네이션 지원을 위한 offset 추가
            results = self.collection.query(expr=sanitized_expr, output_fields=output_fields, limit=limit, offset=offset)
            
            # 원래 표현식과 처리된 표현식이 다른 경우 후처리 필요
            if sanitized_expr != original_expr:
                results = self._post_filter_results(results, original_expr)
                
            return results
        except Exception as e:
            logger.error(f"쿼리 실행 오류: {e}")
            # 오류 발생 시 기본 쿼리로 시도
            if sanitized_expr != "id >= 0":
                logger.info("기본 쿼리로 재시도 중...")
                results = self.collection.query(expr="id >= 0", output_fields=output_fields, limit=limit, offset=offset)
                return self._post_filter_results(results, original_expr)
            else:
                # 기본 쿼리도 실패하면 빈 결과 반환
                return []
        
    def check_file_exists(self, file_path):
        """파일 경로로 문서 존재 여부 확인 (안전한 방법)"""
        # 파일 경로가 None이면 바로 False 반환
        if file_path is None:
            return False
            
        # 전체 재색인 모드인 경우 항상 False 반환 (파일이 존재하지 않는 것으로 처리)
        # ObsidianProcessor 객체에서 embedding_progress 딕셔너리의 is_full_reindex 값을 확인
        import inspect
        for frame in inspect.stack():
            if 'self' in frame.frame.f_locals:
                instance = frame.frame.f_locals['self']
                if hasattr(instance, 'embedding_progress') and isinstance(instance.embedding_progress, dict) and instance.embedding_progress.get('is_full_reindex', False):
                    # 전체 재색인 모드에서는 파일이 존재하지 않는 것으로 처리 (새로 처리하도록)
                    return False
            
        try:
            # 방법 1: 직접 해당 경로를 쿼리하여 처리 속도 개선
            # 경로에 특수 문자가 있을 수 있으므로 안전하게 처리
            try:
                # 특수 문자 이스케이핑을 위해 따옴표 처리 
                escaped_path = file_path.replace("'", "\\'").replace("\\", "\\\\")
                expr = f"path == '{escaped_path}'"
                count_results = self.collection.query(
                    output_fields=["count(*) as count"],
                    limit=1,
                    expr=expr
                )
                
                # count 결과가 있으면 해당 파일이 존재
                if count_results and len(count_results) > 0 and count_results[0].get("count", 0) > 0:
                    return True
            except Exception as e:
                # 직접 쿼리가 실패하면 백업 방법 사용
                print(f"Direct query failed, using fallback method: {e}")
                
            # 방법 2: 페이지네이션을 사용한 백업 방법
            # 한 번에 처리할 수 있는 최대 문서 수
            max_limit = 1000
            offset = 0
            
            while True:
                results = self.collection.query(
                    output_fields=["id", "path"],
                    limit=max_limit,
                    offset=offset,
                    expr="id >= 0"  # 모든 문서 조회
                )
                
                if not results:  # 결과가 없으면 중단
                    break
                    
                # 결과에서 해당 경로와 일치하는 문서가 있는지 확인
                for doc in results:
                    if doc.get("path") == file_path:
                        return True
                        
                offset += max_limit
                # 만약 결과가 한계보다 적으면 더 이상 없는 것으로 간주
                if len(results) < max_limit:
                    break
                    
            return False
            
        except Exception as e:
            print(f"Warning: Error checking existing file {file_path}: {e}")
            return False
    
    # ==================== 고급 기능 패치 ====================
    
    def search_with_params(self, vector, limit=5, filter_expr=None, search_params=None):
        """HNSW 파라미터를 지원하는 고급 검색"""
        if search_params is None:
            # 기본 HNSW 최적화 파라미터
            if config.USE_GPU:
                search_params = {
                    "metric_type": "COSINE",
                    "params": {"nprobe": 16}  # GPU IVF 파라미터
                }
            else:
                search_params = {
                    "metric_type": "COSINE", 
                    "params": {"ef": 128}  # CPU HNSW 파라미터
                }
        
        try:
            search_args = {
                "data": [vector],
                "anns_field": "vector",
                "param": search_params,
                "limit": limit,
                "output_fields": ["id", "path", "title", "content", "chunk_text", "tags", "file_type", "chunk_index", "created_at", "updated_at"]
            }
            
            if filter_expr:
                search_args["expr"] = filter_expr
                
            # GPU 최적화
            if config.USE_GPU and self._is_gpu_available():
                gpu_device_id = getattr(config, 'GPU_DEVICE_ID', 0)
                search_args["search_options"] = {"device_id": gpu_device_id}
            
            results = self.collection.search(**search_args)
            return results[0] if results else []
            
        except Exception as e:
            logger.error(f"고급 검색 실패, 기본 검색으로 폴백: {e}")
            return self.search(vector, limit, filter_expr)
    
    def get_performance_stats(self):
        """성능 통계 수집"""
        try:
            stats = {
                'total_entities': self.count_entities(),
                'file_types': self.get_file_type_counts()
            }
            
            # 인덱스 정보
            try:
                indexes = self.collection.indexes
                if indexes:
                    index_info = indexes[0]
                    stats['index_type'] = index_info.params.get('index_type', 'Unknown')
                    stats['metric_type'] = index_info.params.get('metric_type', 'Unknown')
                else:
                    stats['index_type'] = 'No Index'
                    stats['metric_type'] = 'N/A'
            except Exception as e:
                stats['index_error'] = str(e)
            
            # 메모리 추정
            vector_size = self.dimension * 4  # float32
            estimated_mb = (stats['total_entities'] * vector_size) / (1024 * 1024)
            stats['estimated_memory_mb'] = round(estimated_mb, 2)
            
            # GPU 상태
            stats['gpu_available'] = self._is_gpu_available()
            stats['gpu_enabled'] = config.USE_GPU
            
            return stats
            
        except Exception as e:
            return {'error': str(e)}
    
    def benchmark_search_strategies(self, test_queries=3):
        """다양한 검색 전략 성능 벤치마크"""
        import time
        
        sample_vector = [0.1] * self.dimension
        
        strategies = {
            "fast": {"ef": 64, "nprobe": 8},
            "balanced": {"ef": 128, "nprobe": 16}, 
            "precise": {"ef": 256, "nprobe": 32}
        }
        
        results = {}
        
        for strategy_name, params in strategies.items():
            if config.USE_GPU:
                search_params = {
                    "metric_type": "COSINE",
                    "params": {"nprobe": params["nprobe"]}
                }
            else:
                search_params = {
                    "metric_type": "COSINE",
                    "params": {"ef": params["ef"]}
                }
            
            latencies = []
            for _ in range(test_queries):
                start_time = time.time()
                try:
                    self.search_with_params(sample_vector, limit=10, search_params=search_params)
                    latency = (time.time() - start_time) * 1000
                    latencies.append(latency)
                except Exception as e:
                    logger.error(f"벤치마크 {strategy_name} 실패: {e}")
                    latencies.append(float('inf'))  # 실패한 경우
            
            results[strategy_name] = {
                "avg_latency_ms": sum(latencies) / len(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "success_rate": len([l for l in latencies if l != float('inf')]) / len(latencies)
            }
        
        return results
    
    def advanced_metadata_search(self, query_vector, metadata_filters):
        """고급 메타데이터 필터링 검색"""
        try:
            # 메타데이터 필터를 Milvus 표현식으로 변환
            filter_expr = self._build_filter_expression(metadata_filters)
            
            # 고급 검색 파라미터 사용
            search_params = {"metric_type": "COSINE", "params": {"ef": 256}}
            
            return self.search_with_params(
                vector=query_vector,
                limit=metadata_filters.get('limit', 20),
                filter_expr=filter_expr,
                search_params=search_params
            )
            
        except Exception as e:
            logger.error(f"고급 메타데이터 검색 중 오류: {e}")
            return []
    
    def _build_filter_expression(self, metadata_filters):
        """메타데이터 필터를 Milvus 표현식으로 변환"""
        expressions = []
        
        # 시간 범위 필터
        if 'time_range' in metadata_filters:
            start_time, end_time = metadata_filters['time_range']
            expressions.append(f"created_at >= '{start_time}' and created_at <= '{end_time}'")
        
        # 파일 타입 필터
        if 'file_types' in metadata_filters:
            file_types = metadata_filters['file_types']
            if len(file_types) == 1:
                expressions.append(f"file_type == '{file_types[0]}'")
            else:
                type_expr = " or ".join([f"file_type == '{ft}'" for ft in file_types])
                expressions.append(f"({type_expr})")
        
        # 태그 필터 (간단한 문자열 포함 검색)
        if 'tags' in metadata_filters:
            tags = metadata_filters['tags']
            for tag in tags:
                expressions.append(f"tags like '%{tag}%'")
        
        # 모든 표현식을 AND로 결합
        if expressions:
            return " and ".join(expressions)
        else:
            return "id >= 0"  # 기본값
    
    def build_knowledge_graph(self, start_doc_id, max_depth=3, similarity_threshold=0.8):
        """벡터 유사도 기반 지식 그래프 구축"""
        
        graph = {"nodes": [], "edges": [], "clusters": {}}
        visited = set()
        
        # BFS로 지식 그래프 탐색
        queue = [(start_doc_id, 0)]  # (doc_id, depth)
        
        while queue and len(graph["nodes"]) < 100:  # 최대 100개 노드
            current_id, depth = queue.pop(0)
            
            if current_id in visited or depth > max_depth:
                continue
                
            visited.add(current_id)
            
            # 현재 문서 정보 가져오기
            doc_info = self.query(
                expr=f"id == {current_id}",
                output_fields=["id", "path", "title", "chunk_text"],
                limit=1
            )
            
            if not doc_info:
                continue
                
            doc = doc_info[0]
            
            # 노드 추가
            graph["nodes"].append({
                "id": current_id,
                "title": doc.get("title", ""),
                "path": doc.get("path", ""),
                "depth": depth
            })
            
            # 유사한 문서들 찾기
            if depth < max_depth:
                # 임베딩 모델이 있다면 사용, 없다면 스킵
                try:
                    from embeddings import EmbeddingModel
                    embedding_model = EmbeddingModel()
                    doc_vector = embedding_model.get_embedding(doc.get("chunk_text", ""))
                    
                    similar_docs = self.search_with_params(
                        vector=doc_vector,
                        limit=10,
                        search_params={"metric_type": "COSINE", "params": {"ef": 256}}
                    )
                    
                    for hit in similar_docs:
                        if (hit.score >= similarity_threshold and 
                            hit.id not in visited and
                            hit.id != current_id):
                            
                            # 엣지 추가
                            graph["edges"].append({
                                "source": current_id,
                                "target": hit.id,
                                "weight": float(hit.score),
                                "type": "semantic_similarity"
                            })
                            
                            # 다음 탐색 대상에 추가
                            queue.append((hit.id, depth + 1))
                except ImportError:
                    logger.warning("임베딩 모델을 사용할 수 없어 지식 그래프 구축을 중단합니다.")
                    break
                except Exception as e:
                    logger.error(f"지식 그래프 구축 중 오류: {e}")
                    continue
        
        return graph