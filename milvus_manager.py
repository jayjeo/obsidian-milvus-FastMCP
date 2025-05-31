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
import torch
import psutil

# Import centralized logger
from logger import get_logger

# Initialize module logger
logger = get_logger(__name__)


class SystemMonitor:
    """System monitor for tracking resource usage"""
    def __init__(self, *args, **kwargs):
        try:
            import torch
            self.gpu_available = torch.cuda.is_available()
        except ImportError:
            self.gpu_available = False
        
    def start_monitoring(self, *args, **kwargs):
        """Start system monitoring (placeholder)"""
        pass
        
    def stop_monitoring(self):
        """Stop system monitoring (placeholder)"""
        pass
        
    def get_system_status(self):
        """Get overall system status"""
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            return {
                "memory_status": "normal" if memory_info.percent < 80 else "high",
                "memory_percent": memory_info.percent,
                "cpu_percent": cpu_percent,
                "gpu_percent": 0,
                "gpu_available": self.gpu_available
            }
        except ImportError:
            # Fallback if psutil is not available
            return {
                "memory_status": "normal",
                "memory_percent": 50,
                "cpu_percent": 50,
                "gpu_percent": 0,
                "gpu_available": self.gpu_available
            }
        
    def get_memory_status(self):
        """Get memory-specific status information - THIS WAS MISSING"""
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            return {
                "memory_status": "normal" if memory_info.percent < 80 else "high",
                "memory_percent": memory_info.percent,
                "available_memory_gb": memory_info.available / (1024**3),
                "used_memory_gb": memory_info.used / (1024**3),
                "total_memory_gb": memory_info.total / (1024**3)
            }
        except ImportError:
            # Fallback if psutil is not available
            return {
                "memory_status": "normal",
                "memory_percent": 50,
                "available_memory_gb": 8.0,
                "used_memory_gb": 4.0,
                "total_memory_gb": 16.0
            }
        
    def get_cpu_status(self):
        """Get CPU-specific status information"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            return {
                "cpu_percent": cpu_percent,
                "cpu_cores": cpu_count,
                "cpu_temperature": 65
            }
        except ImportError:
            return {
                "cpu_percent": 50,
                "cpu_cores": 8,
                "cpu_temperature": 65
            }
        
    def get_gpu_status(self):
        """Get GPU-specific status information"""
        gpu_info = {
            "gpu_available": self.gpu_available,
            "gpu_percent": 0,
            "gpu_memory_used": 0,
            "gpu_memory_total": 0
        }
        
        if self.gpu_available:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_info["gpu_memory_used"] = torch.cuda.memory_allocated() / (1024**3)
                    gpu_info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except Exception:
                pass
                
        return gpu_info
        
    def get_history(self):
        """Get historical monitoring data"""
        return {
            'cpu': [50] * 30,
            'memory': [50] * 30,
            'gpu': [0] * 30,
            'timestamps': ['00:00:00'] * 30,
            'gpu_available': self.gpu_available
        }
class MilvusManager:
    def __init__(self):
        self.host = config.MILVUS_HOST
        self.port = config.MILVUS_PORT
        self.collection_name = config.COLLECTION_NAME
        self.dimension = getattr(config, 'VECTOR_DIM', 768)  # 벡터 차원 추가 (768로 기본값 변경)
        self.collection = None  # Initialize collection attribute
        
        # Import and initialize batch optimizer for intelligent query sizing
        try:
            from embeddings import HardwareProfiler, DynamicBatchOptimizer
            self.hardware_profiler = HardwareProfiler()
            self.batch_optimizer = DynamicBatchOptimizer(self.hardware_profiler)
            logger.info(f"Milvus using intelligent batch sizing: {self.batch_optimizer.current_batch_size}")
            print(f"Milvus using intelligent batch sizing: {self.batch_optimizer.current_batch_size}")
        except ImportError:
            # Fallback if embeddings not available
            self.batch_optimizer = None
            logger.warning("Failed to load embeddings module. Using fallback batch sizing")
            print("Using fallback batch sizing")
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
        
        # 삭제 작업 배치 처리를 위한 추가 필드
        self.pending_deletions = set()
        
        # 초기 설정 - 서비스가 이미 실행 중인 경우 스킵
        try:
            if not self.is_port_in_use(self.port):
                logger.info("Milvus is not running, starting services...")
                self.ensure_milvus_running()
            else:
                logger.info(f"Milvus is already running on port {self.port}")
                
            # Milvus 서비스가 완전히 준비될 때까지 대기
            self.wait_for_milvus_ready()
            
            self.connect()
            self.ensure_collection()
            
        except Exception as e:
            logger.error(f"Error during MilvusManager initialization: {e}")
            # 초기화 실패 시 사용자에게 더 명확한 안내 제공
            if "nodes not enough" in str(e):
                logger.error("Milvus services are still initializing. Please wait 1-2 minutes and try again.")
                logger.error("If the problem persists, try restarting Milvus with: start-milvus.bat")
            raise
        
        # 모니터링 스레드 시작
        self.start_monitoring()
        
    def _get_optimal_query_limit(self):
        """Get optimal query limit from DynamicBatchOptimizer or config fallback"""
        if self.batch_optimizer:
            # Use DynamicBatchOptimizer's intelligent sizing - apply Milvus limit from start
            optimal_limit = self.batch_optimizer.current_batch_size
            # Ensure it doesn't exceed Milvus safety limit from the start
            # Milvus 제한: offset + limit <= 16384이므로 안전하게 12000으로 설정
            milvus_limit = 12000  # Hard limit - leave room for offset
            return min(optimal_limit, milvus_limit)
        else:
            # Fallback to safe limit
            return 16000  # Safe fallback
        
    def connect(self):
        """Milvus 서버에 연결 (스레드 안전)"""
        with self.connection_lock:
            try:
                connections.connect(
                    alias="default", 
                    host=self.host, 
                    port=self.port
                )
                logger.info(f"Connected to Milvus server at {self.host}:{self.port}")
                return True
            except Exception as e:
                logger.error(f"Failed to connect to Milvus server: {e}")
                raise
    
    def ensure_collection(self):
        """컬렉션이 없으면 생성"""
        try:
            # 컬렉션 존재 여부 확인
            if utility.has_collection(self.collection_name):
                logger.info(f"Collection '{self.collection_name}' exists")
                # 기존 컬렉션 로드
                self.collection = Collection(self.collection_name)
                return True
            else:
                # 컬렉션 생성
                logger.info(f"Collection '{self.collection_name}' does not exist, creating it...")
                return self.create_collection(self.collection_name, self.dimension)
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise
            
    def create_collection(self, collection_name=None, dimension=None):
        """컬렉션 생성 (없는 경우)"""
        try:
            if collection_name is None:
                collection_name = self.collection_name
            if dimension is None:
                dimension = self.dimension
                
            # 필드 스키마 정의
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
            ]
            
            # 컬렉션 스키마 생성
            schema = CollectionSchema(fields=fields)
            
            # 컬렉션 생성
            logger.info(f"Creating collection: {collection_name} with dimension {dimension}")
            self.collection = Collection(name=collection_name, schema=schema)
            
            # 인덱스 생성
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "IP",
                "params": {"nlist": 1024}
            }
            
            # GPU 사용 가능 여부에 따라 인덱스 타입 조정
            if self._is_gpu_available() and hasattr(config, 'GPU_INDEX_TYPE'):
                index_params["index_type"] = config.GPU_INDEX_TYPE
                logger.info(f"Using GPU index type: {config.GPU_INDEX_TYPE}")
            
            self.collection.create_index("embedding", index_params)
            logger.info(f"Index created on 'embedding' field")
            
            # 컬렉션 로드
            self._load_collection()
            
            return True
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
            
    def _load_collection(self, use_gpu=True, index_params=None):
        """컬렉션을 로드하는 내부 메서드"""
        try:
            # GPU 사용 설정
            search_params = {}
            if use_gpu and self._is_gpu_available() and hasattr(config, 'GPU_DEVICE_ID'):
                search_params = {
                    "gpu_id": config.GPU_DEVICE_ID
                }
                logger.info(f"Loading collection with GPU (device_id: {config.GPU_DEVICE_ID})")
            else:
                logger.info("Loading collection with CPU")
                
            # 컬렉션 로드
            self.collection.load(search_params=search_params)
            logger.info(f"Collection '{self.collection_name}' loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading collection: {e}")
            raise
            
    def _is_gpu_available(self):
        """시스템에 GPU가 사용 가능한지 확인"""
        try:
            if not hasattr(config, 'USE_GPU') or not config.USE_GPU:
                return False
                
            # PyTorch를 통한 GPU 확인
            if torch.cuda.is_available():
                return True
            return False
        except Exception as e:
            logger.warning(f"GPU availability check failed: {e}")
            return False
    
    def wait_for_milvus_ready(self, max_wait_time=120):
        """
        Milvus 서비스가 완전히 준비될 때까지 대기
        'nodes not enough' 오류를 방지하기 위해 사용
        """
        logger.info("Waiting for Milvus services to be fully ready...")
        start_time = time.time()
        
        while (time.time() - start_time) < max_wait_time:
            try:
                # 포트가 열려있는지 확인
                if not self.is_port_in_use(self.port):
                    logger.debug("Port not ready yet, waiting...")
                    time.sleep(5)
                    continue
                
                # 간단한 연결 테스트
                temp_connection = None
                try:
                    from pymilvus import connections
                    temp_connection = connections.connect(
                        alias="temp_check",
                        host=self.host, 
                        port=self.port
                    )
                    
                    # 간단한 작업 수행으로 준비 상태 확인
                    from pymilvus import utility
                    # 컴렉션 목록 가져오기 시도 (Milvus가 준비되었는지 확인)
                    collections = utility.list_collections(using="temp_check")
                    
                    logger.info("Milvus services are ready!")
                    return True
                    
                except Exception as e:
                    if "nodes not enough" in str(e):
                        elapsed = int(time.time() - start_time)
                        logger.info(f"Milvus still initializing... ({elapsed}s/{max_wait_time}s)")
                        time.sleep(10)
                        continue
                    else:
                        # 다른 종류의 오류는 바로 다시 시도
                        time.sleep(5)
                        continue
                finally:
                    # 임시 연결 정리
                    try:
                        if temp_connection:
                            connections.disconnect("temp_check")
                    except:
                        pass
                        
            except Exception as e:
                logger.debug(f"Wait check failed: {e}")
                time.sleep(5)
                continue
        
        # 타임아웃
        logger.warning(f"Milvus readiness check timed out after {max_wait_time} seconds")
        logger.warning("Proceeding anyway - you may encounter 'nodes not enough' errors")
        return False

    def is_port_in_use(self, port):
        """포트가 사용 중인지 확인"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    def get_container_runtime_path(self):
        """Podman 실행 파일 경로 찾기"""
        # config에서 Podman 경로 가져오기 (자동 탐지 포함)
        try:
            return config.get_podman_path()
        except FileNotFoundError:
            logger.error("Podman not found. Please install Podman or set PODMAN_PATH in config.")
            raise
        
        # 일반적인 Podman 설치 경로 목록 - 이 부분은 config.py의 find_podman_path() 함수에서 처리됨
        # 따라서 이미 config에서 처리된 경로를 사용
        return None
    
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
                        # Podman의 상태 문자열 확인 - "Up" 으로 시작하는 경우 실행 중
                        is_running = status.strip().lower().startswith('up')
                        container_status[name] = is_running
                        logger.debug(f"Container {name}: status='{status}' -> running={is_running}")
            
            return container_status
        except subprocess.CalledProcessError as e:
            logger.error(f"Error checking container status: {e}")
            logger.error(f"Command error: {e.stderr}")
            return {}
    
    def create_missing_container(self, container_type, container_name):
        """누락된 컨테이너를 개별적으로 생성"""
        try:
            runtime_path = self.get_container_runtime_path()
            project_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 네트워크 확인 및 생성
            network_name = "milvus"
            self.ensure_network_exists(network_name)
            
            if container_type == "etcd":
                # Create the volumes/etcd directory if it doesn't exist
                etcd_dir = os.path.join(project_dir, "volumes", "etcd")
                os.makedirs(etcd_dir, exist_ok=True)
                
                # Use a relative path instead of absolute path
                # This is equivalent to "./volumes/etcd:/etcd"
                cmd = [
                    runtime_path, "run", "-d", "--name", container_name,
                    "--network", network_name,
                    "-v", "./volumes/etcd:/etcd",  # Use relative path
                    "-e", "ETCD_AUTO_COMPACTION_MODE=revision",
                    "-e", "ETCD_AUTO_COMPACTION_RETENTION=1000",
                    "-e", "ETCD_QUOTA_BACKEND_BYTES=4294967296",
                    "--user", "0:0",
                    "quay.io/coreos/etcd:v3.5.0",
                    "etcd", "-advertise-client-urls=http://127.0.0.1:2379",
                    "-listen-client-urls", "http://0.0.0.0:2379", "--data-dir", "/etcd"
                ]
            elif container_type == "minio":
                # Create the minio directory if it doesn't exist
                minio_dir = os.path.join(project_dir, "volumes", "minio")
                os.makedirs(minio_dir, exist_ok=True)
                
                cmd = [
                    runtime_path, "run", "-d", "--name", container_name,
                    "--network", network_name,
                    "-v", "./volumes/minio:/minio_data",  # Use relative path
                    "-e", "MINIO_ACCESS_KEY=minioadmin",
                    "-e", "MINIO_SECRET_KEY=minioadmin",
                    "--user", "0:0",
                    "minio/minio:RELEASE.2023-03-20T20-16-18Z",
                    "server", "/minio_data"
                ]
            elif container_type == "standalone":
                # Create the milvus data directory if it doesn't exist
                milvus_dir = os.path.join(project_dir, "volumes", "milvus")
                os.makedirs(milvus_dir, exist_ok=True)
                
                cmd = [
                    runtime_path, "run", "-d", "--name", container_name,
                    "--network", network_name,
                    "-p", "19530:19530", "-p", "9091:9091",
                    "-v", "./volumes/milvus:/var/lib/milvus",  # Use relative path
                    "-e", "ETCD_ENDPOINTS=milvus-etcd:2379",
                    "-e", "MINIO_ADDRESS=milvus-minio:9000",
                    "--user", "0:0",
                    "milvusdb/milvus:v2.3.4",
                    "milvus", "run", "standalone"
                ]
            else:
                logger.error(f"Unknown container type: {container_type}")
                return False
            
            # 컨테이너 생성
            logger.info(f"Creating {container_type} container: {container_name}")
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)
            logger.info(f"Successfully created {container_type} container")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating {container_type} container: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                logger.error(f"Command error: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating {container_type} container: {e}")
            return False
    
    def ensure_network_exists(self, network_name):
        """네트워크가 존재하는지 확인하고 없으면 생성"""
        try:
            runtime_path = self.get_container_runtime_path()
            
            # 네트워크 존재 확인
            result = subprocess.run(
                [runtime_path, "network", "exists", network_name],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                # 네트워크 생성
                logger.info(f"Creating network: {network_name}")
                result = subprocess.run(
                    [runtime_path, "network", "create", network_name],
                    check=True, text=True, capture_output=True
                )
                logger.info(f"Successfully created network: {network_name}")
            else:
                logger.debug(f"Network {network_name} already exists")
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"Network operation failed for {network_name}: {e}")
            # 네트워크 오류는 중요하지 않을 수 있으므로 계속 진행
        except Exception as e:
            logger.warning(f"Unexpected network error: {e}")
    
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
            # 컨테이너가 존재하지 않는 경우는 정상적인 상황 (reset 후 등)
            if "no container" in str(e).lower() or "not found" in str(e).lower():
                logger.info(f"Container {container_name} not found (expected after reset) - will create new one")
            else:
                logger.warning(f"Unexpected error starting container {container_name}: {e}")
                if hasattr(e, 'stderr') and e.stderr:
                    logger.warning(f"Command error: {e.stderr}")
            return False
    
    def _cleanup_conflicting_containers(self):
        """충돌하는 컨테이너들을 자동으로 정리 (강화된 버전)"""
        try:
            runtime_path = self.get_container_runtime_path()
            
            logger.info("Cleaning up conflicting containers...")
            
            # 1단계: 모든 Milvus 관련 컨테이너 강제 중지 및 제거
            for container_name in self.milvus_containers.values():
                try:
                    # 컨테이너 강제 중지
                    subprocess.run(
                        [runtime_path, "stop", container_name, "--time", "0"],
                        check=False,  # 오류 무시
                        text=True,
                        capture_output=True
                    )
                    
                    # 컨테이너 강제 제거
                    subprocess.run(
                        [runtime_path, "rm", container_name, "--force"],
                        check=False,  # 오류 무시
                        text=True,
                        capture_output=True
                    )
                    logger.debug(f"Cleaned up container: {container_name}")
                except Exception as e:
                    logger.debug(f"Error cleaning container {container_name}: {e}")
            
            # 2단계: 모든 Pod 강제 정리
            try:
                subprocess.run(
                    [runtime_path, "pod", "stop", "--all", "--time", "0"],
                    check=False,
                    text=True,
                    capture_output=True
                )
                subprocess.run(
                    [runtime_path, "pod", "rm", "--all", "--force"],
                    check=False,
                    text=True,
                    capture_output=True
                )
            except Exception as e:
                logger.debug(f"Error cleaning pods: {e}")
            
            # 3단계: Milvus 네트워크 정리
            try:
                subprocess.run(
                    [runtime_path, "network", "rm", "milvus", "--force"],
                    check=False,
                    text=True,
                    capture_output=True
                )
            except Exception as e:
                logger.debug(f"Error cleaning network: {e}")
            
            # 4단계: 시스템 정리
            try:
                subprocess.run(
                    [runtime_path, "system", "prune", "--force"],
                    check=False,
                    text=True,
                    capture_output=True
                )
            except Exception as e:
                logger.debug(f"Error in system prune: {e}")
                
            logger.info("Container cleanup completed")
            time.sleep(3)  # 정리 완료를 위한 충분한 대기시간
            
        except Exception as e:
            logger.warning(f"Error during container cleanup: {e}")
    
    # Implementation moved to line ~687
    
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
            # 포트가 열려있으면 이미 실행 중이므로 추가 작업 없이 반환
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
                    logger.info(f"Attempting to start existing container: {container_name}")
                    if self.start_container(container_name):
                        logger.info(f"Successfully started container: {container_name}")
                    else:
                        # 컨테이너가 존재하지 않는 경우 새로 생성 시도
                        logger.info(f"Container {container_name} not available, attempting to recreate...")
                        if self.create_missing_container(container_type, container_name):
                            logger.info(f"Successfully recreated {container_type} container")
                        else:
                            logger.error(f"Failed to recreate {container_type} container")
                            all_running = False
                            break
                else:
                    logger.info(f"Container {container_name} does not exist (normal after reset), creating it...")
                    # 컨테이너가 없으므로 새로 생성
                    if self.create_missing_container(container_type, container_name):
                        logger.info(f"Successfully created {container_type} container")
                    else:
                        logger.error(f"Failed to create {container_type} container")
                        all_running = False
                        break
        
        # 개별 시작이 실패했거나 일부 컨테이너가 없는 경우에만 podman-compose로 전체 시작
        if not all_running:
            logger.info("Some containers are missing or failed to start individually. Starting all with podman-compose...")
            
            # 시작하기 전에 충돌 가능성 미리 제거
            logger.info("Pre-emptively cleaning up any existing containers to avoid conflicts...")
            self._cleanup_conflicting_containers()
            
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
                error_msg = f"Error starting Milvus containers with Podman: {e}"
                if hasattr(e, 'stdout') and e.stdout:
                    error_msg += f"\nCommand output: {e.stdout}"
                if hasattr(e, 'stderr') and e.stderr:
                    error_msg += f"\nCommand error: {e.stderr}"
                
                logger.error(error_msg)
                
                # 컨테이너 이름 충돌 감지 개선 - stderr와 전체 에러 메시지에서 검색
                error_text = str(e) + (e.stderr if hasattr(e, 'stderr') and e.stderr else "")
                if "already in use" in error_text or "name is already in use" in error_text:
                    logger.warning("Container name conflict detected. Cleaning up conflicting containers...")
                    # 자동으로 충돌하는 컨테이너 제거 시도
                    self._cleanup_conflicting_containers()
                    # 다시 시도
                    try:
                        logger.info("Retrying container startup after cleanup...")
                        if not compose_path:
                            subprocess.run(
                                [runtime_path, "compose", "-f", compose_file, "up", "-d"],
                                check=True,
                                text=True,
                                capture_output=True
                            )
                        else:
                            subprocess.run(
                                [compose_path, "-f", compose_file, "up", "-d"],
                                check=True,
                                text=True,
                                capture_output=True
                            )
                        logger.info("Successfully started containers after conflict resolution")
                    except Exception as retry_e:
                        logger.error(f"Failed to start containers even after cleanup: {retry_e}")
                        logger.error("Please restart the application or run emergency-reset.bat")
                        raise RuntimeError(f"Failed to start Milvus containers after cleanup. Please try restarting.")
                else:
                    logger.error("Container startup failed for other reasons. Please check Podman installation.")
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
            # Ensure collection is loaded - use proper Milvus method
            try:
                # Try a simple operation that requires the collection to be loaded
                self.collection.num_entities
                logger.debug(f"Collection '{self.collection_name}' is already loaded")
            except Exception as e:
                # Collection is not loaded, so load it
                logger.info(f"Loading collection '{self.collection_name}'...")
                self.collection.load()
                logger.info(f"Collection '{self.collection_name}' loaded successfully")
        
        # Ensure collection attribute is set
        if self.collection is None:
            logger.error("Failed to set collection attribute")
            raise RuntimeError("Collection could not be initialized properly")
            
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
                    limit=12000  # 안전한 쿼리 제한 (offset + limit <= 16384)
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
                    limit=12000  # 안전한 쿼리 제한 (offset + limit <= 16384)
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
        # 외부 저장소 경로 설정 (config에서 가져오기)
        external_storage_path = config.get_external_storage_path()
        
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
                    
                    # 컨테이너 상태 확인
                    container_status = self.get_container_status()
                    
                    # 개별 컨테이너 시작 시도 (compose 대신)
                    for container_type, container_name in self.milvus_containers.items():
                        if container_name not in container_status or not container_status[container_name]:
                            logger.info(f"Attempting to start {container_type} container: {container_name}")
                            if self.start_container(container_name):
                                logger.info(f"Successfully started {container_type} container")
                            else:
                                logger.info(f"Container {container_name} not available for restart (normal after reset)")
                    
                    # 재연결 시도
                    time.sleep(5)  # 컨테이너가 시작될 시간을 줌
                    if self.is_port_in_use(self.port):
                        try:
                            self.connect()
                            logger.info("Successfully reconnected to Milvus after recovery")
                        except Exception as e:
                            logger.warning(f"Failed to reconnect to Milvus after recovery: {e}")
                else:
                    # 서비스는 실행 중이지만 컨테이너 상태 확인 (로깅만)
                    container_status = self.get_container_status()
                    running_containers = [name for name, running in container_status.items() if running and name in self.milvus_containers.values()]
                    if len(running_containers) < len(self.milvus_containers):
                        missing_containers = [name for name in self.milvus_containers.values() if name not in running_containers]
                        logger.debug(f"Some Milvus containers are not detected as running: {missing_containers}")
                        # 포트가 열려있으면 실제로는 정상 작동 중일 가능성이 높으므로 자동 복구하지 않음
                        
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
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="original_path", dtype=DataType.VARCHAR, max_length=1024)  # 원본 경로 필드 추가
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
        logger.info(f"Creating index on 'vector' field with params: {index_params}")
        collection.create_index(field_name="vector", index_params=index_params)
        
        # 컬렉션 로드
        logger.info(f"Loading collection '{collection_name}'")
        collection.load()
        
        logger.info(f"Collection '{collection_name}' created and loaded successfully")
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
            logger.info(f"Dropping existing collection '{collection_name}'")
            utility.drop_collection(collection_name)
            print(f"Collection '{collection_name}' has been dropped.")
        
        # 새 컬렉션 생성
        logger.info(f"Creating new collection '{collection_name}' with dimension {dimension}")
        self.create_collection(collection_name, dimension)
        logger.info(f"Collection '{collection_name}' has been recreated")
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
        """데이터를 Milvus에 삽입 (배치 처리 지원)
        오류 로깅 및 예외 처리 개선
        """
        # 컴파일 시점에 기본 값 설정
        vector_dimension = getattr(self, 'dimension', 768)  # 기본값 768
        
        try:
            # 1. 콜렉션 객체 초기화 여부 확인
            if self.collection is None:
                if not utility.has_collection(self.collection_name):
                    error_msg = f"Collection '{self.collection_name}' does not exist"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # 콜렉션 로드 시도
                try:
                    self.collection = Collection(self.collection_name)
                    logger.info(f"Collection '{self.collection_name}' loaded successfully")
                except Exception as coll_error:
                    error_msg = f"Failed to load collection '{self.collection_name}': {coll_error}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # 2. 데이터 형식 검증 (상세한 오류 로깅 추가)
            if not data:
                error_msg = "Empty data provided for insertion"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            if not isinstance(data, dict):
                error_msg = f"Invalid data format: expected dict, got {type(data)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # 3. 데이터의 필수 필드 확인
            required_fields = ["path", "vector"]
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                error_msg = f"Missing required fields for insertion: {missing_fields}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # 4. 벡터 형식 및 차원 확인
            vector_data = data.get("vector")
            if not isinstance(vector_data, list):
                error_msg = f"Vector data must be a list, got {type(vector_data)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            actual_dimension = len(vector_data)
            if actual_dimension != vector_dimension:
                error_msg = f"Vector dimension mismatch: expected {vector_dimension}, got {actual_dimension}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # auto_id=True로 설정되어 있으므로 항상 'id' 필드 제거
            data_copy = data.copy()
            if 'id' in data_copy:
                del data_copy['id']
                logger.debug(f"Removed 'id' field from data as auto_id=True is enabled")
            
            # 파일 경로 확인 및 추가 진단
            if 'path' in data_copy:
                path = data_copy['path']
                file_name = os.path.basename(path) if path else ''
                
                # 파일명 관련 진단 정보 로깅
                if file_name:
                    import re
                    starts_with_number = bool(re.match(r'^\d', file_name))
                    has_special_chars = any(c in file_name for c in '[](){}#$%^&*;:<>?/|\\=')
                    
                    if starts_with_number:
                        logger.warning(f"File '{file_name}' starts with a number - may cause schema issues")
                    
                    if has_special_chars:
                        logger.warning(f"File '{file_name}' contains special characters - may cause issues")
                        
                    # 로깅에만 사용하기 위한 진단 정보
                    logger.debug(f"File diagnostics - Starts with number: {starts_with_number}, Has special chars: {has_special_chars}")
            try:
                result = self.collection.insert(data_copy)
                logger.debug(f"Successfully inserted data: {result}")
                return result
            except Exception as insert_error:
                error_msg = str(insert_error)
                
                # DataNotMatchException 처리
                if "DataNotMatchException" in str(type(insert_error)) and "original_path" in error_msg:
                    # original_path 필드 문제인 경우 특별 처리
                    logger.warning(f"Schema mismatch with 'original_path' field, attempting to fix...")
                    
                    # 다시 한 번 시도
                    if 'path' in data_copy:
                        data_copy['original_path'] = data_copy['path']
                        logger.debug(f"Retry: Added 'original_path' field with value from 'path'")
                        result = self.collection.insert(data_copy)
                        logger.info(f"Successfully inserted data after fixing original_path field")
                        return result
                
                # 'id' 필드 관련 문제인 경우 다시 시도
                if "id" in str(insert_error).lower():
                    logger.warning("'id' field error detected in insert operation, attempting to fix...")
                    # 'id' 필드 제거 재시도
                    if 'id' in data_copy:
                        logger.info("Removing 'id' field from data and retrying insertion")
                        del data_copy['id']
                        try:
                            result = self.collection.insert(data_copy)
                            logger.info("Successfully inserted data after removing 'id' field")
                            return result
                        except Exception as retry_error:
                            logger.error(f"Still failed after removing 'id' field: {retry_error}")
                
                # 실패 시 상세 로깅
                logger.error(f"Failed to insert data: {insert_error}")
                logger.debug(f"Failed data fields: {[k for k in data_copy.keys() if k != 'vector']}")
                
                # 'id' 필드 관련 문제 확인 
                if "id" in str(insert_error).lower():
                    logger.error("'id' field error detected in insert operation")
                    # 'id' 키가 있는지 다시 확인
                    if 'id' in data_copy:
                        logger.error("CRITICAL: 'id' field is still present in data_copy - this will cause schema errors")
                    # schema 구조 확인
                    try:
                        schema_fields = [field.name for field in schema.fields]
                        logger.error(f"Collection schema fields: {schema_fields}")
                    except Exception as schema_err:
                        logger.error(f"Could not retrieve schema fields: {schema_err}")
                
                # 파일 경로 상세 진단
                if 'path' in data_copy:
                    path_value = data_copy['path']
                    logger.debug(f"Path value: {path_value[:100]}")
                    
                    # 파일명 추가 진단
                    file_name = os.path.basename(path_value) if path_value else ''
                    if file_name:
                        import re
                        if bool(re.match(r'^\d', file_name)):
                            logger.error(f"File '{file_name}' starts with a number - recommend adding 'file_' prefix")
                        
                        special_chars = [c for c in '[](){}#$%^&*;:<>?/|\\=' if c in file_name]
                        if special_chars:
                            logger.error(f"File contains problematic characters: {special_chars}")
                
                raise insert_error
        except Exception as schema_error:
            # 스키마 처리 중 오류 발생
            logger.error(f"Error processing schema compatibility: {schema_error}")
            
            # 'id' 필드 다시 한번 확인하고 제거
            if 'id' in data_copy:
                logger.info("Removing 'id' field from data before final retry")
                del data_copy['id']
            
            # 원래 데이터로 다시 시도
            try:
                result = self.collection.insert(data_copy)
                return result
            except Exception as insert_error:
                # 삽입 오류에 대한 자세한 로깅
                error_msg = f"Failed to insert data: {insert_error}"
                logger.error(error_msg, exc_info=True)
                
                # 데이터 삽입 상태 로깅 (디버깅용)
                logger.debug(f"Data path being inserted: {data_copy.get('path', 'N/A')}")
                logger.debug(f"Vector dimension: {len(data_copy.get('vector', []))}")
                
                # 메모리 사용량 확인
                try:
                    import psutil
                    process = psutil.Process()
                    logger.debug(f"Memory usage during insert: {process.memory_info().rss / (1024 * 1024):.2f} MB")
                except (ImportError, Exception) as e:
                    logger.debug(f"Could not log memory usage: {e}")
                    
                # 삽입 시도 중 데이터 구조 분석
                data_keys = list(data_copy.keys())
                logger.debug(f"Data structure during insert: {data_keys}")
                
                # 마지막으로 중요 필드 값 분석
                for key in ['path', 'title']:
                    if key in data_copy:
                        value = data_copy[key]
                        if value and isinstance(value, str):
                            # 값의 앞부분 로깅 (너무 길면 자름)
                            logger.debug(f"Field '{key}' starts with: {value[:50]}{'...' if len(value) > 50 else ''}")
                            # 숫자로 시작하는지 확인
                            import re
                            if re.match(r'^\d', value):
                                logger.error(f"Field '{key}' starts with a number - may cause schema issues")
                
                raise insert_error
            
        except Exception as e:
                    
                    # 데이터 삽입 상태 로깅 (디버깅용)
                    logger.debug(f"Data path being inserted: {data_copy.get('path', 'N/A')}")
                    logger.debug(f"Vector dimension: {len(data_copy.get('vector', []))}")
                    
                    # 메모리 사용량 확인
                    try:
                        import psutil
                        process = psutil.Process()
                        logger.debug(f"Memory usage during insert: {process.memory_info().rss / (1024 * 1024):.2f} MB")
                    except (ImportError, Exception) as e:
                        logger.debug(f"Could not log memory usage: {e}")
                        
                    # 삽입 시도 중 데이터 구조 분석
                    data_keys = list(data_copy.keys())
                    logger.debug(f"Data structure during insert: {data_keys}")
                    
                    # 마지막으로 중요 필드 값 분석
                    for key in ['path', 'title']:
                        if key in data_copy:
                            value = data_copy[key]
                            if value and isinstance(value, str):
                                # 값의 앞부분 로깅 (너무 길면 자름)
                                logger.debug(f"Field '{key}' starts with: {value[:50]}{'...' if len(value) > 50 else ''}")
                                # 숫자로 시작하는지 확인
                                import re
                                if re.match(r'^\d', value):
                                    logger.error(f"Field '{key}' starts with a number - may cause schema issues")
                    
                    raise insert_error
                
        except Exception as e:
            # 총괄적인 오류 처리
            error_msg = f"Error inserting data: {e}"
            logger.error(error_msg, exc_info=True)
            # 예외를 상위로 전파하여 정확한 오류 파악
            raise
    
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
            # Ensure collection is loaded
            try:
                self.collection.num_entities  # Test if loaded
                logger.debug("Collection is loaded")
            except Exception:
                logger.info("Loading collection for delete operation...")
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
        """삭제 대기열에 있는 모든 파일 삭제 (배치 처리) - 오류 처리 강화"""
        if not self.pending_deletions:
            return
            
        logger.info(f"Executing batch deletion for {len(self.pending_deletions)} files")
        print(f"Executing batch deletion for {len(self.pending_deletions)} files...")
        
        # 성공적으로 삭제된 파일들을 추적
        successful_deletions = []
        failed_deletions = []
        
        try:
            # Use intelligent batch sizing from DynamicBatchOptimizer
            max_limit = self._get_optimal_query_limit()  # Uses batch optimizer
            offset = 0
            all_results = []
            
            try:
                # Pagination with intelligent batch sizing
                while True:
                    try:
                        # Use intelligent limit - already capped at 16000
                        current_limit = max_limit
                        
                        results = self.collection.query(
                            output_fields=["id", "path"],
                            limit=12000,  # 안전한 쿼리 제한 (offset + limit <= 16384)
                            offset=offset,
                            expr="id >= 0"  # Query all documents
                        )
                        
                        if not results:  # No more results
                            break
                            
                        all_results.extend(results)
                        offset += current_limit
                        
                        # If results are fewer than limit, we've reached the end
                        if len(results) < current_limit:
                            break
                    except Exception as e:
                        logger.error(f"Error during query iteration at offset {offset}: {e}")
                        # Break on any error since we're using safe limits
                        break
                        
            except Exception as e:
                logger.error(f"Error querying documents for deletion: {e}")
                # 에러가 있어도 계속 진행
            
            logger.info(f"Found {len(all_results)} documents to process for deletion")
            
            # 파일별 ID 수집
            files_to_delete = {}
            
            for doc in all_results:
                try:
                    path = doc.get("path")
                    doc_id = doc.get("id")
                    
                    if path in self.pending_deletions and doc_id is not None:
                        if path not in files_to_delete:
                            files_to_delete[path] = []
                        files_to_delete[path].append(doc_id)
                except Exception as e:
                    logger.error(f"Error processing document for deletion: {e}")
            
            # 파일별 삭제 실행
            for path, ids in files_to_delete.items():
                try:
                    if ids:
                        # Use intelligent delete batch size from DynamicBatchOptimizer
                        if hasattr(self, 'batch_optimizer') and self.batch_optimizer:
                            batch_size = min(self.batch_optimizer.current_batch_size // 4, 500)  # Conservative for deletes
                        else:
                            batch_size = 500  # Safe fallback for deletes
                        for i in range(0, len(ids), batch_size):
                            batch = ids[i:i+batch_size]
                            try:
                                # id 직접 삭제 방식 사용
                                self.collection.delete(f"id in {batch}")
                            except Exception as e:
                                logger.error(f"Error deleting batch for {path}: {e}")
                                # 배치 삭제 실패 시 개별 삭제 시도
                                for single_id in batch:
                                    try:
                                        self.collection.delete(f"id == {single_id}")
                                    except:
                                        pass
                        
                        successful_deletions.append(path)
                        logger.info(f"Deleted {len(ids)} chunks for file {path}")
                        print(f"Deleted {len(ids)} chunks for file {path}")
                    else:
                        logger.warning(f"No chunks found for file {path}")
                        failed_deletions.append(path)
                except Exception as e:
                    logger.error(f"Error processing deletion for {path}: {e}")
                    failed_deletions.append(path)
            
            # 요약 출력
            if successful_deletions:
                logger.info(f"Successfully deleted chunks from {len(successful_deletions)} files")
                print(f"\n✔ Successfully removed: {len(successful_deletions)} files")
            
            if failed_deletions:
                logger.warning(f"Failed to delete {len(failed_deletions)} files")
                print(f"\n⚠ Failed to remove: {len(failed_deletions)} files")
                print("Files that could not be deleted:")
                
                # Log all failed deletions but only display first 10 to the user
                for f in failed_deletions:
                    logger.warning(f"Failed to delete file: {f}")
                    
                for f in failed_deletions[:10]:  # 처음 10개만 표시
                    print(f"- {f}")
                    
                if len(failed_deletions) > 10:
                    logger.warning(f"...and {len(failed_deletions) - 10} more files could not be deleted")
                    print(f"...and {len(failed_deletions) - 10} more files")
            
            # 성공적으로 삭제된 파일만 대기열에서 제거
            for path in successful_deletions:
                self.pending_deletions.discard(path)
            
            print("Cleanup Results:")
            print(f"\u2714 Successfully removed: {len(successful_deletions)} files")
            if failed_deletions:
                print(f"\u26a0 Failed to remove: {len(failed_deletions)} files")
                print("Files that could not be deleted:")
                for f in failed_deletions[:10]:
                    print(f"- {f}")
                if len(failed_deletions) > 10:
                    print(f"...and {len(failed_deletions) - 10} more files")

            # ✅ 성공적으로 삭제된 파일만 대기열에서 제거
            for path in successful_deletions:
                self.pending_deletions.discard(path)

            # ✅ 변경 사항을 확실히 반영하기 위해 flush 추가
            try:
                self.collection.flush()
                logger.info("Milvus collection flushed after deletions")
                print("✓ Milvus collection flushed after deletions.")
            except Exception as e:
                logger.error(f"Flush failed: {e}")
                print(f"⚠️ Flush failed: {e}")
        
        except Exception as e:
            logger.error(f"Error in batch deletion: {e}", exc_info=True)
            print(f"Warning: Error in batch deletion: {e}")
            
        finally:
            logger.info(f"Batch deletion complete. Successful: {len(successful_deletions)}, Failed: {len(failed_deletions)}")
            logger.info(f"Remaining in pending queue: {len(self.pending_deletions)}")

    
    def delete_by_path(self, file_path):
        """파일 경로로 데이터 삭제 (레거시 지원)"""
        # 파일 경로가 None이면 바로 리턴
        if file_path is None:
            logger.warning("Attempted to delete with a None file path")
            print("Warning: Attempted to delete with a None file path")
            return
            
        try:
            # 메모리 효율성 개선을 위해 필터링 최적화
            # path에 대한 직접 필터링 시도
            logger.debug(f"Attempting to delete file by path: {file_path}")
            expr = f"path == '{file_path}'"
            count = self.collection.query(expr=expr, output_fields=["count(*)"]).get("count")
            
            if count and count > 0:
                logger.info(f"Deleting {count} chunks for file {file_path} using direct path filter")
                self.collection.delete(expr)
                print(f"Deleted {count} chunks for file {file_path}")
                return
                
            # 직접 필터링이 실패한 경우 백업 방법 사용
            logger.info(f"Direct path filter failed for {file_path}, using backup method")

            results = self.collection.query(
                output_fields=["id", "path"],
                limit=1000,
                expr="id >= 0"
            )
            
            ids_to_delete = [doc.get("id") for doc in results if doc.get("path") == file_path and doc.get("id") is not None]
            
            if ids_to_delete:
                logger.info(f"Deleting {len(ids_to_delete)} chunks for file {file_path} using backup method")
                self.collection.delete(f"id in {ids_to_delete}")
                print(f"Deleted {len(ids_to_delete)} chunks for file {file_path}")
            else:
                logger.warning(f"No documents found for path: {file_path}")
                print(f"No documents found for path: {file_path}")
                
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}", exc_info=True)
            print(f"Warning: Error deleting file {file_path}: {e}")
            # 오류가 발생해도 계속 진행
    
    def search(self, vector, limit=5, filter_expr=None):
        """벡터 유사도 검색 수행 (GPU 검색 추가)"""
        # Ensure collection exists
        if self.collection is None:
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                try:
                    self.collection.num_entities  # Test if loaded
                except Exception:
                    self.collection.load()
            else:
                logger.error(f"Collection '{self.collection_name}' does not exist")
                return []
                
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
        """쿼리 표현식에서 특수문자와 경로를 안전하게 처리
        영어와 한글 파일명에 있는 특수문자를 모두 처리합니다.
        """
        import re
        
        # 이미 안전한 쿼리인 경우 그대로 반환
        if expr is None or expr == "id >= 0":
            return expr
        
        try:
            # 문제가 될 수 있는 특수 문자 정의
            problem_chars = ",'\"()[]{},;" 
            
            # 1. 파일 경로 제이터 필드 쿼리 처리 (path, file_path)
            path_patterns = [
                r"path\s*=\s*['\"](.+?)['\"]",  # path = 'value'
                r"path\s+==\s*['\"](.+?)['\"]",  # path == 'value'
                r"path\s+like\s+['\"](.+?)['\"]",  # path like 'value%'
                r"file_path\s*=\s*['\"](.+?)['\"]",  # file_path = 'value'
                r"file_path\s+==\s*['\"](.+?)['\"]",  # file_path == 'value'
            ]
            
            for pattern in path_patterns:
                match = re.search(pattern, expr)
                if match:
                    path_value = match.group(1)
                    
                    # 한글 포함 확인
                    has_korean = bool(re.search(r'[가-힣]', path_value))
                    # 특수 문자 포함 확인
                    has_special_chars = any(c in path_value for c in problem_chars) or " " in path_value
                    
                    # 한글이나 특수 문자가 포함된 경우
                    if has_korean or has_special_chars:
                        logger.debug(f"특수 문자 또는 한글이 포함된 경로 쿼리 감지: {expr}")
                        
                        try:
                            # 파일명만 추출하여 부분 일치 검색으로 대체
                            import os
                            file_name = os.path.basename(path_value)
                            
                            # 파일명에 특수 문자가 많으면 안전한 쿼리로 대체
                            special_char_count = sum(1 for c in file_name if c in problem_chars)
                            
                            # 특수 문자의 수에 따라 처리 방식 결정
                            if special_char_count > 2 or len(file_name) < 3:
                                logger.debug(f"다수의 특수 문자 발견 또는 짧은 파일명, 안전한 쿼리로 대체: {file_name}")
                                return "id >= 0"  # 후처리에서 필터링할 수 있도록 모든 항목 조회
                            else:
                                # 특수 문자가 적은 경우 파일명으로 부분 일치 검색
                                # SQL 인정부호 이스케이프 처리
                                safe_file_name = file_name.replace("'", "\\'")
                                new_expr = f"path like '%{safe_file_name}%'"
                                logger.debug(f"쿼리 변환: {expr} -> {new_expr}")
                                return new_expr
                        except Exception as e:
                            logger.warning(f"파일명 추출 중 오류: {e}, 안전한 쿼리 사용")
                            return "id >= 0"
            
            # 2. 제목 필드 쿼리 처리 (title)
            title_patterns = [
                r"title\s*=\s*['\"](.+?)['\"]",  # title = 'value'
                r"title\s+==\s*['\"](.+?)['\"]",  # title == 'value'
                r"title\s+like\s+['\"](.+?)['\"]",  # title like 'value%'
            ]
            
            for pattern in title_patterns:
                match = re.search(pattern, expr)
                if match:
                    title_value = match.group(1)
                    
                    # 한글 포함 확인
                    has_korean = bool(re.search(r'[가-힣]', title_value))
                    # 특수 문자 포함 확인
                    has_special_chars = any(c in title_value for c in problem_chars)
                    
                    # 한글이나 특수 문자가 포함된 경우
                    if has_korean or has_special_chars:
                        logger.debug(f"특수 문자 또는 한글이 포함된 제목 쿼리 감지: {expr}")
                        
                        # 특수 문자의 수에 따라 처리 방식 결정
                        special_char_count = sum(1 for c in title_value if c in problem_chars)
                        if special_char_count > 2 or len(title_value) < 3:
                            logger.debug(f"다수의 특수 문자 발견 또는 짧은 제목, 안전한 쿼리로 대체: {title_value}")
                            return "id >= 0"  # 후처리에서 필터링
                        else:
                            # 특수 문자가 적은 경우 부분 일치 검색
                            # SQL 인정부호 이스케이프 처리
                            safe_title = title_value.replace("'", "\\'")
                            new_expr = f"title like '%{safe_title}%'"
                            logger.debug(f"쿼리 변환: {expr} -> {new_expr}")
                            return new_expr
                            
        except Exception as e:
            logger.error(f"쿼리 표현식 정제 중 오류: {e}")
            return "id >= 0"  # 오류 발생시 안전한 쿼리 반환
        
        # 변경이 필요 없는 경우 원래 표현식 반환
        return expr
    
    def _post_filter_results(self, results, original_expr):
        """쿼리 결과를 원래 표현식에 맞게 후처리"""
        # 원본 표현식이 경로 비교인 경우, 정확히 일치하는 경로만 반환
        if not results or not original_expr:
            return results
        
        # 'path == 'value'' 패턴 감지
        import re
        path_eq_pattern = re.compile(r"path\s*==\s*['\"](.+?)['\"]")
        path_eq_match = path_eq_pattern.search(original_expr)
        
        if path_eq_match:
            eq_value = path_eq_match.group(1)
            # 정확히 일치하는 경로만 필터링
            filtered_results = [r for r in results if r.get('path') == eq_value]
            logger.debug(f"경로 일치 필터링: {len(filtered_results)}/{len(results)} 결과 남음")
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
        # Ensure collection exists
        if self.collection is None:
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                try:
                    self.collection.num_entities  # Test if loaded
                except Exception:
                    self.collection.load()
            else:
                logger.error(f"Collection '{self.collection_name}' does not exist")
                return []
                
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
            # 경로에 특수 문자나 한글이 있을 수 있으므로 안전하게 처리
            try:
                # 한글 및 특수 문자 안전 처리 향상
                # 1. Base64 인코딩 방식으로 경로 비교 (가장 안전한 방법)
                import base64
                import re
                
                # 경로에 특수 문자나 한글이 포함되어 있는지 확인
                has_special_chars = bool(re.search(r'[^a-zA-Z0-9_\-\./]', file_path))
                
                if has_special_chars:
                    # 특수 문자나 한글이 포함된 경우: ID 기반 우회 검색
                    # 파일명만 추출해서 더 간단한 쿼리 구성
                    file_name = os.path.basename(file_path)
                    # 다른 방식으로 시도 - 파일명 부분 일치 검색
                    expr = f"path like '%{file_name}%'"  # 파일명 부분 일치 검색
                else:
                    # 특수 문자가 없는 경우: 일반 경로 이스케이핑
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