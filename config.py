import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (for backward compatibility)
load_dotenv()

# Base path settings
BASE_DIR = Path(__file__).resolve().parent



##########################################################
# User settings - configure directly here

# Logging settings
LOG_LEVEL = "ERROR"  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# Obsidian settings
OBSIDIAN_VAULT_PATH = "G:\\jayjeo"  # Obsidian vault path
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
CHUNK_MIN_SIZE = 100  # Minimum chunk size

# Milvus settings
MILVUS_HOST = "localhost"  # Milvus server host
MILVUS_PORT = 19530  # Milvus server port
COLLECTION_NAME = "obsidian_notes"  # Milvus collection name

# Search settings
SEARCH_RESULTS_LIMIT = 100  # Limit the number of searcherable notes 

# Container runtime settings
CONTAINER_RUNTIME = "podman"  # Container runtime to use: "podman" only
PODMAN_PATH = "C:\\Program Files\\RedHat\\Podman\\podman.exe"  # Podman execute file path (default assumes it's in PATH)
PODMAN_COMPOSE_PATH = "podman-compose"  # Podman Compose execute file path (if available)
# Legacy Docker settings (no longer used)
DOCKER_PATH = ""  # Not used anymore, kept for backward compatibility
DOCKER_COMPOSE_PATH = ""  # Not used anymore, kept for backward compatibility

# Embedding model settings
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # 다국어 지원 임베딩 모델
VECTOR_DIM = 768  # Vector dimension
EMBEDDING_CACHE_SIZE = 100  # Reduced cache size to decrease RAM usage

# LLM settings
LLM_PROVIDER = "claude"  # LLM provider
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama server URL (kept for backward compatibility)
OLLAMA_MODEL = "qwen3:8b"  # Ollama model (kept for backward compatibility)

# Web server settings
FLASK_PORT = 5678  # Web server port (e.g., http://localhost:5678/ or http://127.0.0.1:5678/)

# FastMCP 2.0 and Claude Desktop settings
FASTMCP_URL = "http://localhost:5680"  # FastMCP URL
FASTMCP_API_KEY = ""  # FastMCP API key (if required)
CLAUDE_MODEL = "claude-3-opus-20240229"  # Claude model to use
FASTMCP_SERVER_NAME = "obsidian-assistant"  # FastMCP server name
FASTMCP_TRANSPORT = "stdio"  # FastMCP transport (stdio, sse, or streamable-http)
FASTMCP_HOST = "127.0.0.1"  # FastMCP host (for sse and streamable-http transports)
FASTMCP_PORT = 5680  # FastMCP port (for sse and streamable-http transports)

# Authentication settings
ENABLE_AUTH = False  # Enable authentication
USERNAME = "admin"  # Admin username
PASSWORD = "password"  # Admin password

# FastMCP authentication settings
FASTMCP_AUTH_ENABLED = False  # FastMCP 인증 사용 여부
FASTMCP_USERNAME = ""  # FastMCP 사용자 이름 (인증 사용 시)
FASTMCP_PASSWORD = ""  # FastMCP 비밀번호 (인증 사용 시)

# Performance settings
BATCH_SIZE = 2000
EMBEDDING_BATCH_SIZE = 2000

# Full embedding settings
SKIP_PDF_IN_FULL_EMBEDDING = False  # PDF 파일을 전체 임베딩에서 건너뛰지 여부

# GPU settings
USE_GPU = True  # Enable GPU usage
GPU_MEMORY_FRACTION = 0.95
GPU_INDEX_TYPE = "GPU_IVF_FLAT"  # GPU index type for Milvus (GPU_IVF_FLAT is more compatible with Podman)
GPU_DEVICE_ID = 0  # GPU device ID to use (0 for first GPU)

# Advanced GPU optimization settings
GPU_FORCE_TENSOR_CORES = True  # Force tensor core usage to accelerate computations
GPU_ENABLE_CUDNN_BENCHMARK = True  # Enable cuDNN benchmark for performance optimization
GPU_ENABLE_MEMORY_EFFICIENT_ATTENTION = True  # Re-enable memory-efficient attention for stability
GPU_ENABLE_FLASH_ATTENTION = True  # Enable flash attention to accelerate transformer operations

# Balanced GPU memory usage settings
MAX_SEQ_LENGTH = 4096  # Balanced maximum sequence length
GPU_MEMORY_GROWTH = True  # Enable dynamic GPU memory allocation
GPU_MIXED_PRECISION = True  # Re-enable mixed precision for better stability
GPU_CACHE_ALL_TENSORS = True  # Cache all tensors in GPU memory
GPU_PARALLEL_PROCESSING = True  # Enable parallel processing
GPU_MEMORY_PREALLOCATION = True  # Preallocate GPU memory to maximize utilization
GPU_AGGRESSIVE_OPTIMIZATION = False  # Disable aggressive optimization for stability
GPU_MULTI_STREAM_EXECUTION = True  # Enable multiple CUDA streams


########################################################## Do not change below
# Podman Configuration
COMPOSE_COMMAND = "podman compose"

# Alternative configurations for different setups
PODMAN_CONFIGS = {
    "system_install": {
        "podman_path": r"C:\\Program Files\\RedHat\\Podman\\podman.exe",
        "compose_command": "podman compose"
    },
    "user_install": {
        "podman_path": r"C:\\Users\\{username}\\AppData\\Local\\Programs\\RedHat\\Podman\\podman.exe",
        "compose_command": "podman compose"
    },
    "portable": {
        "podman_path": "podman",  # Uses PATH
        "compose_command": "podman compose"
    }
}

# Use system install by default
DEFAULT_CONFIG = PODMAN_CONFIGS["system_install"]

# Container settings
MILVUS_NETWORK = "milvus"
MILVUS_VOLUMES = {
    "etcd": "milvus-etcd-data",
    "minio": "milvus-minio-data", 
    "milvus": "milvus-db-data"
}

# Ports
MILVUS_API_PORT = 19530
MILVUS_WEB_PORT = 9091

# Images
IMAGES = {
    "etcd": "quay.io/coreos/etcd:v3.5.0",
    "minio": "minio/minio:RELEASE.2023-03-20T20-16-18Z",
    "milvus": "milvusdb/milvus:v2.3.4"
}

########################################################## Do not change below
# Compatibility functions - for existing code that might call these functions

def get_obsidian_vault_path():
    return OBSIDIAN_VAULT_PATH

def get_milvus_host():
    return MILVUS_HOST

def get_milvus_port():
    return MILVUS_PORT

def get_collection_name():
    return COLLECTION_NAME

def get_embedding_model():
    return EMBEDDING_MODEL

def get_vector_dim():
    return VECTOR_DIM

def get_ollama_base_url():
    return OLLAMA_BASE_URL

def get_ollama_model():
    return OLLAMA_MODEL

def get_chunk_size():
    return CHUNK_SIZE

def get_chunk_overlap():
    return CHUNK_OVERLAP

def get_chunk_min_size():
    return CHUNK_MIN_SIZE

def get_batch_size():
    return BATCH_SIZE

def get_embedding_cache_size():
    return EMBEDDING_CACHE_SIZE

def get_flask_port():
    return FLASK_PORT

def get_search_results_limit():
    return SEARCH_RESULTS_LIMIT

def get_fastmcp_url():
    return FASTMCP_URL

def get_enable_auth():
    return ENABLE_AUTH

def get_username():
    return USERNAME

def get_password():
    return PASSWORD
# 🚀 자동 속도 최적화 설정
FAST_MODE = True
DISABLE_COMPLEX_GPU_OPTIMIZATION = True
MEMORY_CHECK_INTERVAL = 30
DISABLE_PROGRESS_MONITORING = False  # 진행률은 유지
MAX_WORKERS = 1
EMBEDDING_CACHE_SIZE = 10000

# 🎯 고급 Milvus 최적화 설정
ADVANCED_SEARCH_ENABLED = True  # 고급 검색 기능 활성화
KNOWLEDGE_GRAPH_ENABLED = True  # 지식 그래프 기능 활성화
MULTI_QUERY_FUSION_ENABLED = True  # 다중 쿼리 융합 활성화
PERFORMANCE_MONITORING_ENABLED = True  # 성능 모니터링 활성화
HNSW_AUTO_OPTIMIZATION = True  # HNSW 자동 최적화

# 검색 성능 최적화
SEARCH_CACHE_SIZE = 1000  # 검색 결과 캐시 크기
SEARCH_TIMEOUT = 30  # 검색 타임아웃 (초)
MAX_SEARCH_RESULTS = 1000  # 최대 검색 결과 수

# 메타데이터 필터링 설정
ENABLE_COMPLEX_FILTERING = True  # 복잡한 필터링 활성화
FILTER_CACHE_SIZE = 500  # 필터 캐시 크기

# RAG 최적화 설정
RAG_CONTEXT_WINDOW = 4096  # RAG 컨텍스트 윈도우 크기
RAG_CHUNK_OVERLAP_RATIO = 0.1  # RAG 청크 오버랩 비율
RAG_SIMILARITY_THRESHOLD = 0.7  # RAG 유사도 임계값

# 지식 그래프 설정
KG_MAX_DEPTH = 3  # 지식 그래프 최대 깊이
KG_MAX_CONNECTIONS = 100  # 최대 연결 수
KG_SIMILARITY_THRESHOLD = 0.8  # 지식 그래프 유사도 임계값
