import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (optional - auto-loads if .env file exists)
load_dotenv()

# Base path settings - Project root directory
BASE_DIR = Path(__file__).resolve().parent

# Project absolute path for Claude Desktop config (auto-detected)
# This path will be used when setup.py option 5 generates claude_desktop_config.json
PROJECT_ABSOLUTE_PATH = str(BASE_DIR)

##########################################################
# 🔧 USER SETTINGS - Only modify this section!

# 🗂️ Obsidian Vault Path (REQUIRED!)
# Change the path below to your Obsidian vault path
OBSIDIAN_VAULT_PATH = "G:\\jayjeo"  # ← Change this to your path!

# 🔧 Podman Path (Usually auto-detected. Only modify if there are issues)
PODMAN_PATH = ""  # Leave empty for auto-detection, or enter full path if needed

##########################################################
# Other settings (modify if needed)

# 📁 External storage path (Milvus data storage - usually no need to modify)
EXTERNAL_STORAGE_PATH = str(BASE_DIR / "MilvusData")  # DO NOT CHANGE THIS!!!!

# Logging settings
LOG_LEVEL = "ERROR"  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# Obsidian settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
CHUNK_MIN_SIZE = 100  # Minimum chunk size

# Milvus settings
MILVUS_HOST = "localhost"  # Milvus server host
MILVUS_PORT = 19530  # Milvus server port
COLLECTION_NAME = "obsidian_notes"  # Milvus collection name

# Search settings
SEARCH_RESULTS_LIMIT = 100  # Limit the number of searchable notes 

# Container runtime settings
CONTAINER_RUNTIME = "podman"  # Container runtime to use: "podman" only
PODMAN_COMPOSE_PATH = "podman-compose"  # Podman Compose executable path
# Legacy Docker settings (no longer used)
DOCKER_PATH = ""  # Not used anymore, kept for backward compatibility
DOCKER_COMPOSE_PATH = ""  # Not used anymore, kept for backward compatibility

# Embedding model settings
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # Multilingual embedding model
VECTOR_DIM = 768  # Vector dimension
EMBEDDING_CACHE_SIZE = 100  # Cache size to reduce RAM usage

# Model cache directory (relative to project directory)
MODEL_CACHE_DIR = str(BASE_DIR / "model_cache")  # Local model cache directory
MODEL_LOAD_TIMEOUT = 120  # Model loading timeout in seconds

# LLM settings
LLM_PROVIDER = "claude"  # LLM provider
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama server URL (backward compatibility)
OLLAMA_MODEL = "qwen3:8b"  # Ollama model (backward compatibility)

# Web server settings
FLASK_PORT = 5678  # Web server port (e.g., http://localhost:5678/)

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
FASTMCP_AUTH_ENABLED = False  # Enable FastMCP authentication
FASTMCP_USERNAME = ""  # FastMCP username (when auth enabled)
FASTMCP_PASSWORD = ""  # FastMCP password (when auth enabled)

# Performance settings
BATCH_SIZE = 2000
EMBEDDING_BATCH_SIZE = 2000

# Full embedding settings
SKIP_PDF_IN_FULL_EMBEDDING = False  # Skip PDF files in full embedding

# GPU settings
USE_GPU = True  # Enable GPU usage
GPU_MEMORY_FRACTION = 0.95
GPU_INDEX_TYPE = "GPU_IVF_FLAT"  # GPU index type for Milvus
GPU_DEVICE_ID = 0  # GPU device ID to use (0 for first GPU)

# Advanced GPU optimization settings
GPU_FORCE_TENSOR_CORES = True  # Force tensor core usage
GPU_ENABLE_CUDNN_BENCHMARK = True  # Enable cuDNN benchmark
GPU_ENABLE_MEMORY_EFFICIENT_ATTENTION = True  # Enable memory-efficient attention
GPU_ENABLE_FLASH_ATTENTION = True  # Enable flash attention

# Balanced GPU memory usage settings
MAX_SEQ_LENGTH = 4096  # Maximum sequence length
GPU_MEMORY_GROWTH = True  # Enable dynamic GPU memory allocation
GPU_MIXED_PRECISION = True  # Enable mixed precision
GPU_CACHE_ALL_TENSORS = True  # Cache all tensors in GPU memory
GPU_PARALLEL_PROCESSING = True  # Enable parallel processing
GPU_MEMORY_PREALLOCATION = True  # Preallocate GPU memory
GPU_AGGRESSIVE_OPTIMIZATION = False  # Disable aggressive optimization for stability
GPU_MULTI_STREAM_EXECUTION = True  # Enable multiple CUDA streams

########################################################## Do not change below
# Podman Configuration
COMPOSE_COMMAND = "podman compose"

# Alternative configurations for different setups
PODMAN_CONFIGS = {
    "system_install": {
        "podman_path": r"C:\Program Files\RedHat\Podman\podman.exe",
        "compose_command": "podman compose"
    },
    "user_install": {
        "podman_path": r"C:\Users\{username}\AppData\Local\Programs\RedHat\Podman\podman.exe",
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

########################################################## Advanced user environment variable support (optional)
# The functions below use .env files or system environment variables if available
# Regular users don't need to worry about this!

def get_obsidian_vault_path():
    """Return Obsidian vault path (environment variable first, then config value)"""
    return os.getenv('OBSIDIAN_VAULT_PATH', OBSIDIAN_VAULT_PATH)

def get_external_storage_path():
    """Return external storage path (environment variable first, then config value)"""
    return os.getenv('EXTERNAL_STORAGE_PATH', EXTERNAL_STORAGE_PATH)

def find_podman_path():
    """Automatically find Podman executable path."""
    import shutil
    import subprocess
    
    # 1. Use path set in config first
    if PODMAN_PATH and os.path.exists(PODMAN_PATH):
        return PODMAN_PATH
    
    # 2. Check environment variable
    env_path = os.getenv('PODMAN_PATH')
    if env_path and os.path.exists(env_path):
        return env_path
    
    # 3. Find Podman in PATH
    podman_path = shutil.which("podman")
    if podman_path:
        return podman_path
    
    # 4. Check common installation paths
    possible_paths = [
        r"C:\Program Files\RedHat\Podman\podman.exe",
        r"C:\Users\{}\AppData\Local\Programs\RedHat\Podman\podman.exe".format(os.environ.get("USERNAME", "")),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                # Test execution
                result = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return path
            except:
                continue
    
    return None

def get_podman_path():
    """Return Podman path (with auto-detection)."""
    podman_path = find_podman_path()
    if podman_path:
        return podman_path
    
    raise FileNotFoundError("Podman executable not found. Please install Podman or set PODMAN_PATH in config.py")

def validate_paths():
    """Validate configured paths."""
    errors = []
    
    # Validate Obsidian vault path
    vault_path = get_obsidian_vault_path()
    if not os.path.exists(vault_path):
        errors.append(f"❌ Obsidian vault path does not exist: {vault_path}")
        errors.append(f"   → Please modify OBSIDIAN_VAULT_PATH in config.py to the correct path")
    else:
        print(f"✅ Obsidian vault path: {vault_path}")
    
    # Create storage path
    storage_path = Path(get_external_storage_path())
    storage_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    for subdir in ['etcd', 'minio', 'milvus']:
        (storage_path / subdir).mkdir(exist_ok=True)
    
    print(f"✅ Data storage path: {storage_path}")
    
    # Validate Podman path
    try:
        podman_path = get_podman_path()
        print(f"✅ Podman path: {podman_path}")
    except FileNotFoundError:
        errors.append("❌ Podman not found")
        errors.append("   → Please install Podman or set PODMAN_PATH in config.py")
    
    if errors:
        print("\n🚨 Configuration errors:")
        for error in errors:
            print(error)
        print("\n📝 Please open config.py and fix the issues above!")
        return False
    
    print("\n✅ All configurations are correct!")
    return True

########################################################## Compatibility functions

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

def get_project_absolute_path():
    """프로젝트 절대 경로 반환 (Claude Desktop 설정용)"""
    return PROJECT_ABSOLUTE_PATH

# 🚀 Auto speed optimization settings
FAST_MODE = True
DISABLE_COMPLEX_GPU_OPTIMIZATION = True
MEMORY_CHECK_INTERVAL = 30
DISABLE_PROGRESS_MONITORING = False  # Keep progress monitoring
MAX_WORKERS = 1
EMBEDDING_CACHE_SIZE = 10000

# 🎯 Advanced Milvus optimization settings
ADVANCED_SEARCH_ENABLED = True  # Enable advanced search features
KNOWLEDGE_GRAPH_ENABLED = True  # Enable knowledge graph features
MULTI_QUERY_FUSION_ENABLED = True  # Enable multi-query fusion
PERFORMANCE_MONITORING_ENABLED = True  # Enable performance monitoring
HNSW_AUTO_OPTIMIZATION = True  # Enable HNSW auto optimization

# Search performance optimization
SEARCH_CACHE_SIZE = 1000  # Search result cache size
SEARCH_TIMEOUT = 30  # Search timeout (seconds)
MAX_SEARCH_RESULTS = 1000  # Maximum search results

# Metadata filtering settings
ENABLE_COMPLEX_FILTERING = True  # Enable complex filtering
FILTER_CACHE_SIZE = 500  # Filter cache size

# RAG optimization settings
RAG_CONTEXT_WINDOW = 4096  # RAG context window size
RAG_CHUNK_OVERLAP_RATIO = 0.1  # RAG chunk overlap ratio
RAG_SIMILARITY_THRESHOLD = 0.7  # RAG similarity threshold

# Knowledge graph settings
KG_MAX_DEPTH = 3  # Knowledge graph maximum depth
KG_MAX_CONNECTIONS = 100  # Maximum connections
KG_SIMILARITY_THRESHOLD = 0.8  # Knowledge graph similarity threshold

########################################################## Auto validation on startup
if __name__ == "__main__":
    print("🔧 Obsidian-Milvus-FastMCP Configuration Validation")
    print("=" * 52)
    validate_paths()