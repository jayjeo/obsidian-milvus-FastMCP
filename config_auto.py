import os
import shutil
import getpass

def find_podman_path():
    """Automatically find Podman installation"""
    
    # Common installation paths
    paths_to_check = [
        r"C:\Program Files\RedHat\Podman\podman.exe",
        rf"C:\Users\{getpass.getuser()}\AppData\Local\Programs\RedHat\Podman\podman.exe",
        "podman"  # Check if in PATH
    ]
    
    for path in paths_to_check:
        if path == "podman":
            # Check if podman is in PATH
            if shutil.which("podman"):
                return "podman"
        else:
            # Check if file exists
            if os.path.exists(path):
                return path
    
    raise FileNotFoundError("Podman not found. Please install Podman first.")

# Auto-detect or use manual setting
try:
    PODMAN_PATH = find_podman_path()
    print(f"✅ Auto-detected Podman: {PODMAN_PATH}")
except FileNotFoundError:
    # Fallback to manual setting
    PODMAN_PATH = r"C:\Program Files\RedHat\Podman\podman.exe"
    print(f"⚠️  Using default path: {PODMAN_PATH}")

COMPOSE_COMMAND = "podman compose"

# Rest of configuration
MILVUS_NETWORK = "milvus"
MILVUS_VOLUMES = {
    "etcd": "milvus-etcd-data",
    "minio": "milvus-minio-data", 
    "milvus": "milvus-db-data"
}

MILVUS_API_PORT = 19530
MILVUS_WEB_PORT = 9091

IMAGES = {
    "etcd": "quay.io/coreos/etcd:v3.5.0",
    "minio": "minio/minio:RELEASE.2023-03-20T20-16-18Z",
    "milvus": "milvusdb/milvus:v2.3.4"
}