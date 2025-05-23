#!/bin/bash

# 필요한 패키지 설치
pip install pymilvus sentence-transformers python-dotenv watchdog PyPDF2 markdown flask requests

# 디렉토리 구조 생성
mkdir -p obsidian-milvus-ollama/static/css
mkdir -p obsidian-milvus-ollama/static/js
mkdir -p obsidian-milvus-ollama/templates

# 환경 설정 파일 생성
cat > obsidian-milvus-ollama/.env << EOF
OBSIDIAN_VAULT_PATH=~/Obsidian/MyVault
MILVUS_HOST=localhost
MILVUS_PORT=19530
COLLECTION_NAME=obsidian_notes
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DIM=384
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
CHUNK_SIZE=512
CHUNK_OVERLAP=128
FLASK_PORT=5000
OPENWEBUI_URL=http://localhost:8080
EOF

# 시작 스크립트 생성
cat > obsidian-milvus-ollama/start.sh << EOF
#!/bin/bash

# Milvus 실행 (Podman 사용)
echo "Milvus 시작 중..."
podman compose -f milvus-podman-compose.yml up -d

# 웹 서버 시작
echo "웹 서버 시작 중..."
python main.py
EOF

chmod +x obsidian-milvus-ollama/start.sh

# Milvus Podman Compose 파일 생성
cat > obsidian-milvus-ollama/milvus-podman-compose.yml << EOF
version: '3'
services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.0
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - ${PODMAN_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ${PODMAN_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.3.4
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${PODMAN_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"
EOF

echo "설치 완료!"
echo "시스템을 시작하려면: cd obsidian-milvus-ollama && ./start.sh"