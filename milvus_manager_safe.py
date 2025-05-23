# Milvus Manager - Safe Mode Version
# 이 파일은 기존 milvus_manager.py를 대체할 임시 안전 모드 버전입니다.

import time
import socket
import threading
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, utility
import config
import logging

logger = logging.getLogger('MilvusManager-SafeMode')

class MilvusManagerSafeMode:
    """안전 모드 Milvus Manager - 컨테이너 시작 없이 연결만 시도"""
    
    def __init__(self):
        self.host = config.MILVUS_HOST
        self.port = config.MILVUS_PORT
        self.collection_name = config.COLLECTION_NAME
        self.dimension = getattr(config, 'VECTOR_DIM', 768)
        self.connection_lock = threading.Lock()
        self.pending_deletions = set()
        
        # 컨테이너 시작 없이 연결만 시도
        logger.info("Safe mode: Connecting to existing Milvus instance...")
        max_retries = 30
        for i in range(max_retries):
            if self.is_port_in_use(self.port):
                try:
                    self.connect()
                    self.ensure_collection()
                    logger.info("Successfully connected in safe mode")
                    return
                except Exception as e:
                    logger.warning(f"Connection attempt {i+1} failed: {e}")
            time.sleep(2)
        
        raise ConnectionError("Could not connect to Milvus in safe mode")
    
    def is_port_in_use(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    def connect(self):
        with self.connection_lock:
            try:
                if connections.has_connection():
                    return
            except:
                pass
            connections.connect(host=self.host, port=self.port)
            
    # 나머지 메서드들은 기존과 동일하게 구현...
    # (여기서는 핵심 부분만 표시)
