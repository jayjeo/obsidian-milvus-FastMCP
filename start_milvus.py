#!/usr/bin/env python3
"""
Milvus with Podman - Python Controller
"""

import subprocess
import os
import time
from config import *

class MilvusPodmanController:
    def __init__(self):
        self.podman_path = PODMAN_PATH
        self.compose_command = COMPOSE_COMMAND
        self.network = MILVUS_NETWORK
        self.volumes = MILVUS_VOLUMES
        self.images = IMAGES
        
    def run_command(self, cmd):
        """Execute command and return result"""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode == 0, result.stdout, result.stderr
        except Exception as e:
            return False, "", str(e)
    
    def check_podman(self):
        """Check if Podman is available"""
        success, stdout, stderr = self.run_command(f'"{self.podman_path}" --version')
        if success:
            print(f"✅ Podman found: {stdout.strip()}")
            return True
        else:
            print(f"❌ Podman not found at: {self.podman_path}")
            return False
    
    def start_machine(self):
        """Start Podman machine"""
        print("Starting Podman machine...")
        success, stdout, stderr = self.run_command(f'"{self.podman_path}" machine start')
        return success
    
    def create_network(self):
        """Create Milvus network"""
        print(f"Creating network: {self.network}")
        cmd = f'"{self.podman_path}" network exists {self.network} || "{self.podman_path}" network create {self.network}'
        return self.run_command(cmd)[0]
    
    def create_volumes(self):
        """Create persistent volumes"""
        print("Creating persistent volumes...")
        for name, volume in self.volumes.items():
            cmd = f'"{self.podman_path}" volume exists {volume} || "{self.podman_path}" volume create {volume}'
            success, _, _ = self.run_command(cmd)
            if success:
                print(f"  ✅ Volume {volume} ready")
            else:
                print(f"  ❌ Failed to create volume {volume}")
                return False
        return True
    
    def stop_containers(self):
        """Stop existing containers"""
        print("Stopping existing containers...")
        containers = ["milvus-standalone", "milvus-minio", "milvus-etcd"]
        for container in containers:
            self.run_command(f'"{self.podman_path}" stop {container}')
            self.run_command(f'"{self.podman_path}" rm {container}')
    
    def start_etcd(self):
        """Start etcd container"""
        print("[1/3] Starting etcd...")
        cmd = f'''"{self.podman_path}" run -d --name milvus-etcd --network {self.network} \
        -v {self.volumes["etcd"]}:/etcd \
        -e ETCD_AUTO_COMPACTION_MODE=revision \
        -e ETCD_AUTO_COMPACTION_RETENTION=1000 \
        -e ETCD_QUOTA_BACKEND_BYTES=4294967296 \
        --user 0:0 \
        {self.images["etcd"]} \
        etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd'''
        return self.run_command(cmd)[0]
    
    def start_minio(self):
        """Start MinIO container"""
        print("[2/3] Starting MinIO...")
        cmd = f'''"{self.podman_path}" run -d --name milvus-minio --network {self.network} \
        -v {self.volumes["minio"]}:/minio_data \
        -e MINIO_ACCESS_KEY=minioadmin \
        -e MINIO_SECRET_KEY=minioadmin \
        --user 0:0 \
        {self.images["minio"]} \
        server /minio_data'''
        return self.run_command(cmd)[0]
    
    def start_milvus(self):
        """Start Milvus container"""
        print("[3/3] Starting Milvus...")
        cmd = f'''"{self.podman_path}" run -d --name milvus-standalone --network {self.network} \
        -v {self.volumes["milvus"]}:/var/lib/milvus \
        -p {MILVUS_API_PORT}:{MILVUS_API_PORT} \
        -p {MILVUS_WEB_PORT}:{MILVUS_WEB_PORT} \
        -e ETCD_ENDPOINTS=milvus-etcd:2379 \
        -e MINIO_ADDRESS=milvus-minio:9000 \
        --user 0:0 \
        {self.images["milvus"]} \
        milvus run standalone'''
        return self.run_command(cmd)[0]
    
    def check_status(self):
        """Check container status"""
        print("\nContainer Status:")
        success, stdout, stderr = self.run_command(f'"{self.podman_path}" ps --format "table {{{{.Names}}}}\\t{{{{.Status}}}}\\t{{{{.Ports}}}}"')
        if success:
            print(stdout)
        return success
    
    def start_all(self):
        """Start complete Milvus stack"""
        print("="*60)
        print("         Starting Milvus with Podman (Python)")
        print("="*60)
        
        # Check prerequisites
        if not self.check_podman():
            return False
        
        # Start machine
        self.start_machine()
        
        # Setup infrastructure
        self.stop_containers()
        if not self.create_network():
            print("❌ Failed to create network")
            return False
        
        if not self.create_volumes():
            print("❌ Failed to create volumes")
            return False
        
        # Start services
        if not self.start_etcd():
            print("❌ Failed to start etcd")
            return False
        
        if not self.start_minio():
            print("❌ Failed to start MinIO")
            return False
        
        print("Waiting for dependencies...")
        time.sleep(15)
        
        if not self.start_milvus():
            print("❌ Failed to start Milvus")
            return False
        
        print("\nWaiting for services to be ready...")
        time.sleep(20)
        
        # Check final status
        self.check_status()
        
        print("\n" + "="*60)
        print("                    🎉 SUCCESS! 🎉")
        print("="*60)
        print(f"🌐 Milvus API:    http://localhost:{MILVUS_API_PORT}")
        print(f"🌐 Web Interface: http://localhost:{MILVUS_WEB_PORT}")
        print("💾 Data is persistent across restarts")
        print("="*60)
        
        return True

if __name__ == "__main__":
    controller = MilvusPodmanController()
    controller.start_all()