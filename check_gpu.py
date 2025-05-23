"""
GPU 상태 확인 스크립트
이 스크립트는 PyTorch에서 CUDA GPU를 인식하고 사용할 수 있는지 확인합니다.
"""

import torch
import sys

def check_gpu_status():
    """GPU 상태 확인 및 정보 출력"""
    print("\n===== GPU 상태 확인 =====")
    
    # CUDA 사용 가능 여부 확인
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 사용 가능: {cuda_available}")
    
    if not cuda_available:
        print("CUDA를 사용할 수 없습니다. 다음을 확인하세요:")
        print("1. NVIDIA GPU가 설치되어 있는지")
        print("2. NVIDIA 드라이버가 설치되어 있는지")
        print("3. CUDA 툴킷이 설치되어 있는지")
        print("4. PyTorch가 CUDA 버전으로 설치되어 있는지")
        return False
    
    # GPU 정보 출력
    gpu_count = torch.cuda.device_count()
    print(f"사용 가능한 GPU 개수: {gpu_count}")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB 단위
        print(f"GPU {i}: {gpu_name} (메모리: {gpu_memory:.2f} GB)")
    
    # CUDA 버전 확인
    cuda_version = torch.version.cuda
    print(f"CUDA 버전: {cuda_version}")
    
    # PyTorch 버전 확인
    pytorch_version = torch.__version__
    print(f"PyTorch 버전: {pytorch_version}")
    
    # 간단한 GPU 테스트
    try:
        print("\n간단한 GPU 테스트 실행 중...")
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"테스트 결과 (행렬 곱 결과의 합): {z.sum().item()}")
        print("GPU 테스트 성공!")
        return True
    except Exception as e:
        print(f"GPU 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    success = check_gpu_status()
    sys.exit(0 if success else 1)
