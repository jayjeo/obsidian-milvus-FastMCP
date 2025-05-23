"""
PyTorch CUDA 버전 설치 스크립트
이 스크립트는 CUDA 지원이 포함된 PyTorch를 설치합니다.
"""

import subprocess
import sys
import platform
import os

def install_pytorch_cuda():
    """CUDA 지원이 포함된 PyTorch 설치"""
    print("\n===== PyTorch CUDA 버전 설치 =====")
    
    # 현재 Python 버전 확인
    python_version = platform.python_version()
    print(f"Python 버전: {python_version}")
    
    # 운영체제 확인
    os_name = platform.system()
    print(f"운영체제: {os_name}")
    
    # CUDA 버전 선택 (최신 안정 버전)
    cuda_version = "11.8"
    print(f"설치할 CUDA 버전: {cuda_version}")
    
    # 설치 명령어 구성
    if os_name == "Windows":
        install_cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}"
        ]
    else:  # Linux/MacOS
        install_cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}"
        ]
    
    print("\n설치 명령어:", " ".join(install_cmd))
    print("\n설치 진행 중...")
    
    try:
        # 설치 실행 (실시간 출력을 위해 capture_output=False 사용)
        print("\n설치 진행 중... (이 과정은 몇 분 정도 소요될 수 있습니다)")
        process = subprocess.Popen(
            install_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 실시간으로 출력 표시
        for line in process.stdout:
            print(line, end='')
        
        # 프로세스 완료 대기
        return_code = process.wait()
        
        if return_code == 0:
            print("\n설치 성공!")
            return True
        else:
            print(f"\n설치 실패: 반환 코드 {return_code}")
            return False
    except Exception as e:
        print(f"\n설치 중 오류 발생: {e}")
        return False

def verify_installation():
    """설치 확인"""
    print("\n===== 설치 확인 =====")
    
    try:
        # PyTorch 임포트
        import torch
        print(f"PyTorch 버전: {torch.__version__}")
        
        # CUDA 사용 가능 여부 확인
        cuda_available = torch.cuda.is_available()
        print(f"CUDA 사용 가능: {cuda_available}")
        
        if cuda_available:
            # CUDA 버전 확인
            cuda_version = torch.version.cuda
            print(f"CUDA 버전: {cuda_version}")
            
            # GPU 정보 출력
            gpu_count = torch.cuda.device_count()
            print(f"사용 가능한 GPU 개수: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"GPU {i}: {gpu_name}")
            
            return True
        else:
            print("CUDA를 사용할 수 없습니다. 다음을 확인하세요:")
            print("1. NVIDIA GPU가 설치되어 있는지")
            print("2. NVIDIA 드라이버가 설치되어 있는지")
            print("3. CUDA 툴킷이 설치되어 있는지")
            return False
    except ImportError:
        print("PyTorch를 임포트할 수 없습니다. 설치가 실패했을 수 있습니다.")
        return False

if __name__ == "__main__":
    print("이 스크립트는 CUDA 지원이 포함된 PyTorch를 설치합니다.")
    print("계속하려면 Enter 키를 누르세요. 취소하려면 Ctrl+C를 누르세요.")
    input()
    
    success = install_pytorch_cuda()
    if success:
        verify_installation()
        
        # 설치 후 GPU 상태 확인 스크립트 실행
        print("\n\nGPU 상태 확인 스크립트를 실행합니다...")
        check_gpu_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_gpu.py")
        if os.path.exists(check_gpu_script):
            subprocess.run([sys.executable, check_gpu_script])
        else:
            print(f"GPU 상태 확인 스크립트를 찾을 수 없습니다: {check_gpu_script}")
    
    sys.exit(0 if success else 1)
