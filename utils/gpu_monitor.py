"""
시스템 리소스 모니터링 도구
이 스크립트는 CPU, 메모리, GPU 사용량 및 GPU 메모리 사용량을 모니터링합니다.
"""

import os
import time
import subprocess
import psutil
import sys

def clear_screen():
    """화면 지우기"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_gpu_info():
    """NVIDIA-SMI를 사용하여 GPU 정보 가져오기"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, text=True, check=True
        )
        gpu_info = result.stdout.strip().split(',')
        gpu_util = float(gpu_info[0])
        gpu_mem_used = float(gpu_info[1])
        gpu_mem_total = float(gpu_info[2])
        gpu_mem_percent = (gpu_mem_used / gpu_mem_total) * 100
        
        return gpu_util, gpu_mem_used, gpu_mem_total, gpu_mem_percent
    except (subprocess.SubprocessError, IndexError, ValueError) as e:
        print(f"GPU 정보 가져오기 실패: {e}")
        return 0, 0, 0, 0

def generate_bar(percent, length=20):
    """퍼센트 값에 따른 진행 막대 생성"""
    filled_length = int(length * percent / 100)
    bar = '█' * filled_length + '░' * (length - filled_length)
    return bar

def main():
    """메인 함수"""
    try:
        while True:
            # CPU 및 메모리 사용량 가져오기
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # GPU 정보 가져오기
            gpu_util, gpu_mem_used, gpu_mem_total, gpu_mem_percent = get_gpu_info()
            
            # 화면 지우기
            clear_screen()
            
            # 막대 그래프 생성
            cpu_bar = generate_bar(cpu_percent)
            memory_bar = generate_bar(memory_percent)
            gpu_bar = generate_bar(gpu_util)
            gpu_mem_bar = generate_bar(gpu_mem_percent)
            
            # 정확히 요청된 형식으로 출력
            print("System Resources:")
            print(f"CPU Usage:   [{cpu_bar}]   {cpu_percent:.1f}%")
            print(f"Memory:      [{memory_bar}]  {memory_percent:.1f}%")
            print(f"GPU Usage:    [{gpu_bar}]  {gpu_util:.1f}%")
            print(f"GPU Memory:   [{gpu_mem_bar}]  {gpu_mem_percent:.1f}%")
            print(f"GPU Memory:   {gpu_mem_used:.1f} MB / {gpu_mem_total:.1f} MB")
            
            # 1초 대기
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n사용자에 의해 종료됨")
        return 0
    except Exception as e:
        print(f"\n오류 발생: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
