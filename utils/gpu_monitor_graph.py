"""
GPU 사용량 및 메모리 사용량 그래프 모니터링 도구
이 스크립트는 실시간으로 GPU 사용량과 메모리 사용량을 그래프로 표시합니다.
"""

import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import psutil
import os
import sys
from datetime import datetime

# 데이터 저장용 리스트
timestamps = []
cpu_percentages = []
memory_percentages = []
gpu_usages = []
gpu_memory_usages = []

# 그래프 설정
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('시스템 리소스 모니터링', fontsize=16)

# 그래프 초기화
cpu_line, = ax1.plot([], [], label='CPU 사용량 (%)', color='blue')
memory_line, = ax1.plot([], [], label='메모리 사용량 (%)', color='green')
gpu_line, = ax2.plot([], [], label='GPU 사용량 (%)', color='red')
gpu_memory_line, = ax2.plot([], [], label='GPU 메모리 사용량 (%)', color='purple')

# 그래프 레이블 및 범례 설정
ax1.set_ylabel('사용률 (%)')
ax1.set_title('CPU 및 메모리 사용량')
ax1.legend(loc='upper left')
ax1.grid(True)

ax2.set_xlabel('시간 (초)')
ax2.set_ylabel('사용률 (%)')
ax2.set_title('GPU 사용량 및 메모리 사용량')
ax2.legend(loc='upper left')
ax2.grid(True)

# 데이터 수집 시작 시간
start_time = time.time()

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
        
        return gpu_util, gpu_mem_percent, gpu_mem_used, gpu_mem_total
    except (subprocess.SubprocessError, IndexError, ValueError) as e:
        print(f"GPU 정보 가져오기 실패: {e}")
        return 0, 0, 0, 0

def update(frame):
    """그래프 업데이트 함수"""
    current_time = time.time() - start_time
    
    # CPU 및 메모리 사용량 가져오기
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    
    # GPU 정보 가져오기
    gpu_util, gpu_mem_percent, gpu_mem_used, gpu_mem_total = get_gpu_info()
    
    # 데이터 저장
    timestamps.append(current_time)
    cpu_percentages.append(cpu_percent)
    memory_percentages.append(memory_percent)
    gpu_usages.append(gpu_util)
    gpu_memory_usages.append(gpu_mem_percent)
    
    # 콘솔에 현재 상태 출력
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"===== 시스템 리소스 모니터링 ({datetime.now().strftime('%H:%M:%S')}) =====")
    print(f"CPU 사용량: {cpu_percent:.1f}%")
    print(f"메모리 사용량: {memory_percent:.1f}%")
    print(f"GPU 사용량: {gpu_util:.1f}%")
    print(f"GPU 메모리: {gpu_mem_used:.1f} MB / {gpu_mem_total:.1f} MB ({gpu_mem_percent:.1f}%)")
    print("============================================")
    print("그래프 창을 닫거나 Ctrl+C를 눌러 종료하세요.")
    
    # 최대 표시할 데이터 포인트 수 (30초 분량)
    max_points = 30
    if len(timestamps) > max_points:
        timestamps.pop(0)
        cpu_percentages.pop(0)
        memory_percentages.pop(0)
        gpu_usages.pop(0)
        gpu_memory_usages.pop(0)
    
    # 그래프 데이터 업데이트
    cpu_line.set_data(timestamps, cpu_percentages)
    memory_line.set_data(timestamps, memory_percentages)
    gpu_line.set_data(timestamps, gpu_usages)
    gpu_memory_line.set_data(timestamps, gpu_memory_usages)
    
    # x축 범위 업데이트
    ax1.set_xlim(max(0, current_time - 30), max(30, current_time))
    ax2.set_xlim(max(0, current_time - 30), max(30, current_time))
    
    # y축 범위 업데이트
    ax1.set_ylim(0, 105)
    ax2.set_ylim(0, 105)
    
    return cpu_line, memory_line, gpu_line, gpu_memory_line

def main():
    """메인 함수"""
    print("\n===== GPU 사용량 및 메모리 사용량 그래프 모니터링 =====")
    print("이 도구는 실시간으로 GPU 사용량과 메모리 사용량을 그래프로 표시합니다.")
    print("그래프 창을 닫거나 Ctrl+C를 눌러 종료하세요.")
    print("========================================================\n")
    
    try:
        # GPU 정보 확인
        gpu_util, gpu_mem_percent, gpu_mem_used, gpu_mem_total = get_gpu_info()
        if gpu_mem_total == 0:
            print("GPU를 감지할 수 없습니다. NVIDIA GPU가 설치되어 있고 드라이버가 올바르게 설치되었는지 확인하세요.")
            return 1
            
        print(f"감지된 GPU: NVIDIA GPU ({gpu_mem_total:.1f} MB)")
        print("그래프 창이 열립니다. 잠시 기다려주세요...")
        
        # 애니메이션 설정 및 실행
        ani = FuncAnimation(fig, update, interval=1000, blit=True)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
        
        return 0
    except KeyboardInterrupt:
        print("\n사용자에 의해 종료됨")
        return 0
    except Exception as e:
        print(f"\n오류 발생: {e}")
        return 1

if __name__ == "__main__":
    # matplotlib 경고 무시
    import warnings
    warnings.filterwarnings("ignore")
    
    sys.exit(main())
