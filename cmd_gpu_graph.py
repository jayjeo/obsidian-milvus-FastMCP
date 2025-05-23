"""
CMD 창에서 GPU 사용량 및 메모리 사용량 그래프 출력
이 스크립트는 명령 프롬프트에서 ASCII 그래프로 GPU 사용량과 메모리 사용량을 표시합니다.
"""

import os
import time
import subprocess
import psutil
import sys
from datetime import datetime

# 그래프 문자
BLOCK_CHARS = ['░', '▒', '▓', '█']

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
        
        return gpu_util, gpu_mem_percent, gpu_mem_used, gpu_mem_total
    except (subprocess.SubprocessError, IndexError, ValueError) as e:
        print(f"GPU 정보 가져오기 실패: {e}")
        return 0, 0, 0, 0

def generate_bar(percent, length=20):
    """퍼센트 값에 따른 진행 막대 생성"""
    filled_length = int(length * percent / 100)
    bar = '█' * filled_length + '░' * (length - filled_length)
    return bar

def generate_horizontal_graph(data, max_value, width=70, title=""):
    """수평 그래프 생성"""
    result = f"{title}\n"
    for i, value in enumerate(data):
        bar_length = int((value / max_value) * width)
        bar = '█' * bar_length
        result += f"{i:2d} | {bar} {value:.1f}\n"
    return result

def generate_vertical_graph(data, max_value, height=10, width=70, labels=None):
    """수직 그래프 생성"""
    if not data:
        return "데이터 없음"
        
    # 그래프 초기화
    graph = [[' ' for _ in range(width)] for _ in range(height)]
    
    # 데이터 포인트 그리기
    for x, value in enumerate(data[-width:]):
        if x >= width:
            break
            
        # 값을 높이에 맞게 스케일링
        y_pos = height - 1 - int((value / max_value) * (height - 1))
        y_pos = max(0, min(y_pos, height - 1))
        
        # 그래프에 데이터 포인트 표시
        for y in range(y_pos, height):
            graph[y][x] = '█'
    
    # 그래프를 문자열로 변환
    result = ""
    for row in graph:
        result += ''.join(row) + '\n'
    
    # 축 레이블 추가
    axis = '-' * width
    result += axis + '\n'
    
    # 시간 레이블 추가
    if labels and len(labels) > 0:
        time_labels = ''
        step = max(1, len(labels) // 10)
        for i in range(0, min(len(labels), width), step):
            idx = len(labels) - width + i if len(labels) > width else i
            if idx >= 0 and idx < len(labels):
                label = f"{labels[idx]:.0f}s"
                time_labels += label + ' ' * (step - len(label))
        result += time_labels
    
    return result

def main():
    """메인 함수"""
    # 데이터 저장용 리스트
    timestamps = []
    cpu_percentages = []
    memory_percentages = []
    gpu_usages = []
    gpu_memory_usages = []
    
    # 시작 시간
    start_time = time.time()
    
    try:
        # GPU 정보 확인
        gpu_util, gpu_mem_percent, gpu_mem_used, gpu_mem_total = get_gpu_info()
        if gpu_mem_total == 0:
            print("GPU를 감지할 수 없습니다. NVIDIA GPU가 설치되어 있고 드라이버가 올바르게 설치되었는지 확인하세요.")
            return 1
            
        print(f"감지된 GPU: NVIDIA GPU ({gpu_mem_total:.1f} MB)")
        print("Ctrl+C를 눌러 종료하세요.")
        time.sleep(2)
        
        while True:
            # 현재 시간
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
            
            # 최대 표시할 데이터 포인트 수 (70개)
            max_points = 70
            if len(timestamps) > max_points:
                timestamps = timestamps[-max_points:]
                cpu_percentages = cpu_percentages[-max_points:]
                memory_percentages = memory_percentages[-max_points:]
                gpu_usages = gpu_usages[-max_points:]
                gpu_memory_usages = gpu_memory_usages[-max_points:]
            
            # 화면 지우기
            clear_screen()
            
            # 현재 상태 출력
            print(f"===== 시스템 리소스 모니터링 ({datetime.now().strftime('%H:%M:%S')}) =====")
            print(f"실행 시간: {int(current_time)}초")
            print()
            
            # 막대 그래프로 현재 상태 표시
            cpu_bar = generate_bar(cpu_percent)
            memory_bar = generate_bar(memory_percent)
            gpu_bar = generate_bar(gpu_util)
            gpu_mem_bar = generate_bar(gpu_mem_percent)
            
            print(f"CPU 사용량:    [{cpu_bar}] {cpu_percent:.1f}%")
            print(f"메모리 사용량:  [{memory_bar}] {memory_percent:.1f}%")
            print(f"GPU 사용량:    [{gpu_bar}] {gpu_util:.1f}%")
            print(f"GPU 메모리:    [{gpu_mem_bar}] {gpu_mem_percent:.1f}% ({gpu_mem_used:.1f} MB / {gpu_mem_total:.1f} MB)")
            print()
            
            # 수직 그래프 표시
            print("===== GPU 사용량 그래프 =====")
            print(generate_vertical_graph(gpu_usages, 100, 10, 70, timestamps))
            print()
            
            print("===== GPU 메모리 사용량 그래프 =====")
            print(generate_vertical_graph(gpu_memory_usages, 100, 10, 70, timestamps))
            
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
