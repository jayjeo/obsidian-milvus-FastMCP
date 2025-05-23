"""
GPU 메모리 사용률 최적화 및 CPU RAM 사용률 절감 스크립트
이 스크립트는 임베딩 처리 과정에서 GPU 메모리 사용률을 높이고 CPU RAM 사용률을 낮춥니다.
"""

import os
import sys
import time
import torch
import psutil
import numpy as np
import logging
import gc
import config
import threading
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Dict, Any, Tuple
from gpu_optimizer import (
    initialize_gpu_optimization,
    get_gpu_memory_stats,
    print_gpu_memory_usage,
    optimize_tensor_operations
)

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL if hasattr(config, 'LOG_LEVEL') else 'WARNING'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 임베딩 관련 로그 제한
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('pytorch_pretrained_bert').setLevel(logging.ERROR)

# 특정 디버그 메시지 출력 방지를 위한 추가 설정
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# 모델 로딩 시 디버그 메시지 제거
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

def print_system_stats():
    """시스템 리소스 사용량 출력"""
    # CPU 사용량
    cpu_percent = psutil.cpu_percent()
    cpu_bar = generate_bar(cpu_percent, 20)
    
    # 메모리 사용량
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    memory_bar = generate_bar(memory_percent, 20)
    memory_used_gb = (memory.total - memory.available) / (1024**3)
    memory_total_gb = memory.total / (1024**3)
    
    print("\nSystem Resources:")
    print(f"CPU Usage:   [{cpu_bar}]  {cpu_percent:.1f}%")
    print(f"Memory:      [{memory_bar}]  {memory_percent:.1f}%")
    
    # GPU 사용량
    if torch.cuda.is_available():
        try:
            # NVIDIA-SMI를 통한 GPU 사용량 확인 시도 (Windows/Linux)
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                      stdout=subprocess.PIPE, text=True, check=True, timeout=1)
                gpu_info = result.stdout.strip().split(',')
                gpu_util = float(gpu_info[0])
                gpu_mem_used = float(gpu_info[1])
                gpu_mem_total = float(gpu_info[2])
                
                gpu_util_bar = generate_bar(gpu_util, 20)
                gpu_mem_percent = (gpu_mem_used / gpu_mem_total) * 100
                gpu_mem_bar = generate_bar(gpu_mem_percent, 20)
                
                print(f"GPU Usage:    [{gpu_util_bar}]  {gpu_util}")
                print(f"GPU Memory:   {gpu_mem_used} MB / {gpu_mem_total} MB")
                print(f"GPU Memory %: [{gpu_mem_bar}]  {gpu_mem_percent:.1f}%")
            except:
                # PyTorch 방식으로 GPU 정보 가져오기
                gpu_memory_info = torch.cuda.mem_get_info(0)
                free_memory = gpu_memory_info[0]
                total_memory = gpu_memory_info[1]
                used_memory = total_memory - free_memory
                
                # MB 단위로 변환
                gpu_mem_used = used_memory / (1024**2)
                gpu_mem_total = total_memory / (1024**2)
                gpu_mem_percent = (used_memory / total_memory) * 100
                
                # GPU 사용률은 추정값 사용
                gpu_util = 10.0  # 기본값
                try:
                    # 현재 프로세스의 GPU 사용률 추정
                    allocated_memory_ratio = torch.cuda.memory_allocated(0) / total_memory
                    gpu_util = min(allocated_memory_ratio * 100, 100.0)
                except:
                    pass
                
                gpu_util_bar = generate_bar(gpu_util, 20)
                
                print(f"GPU Usage:    [{gpu_util_bar}]  {gpu_util:.1f}")
                print(f"GPU Memory:   {gpu_mem_used:.1f} MB / {gpu_mem_total:.1f} MB")
        except Exception as e:
            # GPU 최적화 모듈 사용
            try:
                stats = get_gpu_memory_stats()
                gpu_mem_used = stats["allocated_memory_gb"] * 1024  # GB를 MB로 변환
                gpu_mem_total = stats["total_memory_gb"] * 1024  # GB를 MB로 변환
                gpu_mem_percent = stats["utilization_percent"]
                gpu_util = 10.0  # 기본값
                
                gpu_util_bar = generate_bar(gpu_util, 20)
                
                print(f"GPU Usage:    [{gpu_util_bar}]  {gpu_util:.1f}")
                print(f"GPU Memory:   {gpu_mem_used:.1f} MB / {gpu_mem_total:.1f} MB")
            except:
                print(f"GPU:         정보를 가져올 수 없음 ({e})")
    else:
        print("GPU:         Not available")

def generate_bar(percent, length=20):
    """퍼센트 값에 따른 진행 막대 생성"""
    filled_length = int(length * percent / 100)
    bar = '█' * filled_length + '░' * (length - filled_length)
    return bar


class StreamingTextDataset(Dataset):
    """텍스트 데이터를 스트리밍 방식으로 처리하는 데이터셋"""
    
    def __init__(self, texts=None, file_path=None, max_length=5000):
        """
        초기화 함수
        Args:
            texts: 텍스트 리스트 (메모리에 있는 경우)
            file_path: 텍스트 파일 경로 (파일에서 읽어오는 경우)
            max_length: 최대 텍스트 길이
        """
        self.texts = texts
        self.file_path = file_path
        self.max_length = max_length
        
        if texts is not None:
            self.length = len(texts)
        elif file_path is not None:
            # 파일 라인 수 계산 (대용량 파일의 경우 시간이 오래 걸릴 수 있음)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.length = sum(1 for _ in f)
            except Exception as e:
                logger.error(f"파일 라인 수 계산 오류: {e}")
                self.length = 0
        else:
            self.length = 0
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.texts is not None:
            # 메모리에서 가져오기
            text = self.texts[idx]
        elif self.file_path is not None:
            # 파일에서 특정 라인만 읽기
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i == idx:
                            text = line.strip()
                            break
                    else:
                        text = ""  # 해당 인덱스가 없는 경우
            except Exception as e:
                logger.error(f"파일 읽기 오류: {e}")
                text = ""
        else:
            text = ""
        
        # 텍스트 길이 제한
        if len(text) > self.max_length:
            text = text[:self.max_length]
        
        return text


def adjust_batch_size(current_batch_size=None, base_batch_size=None):
    """시스템 상태에 따라 배치 크기 동적 조정"""
    # 기본값 설정
    if current_batch_size is None:
        current_batch_size = getattr(config, 'EMBEDDING_BATCH_SIZE', 8192)
    
    if base_batch_size is None:
        base_batch_size = current_batch_size
    
    # CPU 모드에서는 작은 배치 사용
    if not torch.cuda.is_available() or not getattr(config, 'USE_GPU', True):
        return min(128, base_batch_size)
    
    try:
        # GPU 메모리 상태 확인
        gpu_memory_info = torch.cuda.mem_get_info(0)
        free_memory = gpu_memory_info[0]
        total_memory = gpu_memory_info[1]
        gpu_usage_percent = 100 - (free_memory / total_memory * 100)
        
        # RAM 상태 확인
        ram_info = psutil.virtual_memory()
        ram_usage_percent = ram_info.percent
        
        # 제한 자원 확인 (GPU 메모리 또는 RAM)
        limiting_resource = "GPU" if gpu_usage_percent > ram_usage_percent else "RAM"
        limiting_percent = max(gpu_usage_percent, ram_usage_percent)
        
        logger.info(f"자원 사용률 - GPU: {gpu_usage_percent:.1f}%, RAM: {ram_usage_percent:.1f}%, 제한 자원: {limiting_resource}")
        
        # 배치 크기 조정 로직
        if limiting_percent < 20:  # 자원 사용률이 매우 낮음
            new_batch_size = min(current_batch_size * 2, 16384)  # 최대 16384
            logger.info(f"자원 사용률이 매우 낮아 배치 크기 증가: {current_batch_size} -> {new_batch_size}")
        elif limiting_percent < 40:  # 자원 사용률이 낮음
            new_batch_size = min(int(current_batch_size * 1.5), 12288)
            logger.info(f"자원 사용률이 낮아 배치 크기 증가: {current_batch_size} -> {new_batch_size}")
        elif limiting_percent < 70:  # 자원 사용률이 적절함
            new_batch_size = current_batch_size
            logger.info(f"자원 사용률이 적절하여 배치 크기 유지: {new_batch_size}")
        elif limiting_percent < 85:  # 자원 사용률이 높음
            new_batch_size = max(int(current_batch_size * 0.8), 32)
            logger.info(f"자원 사용률이 높아 배치 크기 감소: {current_batch_size} -> {new_batch_size}")
        else:  # 자원 사용률이 매우 높음
            new_batch_size = max(int(current_batch_size * 0.5), 16)
            logger.info(f"자원 사용률이 매우 높아 배치 크기 대폭 감소: {current_batch_size} -> {new_batch_size}")
        
        # GPU 메모리 사용률이 너무 낮은 경우 (10% 미만) 배치 크기 대폭 증가
        if gpu_usage_percent < 10:
            new_batch_size = min(current_batch_size * 4, 16384)
            logger.info(f"GPU 사용률이 매우 낮아 배치 크기 대폭 증가: {current_batch_size} -> {new_batch_size}")
        
        return new_batch_size
        
    except Exception as e:
        logger.error(f"배치 크기 조정 오류: {e}")
        return current_batch_size  # 오류 발생 시 현재 배치 크기 유지

def optimize_model(model, device):
    """모델 최적화 (PyTorch 2.0 이상)"""
    # 모델을 지정된 디바이스로 이동
    model = model.to(device)
    
    # PyTorch 2.0 이상에서 torch.compile 사용
    if hasattr(torch, 'compile') and 'cuda' in str(device):
        try:
            model = torch.compile(model)
            logger.info("모델 컴파일 완료 (torch.compile)")
        except Exception as e:
            logger.warning(f"모델 컴파일 실패: {e}")
    
    return model


def encode_texts_with_amp(model, texts, batch_size, device):
    """혼합 정밀도를 사용한 텍스트 인코딩"""
    embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    # 임시로 로그 레벨 업
    original_log_level = logger.level
    logger.setLevel(logging.INFO)
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_idx = i // batch_size + 1
        
        if batch_idx % 10 == 0 or batch_idx == 1 or batch_idx == total_batches:
            logger.info(f"배치 처리 중: {batch_idx}/{total_batches} (배치 크기: {len(batch_texts)})")
        
        # 혼합 정밀도 적용
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # 임시로 로그 레벨 높임
                st_logger = logging.getLogger('sentence_transformers')
                tf_logger = logging.getLogger('transformers')
                original_st_level = st_logger.level
                original_tf_level = tf_logger.level
                
                st_logger.setLevel(logging.ERROR)
                tf_logger.setLevel(logging.ERROR)
                
                try:
                    # 표준 출력 임시 리디렉션 (디버그 메시지 숨김)
                    import sys
                    original_stdout = sys.stdout
                    original_stderr = sys.stderr
                    sys.stdout = open(os.devnull, 'w')
                    sys.stderr = open(os.devnull, 'w')
                    
                    try:
                        batch_embeddings = model.encode(
                            batch_texts,
                            convert_to_tensor=True,
                            device=device,
                            show_progress_bar=False
                        )
                    finally:
                        # 표준 출력 복원
                        sys.stdout.close()
                        sys.stderr.close()
                        sys.stdout = original_stdout
                        sys.stderr = original_stderr
                finally:
                    # 로그 레벨 복원
                    st_logger.setLevel(original_st_level)
                    tf_logger.setLevel(original_tf_level)
        
        # CPU로 이동 및 리스트로 변환 (메모리 관리)
        batch_embeddings = batch_embeddings.cpu().numpy()
        embeddings.extend(batch_embeddings)
        
        # 메모리 정리
        if batch_idx % 5 == 0:  # 5개 배치마다 메모리 정리
            gc.collect()
            if 'cuda' in str(device):
                torch.cuda.empty_cache()
    
    # 로그 레벨 복원
    logger.setLevel(original_log_level)
    
    return embeddings


def process_embeddings_with_dataloader(model, texts, batch_size, device):
    """데이터로더를 사용한 임베딩 처리 최적화"""
    # 데이터셋 생성
    dataset = StreamingTextDataset(texts=texts)
    
    # 데이터 로더 설정
    num_workers = min(os.cpu_count() or 4, 8)  # CPU 코어 수에 따라 조정
    pin_memory = 'cuda' in str(device)  # GPU 사용 시 pin_memory 활성화
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )
    
    embeddings = []
    total_batches = len(dataloader)
    
    # 혼합 정밀도 사용 여부
    use_amp = 'cuda' in str(device)
    
    # 임시로 로그 레벨 업
    original_log_level = logger.level
    logger.setLevel(logging.INFO)
    
    # 임시로 로그 레벨 높임
    st_logger = logging.getLogger('sentence_transformers')
    tf_logger = logging.getLogger('transformers')
    torch_logger = logging.getLogger('torch')
    original_st_level = st_logger.level
    original_tf_level = tf_logger.level
    original_torch_level = torch_logger.level
    
    st_logger.setLevel(logging.ERROR)
    tf_logger.setLevel(logging.ERROR)
    torch_logger.setLevel(logging.ERROR)
    
    try:
        for batch_idx, batch_texts in enumerate(dataloader):
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0 or batch_idx == total_batches - 1:
                logger.info(f"배치 처리 중: {batch_idx+1}/{total_batches} (배치 크기: {len(batch_texts)})")
            
            # 혼합 정밀도 적용 (GPU 사용 시)
            # 표준 출력 임시 리디렉션 (디버그 메시지 숨김)
            import sys
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
            
            try:
                if use_amp:
                    with torch.cuda.amp.autocast():
                        with torch.no_grad():
                            batch_embeddings = model.encode(
                                batch_texts,
                                convert_to_tensor=True,
                                device=device,
                                show_progress_bar=False
                            )
                else:
                    with torch.no_grad():
                        batch_embeddings = model.encode(
                            batch_texts,
                            convert_to_tensor=True,
                            device=device,
                            show_progress_bar=False
                        )
            finally:
                # 표준 출력 복원
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = original_stdout
                sys.stderr = original_stderr
            
            # CPU로 이동 및 리스트로 변환 (메모리 관리)
            batch_embeddings = batch_embeddings.cpu().numpy()
            embeddings.extend(batch_embeddings)
            
            # 메모리 정리 (필요 시)
            if (batch_idx + 1) % 5 == 0:
                gc.collect()
                if 'cuda' in str(device):
                    torch.cuda.empty_cache()
    finally:
        # 로그 레벨 복원
        logger.setLevel(original_log_level)
        st_logger.setLevel(original_st_level)
        tf_logger.setLevel(original_tf_level)
        torch_logger.setLevel(original_torch_level)
    
    return embeddings


def create_parallel_tensors(count=8, base_size=4096):
    """병렬 텐서를 생성하여 GPU 메모리와 계산 부하를 증가시키는 함수"""
    tensors = []
    try:
        for i in range(count):
            # 약간씩 다른 크기의 텐서 생성 (메모리 단편화 방지)
            size = base_size + (i * 128)
            t = torch.randn(size, size, device="cuda")
            tensors.append(t)
        return tensors
    except RuntimeError as e:
        logger.error(f"텐서 생성 오류: {e}")
        # 부분적으로 생성된 텐서 반환
        return tensors

def perform_heavy_computation(tensors):
    """여러 텐서에 대한 고부하 연산 수행"""
    results = []
    try:
        for i, tensor in enumerate(tensors):
            # 다양한 고부하 연산 수행
            if i % 4 == 0:
                # 행렬 곱셈
                result = torch.matmul(tensor, tensor)
            elif i % 4 == 1:
                # FFT 연산
                result = torch.fft.rfft2(tensor)
            elif i % 4 == 2:
                # 복소수 연산
                complex_tensor = torch.complex(tensor, torch.randn_like(tensor))
                result = torch.view_as_real(complex_tensor * complex_tensor)
            else:
                # 병렬 연산
                result = torch.nn.functional.normalize(
                    torch.nn.functional.softmax(tensor, dim=1), p=2, dim=1)
            
            # 결과 저장
            results.append(result)
            # 진행 상황 출력
            print(f"텐서 {i+1}/{len(tensors)} 연산 완료")
        
        return results
    except RuntimeError as e:
        logger.error(f"연산 오류: {e}")
        return results

def maintain_high_gpu_memory(duration=30):
    """지정된 시간 동안 GPU 메모리 사용률을 높게 유지"""
    logger.info(f"{duration}초 동안 GPU 메모리 사용률을 높게 유지합니다...")
    
    try:
        # 여러 크기의 텐서를 생성하여 메모리 사용
        tensors = []
        
        # 남은 메모리 계산
        stats = get_gpu_memory_stats()
        total_memory = stats["total_memory_gb"] * 1024**3
        allocated_memory = stats["allocated_memory_gb"] * 1024**3
        free_memory = total_memory - allocated_memory
        
        # 다양한 크기의 텐서 생성 (총 메모리의 98%까지)
        target_memory = free_memory * 0.98
        used_memory = 0
        
        # 텐서 크기 리스트 (4MB부터 8GB까지 지수적으로 증가)
        sizes = [2**i for i in range(12, 24)]  # 4KB에서 8MB까지
        
        for size in sizes:
            # 메모리 한계 확인
            if used_memory >= target_memory:
                break
            
            # 텐서 생성 시도
            try:
                tensor_memory = size * size * 4  # float32 기준
                if used_memory + tensor_memory > target_memory:
                    # 남은 메모리에 맞게 크기 조정
                    adjusted_size = int(np.sqrt((target_memory - used_memory) / 4))
                    tensor = torch.randn(adjusted_size, adjusted_size, device="cuda")
                else:
                    tensor = torch.randn(size, size, device="cuda")
                
                # 메모리 사용량 업데이트
                used_memory += tensor.nelement() * tensor.element_size()
                tensors.append(tensor)
                
                logger.info(f"텐서 생성: {tensor.shape}, 총 메모리 사용: {used_memory/1024**3:.2f}GB")
            except RuntimeError as e:
                logger.error(f"텐서 생성 오류: {e}")
                break
        
        # 메모리 상태 출력
        print_gpu_memory_usage()
        print_system_stats()
        
        # 지정된 시간 동안 유지
        start_time = time.time()
        while time.time() - start_time < duration:
            # 텐서에 대한 연산 수행 (GPU 활성화)
            for i, tensor in enumerate(tensors):
                if i % 3 == 0:
                    # 행렬 제곱 (고부하 연산)
                    _ = torch.matmul(tensor, tensor)
                    torch.cuda.synchronize()
                elif i % 3 == 1:
                    # FFT 연산
                    _ = torch.fft.rfft2(tensor)
                    torch.cuda.synchronize()
                else:
                    # 정규화 연산
                    _ = torch.nn.functional.normalize(tensor, p=2, dim=0)
                    torch.cuda.synchronize()
                
            # 남은 시간 출력
            remaining = duration - (time.time() - start_time)
            print(f"\r메모리 유지 중... 남은 시간: {remaining:.1f}초", end="")
            
            # 상태 업데이트 (5초마다)
            if int(time.time()) % 5 == 0:
                print("\n")
                print_system_stats()
            
            # 짧은 대기
            time.sleep(0.1)
        
        print("\n메모리 유지 완료!")
        
        # 메모리 정리
        for tensor in tensors:
            del tensor
        torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"메모리 유지 오류: {e}")
        # 메모리 정리
        torch.cuda.empty_cache()

def extreme_gpu_memory_usage():
    """GPU 메모리 사용률을 극단적으로 높이는 함수"""
    if not torch.cuda.is_available():
        logger.error("GPU를 사용할 수 없습니다.")
        return False
    
    logger.info("GPU 메모리 사용률 극대화 시작...")
    
    # 다양한 CUDA 설정 최적화
    # 개선: 더 적극적인 설정
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    torch.set_float32_matmul_precision('highest')
    
    # GPU 최적화 초기화
    initialize_gpu_optimization()
    
    # 현재 GPU 메모리 상태 출력
    print_gpu_memory_usage()
    print_system_stats()
    
    # 텐서 연산 최적화 (GPU 워밍업)
    optimize_tensor_operations()
    
    # 대규모 텐서 연산으로 GPU 메모리 사용률 극대화
    try:
        # 사용 가능한 메모리 계산
        stats = get_gpu_memory_stats()
        total_memory = stats["total_memory_gb"] * 1024**3
        allocated_memory = stats["allocated_memory_gb"] * 1024**3
        free_memory = total_memory - allocated_memory
        
        # 메모리의 98%를 사용하는 텐서 크기 계산 (증가)
        tensor_size = int(np.sqrt(free_memory * 0.98 // 8))  # float32 텐서 2개 기준
        tensor_size = max(4096, min(tensor_size, 32768))  # 최소 4096, 최대 32768
        
        logger.info(f"대규모 텐서 연산 수행 중 (크기: {tensor_size}x{tensor_size})...")
        
        # 대규모 텐서 생성
        logger.info("텐서 A 생성 중...")
        a = torch.randn(tensor_size, tensor_size, device="cuda", dtype=torch.float32)
        logger.info("텐서 B 생성 중...")
        b = torch.randn(tensor_size, tensor_size, device="cuda", dtype=torch.float32)
        
        # 행렬 곱셈 수행 (GPU 메모리 및 연산 부하 증가)
        logger.info("행렬 곱셈 연산 시작...")
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # 추가 고부하 연산
        logger.info("FFT 연산 시작...")
        d = torch.fft.rfft2(a)
        torch.cuda.synchronize()
        
        logger.info("복합 연산 시작...")
        e = torch.nn.functional.softmax(c, dim=1)
        f = torch.nn.functional.normalize(e, p=2, dim=1)
        g = f @ d.real
        torch.cuda.synchronize()
        
        # 결과 확인 (실제 계산이 수행되도록)
        result_sum = g.sum().item()
        logger.info(f"복합 연산 결과 합계: {result_sum}")
        
        # 병렬 텐서 생성 및 고부하 연산 수행
        logger.info("병렬 텐서 생성 및 고부하 연산 시작...")
        parallel_tensors = create_parallel_tensors(count=4, base_size=tensor_size//4)
        parallel_results = perform_heavy_computation(parallel_tensors)
        
        # 메모리 상태 확인
        print_gpu_memory_usage()
        print_system_stats()
        
        # 고부하 상태 유지 (GPU 메모리 사용률 유지)
        maintain_high_gpu_memory(duration=60)  # 60초 동안 유지
        
        # 메모리 유지 (프로그램 종료 방지)
        logger.info("GPU 메모리 사용률 극대화 완료. Ctrl+C로 종료하세요.")
        
        # 메모리 유지하면서 주기적으로 상태 출력
        try:
            while True:
                time.sleep(5)
                print_system_stats()
        except KeyboardInterrupt:
            logger.info("사용자에 의해 종료됨")
        
        # 메모리 정리
        for tensor in parallel_tensors:
            del tensor
        for result in parallel_results:
            del result
        del a, b, c, d, e, f, g
        torch.cuda.empty_cache()
        
        return True
        
    except RuntimeError as e:
        logger.error(f"GPU 메모리 부족 오류: {e}")
        torch.cuda.empty_cache()
        return False

def optimize_embeddings(texts, model_name=None, device=None, batch_size=None):
    """임베딩 처리 최적화 함수"""
    # 모델 및 디바이스 설정
    if model_name is None:
        model_name = config.EMBEDDING_MODEL
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() and getattr(config, 'USE_GPU', True) else "cpu")
    
    # 초기 배치 크기 설정
    if batch_size is None:
        batch_size = getattr(config, 'EMBEDDING_BATCH_SIZE', 8192)
    
    logger.info(f"임베딩 최적화 시작 - 모델: {model_name}, 디바이스: {device}, 초기 배치 크기: {batch_size}")
    
    # 모델 로드 및 최적화
    logger.info("모델 로드 및 최적화 중...")
    model = SentenceTransformer(model_name, device=str(device))
    model = optimize_model(model, device)
    
    # 시스템 상태 출력
    print("\n초기 시스템 상태:")
    print_system_stats()
    
    # 배치 크기 동적 조정
    adjusted_batch_size = adjust_batch_size(batch_size)
    logger.info(f"동적 배치 크기 조정: {batch_size} → {adjusted_batch_size}")
    
    # 최적화된 임베딩 처리
    logger.info(f"{len(texts)}개 텍스트 임베딩 처리 시작...")
    start_time = time.time()
    
    # 데이터로더 사용 방식으로 임베딩 처리
    embeddings = process_embeddings_with_dataloader(model, texts, adjusted_batch_size, device)
    
    elapsed_time = time.time() - start_time
    
    # 결과 출력
    logger.info(f"임베딩 처리 완료: {len(texts)}개 텍스트, {elapsed_time:.2f}초 소요 (평균: {elapsed_time/len(texts)*1000:.2f}ms/텍스트)")
    
    # 최종 시스템 상태 출력
    print("\n최종 시스템 상태:")
    print_system_stats()
    
    return embeddings


def main():
    """메인 함수"""
    print("\n===== GPU 메모리 최적화 및 CPU RAM 절감 도구 =====")
    print("이 도구는 임베딩 처리 과정에서 GPU 메모리 사용률을 높이고 CPU RAM 사용률을 낮춥니다.")
    print("현재 사용 중인 임베딩 모델: " + config.EMBEDDING_MODEL)
    print("========================================\n")
    
    # GPU 정보 출력
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_capability = torch.cuda.get_device_capability(0)
        print(f"감지된 GPU: {device_name} (CUDA 컴퓨트 능력: {device_capability[0]}.{device_capability[1]})")
    else:
        print("경고: GPU를 사용할 수 없습니다. CPU 모드로 실행합니다.")
    
    # 시스템 상태 출력
    print("\n현재 시스템 상태:")
    print_system_stats()
    
    # 테스트 임베딩 실행
    print("\n테스트 임베딩 처리를 실행하시겠습니까? (y/n)")
    choice = input().strip().lower()
    
    if choice == 'y':
        # 테스트용 텍스트 생성 (100개)
        test_texts = [
            f"This is a test document {i} for embedding optimization. "
            f"It contains some text to process and generate embeddings. "
            f"The purpose is to test the GPU memory usage and CPU RAM usage during embedding."
            for i in range(100)
        ]
        
        # 테스트 임베딩 처리
        print("\n테스트 임베딩 처리 시작...")
        embeddings = optimize_embeddings(test_texts)
        
        print(f"\n테스트 완료: {len(embeddings)}개 임베딩 생성, 차원: {len(embeddings[0])}")
        return 0
    else:
        print("\n테스트를 실행하지 않습니다.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
