"""
GPU 메모리 사용 최적화 스크립트
이 스크립트는 GPU 메모리 사용률을 극대화하여 처리 속도를 향상시킵니다.
"""

import os
import torch
import gc
import config
import logging
import numpy as np
from typing import Optional

# 로깅 설정
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class GPUOptimizer:
    """GPU 메모리 사용을 극대화하는 최적화 클래스"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu")
        self.initialized = False
        self.reserved_memory = None
        self.original_memory_stats = None
        self.tensor_cache = []  # 텐서 캐시 추가
    
    def initialize(self):
        """GPU 최적화 초기화"""
        if not torch.cuda.is_available() or not config.USE_GPU:
            logger.warning("GPU를 사용할 수 없습니다. CPU 모드로 실행합니다.")
            return False
            
        if self.initialized:
            return True
            
        logger.info("GPU 메모리 사용 최적화 시작...")
        
        # 현재 GPU 메모리 상태 저장
        self.original_memory_stats = self.get_gpu_memory_stats()
        
        # CUDA 메모리 할당 전략 변경
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.8'
        
        # cuDNN 최적화 설정
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # 텐서 코어 강제 사용 (Ampere 아키텍처 이상에서 작동)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # 혼합 정밀도 비활성화 (FP32 사용으로 메모리 사용량 증가)
        torch.set_float32_matmul_precision('highest')
        
        # 모든 가능한 CUDA 스트림 설정 최적화
        torch.cuda.set_device(config.GPU_DEVICE_ID)
        
        # 메모리 단편화 방지를 위한 초기 할당
        self._fragment_memory()
        
        # GPU 메모리 예약 (사용 가능한 메모리의 대부분을 미리 할당)
        self._reserve_gpu_memory(fraction=0.98)  # 98%로 증가
        
        # 추가 텐서 연산 최적화
        self._preload_tensor_kernels()
        
        self.initialized = True
        logger.info(f"GPU 최적화 완료: {torch.cuda.get_device_name(config.GPU_DEVICE_ID)}")
        
        # 최적화 후 메모리 상태 출력
        self.print_gpu_memory_usage()
        
        return True
    
    def _fragment_memory(self):
        """메모리 단편화를 방지하기 위한 초기 할당 패턴"""
        logger.info("메모리 단편화 방지 초기화 중...")
        
        try:
            # 다양한 크기의 텐서를 할당하여 메모리 단편화 방지
            sizes = [128, 256, 512, 1024, 2048, 4096]
            for size in sizes:
                # 텐서를 생성하고 바로 해제
                tensor = torch.randn(size, size, device=self.device)
                del tensor
                
            # 메모리 정리
            torch.cuda.empty_cache()
            logger.info("메모리 단편화 방지 초기화 완료")
        except RuntimeError as e:
            logger.warning(f"메모리 단편화 방지 초기화 실패: {e}")
    
    def _preload_tensor_kernels(self):
        """CUDA 커널 예열을 위한 다양한 텐서 연산 수행"""
        logger.info("CUDA 커널 예열 중...")
        
        try:
            # 작은 크기로 다양한 텐서 연산 수행
            a = torch.randn(1024, 1024, device=self.device)
            b = torch.randn(1024, 1024, device=self.device)
            
            # 다양한 연산으로 커널 로드
            operations = [
                lambda: torch.matmul(a, b),
                lambda: a + b,
                lambda: a * b,
                lambda: torch.nn.functional.relu(a),
                lambda: torch.nn.functional.softmax(a, dim=1),
                lambda: torch.nn.functional.normalize(a, p=2, dim=1),
                lambda: torch.fft.rfft2(a),
                lambda: torch.transpose(a, 0, 1)
            ]
            
            for op in operations:
                result = op()
                # 실제 계산이 수행되도록 함
                _ = result.sum().item()
                del result
            
            # 메모리 정리
            del a, b
            torch.cuda.empty_cache()
            
            logger.info("CUDA 커널 예열 완료")
        except Exception as e:
            logger.warning(f"CUDA 커널 예열 실패: {e}")
    
    def _reserve_gpu_memory(self, fraction: float = 0.98):
        """
        GPU 메모리를 미리 할당하여 메모리 사용률을 높임
        
        Args:
            fraction: 할당할 GPU 메모리 비율 (0.0-1.0)
        """
        if not torch.cuda.is_available():
            return
            
        # 현재 사용 가능한 메모리 계산
        total_memory = torch.cuda.get_device_properties(config.GPU_DEVICE_ID).total_memory
        allocated_memory = torch.cuda.memory_allocated(config.GPU_DEVICE_ID)
        free_memory = total_memory - allocated_memory
        
        # 예약할 메모리 계산 (전체 메모리의 fraction 비율)
        reserve_size = int(free_memory * fraction)
        
        try:
            # 복잡한 메모리 할당 패턴 사용
            logger.info(f"GPU 메모리 {reserve_size / (1024**3):.2f} GB 예약 중...")
            
            # 멀티 텐서 할당을 통한, 메모리 사용 패턴 최적화
            chunk_size = reserve_size // 5  # 5개의 다른 텐서로 분할
            
            # 다양한 크기의 텐서 할당
            self.tensor_cache = []
            sizes = [
                (chunk_size // 8, torch.float32),      # 1/8 크기의 FP32 텐서
                (chunk_size // 4, torch.float32),      # 1/4 크기의 FP32 텐서
                (chunk_size // 2, torch.float32),      # 1/2 크기의 FP32 텐서
                (chunk_size, torch.float32),           # 전체 크기의 FP32 텐서
                (chunk_size * 2, torch.int8),          # 2배 크기의 INT8 텐서 (메모리 적게 사용)
            ]
            
            for size, dtype in sizes:
                tensor = torch.zeros(size // max(1, torch.tensor([], dtype=dtype).element_size()), 
                                    dtype=dtype, device=self.device)
                self.tensor_cache.append(tensor)
            
            # 마지막으로 주 예약 메모리 텐서 할당
            remaining = free_memory - sum(t.nelement() * t.element_size() for t in self.tensor_cache)
            if remaining > 0:
                self.reserved_memory = torch.zeros(remaining // 4, dtype=torch.float32, device=self.device)
                self.tensor_cache.append(self.reserved_memory)
            
            # 메모리 할당 확인
            torch.cuda.synchronize()
            logger.info(f"GPU 메모리 예약 완료: {torch.cuda.memory_allocated(config.GPU_DEVICE_ID) / (1024**3):.2f} GB")
            
        except RuntimeError as e:
            logger.warning(f"메모리 예약 실패: {e}")
            # 메모리 정리
            self.release_cache()
            gc.collect()
            torch.cuda.empty_cache()
            
            # 더 작은 크기로 재시도
            try:
                reserve_size = int(free_memory * 0.9)  # 더 작은 비율로 재시도
                self.reserved_memory = torch.zeros(reserve_size // 4, dtype=torch.float32, device=self.device)
                torch.cuda.synchronize()
                logger.info(f"GPU 메모리 예약 재시도 성공: {torch.cuda.memory_allocated(config.GPU_DEVICE_ID) / (1024**3):.2f} GB")
            except RuntimeError:
                logger.error("GPU 메모리 예약 완전 실패")
    
    def release_cache(self):
        """텐서 캐시 해제"""
        if self.tensor_cache:
            logger.info("텐서 캐시 해제 중...")
            for tensor in self.tensor_cache:
                del tensor
            self.tensor_cache = []
    
    def release_reserved_memory(self):
        """예약된 GPU 메모리 해제"""
        self.release_cache()
        
        if self.reserved_memory is not None:
            logger.info("예약된 GPU 메모리 해제 중...")
            del self.reserved_memory
            self.reserved_memory = None
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"메모리 해제 후 GPU 메모리 사용량: {torch.cuda.memory_allocated(config.GPU_DEVICE_ID) / (1024**3):.2f} GB")
    
    def get_gpu_memory_stats(self):
        """GPU 메모리 통계 반환"""
        if not torch.cuda.is_available():
            return {"available": False}
            
        # 현재 CUDA 상태 동기화
        torch.cuda.synchronize()
        
        stats = {
            "available": True,
            "device_name": torch.cuda.get_device_name(config.GPU_DEVICE_ID),
            "total_memory_gb": torch.cuda.get_device_properties(config.GPU_DEVICE_ID).total_memory / (1024**3),
            "allocated_memory_gb": torch.cuda.memory_allocated(config.GPU_DEVICE_ID) / (1024**3),
            "cached_memory_gb": torch.cuda.memory_reserved(config.GPU_DEVICE_ID) / (1024**3),
            "free_memory_gb": (torch.cuda.get_device_properties(config.GPU_DEVICE_ID).total_memory - 
                              torch.cuda.memory_allocated(config.GPU_DEVICE_ID)) / (1024**3),
            "utilization_percent": torch.cuda.memory_allocated(config.GPU_DEVICE_ID) / 
                                  torch.cuda.get_device_properties(config.GPU_DEVICE_ID).total_memory * 100
        }
        return stats
    
    def print_gpu_memory_usage(self):
        """GPU 메모리 사용량 출력"""
        stats = self.get_gpu_memory_stats()
        if not stats["available"]:
            logger.warning("GPU를 사용할 수 없습니다.")
            return
            
        logger.info("===== GPU 메모리 사용 현황 =====")
        logger.info(f"GPU: {stats['device_name']}")
        logger.info(f"총 메모리: {stats['total_memory_gb']:.2f} GB")
        logger.info(f"할당된 메모리: {stats['allocated_memory_gb']:.2f} GB")
        logger.info(f"캐시된 메모리: {stats['cached_memory_gb']:.2f} GB")
        logger.info(f"남은 메모리: {stats['free_memory_gb']:.2f} GB")
        logger.info(f"메모리 사용률: {stats['utilization_percent']:.2f}%")
        logger.info("===============================")
    
    def optimize_tensor_operations(self, tensor_size: Optional[int] = None):
        """
        텐서 연산 최적화를 위한 대규모 연산 수행
        
        Args:
            tensor_size: 더미 텐서 크기 (None이면 자동 계산)
        """
        if not torch.cuda.is_available() or not config.USE_GPU:
            return
            
        # 텐서 크기 자동 계산 (남은 메모리의 80%를 사용)
        if tensor_size is None:
            free_memory = torch.cuda.get_device_properties(config.GPU_DEVICE_ID).total_memory - torch.cuda.memory_allocated(config.GPU_DEVICE_ID)
            tensor_size = int(np.sqrt(free_memory * 0.8 // 4))  # float32 텐서 기준
            tensor_size = max(4096, min(tensor_size, 32768))  # 최소 4096, 최대 32768
        
        logger.info(f"텐서 연산 최적화 중 (크기: {tensor_size}x{tensor_size})...")
        
        try:
            # 다양한 고부하 연산으로 GPU 활성화
            for _ in range(3):  # 여러 번 반복하여 GPU 워밍업
                # 대규모 행렬 곱셈
                a = torch.randn(tensor_size, tensor_size, device=self.device)
                b = torch.randn(tensor_size, tensor_size, device=self.device)
                
                # 행렬 곱셈 수행 (고부하 연산)
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
                
                # FFT 연산 (GPU 메모리 사용 증가)
                d = torch.fft.rfft2(a)
                torch.cuda.synchronize()
                
                # 복잡한 연산 조합 (GPU 사용률 극대화)
                e = torch.nn.functional.softmax(c, dim=1)
                f = torch.nn.functional.normalize(e, p=2, dim=1)
                g = f @ d.real
                torch.cuda.synchronize()
                
                # 결과 확인 (실제 계산이 수행되도록)
                result_sum = g.sum().item()
                
                # 메모리 정리 (다음 반복을 위해)
                del a, b, c, d, e, f, g
                
            logger.info("텐서 연산 최적화 완료")
            
        except RuntimeError as e:
            logger.warning(f"텐서 연산 최적화 실패: {e}")
            # 메모리 정리
            gc.collect()
            torch.cuda.empty_cache()
            
            # 더 작은 크기로 재시도
            try:
                tensor_size = tensor_size // 2
                # 간단한 연산만 시도
                a = torch.randn(tensor_size, tensor_size, device=self.device)
                b = torch.randn(tensor_size, tensor_size, device=self.device)
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
                del a, b, c
                logger.info("텐서 연산 최적화 재시도 성공 (작은 크기)")
            except RuntimeError:
                logger.error("텐서 연산 최적화 완전 실패")

# 싱글톤 인스턴스
gpu_optimizer = GPUOptimizer()

def initialize_gpu_optimization():
    """GPU 최적화 초기화 함수"""
    return gpu_optimizer.initialize()

def get_gpu_memory_stats():
    """GPU 메모리 통계 반환 함수"""
    return gpu_optimizer.get_gpu_memory_stats()

def print_gpu_memory_usage():
    """GPU 메모리 사용량 출력 함수"""
    gpu_optimizer.print_gpu_memory_usage()

def release_reserved_memory():
    """예약된 GPU 메모리 해제 함수"""
    gpu_optimizer.release_reserved_memory()

def optimize_tensor_operations(tensor_size=None):
    """텐서 연산 최적화 함수"""
    gpu_optimizer.optimize_tensor_operations(tensor_size)

if __name__ == "__main__":
    # 스크립트 직접 실행 시 GPU 최적화 테스트
    initialize_gpu_optimization()
    optimize_tensor_operations()
    print_gpu_memory_usage()
