# Import warning suppressor first to suppress all warnings
import warning_suppressor

# Import NumPy without compatibility check as sentence-transformers now supports NumPy 2.x
try:
    import numpy as np
except ImportError:
    print("WARNING: NumPy not found")

from sentence_transformers import SentenceTransformer
import config
from functools import lru_cache
import hashlib
import numpy as np
import torch
import psutil
import os
import datetime
import time
import gc
import threading
import logging
import warnings
import re

# Additional warning suppression
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', module='huggingface_hub')
warnings.filterwarnings('ignore', module='transformers')

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL if hasattr(config, 'LOG_LEVEL') else 'WARNING'),
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class HardwareProfiler:
    """Enhanced hardware detection and profiling system for automatic optimization"""
    def __init__(self):
        self.cpu_cores = psutil.cpu_count()
        self.cpu_cores_physical = psutil.cpu_count(logical=False) or self.cpu_cores
        self.total_ram_gb = psutil.virtual_memory().total / (1024**3)
        self.gpu_info = self._detect_gpu()
        self.performance_profile = self._create_performance_profile()
        self.system_benchmark_score = self._calculate_benchmark_score()
        print(f"🔧 Hardware Profile: {self.performance_profile} (Score: {self.system_benchmark_score})")
        print(f"💻 CPU: {self.cpu_cores_physical}C/{self.cpu_cores}T, RAM: {self.total_ram_gb:.1f}GB")
        
    def _detect_gpu(self):
        """Enhanced GPU detection with detailed performance metrics"""
        if not torch.cuda.is_available():
            return {"available": False}
        
        gpu_count = torch.cuda.device_count()
        gpu_info = {
            "available": True,
            "count": gpu_count,
            "devices": []
        }
        
        for i in range(gpu_count):
            try:
                props = torch.cuda.get_device_properties(i)
                gpu_name = props.name
                memory_gb = props.total_memory / (1024**3)
                
                # Calculate TFLOPS and performance tier based on real data
                tflops, tier = self._calculate_gpu_performance(gpu_name, memory_gb)
                
                device_info = {
                    "name": gpu_name,
                    "memory_gb": memory_gb,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "tflops": tflops,
                    "performance_tier": tier
                }
                
                gpu_info["devices"].append(device_info)
                print(f"🎮 GPU {i}: {gpu_name} ({memory_gb:.1f}GB, {tflops:.1f} TFLOPS, Tier: {tier})")
                
            except Exception as e:
                print(f"Error detecting GPU {i}: {e}")
        
        return gpu_info
    
    def _calculate_gpu_performance(self, gpu_name, memory_gb):
        """Calculate GPU performance based on real TFLOPS data"""
        gpu_name_lower = gpu_name.lower()
        
        # NVIDIA RTX 50 Series (Blackwell - 2025)
        if "rtx 5090" in gpu_name_lower:
            return 125.0, "flagship"  # Shader TFLOPS (official NVIDIA spec)
        elif "rtx 5080" in gpu_name_lower:
            return 70.0, "ultra_high_end"  # Estimated shader TFLOPS
        elif "rtx 5070 ti" in gpu_name_lower:
            return 50.0, "high_end"  # Estimated shader TFLOPS
        elif "rtx 5070" in gpu_name_lower:
            return 38.0, "high_end"  # Estimated shader TFLOPS
        elif "rtx 5060 ti" in gpu_name_lower:
            return 25.0, "mid_range"  # Estimated
        elif "rtx 5060" in gpu_name_lower:
            return 18.0, "mid_range"  # Estimated
        
        # NVIDIA RTX 40 Series (Ada Lovelace)
        elif "rtx 4090" in gpu_name_lower:
            return 83.0, "flagship"
        elif "rtx 4080" in gpu_name_lower:
            return 49.0, "ultra_high_end"
        elif "rtx 4070" in gpu_name_lower:
            return 29.0, "high_end"  # Estimated based on architecture
        elif "rtx 4060" in gpu_name_lower:
            return 15.0, "mid_range"  # Estimated
        
        # NVIDIA RTX 30 Series (Ampere) - Using Shader TFLOPS
        elif "rtx 3090" in gpu_name_lower:
            return 35.6, "flagship"
        elif "rtx 3080" in gpu_name_lower:
            if memory_gb >= 12:  # RTX 3080 12GB
                return 30.0, "ultra_high_end"
            else:  # RTX 3080 10GB
                return 29.8, "ultra_high_end"
        elif "rtx 3070" in gpu_name_lower:
            return 20.3, "high_end"
        elif "rtx 3060 ti" in gpu_name_lower:
            return 16.2, "mid_range"
        elif "rtx 3060" in gpu_name_lower:
            return 12.7, "mid_range"
        
        # NVIDIA RTX 20 Series (Turing)
        elif "rtx 2080 ti" in gpu_name_lower:
            return 13.5, "high_end"
        elif "rtx 2080 super" in gpu_name_lower:
            return 11.2, "mid_range"
        elif "rtx 2080" in gpu_name_lower:
            return 10.1, "mid_range"
        elif "rtx 2070 super" in gpu_name_lower:
            return 9.1, "mid_range"
        elif "rtx 2070" in gpu_name_lower:
            return 7.5, "mid_range"
        elif "rtx 2060 super" in gpu_name_lower:
            return 7.2, "low_mid"
        elif "rtx 2060" in gpu_name_lower:
            return 6.5, "low_mid"
        
        # NVIDIA GTX 16 Series
        elif "gtx 1660 ti" in gpu_name_lower:
            return 5.4, "low_mid"
        elif "gtx 1660" in gpu_name_lower:
            return 5.0, "low_mid"
        elif "gtx 1650 super" in gpu_name_lower:
            return 4.4, "low_end"
        elif "gtx 1650" in gpu_name_lower:
            return 3.0, "low_end"
        
        # NVIDIA GTX 10 Series (Pascal)
        elif "gtx 1080 ti" in gpu_name_lower:
            return 11.3, "mid_range"
        elif "gtx 1080" in gpu_name_lower:
            return 8.9, "low_mid"
        elif "gtx 1070" in gpu_name_lower:
            return 6.5, "low_mid"
        elif "gtx 1060" in gpu_name_lower:
            return 4.4, "low_end"
        elif "gtx 1050 ti" in gpu_name_lower:
            return 2.1, "very_low_end"
        elif "gtx 1050" in gpu_name_lower:
            return 1.9, "very_low_end"
        
        # NVIDIA GTX 900 Series (Maxwell)
        elif "gtx 980 ti" in gpu_name_lower:
            return 6.1, "low_mid"
        elif "gtx 980" in gpu_name_lower:
            return 4.7, "low_end"
        elif "gtx 960" in gpu_name_lower:
            return 2.4, "very_low_end"
        elif "gtx 950" in gpu_name_lower:
            return 1.8, "very_low_end"
        
        # NVIDIA Professional/Tesla Cards
        elif "tesla v100" in gpu_name_lower:
            return 14.1, "professional"
        elif "a100" in gpu_name_lower:
            return 312.0, "professional"  # Tensor TFLOPS
        elif "h100" in gpu_name_lower:
            return 1000.0, "professional"  # Tensor TFLOPS
        
        # AMD Radeon RX 9000 Series (RDNA 4 - 2025)
        elif "rx 9070 xt" in gpu_name_lower:
            return 48.7, "high_end"  # 48.7 TFLOPS confirmed by AMD
        elif "rx 9070" in gpu_name_lower:
            return 36.4, "mid_range"  # Estimated based on reduced specs
        elif "rx 9060 xt" in gpu_name_lower:
            return 28.0, "mid_range"  # Estimated
        elif "rx 9060" in gpu_name_lower:
            return 20.0, "low_mid"  # Estimated
        
        # AMD Radeon RX 7000 Series (RDNA 3)
        elif "rx 7900 xtx" in gpu_name_lower:
            return 61.4, "flagship"
        elif "rx 7900 xt" in gpu_name_lower:
            return 51.5, "ultra_high_end"
        
        # AMD Radeon RX 6000 Series (RDNA 2)
        elif "rx 6900 xt" in gpu_name_lower:
            return 23.0, "high_end"
        elif "rx 6800 xt" in gpu_name_lower:
            return 20.7, "high_end"
        elif "rx 6800" in gpu_name_lower:
            return 16.2, "mid_range"
        
        # AMD Radeon RX 5000 Series (RDNA)
        elif "rx 5700 xt" in gpu_name_lower:
            return 9.8, "low_mid"
        elif "rx 5600 xt" in gpu_name_lower:
            return 7.2, "low_mid"
        elif "rx 5500 xt" in gpu_name_lower:
            return 5.2, "low_end"
        
        # AMD Radeon Vega Series
        elif "radeon vii" in gpu_name_lower:
            return 13.4, "mid_range"
        elif "vega 64" in gpu_name_lower:
            return 12.7, "mid_range"
        
        # AMD Radeon RX 500 Series
        elif "rx 580" in gpu_name_lower:
            return 6.2, "low_end"
        elif "rx 480" in gpu_name_lower:
            return 5.8, "low_end"
        
        # AMD Radeon R9 Series
        elif "r9 fury x" in gpu_name_lower:
            return 8.6, "low_mid"
        elif "r9 390x" in gpu_name_lower:
            return 5.9, "low_end"
        
        # AMD Professional Cards
        elif "instinct mi250x" in gpu_name_lower:
            return 326.0, "professional"  # Matrix TFLOPS
        
        # Fallback: Estimate based on memory and name patterns + Future GPU support
        else:
            # 🚀 Future GPU auto-detection based on model numbers
            future_gpu_detected = False
            estimated_tflops = 0
            estimated_tier = "unknown"
            
            # NVIDIA Future GPU Detection (RTX 60xx, 70xx, etc.)
            nvidia_match = re.search(r'rtx\s*(\d+)(\d{2})', gpu_name_lower)
            if nvidia_match:
                series = int(nvidia_match.group(1))
                model = int(nvidia_match.group(2))
                
                if series >= 60:  # RTX 6000 series and beyond
                    # Assume significant generational improvement
                    base_multiplier = 1.5 ** ((series - 50) / 10)  # 50% improvement per generation
                    
                    if model >= 90:
                        estimated_tflops = 125.0 * base_multiplier
                        estimated_tier = "flagship"
                    elif model >= 80:
                        estimated_tflops = 70.0 * base_multiplier
                        estimated_tier = "ultra_high_end"
                    elif model >= 70:
                        estimated_tflops = 45.0 * base_multiplier
                        estimated_tier = "high_end"
                    elif model >= 60:
                        estimated_tflops = 22.0 * base_multiplier
                        estimated_tier = "mid_range"
                    else:
                        estimated_tflops = 15.0 * base_multiplier
                        estimated_tier = "low_mid"
                    
                    future_gpu_detected = True
                    print(f"🔮 Future NVIDIA GPU detected: {gpu_name} - Estimated {estimated_tflops:.1f} TFLOPS ({estimated_tier})")
            
            # AMD Future GPU Detection (RX 10xxx, 11xxx, etc.)
            amd_match = re.search(r'rx\s*(\d+)(\d{2})', gpu_name_lower)
            if not future_gpu_detected and amd_match:
                series = int(amd_match.group(1))
                model = int(amd_match.group(2))
                
                if series >= 10:  # RX 10000 series and beyond
                    # Assume AMD continues competitive improvement
                    base_multiplier = 1.4 ** ((series - 90) / 10)  # 40% improvement per generation
                    
                    if model >= 90:
                        estimated_tflops = 65.0 * base_multiplier
                        estimated_tier = "flagship"
                    elif model >= 80:
                        estimated_tflops = 55.0 * base_multiplier
                        estimated_tier = "ultra_high_end"
                    elif model >= 70:
                        estimated_tflops = 45.0 * base_multiplier
                        estimated_tier = "high_end"
                    elif model >= 60:
                        estimated_tflops = 30.0 * base_multiplier
                        estimated_tier = "mid_range"
                    else:
                        estimated_tflops = 20.0 * base_multiplier
                        estimated_tier = "low_mid"
                    
                    future_gpu_detected = True
                    print(f"🔮 Future AMD GPU detected: {gpu_name} - Estimated {estimated_tflops:.1f} TFLOPS ({estimated_tier})")
            
            # If future GPU detected, return estimated performance
            if future_gpu_detected:
                return estimated_tflops, estimated_tier
            
            # Traditional memory-based estimation for unknown GPUs
            if memory_gb >= 20:  # High-end cards
                return max(40.0, memory_gb * 2), "flagship"
            elif memory_gb >= 16:
                return max(25.0, memory_gb * 1.8), "ultra_high_end"
            elif memory_gb >= 12:
                return max(15.0, memory_gb * 1.5), "high_end"
            elif memory_gb >= 8:
                return max(8.0, memory_gb * 1.2), "mid_range"
            elif memory_gb >= 6:
                return max(5.0, memory_gb * 1.0), "low_mid"
            elif memory_gb >= 4:
                return max(3.0, memory_gb * 0.8), "low_end"
            else:
                return max(1.5, memory_gb * 0.5), "very_low_end"
    
    def _create_performance_profile(self):
        """Enhanced system performance profile with TFLOPS-based GPU classification"""
        if self.gpu_info["available"] and len(self.gpu_info["devices"]) > 0:
            gpu_device = self.gpu_info["devices"][0]
            gpu_memory = gpu_device["memory_gb"]
            gpu_name = gpu_device["name"]
            tflops = gpu_device.get("tflops", 0)
            tier = gpu_device.get("performance_tier", "unknown")
            
            # 🚀 TFLOPS-based performance classification
            if tier == "professional":
                return "professional_gpu"  # Tesla, A100, H100 etc.
            elif tier == "flagship":
                return "flagship_gpu"      # RTX 4090, RX 7900 XTX etc.
            elif tier == "ultra_high_end":
                return "ultra_high_end_gpu" # RTX 4080, RTX 3080 etc.
            elif tier == "high_end":
                return "high_end_gpu"       # RTX 4070, RTX 3070, RX 6900 XT etc.
            elif tier == "mid_range":
                return "mid_range_gpu"      # RTX 3060, RTX 2080 etc.
            elif tier == "low_mid":
                return "low_mid_gpu"        # RTX 2060, GTX 1660 Ti etc.
            elif tier == "low_end":
                return "low_end_gpu"        # GTX 1650, RX 580 etc.
            elif tier == "very_low_end":
                return "very_low_end_gpu"   # GTX 1050, GTX 950 etc.
            else:
                # Fallback based on TFLOPS if tier is unknown
                if tflops >= 50:
                    return "flagship_gpu"
                elif tflops >= 25:
                    return "ultra_high_end_gpu"
                elif tflops >= 15:
                    return "high_end_gpu"
                elif tflops >= 8:
                    return "mid_range_gpu"
                elif tflops >= 5:
                    return "low_mid_gpu"
                elif tflops >= 3:
                    return "low_end_gpu"
                else:
                    return "very_low_end_gpu"
        else:
            # Enhanced CPU-only classification
            cpu_score = self.cpu_cores_physical * 2 + (self.total_ram_gb / 8)
            if cpu_score >= 20 and self.total_ram_gb >= 32:
                return "high_end_cpu"
            elif cpu_score >= 12 and self.total_ram_gb >= 16:
                return "mid_range_cpu"
            elif cpu_score >= 6 and self.total_ram_gb >= 8:
                return "low_end_cpu"
            else:
                return "very_low_end_cpu"
    
    def _calculate_benchmark_score(self):
        """Enhanced benchmark score calculation with TFLOPS integration"""
        score = 0
        
        # CPU score (30% weight)
        cpu_score = self.cpu_cores_physical * 10 + self.cpu_cores * 5
        score += cpu_score * 0.3
        
        # Memory score (20% weight) 
        memory_score = min(100, self.total_ram_gb * 3)
        score += memory_score * 0.2
        
        # GPU score (50% weight) - Enhanced with TFLOPS data
        if self.gpu_info["available"] and len(self.gpu_info["devices"]) > 0:
            gpu_device = self.gpu_info["devices"][0]
            gpu_memory = gpu_device["memory_gb"]
            gpu_name = gpu_device["name"]
            tflops = gpu_device.get("tflops", 0)
            tier = gpu_device.get("performance_tier", "unknown")
            
            # TFLOPS-based GPU score calculation
            if tier == "professional":
                gpu_score = min(500, tflops * 0.5 + 200)  # Professional cards get bonus
            elif tflops >= 50:  # Flagship cards
                gpu_score = min(400, tflops * 4)
            elif tflops >= 25:  # Ultra high-end
                gpu_score = min(350, tflops * 6)
            elif tflops >= 15:  # High-end
                gpu_score = min(300, tflops * 8)
            elif tflops >= 8:   # Mid-range
                gpu_score = min(250, tflops * 10)
            elif tflops >= 5:   # Low-mid
                gpu_score = min(200, tflops * 12)
            elif tflops >= 3:   # Low-end
                gpu_score = min(150, tflops * 15)
            else:  # Very low-end
                gpu_score = min(100, tflops * 20)
            
            # Memory bonus (up to 20% increase)
            memory_bonus = min(20, gpu_memory * 2) / 100
            gpu_score *= (1 + memory_bonus)
            
            # Architecture bonuses
            if "RTX 40" in gpu_name:  # Ada Lovelace
                gpu_score *= 1.2
            elif "RTX 30" in gpu_name:  # Ampere
                gpu_score *= 1.1
            elif "RX 7900" in gpu_name:  # RDNA 3
                gpu_score *= 1.15
            elif "RX 6" in gpu_name:  # RDNA 2
                gpu_score *= 1.05
            
            score += gpu_score * 0.5
        
        return int(score)

class DynamicBatchOptimizer:
    """Dynamic batch size optimizer that automatically adjusts based on system resources"""
    def __init__(self, hardware_profiler):
        self.profiler = hardware_profiler
        self.performance_history = []
        self.current_batch_size = self._calculate_initial_batch_size()
        self.min_batch_size = self._calculate_min_batch_size()
        self.max_batch_size = self._calculate_max_batch_size()
        self.adjustment_factor = 1.2  # Conservative adjustment
        
        print(f"Batch Optimizer - Initial: {self.current_batch_size}, "
              f"Range: {self.min_batch_size}-{self.max_batch_size}")
        
    def _calculate_initial_batch_size(self):
        """Enhanced initial batch size calculation with benchmark score consideration"""
        profile = self.profiler.performance_profile
        benchmark_score = self.profiler.system_benchmark_score
        
        # 🚀 TFLOPS-based batch sizes with 2025 GPU updates
        base_batch_sizes = {
            "professional_gpu": 2500,    # Tesla V100, A100, H100 - Enterprise workloads
            "flagship_gpu": 2000,        # RTX 5090, RX 7900 XTX - Ultimate performance
            "ultra_high_end_gpu": 1500,  # RTX 5080, RTX 4080, RX 7900 XT - High-end gaming
            "high_end_gpu": 1000,        # RTX 5070 Ti/5070, RTX 4070, RX 9070 XT - Premium gaming
            "mid_range_gpu": 500,        # RTX 5060 Ti/5060, RTX 3060, RX 9070 - Mainstream
            "low_mid_gpu": 250,          # RTX 2060, GTX 1660 Ti, RX 9060 - Budget gaming
            "low_end_gpu": 150,          # GTX 1650, RX 580 - Entry level
            "very_low_end_gpu": 75,      # GTX 1050, GTX 950 - Very low-end
            "high_end_cpu": 100,         # High-end CPU systems
            "mid_range_cpu": 50,         # Mid-range CPU systems
            "low_end_cpu": 25,           # Low-end CPU systems
            "very_low_end_cpu": 12       # Very low-end systems
        }
        
        base_size = base_batch_sizes.get(profile, 32)
        
        # Fine-tune based on benchmark score and TFLOPS
        if hasattr(self.profiler, 'gpu_info') and self.profiler.gpu_info.get("available"):
            gpu_device = self.profiler.gpu_info["devices"][0]
            tflops = gpu_device.get("tflops", 0)
            
            # TFLOPS-based multiplier for more precise tuning (2025 updated + Future GPU support)
            if tflops >= 150:  # Future flagship GPUs (RTX 6090+ tier)
                multiplier = 1.6
            elif tflops >= 120:  # RTX 5090 level - Current flagship tier
                multiplier = 1.5
            elif tflops >= 80:   # Future high-end / RTX 5080+ level
                multiplier = 1.4
            elif tflops >= 60:   # RTX 5080, RX 7900 XTX level
                multiplier = 1.3
            elif tflops >= 40:   # RTX 5070 Ti, RTX 4080, RX 9070 XT level
                multiplier = 1.2
            elif tflops >= 25:   # RTX 5070, RTX 4070 level
                multiplier = 1.1
            elif tflops >= 15:   # RTX 5060 Ti, RTX 2070/3060 level
                multiplier = 1.0
            elif tflops >= 8:    # RTX 5060, GTX 1660 level
                multiplier = 0.95
            elif tflops >= 4:    # GTX 1050 Ti level
                multiplier = 0.9
            else:  # Very low-end
                multiplier = 0.7
        else:
            # CPU-only benchmark score adjustment
            if benchmark_score > 300:
                multiplier = 1.3
            elif benchmark_score > 200:
                multiplier = 1.1
            elif benchmark_score > 100:
                multiplier = 1.0
            elif benchmark_score > 50:
                multiplier = 0.8
            else:
                multiplier = 0.6
        
        optimized_size = int(base_size * multiplier)
        
        print(f"📊 Initial batch size: {base_size} x {multiplier:.1f} = {optimized_size}")
        return optimized_size
    
    def _calculate_min_batch_size(self):
        """Calculate minimum batch size"""
        if self.profiler.gpu_info["available"]:
            return 4  # GPU minimum
        else:
            return 2  # CPU minimum
    
    def _calculate_max_batch_size(self):
        """Enhanced maximum batch size calculation with safety margins"""
        if self.profiler.gpu_info["available"]:
            try:
                gpu_memory = self.profiler.gpu_info["devices"][0]["memory_gb"]
                gpu_name = self.profiler.gpu_info["devices"][0]["name"]
                benchmark_score = self.profiler.system_benchmark_score
                
                # 🚀 2025 Updated GPU max batch sizes with latest RTX 50 and RX 9000 series
                gpu_device = self.profiler.gpu_info["devices"][0]
                tflops = gpu_device.get("tflops", 0)
                tier = gpu_device.get("performance_tier", "unknown")
                
                # Tier-based max batch calculation (2025 updated)
                if tier == "professional":
                    max_batch = 4000  # Tesla V100, A100, H100 - Increased for enterprise
                elif tier == "flagship":
                    max_batch = 3500  # RTX 5090, RX 7900 XTX - New flagship tier
                elif tier == "ultra_high_end":
                    max_batch = 2500  # RTX 5080, RTX 4080, RX 7900 XT
                elif tier == "high_end":
                    max_batch = 2000  # RTX 5070 Ti/5070, RTX 4070, RX 9070 XT
                elif tier == "mid_range":
                    max_batch = 1200  # RTX 5060 Ti/5060, RTX 3060, RX 9070
                elif tier == "low_mid":
                    max_batch = 600   # RTX 2060, GTX 1660 Ti, RX 9060
                elif tier == "low_end":
                    max_batch = 300   # GTX 1650, RX 580
                elif tier == "very_low_end":
                    max_batch = 150   # GTX 1050, GTX 950
                else:
                    # TFLOPS-based fallback (2025 updated + Future GPU support)
                    if tflops >= 200:  # Future flagship GPUs
                        max_batch = 5000   # Next-gen flagship tier
                    elif tflops >= 150:
                        max_batch = 4000   # Future high-end tier
                    elif tflops >= 120:
                        max_batch = 3500  # RTX 5090 tier
                    elif tflops >= 80:
                        max_batch = 3000  # Future upper high-end
                    elif tflops >= 60:
                        max_batch = 2500  # RTX 5080 tier
                    elif tflops >= 40:
                        max_batch = 2000  # RTX 5070 Ti tier
                    elif tflops >= 25:
                        max_batch = 1500  # RTX 5070 tier
                    elif tflops >= 15:
                        max_batch = 1000  # RTX 5060 Ti tier
                    elif tflops >= 8:
                        max_batch = 600   # RTX 5060 tier
                    elif tflops >= 4:
                        max_batch = 300
                    else:
                        max_batch = 150
                
                # Apply benchmark score modifier
                if benchmark_score > 300:
                    max_batch = int(max_batch * 1.2)  # Boost for high-performance systems
                elif benchmark_score < 100:
                    max_batch = int(max_batch * 0.8)  # Conservative for low-performance systems
                
                # Safety bounds with GPU memory consideration
                memory_limit = max(100, int(gpu_memory * 100))  # Memory-based upper limit
                max_batch = max(50, min(memory_limit, max_batch))
                
                print(f"📈 Max batch for {gpu_name} ({tflops:.1f} TFLOPS, {tier}): {max_batch}")
                return max_batch
                
            except Exception as e:
                print(f"Error calculating GPU max batch size: {e}")
                return 500  # Safe fallback
        else:
            # Enhanced CPU mode with better scaling
            cpu_cores = self.profiler.cpu_cores_physical
            ram_gb = self.profiler.total_ram_gb
            
            # Dynamic CPU-based calculation
            max_batch = min(200, max(8, int(cpu_cores * 8 + ram_gb * 2)))
            
            print(f"💻 CPU max batch size: {max_batch} (Cores: {cpu_cores}, RAM: {ram_gb:.1f}GB)")
            return max_batch

    def adjust_batch_size(self, current_metrics):
        """Enhanced dynamic batch size adjustment with performance feedback"""
        memory_percent = current_metrics.get("memory_percent", 50)
        gpu_percent = current_metrics.get("gpu_percent", 0)
        gpu_memory_percent = current_metrics.get("gpu_memory_percent", 0)
        cpu_percent = current_metrics.get("cpu_percent", 50)
        processing_time = current_metrics.get("processing_time", 1.0)
        success_rate = current_metrics.get("success_rate", 1.0)
        
        # Record enhanced performance metrics
        current_timestamp = time.time()
        self.performance_history.append({
            "batch_size": self.current_batch_size,
            "memory_percent": memory_percent,
            "gpu_utilization": gpu_percent,
            "gpu_memory_percent": gpu_memory_percent,
            "cpu_percent": cpu_percent,
            "processing_time": processing_time,
            "success_rate": success_rate,
            "timestamp": current_timestamp,
            "throughput": self.current_batch_size / max(0.1, processing_time)
        })
        
        # Keep only recent 20 records for better trend analysis
        if len(self.performance_history) > 20:
            self.performance_history.pop(0)
        
        # Calculate new batch size with enhanced logic
        old_batch_size = self.current_batch_size
        new_batch_size = self.current_batch_size
        
        if self.profiler.gpu_info["available"]:
            new_batch_size = self._adjust_for_gpu_enhanced(
                memory_percent, gpu_percent, gpu_memory_percent, 
                processing_time, success_rate, new_batch_size
            )
        else:
            new_batch_size = self._adjust_for_cpu_enhanced(
                memory_percent, cpu_percent, processing_time, success_rate, new_batch_size
            )
        
        # Apply advanced performance-based fine-tuning
        new_batch_size = self._apply_advanced_performance_tuning(new_batch_size)
        
        # Safety bounds check with gradual adjustment
        new_batch_size = max(self.min_batch_size, min(self.max_batch_size, new_batch_size))
        
        # Gradual adjustment to prevent sudden changes
        if abs(new_batch_size - old_batch_size) > old_batch_size * 0.3:
            if new_batch_size > old_batch_size:
                new_batch_size = int(old_batch_size * 1.3)  # Max 30% increase
            else:
                new_batch_size = int(old_batch_size * 0.7)  # Max 30% decrease
        
        # Log significant changes
        if abs(new_batch_size - old_batch_size) > 10:
            print(f"🔄 Batch size adjusted: {old_batch_size} → {new_batch_size} "
                  f"(Memory: {memory_percent:.1f}%, GPU: {gpu_percent:.1f}%)")
        
        self.current_batch_size = new_batch_size
        return self.current_batch_size
    
    def _adjust_for_gpu_enhanced(self, memory_percent, gpu_percent, gpu_memory_percent, 
                                processing_time, success_rate, current_size):
        """Enhanced GPU-specific batch size adjustment with performance feedback"""
        # Critical conditions - immediate reduction
        if memory_percent > 90 or gpu_memory_percent > 95 or success_rate < 0.8:
            reduction = 0.5 if success_rate < 0.8 else 0.6
            return max(self.min_batch_size, int(current_size * reduction))
        
        # High memory pressure - conservative reduction
        if memory_percent > 80 or gpu_memory_percent > 85:
            return max(self.min_batch_size, int(current_size * 0.75))
        
        # Performance-based adjustments
        if success_rate > 0.95 and processing_time < 2.0:
            # Excellent performance - can increase aggressively
            if gpu_percent < 40 and memory_percent < 60:
                return min(self.max_batch_size, int(current_size * 1.5))
            elif gpu_percent < 60 and memory_percent < 70:
                return min(self.max_batch_size, int(current_size * 1.2))
        
        # Moderate performance - gradual adjustments
        if success_rate > 0.9:
            if gpu_percent < 50 and memory_percent < 65 and gpu_memory_percent < 70:
                return min(self.max_batch_size, int(current_size * 1.1))
            elif gpu_percent > 90 or gpu_memory_percent > 80:
                return max(self.min_batch_size, int(current_size * 0.9))
        
        # Poor performance - reduce batch size
        if processing_time > 5.0 or gpu_percent > 95:
            return max(self.min_batch_size, int(current_size * 0.8))
        
        return current_size
    
    def _apply_advanced_performance_tuning(self, current_size):
        """Enhanced performance-based fine-tuning with trend analysis"""
        if len(self.performance_history) < 5:
            return current_size
        
        # Analyze recent performance trends
        recent_performance = self.performance_history[-5:]
        avg_gpu_util = sum(p["gpu_utilization"] for p in recent_performance) / len(recent_performance)
        avg_memory = sum(p["memory_percent"] for p in recent_performance) / len(recent_performance)
        avg_throughput = sum(p.get("throughput", 0) for p in recent_performance) / len(recent_performance)
        avg_success_rate = sum(p.get("success_rate", 1.0) for p in recent_performance) / len(recent_performance)
        
        # Performance-based adjustments
        if avg_success_rate > 0.95 and avg_throughput > 0:
            # Excellent performance - can be more aggressive
            if avg_gpu_util < 30 and avg_memory < 60:
                return min(self.max_batch_size, int(current_size * 1.3))
            elif avg_gpu_util < 50 and avg_memory < 70:
                return min(self.max_batch_size, int(current_size * 1.1))
        
        # Poor performance - be conservative
        elif avg_success_rate < 0.9 or avg_gpu_util > 95:
            return max(self.min_batch_size, int(current_size * 0.8))
        
        return current_size
    
    def _adjust_for_gpu(self, memory_percent, gpu_percent, gpu_memory_percent, current_size):
        """Legacy GPU adjustment - kept for compatibility"""
        return self._adjust_for_gpu_enhanced(memory_percent, gpu_percent, gpu_memory_percent, 
                                           1.0, 1.0, current_size)
    
    def _adjust_for_cpu(self, memory_percent, cpu_percent, current_size):
        """CPU-specific batch size adjustment"""
        # Critical memory check
        if memory_percent > 85:
            return max(self.min_batch_size, int(current_size * 0.7))
        
        # High memory usage
        if memory_percent > 75:
            return max(self.min_batch_size, int(current_size * 0.85))
        
        # Low CPU utilization - can increase batch size
        if cpu_percent < 40 and memory_percent < 60:
            return min(self.max_batch_size, int(current_size * 1.3))
        
        # Moderate utilization - slight increase
        if cpu_percent < 60 and memory_percent < 70:
            return min(self.max_batch_size, int(current_size * self.adjustment_factor))
        
        # High CPU usage - reduce batch size
        if cpu_percent > 80:
            return max(self.min_batch_size, int(current_size * 0.9))
        
        return current_size
    
    def _apply_performance_tuning(self, current_size):
        """Apply performance-based fine-tuning"""
        if len(self.performance_history) < 5:
            return current_size
        
        # Analyze recent performance trends
        recent_performance = self.performance_history[-5:]
        avg_gpu_util = sum(p["gpu_utilization"] for p in recent_performance) / len(recent_performance)
        avg_memory = sum(p["memory_percent"] for p in recent_performance) / len(recent_performance)
        
        # If consistently low GPU/CPU utilization, be more aggressive
        if avg_gpu_util < 25 and avg_memory < 60:
            return min(self.max_batch_size, int(current_size * 1.3))
        
        return current_size

class SystemMonitor:
    """System resource monitor for optimization"""
    def __init__(self, update_interval=2.0):
        self.gpu_available = torch.cuda.is_available()
        self.update_interval = update_interval
        self.monitoring = False
        self.monitor_thread = None
        self.current_metrics = {
            "memory_status": "normal",
            "memory_percent": 50,
            "cpu_percent": 50,
            "gpu_percent": 0,
            "gpu_memory_percent": 0,
            "gpu_available": self.gpu_available
        }
        
    def start_monitoring(self):
        """Start system monitoring in background thread"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                self._update_metrics()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.update_interval)
    
    def _update_metrics(self):
        """Update system metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            
            # GPU metrics (simplified)
            gpu_percent = 0
            gpu_memory_percent = 0
            
            if self.gpu_available:
                try:
                    allocated = torch.cuda.memory_allocated()
                    max_memory = torch.cuda.max_memory_allocated()
                    
                    if max_memory > 0:
                        gpu_memory_percent = (allocated / max_memory) * 100
                    
                    # Simple GPU utilization estimation
                    if hasattr(self, '_last_gpu_memory'):
                        memory_change = abs(allocated - self._last_gpu_memory)
                        if memory_change > 1024 * 1024:  # 1MB change
                            gpu_percent = min(100, 20)  # Simplified estimation
                    
                    self._last_gpu_memory = allocated
                    
                except Exception:
                    pass
            
            # Update current metrics
            self.current_metrics.update({
                "cpu_percent": cpu_percent,
                "memory_percent": memory_info.percent,
                "gpu_percent": gpu_percent,
                "gpu_memory_percent": gpu_memory_percent,
                "memory_status": self._get_memory_status(memory_info.percent)
            })
                    
        except Exception as e:
            print(f"Error updating metrics: {e}")
    
    def _get_memory_status(self, memory_percent):
        """Determine memory status"""
        if memory_percent > 85:
            return "critical"
        elif memory_percent > 75:
            return "high"
        elif memory_percent > 60:
            return "moderate"
        else:
            return "normal"
    
    def get_system_status(self):
        """Get current system status"""
        return self.current_metrics.copy()

class EmbeddingModel:
    """Enhanced embedding model with dynamic batch optimization"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the embedding model with dynamic optimization"""
        print("Initializing Enhanced EmbeddingModel with Dynamic Batch Optimization...")
        
        # Hardware profiling
        self.hardware_profiler = HardwareProfiler()
        self.batch_optimizer = DynamicBatchOptimizer(self.hardware_profiler)
        self.system_monitor = SystemMonitor()
        
        # GPU usage settings from config
        use_gpu = getattr(config, 'USE_GPU', True)
        gpu_device_id = getattr(config, 'GPU_DEVICE_ID', 0)
        
        # System information
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Device selection
        self.device = self._select_optimal_device(use_gpu, gpu_device_id)
        
        # Load model
        self._load_model()
        
        # Configure caching
        cache_size = getattr(config, 'EMBEDDING_CACHE_SIZE', 1000)
        self.get_embedding_cached = lru_cache(maxsize=cache_size)(self._get_embedding_impl)
        
        # Initialize threading components
        self._lock = threading.Lock()
        
        # Start monitoring
        self.system_monitor.start_monitoring()
        
        print("Enhanced EmbeddingModel initialization completed!")
        print(f"Current batch size: {self.batch_optimizer.current_batch_size}")
    
    def _select_optimal_device(self, use_gpu, gpu_device_id):
        """Select optimal device"""
        device = 'cpu'  # Default fallback
        device_info = "CPU"
        
        if use_gpu and torch.cuda.is_available():
            try:
                gpu_count = torch.cuda.device_count()
                
                # Validate GPU device ID
                if gpu_device_id >= gpu_count:
                    gpu_device_id = 0
                
                # Get GPU information
                gpu_name = torch.cuda.get_device_name(gpu_device_id)
                total_memory = torch.cuda.get_device_properties(gpu_device_id).total_memory
                total_memory_gb = total_memory / (1024**3)
                
                print(f"Selected GPU {gpu_device_id}: {gpu_name}")
                print(f"GPU total memory: {total_memory_gb:.2f} GB")
                
                # Test GPU functionality
                test_tensor = torch.rand(100, 100, device=f'cuda:{gpu_device_id}')
                test_result = torch.matmul(test_tensor, test_tensor)
                print("GPU test successful!")
                
                device = f'cuda:{gpu_device_id}'
                device_info = f"GPU: {gpu_name} ({total_memory_gb:.1f}GB)"
                
            except Exception as e:
                print(f"GPU setup error: {e}")
                print("Falling back to CPU")
                device = 'cpu'
                device_info = "CPU (GPU fallback)"
        
        # Display device information
        print("\n" + "=" * 50)
        print(f"COMPUTE DEVICE: {device_info}")
        print("=" * 50)
        
        return device

    def _load_model(self):
        """Load the embedding model"""
        model_cache_dir = getattr(config, 'MODEL_CACHE_DIR', None)
        if model_cache_dir:
            os.makedirs(model_cache_dir, exist_ok=True)
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = model_cache_dir
            print(f"Model cache directory: {model_cache_dir}")
        
        model_name = getattr(config, 'EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        print(f"Loading model: {model_name}")
        
        try:
            start_time = time.time()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                self.model = SentenceTransformer(
                    model_name,
                    device=self.device,
                    trust_remote_code=True
                )
                
                # Optimize model for inference
                self.model.eval()
                if 'cuda' in self.device:
                    # Use half precision for GPU to save memory (optional)
                    try:
                        self.model.half()
                    except:
                        print("Half precision not supported, using full precision")
            
            elapsed = time.time() - start_time
            print(f"Model loaded successfully in {elapsed:.1f} seconds!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def get_embedding(self, text):
        """Get embedding for single text with caching"""
        if not text or not isinstance(text, str) or text.isspace():
            return [0] * getattr(config, 'VECTOR_DIM', 384)
        
        # Text length limit for safety
        max_text_length = getattr(config, 'MAX_TEXT_LENGTH', 5000)
        if len(text) > max_text_length:
            text = text[:max_text_length]
        
        # Use hash for caching very long texts
        if len(text) > 1000:
            text_hash = hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()
            return self.get_embedding_cached(text_hash, text)
        else:
            return self.get_embedding_cached(text)
    
    def get_embeddings_batch(self, texts):
        """Legacy method - redirects to adaptive batch processing"""
        return self.get_embeddings_batch_adaptive(texts)
    
    def get_embeddings_batch_adaptive(self, texts):
        """MAIN METHOD: Adaptive batch processing with CONTINUOUS GPU utilization"""
        if not isinstance(texts, list):
            if isinstance(texts, str):
                texts = [texts]
            else:
                return []
        
        if not texts:
            return []
        
        start_time = time.time()
        
        # 🔥 AGGRESSIVE BATCH SIZE for continuous GPU utilization
        # RTX 4070을 위한 초대형 배치 크기 - 테스트 결과 반영!
        if len(texts) < 100:
            # 작은 배치도 최소 크기 보장 - 500 → 800으로 업그레이드!
            effective_batch_size = max(800, len(texts) * 15)  # 테스트 결과 기반 업그레이드
            # 텍스트 복제로 배치 크기 늘리기 (GPU 활용도 극대화)
            extended_texts = texts * (effective_batch_size // len(texts) + 1)
            extended_texts = extended_texts[:effective_batch_size]
            
            print(f"🚀 BOOSTING small batch: {len(texts)} → {len(extended_texts)} for GPU utilization")
            
            # 확장된 배치 처리
            extended_results = self._process_continuous_gpu_batches(extended_texts)
            
            # 원본 결과만 반환
            return extended_results[:len(texts)]
        else:
            # 큰 배치는 연속 처리
            return self._process_continuous_gpu_batches(texts)
        
    def _process_continuous_gpu_batches(self, texts):
        """연속적인 GPU 배치 처리로 GPU 사용률 극대화"""
        print(f"🔥 CONTINUOUS GPU processing: {len(texts)} texts")
        
        start_time = time.time()  # ✅ start_time 변수 추가
        
        # Collect current system metrics
        current_metrics = self._collect_system_metrics()
        
        # 🚀 ULTRA-AGGRESSIVE batch size for RTX 4070
        base_batch_size = 1000  # 기본 1000개씩 처리
        optimal_batch_size = min(base_batch_size, len(texts))
        
        # 시스템 상태에 따른 배치 크기 조정
        if hasattr(self, 'batch_optimizer'):
            suggested_batch = self.batch_optimizer.adjust_batch_size(current_metrics)
            optimal_batch_size = max(optimal_batch_size, suggested_batch)
        
        print(f"📦 Using ULTRA batch size: {optimal_batch_size}")
        
        # Process texts with continuous GPU utilization
        results = self._process_batches_with_continuous_gpu(texts, optimal_batch_size)
        
        # Performance logging
        processing_time = time.time() - start_time
        if processing_time > 0:
            throughput = len(texts) / processing_time
            print(f"⚡ CONTINUOUS processing: {len(texts)} texts in {processing_time:.2f}s ({throughput:.1f} texts/sec)")
            print(f"🎯 GPU should be at HIGH utilization during this process")
        
        return results
        
    def _process_batches_with_continuous_gpu(self, texts, batch_size):
        """GPU를 연속적으로 사용하는 배치 처리"""
        # Validate and prepare texts
        valid_texts = []
        results = [[0] * getattr(config, 'VECTOR_DIM', 384) for _ in range(len(texts))]
        text_mapping = []
        
        max_text_length = getattr(config, 'MAX_TEXT_LENGTH', 5000)
        
        for i, text in enumerate(texts):
            if isinstance(text, str) and text and not text.isspace():
                if len(text) > max_text_length:
                    text = text[:max_text_length]
                valid_texts.append(text)
                text_mapping.append(i)
        
        if not valid_texts:
            return results
            
        # 🔥 CONTINUOUS GPU processing strategy
        processed_count = 0
        gpu_warm_up_done = False
        
        while processed_count < len(valid_texts):
            batch_start = processed_count
            batch_end = min(batch_start + batch_size, len(valid_texts))
            batch_texts = valid_texts[batch_start:batch_end]
            
            try:
                # GPU 워밍업 (첫 번째 배치에서만)
                if not gpu_warm_up_done and torch.cuda.is_available():
                    print("🔥 GPU warm-up for continuous processing...")
                    # 더미 연산으로 GPU 워밍업
                    dummy_tensor = torch.randn(1000, 1000, device=self.device)
                    _ = torch.matmul(dummy_tensor, dummy_tensor)
                    del dummy_tensor
                    torch.cuda.synchronize()
                    gpu_warm_up_done = True
                    print("✅ GPU warmed up")
                
                # Process current batch with continuous GPU usage
                print(f"🚀 Processing GPU batch {processed_count}-{batch_end}: {len(batch_texts)} texts")
                batch_vectors = self._process_single_batch_continuous(batch_texts)
                
                # Store results
                for i, vector in enumerate(batch_vectors):
                    result_idx = text_mapping[batch_start + i]
                    results[result_idx] = vector
                
                processed_count = batch_end
                
                # 짧은 GPU 유지 작업 (유휴 시간 최소화)
                if processed_count < len(valid_texts) and torch.cuda.is_available():
                    # GPU 메모리 유지를 위한 작은 연산
                    maintenance_tensor = torch.ones(100, 100, device=self.device)
                    _ = maintenance_tensor.sum()
                    del maintenance_tensor
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"🚨 GPU memory error, reducing batch size from {batch_size}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    batch_size = max(1, batch_size // 2)
                    continue
                else:
                    # 개별 처리 폴백
                    print(f"⚠️ Falling back to individual processing for batch")
                    for i, text in enumerate(batch_texts):
                        try:
                            vector = self.get_embedding(text)
                            result_idx = text_mapping[batch_start + i]
                            results[result_idx] = vector
                        except Exception:
                            pass
                    processed_count = batch_end
                    
            except Exception as e:
                print(f"❌ Batch processing error: {e}")
                # 개별 처리 폴백
                for i, text in enumerate(batch_texts):
                    try:
                        vector = self.get_embedding(text)
                        result_idx = text_mapping[batch_start + i]
                        results[result_idx] = vector
                    except Exception:
                        pass
                processed_count = batch_end
        
        return results
        
    def _process_single_batch_continuous(self, batch_texts):
        """연속적인 GPU 사용을 위한 단일 배치 처리"""
        if not batch_texts:
            return []
        
        try:
            with torch.no_grad():
                # 🔥 CONTINUOUS GPU processing with memory persistence
                if 'cuda' in self.device:
                    # GPU에서 메모리를 미리 할당해서 연속 처리
                    print(f"🚀 GPU continuous processing: {len(batch_texts)} texts")
                    
                    # 대용량 배치 처리
                    vectors = self.model.encode(
                        batch_texts,
                        batch_size=len(batch_texts),  # 전체를 한 번에 처리
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        device=self.device
                    )
                    
                    # GPU 메모리에서 연산 지속 (사용률 유지)
                    if torch.cuda.is_available():
                        # 추가 GPU 연산으로 사용률 유지
                        gpu_tensor = torch.tensor(vectors, device=self.device)
                        # 정규화 연산 추가
                        gpu_tensor = torch.nn.functional.normalize(gpu_tensor, p=2, dim=1)
                        vectors = gpu_tensor.cpu().numpy()
                        del gpu_tensor
                    
                else:
                    # CPU 처리
                    vectors = self.model.encode(
                        batch_texts,
                        batch_size=min(32, len(batch_texts)),
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                
                # Convert to list format
                if isinstance(vectors, np.ndarray):
                    return [vector.tolist() for vector in vectors]
                else:
                    return [vector.tolist() if isinstance(vector, np.ndarray) else vector for vector in vectors]
                    
        except Exception as e:
            print(f"❌ Continuous batch processing error: {e}")
            raise
    
    def _collect_system_metrics(self):
        """Collect current system metrics for optimization"""
        try:
            system_status = self.system_monitor.get_system_status()
            
            # Add GPU-specific metrics if available
            if torch.cuda.is_available():
                try:
                    allocated_memory = torch.cuda.memory_allocated()
                    max_memory = torch.cuda.max_memory_allocated()
                    if max_memory > 0:
                        gpu_memory_usage = (allocated_memory / max_memory) * 100
                        system_status["gpu_memory_percent"] = gpu_memory_usage
                    
                    # Reset max memory tracking periodically
                    if not hasattr(self, '_last_memory_reset') or time.time() - self._last_memory_reset > 300:
                        torch.cuda.reset_max_memory_allocated()
                        self._last_memory_reset = time.time()
                        
                except Exception:
                    system_status["gpu_memory_percent"] = 0
            
            return system_status
            
        except Exception as e:
            # Return safe defaults
            return {
                "memory_percent": 50,
                "cpu_percent": 50,
                "gpu_percent": 0,
                "gpu_memory_percent": 0
            }
    
    def _process_batches_with_adaptive_size(self, texts, initial_batch_size):
        """Process texts with adaptive batch sizing and error recovery"""
        # Validate and prepare texts
        valid_texts = []
        results = [[0] * getattr(config, 'VECTOR_DIM', 384) for _ in range(len(texts))]
        text_mapping = []
        
        max_text_length = getattr(config, 'MAX_TEXT_LENGTH', 5000)
        
        for i, text in enumerate(texts):
            if isinstance(text, str) and text and not text.isspace():
                # Truncate if necessary
                if len(text) > max_text_length:
                    text = text[:max_text_length]
                valid_texts.append(text)
                text_mapping.append(i)
        
        if not valid_texts:
            return results
        
        # Adaptive batch processing
        current_batch_size = initial_batch_size
        processed_count = 0
        
        while processed_count < len(valid_texts):
            batch_start = processed_count
            batch_end = min(batch_start + current_batch_size, len(valid_texts))
            batch_texts = valid_texts[batch_start:batch_end]
            
            try:
                # Process current batch
                batch_vectors = self._process_single_batch(batch_texts)
                
                # Store results
                for i, vector in enumerate(batch_vectors):
                    result_idx = text_mapping[batch_start + i]
                    results[result_idx] = vector
                
                processed_count = batch_end
                
                # Adaptive batch size adjustment based on success
                if len(batch_texts) == current_batch_size and current_batch_size < self.batch_optimizer.max_batch_size:
                    current_batch_size = min(
                        self.batch_optimizer.max_batch_size,
                        int(current_batch_size * 1.1)
                    )
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # GPU memory issue - reduce batch size
                    print(f"GPU memory error, reducing batch size from {current_batch_size}")
                    
                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Reduce batch size significantly
                    current_batch_size = max(1, current_batch_size // 2)
                    continue
                else:
                    # Other error - process individually
                    for i, text in enumerate(batch_texts):
                        try:
                            vector = self.get_embedding(text)
                            result_idx = text_mapping[batch_start + i]
                            results[result_idx] = vector
                        except Exception:
                            pass  # Use zero vector
                    
                    processed_count = batch_end
                    
            except Exception as e:
                print(f"Batch processing error: {e}")
                # Fallback to individual processing
                for i, text in enumerate(batch_texts):
                    try:
                        vector = self.get_embedding(text)
                        result_idx = text_mapping[batch_start + i]
                        results[result_idx] = vector
                    except Exception:
                        pass  # Use zero vector
                
                processed_count = batch_end
        
        return results
    
    def _process_single_batch(self, batch_texts):
        """Process a single batch of texts"""
        if not batch_texts:
            return []
        
        try:
            with torch.no_grad():
                # Process batch
                if 'cuda' in self.device:
                    # GPU processing
                    vectors = self.model.encode(
                        batch_texts,
                        batch_size=len(batch_texts),
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                else:
                    # CPU processing
                    cpu_batch_size = min(16, len(batch_texts))
                    vectors = self.model.encode(
                        batch_texts,
                        batch_size=cpu_batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                
                # Convert to list format
                if isinstance(vectors, np.ndarray):
                    return [vector.tolist() for vector in vectors]
                else:
                    return [vector.tolist() if isinstance(vector, np.ndarray) else vector for vector in vectors]
                    
        except Exception as e:
            print(f"Single batch processing error: {e}")
            raise

    def _get_embedding_impl(self, text, original_text=None):
        """Implementation for cached embedding generation"""
        compute_text = original_text if original_text is not None else text
        
        # Additional safety check for text length
        max_length = getattr(config, 'MAX_TEXT_LENGTH', 10000)
        if len(compute_text) > max_length:
            compute_text = compute_text[:max_length]
        
        try:
            with torch.no_grad():
                vector = self.model.encode(
                    compute_text,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                
                # Convert to list and validate
                if isinstance(vector, np.ndarray):
                    result = vector.tolist()
                else:
                    result = vector
                
                # Validate result dimensions
                expected_dim = getattr(config, 'VECTOR_DIM', 384)
                if isinstance(result, list) and len(result) == expected_dim:
                    return result
                else:
                    return [0] * expected_dim
                    
        except Exception as e:
            print(f"Error during embedding generation: {e}")
            return [0] * getattr(config, 'VECTOR_DIM', 384)
    
    def clear_cache(self):
        """Clear embedding cache and GPU memory"""
        try:
            # Clear LRU cache
            if hasattr(self, 'get_embedding_cached'):
                self.get_embedding_cached.cache_clear()
            
            # Clear system caches
            gc.collect()
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            return True
            
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False
    
    def get_optimal_chunk_size(self, total_text_length, target_chunks=None):
        """Calculate optimal chunk size based on current system performance"""
        try:
            # Base chunk size from config
            base_chunk_size = getattr(config, 'CHUNK_SIZE', 1000)
            min_chunk_size = getattr(config, 'CHUNK_MIN_SIZE', 100)
            max_chunk_size = min(5000, base_chunk_size * 3)
            
            # Adjust based on system performance profile
            profile = self.hardware_profiler.performance_profile
            
            # Performance-based adjustments
            if 'high_end_gpu' in profile:
                chunk_multiplier = 1.5
            elif 'mid_range_gpu' in profile:
                chunk_multiplier = 1.2
            elif 'gpu' in profile:
                chunk_multiplier = 1.0
            elif 'high_end_cpu' in profile:
                chunk_multiplier = 0.8
            else:
                chunk_multiplier = 0.6
            
            optimal_chunk_size = int(base_chunk_size * chunk_multiplier)
            
            # Consider target number of chunks if specified
            if target_chunks and total_text_length > 0:
                target_chunk_size = total_text_length // target_chunks
                optimal_chunk_size = int((optimal_chunk_size + target_chunk_size) / 2)
            
            # Apply bounds
            optimal_chunk_size = max(min_chunk_size, min(max_chunk_size, optimal_chunk_size))
            
            return optimal_chunk_size
            
        except Exception as e:
            print(f"Error calculating optimal chunk size: {e}")
            return getattr(config, 'CHUNK_SIZE', 1000)
    
    def __del__(self):
        """Enhanced destructor with proper cleanup"""
        try:
            # Stop monitoring
            if hasattr(self, 'system_monitor'):
                self.system_monitor.stop_monitoring()
            
            # Clear caches
            self.clear_cache()
                
        except Exception as e:
            print(f"Cleanup error in destructor: {e}")
