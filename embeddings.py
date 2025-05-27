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
import math

# Additional warning suppression
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', module='huggingface_hub')
warnings.filterwarnings('ignore', module='transformers')

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL if hasattr(config, 'LOG_LEVEL') else 'WARNING'),
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class HardwareProfiler:
    """Hardware detection and profiling system for automatic optimization"""
    def __init__(self):
        self.cpu_cores = psutil.cpu_count()
        self.total_ram_gb = psutil.virtual_memory().total / (1024**3)
        self.gpu_info = self._detect_gpu()
        self.performance_profile = self._create_performance_profile()
        print(f"Hardware Profile: {self.performance_profile}")
        print(f"CPU Cores: {self.cpu_cores}, RAM: {self.total_ram_gb:.1f}GB")
        
    def _detect_gpu(self):
        """Detect GPU detailed information"""
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
                gpu_info["devices"].append({
                    "name": props.name,
                    "memory_gb": props.total_memory / (1024**3),
                    "compute_capability": f"{props.major}.{props.minor}"
                })
                print(f"GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f}GB)")
            except Exception as e:
                print(f"Error detecting GPU {i}: {e}")
        
        return gpu_info
    
    def _create_performance_profile(self):
        """Create system performance profile for automatic optimization"""
        if self.gpu_info["available"] and len(self.gpu_info["devices"]) > 0:
            gpu_memory = self.gpu_info["devices"][0]["memory_gb"]
            if gpu_memory >= 12:  # RTX 4070 Ti, RTX 3080 and above
                return "high_end_gpu"
            elif gpu_memory >= 8:  # RTX 4060 Ti, RTX 3070 and above
                return "mid_range_gpu"
            elif gpu_memory >= 4:  # GTX 1660, RTX 3050 and above
                return "low_end_gpu"
            else:
                return "very_low_end_gpu"
        else:
            if self.total_ram_gb >= 32:
                return "high_end_cpu"
            elif self.total_ram_gb >= 16:
                return "mid_range_cpu"
            elif self.total_ram_gb >= 8:
                return "low_end_cpu"
            else:
                return "very_low_end_cpu"

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
        """Calculate initial batch size based on hardware profile"""
        profile = self.profiler.performance_profile
        
        # Conservative initial batch sizes that work well for most systems
        batch_sizes = {
            "high_end_gpu": 128,      # Reduced from 256 for stability
            "mid_range_gpu": 64,      # Reduced from 128
            "low_end_gpu": 32,        # Reduced from 64
            "very_low_end_gpu": 16,   # New category
            "high_end_cpu": 32,       # Reduced from 64
            "mid_range_cpu": 16,      # Reduced from 32
            "low_end_cpu": 8,         # Reduced from 16
            "very_low_end_cpu": 4     # New category
        }
        
        return batch_sizes.get(profile, 16)
    
    def _calculate_min_batch_size(self):
        """Calculate minimum batch size"""
        if self.profiler.gpu_info["available"]:
            return 4  # GPU minimum
        else:
            return 2  # CPU minimum
    
    def _calculate_max_batch_size(self):
        """Calculate maximum batch size based on available memory"""
        if self.profiler.gpu_info["available"]:
            try:
                gpu_memory = self.profiler.gpu_info["devices"][0]["memory_gb"]
                # Conservative calculation: 50% of GPU memory for embedding
                max_batch = min(512, max(32, int(gpu_memory * 24)))
                return max_batch
            except:
                return 128  # Safe fallback
        else:
            # CPU mode: based on system RAM
            # Conservative calculation: 25% of RAM for embedding
            max_batch = min(64, max(8, int(self.profiler.total_ram_gb * 3)))
            return max_batch

    def adjust_batch_size(self, current_metrics):
        """Dynamically adjust batch size based on real-time system metrics"""
        memory_percent = current_metrics.get("memory_percent", 50)
        gpu_percent = current_metrics.get("gpu_percent", 0)
        gpu_memory_percent = current_metrics.get("gpu_memory_percent", 0)
        cpu_percent = current_metrics.get("cpu_percent", 50)
        
        # Record performance metrics
        self.performance_history.append({
            "batch_size": self.current_batch_size,
            "memory_percent": memory_percent,
            "gpu_utilization": gpu_percent,
            "timestamp": time.time()
        })
        
        # Keep only recent 15 records
        if len(self.performance_history) > 15:
            self.performance_history.pop(0)
        
        # Calculate new batch size
        new_batch_size = self.current_batch_size
        
        if self.profiler.gpu_info["available"]:
            new_batch_size = self._adjust_for_gpu(
                memory_percent, gpu_percent, gpu_memory_percent, new_batch_size
            )
        else:
            new_batch_size = self._adjust_for_cpu(
                memory_percent, cpu_percent, new_batch_size
            )
        
        # Apply performance-based fine-tuning
        new_batch_size = self._apply_performance_tuning(new_batch_size)
        
        # Safety bounds check
        new_batch_size = max(self.min_batch_size, min(self.max_batch_size, new_batch_size))
        
        self.current_batch_size = new_batch_size
        return self.current_batch_size
    
    def _adjust_for_gpu(self, memory_percent, gpu_percent, gpu_memory_percent, current_size):
        """GPU-specific batch size adjustment"""
        # Critical memory check first
        if memory_percent > 85 or gpu_memory_percent > 90:
            return max(self.min_batch_size, int(current_size * 0.6))
        
        # High memory usage - reduce batch size
        if memory_percent > 75 or gpu_memory_percent > 80:
            return max(self.min_batch_size, int(current_size * 0.8))
        
        # GPU utilization is very low - increase batch size
        if gpu_percent < 30 and memory_percent < 65 and gpu_memory_percent < 60:
            return min(self.max_batch_size, int(current_size * 1.4))
        
        # GPU utilization is low - increase batch size moderately
        if gpu_percent < 50 and memory_percent < 70 and gpu_memory_percent < 70:
            return min(self.max_batch_size, int(current_size * self.adjustment_factor))
        
        # GPU utilization is too high - slightly reduce batch size
        if gpu_percent > 95 or gpu_memory_percent > 85:
            return max(self.min_batch_size, int(current_size * 0.9))
        
        return current_size
    
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
        """MAIN METHOD: Adaptive batch processing with dynamic optimization"""
        if not isinstance(texts, list):
            if isinstance(texts, str):
                texts = [texts]
            else:
                return []
        
        if not texts:
            return []
        
        start_time = time.time()
        
        # Collect current system metrics
        current_metrics = self._collect_system_metrics()
        
        # Get optimal batch size through dynamic adjustment
        optimal_batch_size = self.batch_optimizer.adjust_batch_size(current_metrics)
        
        # Process texts with adaptive batching
        results = self._process_batches_with_adaptive_size(texts, optimal_batch_size)
        
        # Performance logging (optional)
        processing_time = time.time() - start_time
        if processing_time > 1.0:  # Only log significant processing times
            print(f"Processed {len(texts)} texts in {processing_time:.2f}s with batch size {optimal_batch_size}")
        
        return results
    
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
