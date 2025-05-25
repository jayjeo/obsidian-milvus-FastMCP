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
import signal
import concurrent.futures
from threading import Event
import logging
import traceback

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL if hasattr(config, 'LOG_LEVEL') else 'WARNING'),
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class SystemMonitor:
    """System resource monitoring class for CPU and memory usage"""
    
    def __init__(self, warning_threshold=75, critical_threshold=90):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.last_status = "normal"
        self._lock = threading.Lock()
        
        # Monitor thread settings
        self.stop_event = threading.Event()
        self.monitor_thread = None
        
        # Data storage for recent monitoring data (last 30 data points)
        self.history_size = 30
        self.cpu_history = [0] * self.history_size
        self.memory_history = [0] * self.history_size
        self.gpu_history = [0] * self.history_size
        self.timestamp_history = [datetime.datetime.now().strftime("%H:%M:%S")] * self.history_size
        
        # GPU availability check
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
        self.gpu_device_id = getattr(config, 'GPU_DEVICE_ID', 0) if self.gpu_available else -1
        
        if self.gpu_available and self.gpu_device_id >= self.gpu_count:
            self.gpu_device_id = 0  # Fallback to default GPU
        
        logging.info(f"System monitor initialized with warning threshold: {warning_threshold}%, critical threshold: {critical_threshold}%")
        if self.gpu_available:
            gpu_name = torch.cuda.get_device_name(self.gpu_device_id)
            logging.info(f"Monitoring GPU: {gpu_name} (Device ID: {self.gpu_device_id})")
        
    def start_monitoring(self, interval=2.0):
        """Start system monitoring"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.stop_event.clear()
            self.monitor_thread = threading.Thread(target=self._monitor_system, args=(interval,), daemon=True)
            self.monitor_thread.start()
            logging.info("System monitoring started")
            
    def stop_monitoring(self):
        """Stop system monitoring"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_event.set()
            self.monitor_thread.join(timeout=3)
            logging.info("System monitoring stopped")
            
    def _monitor_system(self, interval):
        """Monitor system resource usage periodically"""
        while not self.stop_event.is_set():
            try:
                # Get system status
                status = self.get_system_status()
                
                # Update history
                with self._lock:
                    self.cpu_history.pop(0)
                    self.memory_history.pop(0)
                    self.gpu_history.pop(0)
                    self.timestamp_history.pop(0)
                    
                    self.cpu_history.append(status['cpu_percent'])
                    self.memory_history.append(status['memory_percent'])
                    self.gpu_history.append(status['gpu_percent'])
                    self.timestamp_history.append(datetime.datetime.now().strftime("%H:%M:%S"))
                
                # Log only when memory status changes
                if status["memory_status"] != self.last_status:
                    if status["memory_status"] == "critical":
                        logging.warning(f"Memory usage critical: {status['memory_percent']}%")
                        # Force memory cleanup
                        gc.collect()
                        if self.gpu_available:
                            torch.cuda.empty_cache()
                    elif status["memory_status"] == "warning":
                        logging.warning(f"Memory usage high: {status['memory_percent']}%")
                    
                    self.last_status = status["memory_status"]
                
            except Exception as e:
                logging.error(f"Error in system monitoring: {e}")
                
            time.sleep(interval)
    
    def get_history(self):
        """Return monitoring history data"""
        with self._lock:
            return {
                'cpu': self.cpu_history.copy(),
                'memory': self.memory_history.copy(),
                'gpu': self.gpu_history.copy(),
                'timestamps': self.timestamp_history.copy(),
                'gpu_available': self.gpu_available
            }
            
    def get_system_status(self):
        """Return current system status"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            
            # Determine memory status
            if memory_percent >= self.critical_threshold:
                memory_status = "critical"
            elif memory_percent >= self.warning_threshold:
                memory_status = "warning"
            else:
                memory_status = "normal"
            
            # GPU usage
            gpu_percent = 0
            if self.gpu_available:
                try:
                    # Check GPU memory usage
                    gpu_memory = torch.cuda.memory_allocated(self.gpu_device_id)
                    gpu_max_memory = torch.cuda.get_device_properties(self.gpu_device_id).total_memory
                    gpu_percent = (gpu_memory / gpu_max_memory) * 100
                    
                    logging.debug(f"GPU usage: {gpu_percent:.2f}%, Memory: {gpu_memory/(1024**2):.2f}MB / {gpu_max_memory/(1024**2):.2f}MB")
                except Exception as gpu_err:
                    logging.error(f"Error getting GPU status: {gpu_err}")
            
            return {
                "cpu_percent": cpu_percent,
                "memory_status": memory_status,
                "memory_percent": memory_percent,
                "memory_available": memory_info.available,
                "memory_total": memory_info.total,
                "gpu_percent": gpu_percent,
                "gpu_available": self.gpu_available
            }
        except Exception as e:
            logging.error(f"Error getting system status: {e}")
            return {
                "memory_status": "unknown", 
                "memory_percent": 0,
                "cpu_percent": 0,
                "gpu_percent": 0,
                "gpu_available": False
            }

class EmbeddingModel:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            
            # Device configuration with timeout protection
            print("Initializing EmbeddingModel...")
            
            # GPU usage settings from config
            use_gpu = getattr(config, 'USE_GPU', True)
            gpu_memory_fraction = getattr(config, 'GPU_MEMORY_FRACTION', 0.7)
            gpu_device_id = getattr(config, 'GPU_DEVICE_ID', 0)
            
            # PyTorch and CUDA version info
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA version: {torch.version.cuda}")
            
            # Device selection with simplified logic
            device = 'cpu'  # Default to CPU
            device_info = "CPU"
            
            if use_gpu and torch.cuda.is_available():
                try:
                    # Simple GPU configuration
                    gpu_count = torch.cuda.device_count()
                    print(f"Available GPU count: {gpu_count}")
                    
                    if gpu_device_id >= gpu_count:
                        gpu_device_id = 0  # Fallback to default GPU
                    
                    # GPU info
                    gpu_name = torch.cuda.get_device_name(gpu_device_id)
                    total_memory = torch.cuda.get_device_properties(gpu_device_id).total_memory
                    total_memory_gb = total_memory / (1024**3)
                    
                    print(f"Selected GPU {gpu_device_id}: {gpu_name}")
                    print(f"GPU total memory: {total_memory_gb:.2f} GB")
                    
                    # Memory limit setting
                    if 0 < gpu_memory_fraction < 1.0:
                        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, gpu_device_id)
                        reserved_memory_gb = total_memory_gb * gpu_memory_fraction
                        print(f"GPU memory limited to {gpu_memory_fraction:.1%} ({reserved_memory_gb:.2f} GB)")
                    
                    # Simple GPU test
                    print("Testing GPU with simple tensor operation...")
                    test_tensor = torch.rand(10, 10, device=f'cuda:{gpu_device_id}')
                    test_result = test_tensor @ test_tensor
                    print(f"GPU test successful. Result shape: {test_result.shape}")
                    
                    device = f'cuda:{gpu_device_id}'
                    device_info = f"GPU: {gpu_name}"
                    print(f"Using GPU: {gpu_name} (Device ID: {gpu_device_id})")
                    
                except Exception as e:
                    print(f"Error configuring GPU: {e}")
                    device = 'cpu'
                    device_info = "CPU (GPU failed)"
                    print("Falling back to CPU due to GPU configuration error")
            else:
                if not use_gpu:
                    print("GPU usage disabled in config, using CPU")
                elif not torch.cuda.is_available():
                    print("No CUDA-compatible GPU available, using CPU")
                device = 'cpu'
                device_info = "CPU"
            
            # Store device info
            cls._instance.device = device
            
            # Display device information
            print("\n" + "=" * 50)
            print(f"DEVICE INFORMATION: Using {device_info}")
            print("=" * 50 + "\n")
            
            # Load embedding model with timeout protection
            print("Loading embedding model...")
            print(f"Model: {config.EMBEDDING_MODEL}")
            
            # Configure model cache directory
            model_cache_dir = getattr(config, 'MODEL_CACHE_DIR', None)
            if model_cache_dir:
                # Create cache directory if it doesn't exist
                os.makedirs(model_cache_dir, exist_ok=True)
                print(f"Model cache directory: {model_cache_dir}")
                # Set environment variable for SentenceTransformer
                os.environ['SENTENCE_TRANSFORMERS_HOME'] = model_cache_dir
            
            # Model loading with timeout
            model_load_timeout = getattr(config, 'MODEL_LOAD_TIMEOUT', 120)
            
            def load_model_with_timeout():
                """Load model with timeout protection"""
                try:
                    print("Starting model download/loading...")
                    model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
                    print("Model loaded successfully!")
                    return model
                except Exception as e:
                    print(f"Error loading model: {e}")
                    raise
            
            # Use thread executor with timeout for model loading
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                try:
                    print(f"Loading model with {model_load_timeout} second timeout...")
                    future = executor.submit(load_model_with_timeout)
                    cls._instance.model = future.result(timeout=model_load_timeout)
                    print("Model loading completed successfully!")
                    
                except concurrent.futures.TimeoutError:
                    print(f"Model loading timed out after {model_load_timeout} seconds")
                    print("This might be due to slow internet connection or large model download")
                    print("Please check your internet connection and try again")
                    raise RuntimeError(f"Model loading timeout ({model_load_timeout}s)")
                    
                except Exception as e:
                    print(f"Failed to load embedding model: {e}")
                    raise
            
            # Cache configuration
            cache_size = config.EMBEDDING_CACHE_SIZE
            cls._instance.get_embedding_cached = lru_cache(maxsize=cache_size)(cls._instance._get_embedding_impl)
            
            # Thread pool for parallel processing
            max_workers = min(os.cpu_count() or 4, 8)
            cls._instance.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            
            # Memory monitoring system
            cls._instance.memory_monitor = SystemMonitor(warning_threshold=70, critical_threshold=85)
            cls._instance.memory_monitor.start_monitoring(interval=3.0)
            
            # Simplified batch size configuration
            base_batch_size = config.EMBEDDING_BATCH_SIZE
            
            # Simple resource-based batch size adjustment
            ram_info = psutil.virtual_memory()
            ram_usage_percent = ram_info.percent
            
            print(f"System RAM usage: {ram_usage_percent:.1f}%")
            
            if 'cuda' in device:
                # GPU mode batch size
                if ram_usage_percent < 50:
                    cls._instance.optimal_batch_size = min(base_batch_size * 2, 256)
                elif ram_usage_percent < 70:
                    cls._instance.optimal_batch_size = base_batch_size
                else:
                    cls._instance.optimal_batch_size = max(base_batch_size // 2, 32)
            else:
                # CPU mode batch size
                if ram_usage_percent < 50:
                    cls._instance.optimal_batch_size = min(32, base_batch_size)
                elif ram_usage_percent < 70:
                    cls._instance.optimal_batch_size = 16
                else:
                    cls._instance.optimal_batch_size = 8
                    
            cls._instance.current_batch_size = cls._instance.optimal_batch_size
            cls._instance._lock = threading.Lock()
            
            print(f"Optimal batch size: {cls._instance.optimal_batch_size}")
            print("EmbeddingModel initialization completed!")
            
            logging.info(f"Embedding model loaded on {device} with cache size {cache_size}, max workers: {max_workers}, batch size: {cls._instance.optimal_batch_size}")
            
        return cls._instance
    
    def get_embedding(self, text):
        """Convert text to embedding vector (with caching)"""
        if not text or text.isspace():
            return [0] * config.VECTOR_DIM
            
        # Text length limit - truncate overly long text
        max_text_length = 5000
        if len(text) > max_text_length:
            print(f"Warning: Text too long ({len(text)} chars), truncating to {max_text_length} chars")
            text = text[:max_text_length]
            
        # Use hash for very long text caching
        if len(text) > 1000:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            return self.get_embedding_cached(text_hash, text)
        else:
            return self.get_embedding_cached(text)
    
    def get_embeddings_batch(self, texts):
        """Generate embedding vectors for text batch"""
        if not isinstance(texts, list):
            logging.warning(f"Expected list for texts, got {type(texts)}")
            if isinstance(texts, str):
                texts = [texts]
            else:
                return []
                
        if not texts:
            return []
        
        # Filter empty texts and preprocess
        valid_texts = []
        empty_indices = []
        
        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text or text.isspace():
                empty_indices.append(i)
            else:
                # Text length limit
                if len(text) > 5000:
                    text = text[:5000]
                valid_texts.append((i, text))
        
        # Initialize result array (empty texts get zero vectors)
        results = [[0] * config.VECTOR_DIM for _ in range(len(texts))]
        
        if not valid_texts:
            return results
        
        # Batch size adjustment
        batch_size = self._adjust_batch_size()
        
        # Split texts into batches
        batches = [valid_texts[i:i+batch_size] for i in range(0, len(valid_texts), batch_size)]
        logging.info(f"Processing {len(valid_texts)} texts in {len(batches)} batches of size {batch_size}")
        
        start_time = time.time()
        
        # Process each batch
        for batch_idx, batch in enumerate(batches):
            indices, batch_texts = zip(*batch)
            
            try:
                # Batch embedding calculation
                with torch.no_grad():
                    vectors = self.model.encode(batch_texts, show_progress_bar=False)
                
                # Store results
                for i, vector in enumerate(vectors):
                    results[indices[i]] = vector.tolist() if isinstance(vector, np.ndarray) else vector
                    
            except Exception as e:
                logging.error(f"Error in batch {batch_idx}: {e}")
                # Fallback to individual processing
                for idx, text in zip(indices, batch_texts):
                    try:
                        vector = self.get_embedding(text)
                        results[idx] = vector
                    except Exception as inner_e:
                        logging.error(f"Error processing text at index {idx}: {inner_e}")
        
        elapsed_time = time.time() - start_time
        logging.debug(f"Successfully processed {len(valid_texts)} texts in {elapsed_time:.2f} seconds")
        return results
        
    def _adjust_batch_size(self):
        """Dynamically adjust batch size based on current system state"""
        system_status = self.memory_monitor.get_system_status()
        memory_percent = system_status["memory_percent"]
        
        with self._lock:
            if memory_percent > 85:
                self.current_batch_size = 1
            elif memory_percent > 70:
                self.current_batch_size = max(1, self.optimal_batch_size // 2)
            else:
                self.current_batch_size = self.optimal_batch_size
        
        return self.current_batch_size
    
    def _get_embedding_impl(self, text, original_text=None):
        """Actual embedding calculation implementation (called on cache miss)"""
        compute_text = original_text if original_text is not None else text
        
        # Text length limit
        max_text_length = 10000
        if len(compute_text) > max_text_length:
            logging.warning(f"Text too long ({len(compute_text)} chars), truncating to {max_text_length} chars")
            compute_text = compute_text[:max_text_length]
        
        try:
            timeout = 10
            with torch.no_grad():
                vector = self.model.encode(compute_text)
                result = vector.tolist() if isinstance(vector, np.ndarray) else vector
            
            # Validate result
            if isinstance(result, list) and len(result) == config.VECTOR_DIM:
                return result
            else:
                logging.warning(f"Invalid embedding result: {type(result)}")
                return [0] * config.VECTOR_DIM
                
        except Exception as e:
            logging.error(f"Error during embedding: {e}")
            return [0] * config.VECTOR_DIM
            
    def clear_cache(self):
        """Clear embedding cache and cleanup memory"""
        self.get_embedding_cached.cache_clear()
        logging.info("Embedding cache cleared")
        
        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("GPU memory cache cleared")
        
        return True
            
    def __del__(self):
        """Destructor: cleanup thread pool and memory monitoring"""
        try:
            if hasattr(self, 'memory_monitor'):
                self.memory_monitor.stop_monitoring()
                
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
                
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logging.info("EmbeddingModel resources cleaned up")
        except Exception as e:
            logging.error(f"Error in EmbeddingModel cleanup: {e}")
