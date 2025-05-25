# Import warning suppressor first to suppress all warnings
import warning_suppressor

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

# Additional warning suppression
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', module='huggingface_hub')
warnings.filterwarnings('ignore', module='transformers')

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL if hasattr(config, 'LOG_LEVEL') else 'WARNING'),
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class SystemMonitor:
    """Simplified system monitor for compatibility"""
    def __init__(self, *args, **kwargs):
        self.gpu_available = torch.cuda.is_available()
        
    def start_monitoring(self, *args, **kwargs):
        pass
        
    def stop_monitoring(self):
        pass
        
    def get_system_status(self):
        return {
            "memory_status": "normal",
            "memory_percent": 50,
            "cpu_percent": 50,
            "gpu_percent": 0,
            "gpu_available": self.gpu_available
        }
        
    def get_history(self):
        return {
            'cpu': [50] * 30,
            'memory': [50] * 30,
            'gpu': [0] * 30,
            'timestamps': ['00:00:00'] * 30,
            'gpu_available': self.gpu_available
        }

class EmbeddingModel:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            
            print("Initializing EmbeddingModel...")
            
            # GPU usage settings from config
            use_gpu = getattr(config, 'USE_GPU', True)
            gpu_device_id = getattr(config, 'GPU_DEVICE_ID', 0)
            
            # PyTorch and CUDA version info
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA version: {torch.version.cuda}")
            
            # Device selection
            device = 'cpu'  # Default to CPU
            device_info = "CPU"
            
            if use_gpu and torch.cuda.is_available():
                try:
                    gpu_count = torch.cuda.device_count()
                    print(f"Available GPU count: {gpu_count}")
                    
                    if gpu_device_id >= gpu_count:
                        gpu_device_id = 0
                    
                    gpu_name = torch.cuda.get_device_name(gpu_device_id)
                    total_memory = torch.cuda.get_device_properties(gpu_device_id).total_memory
                    total_memory_gb = total_memory / (1024**3)
                    
                    print(f"Selected GPU {gpu_device_id}: {gpu_name}")
                    print(f"GPU total memory: {total_memory_gb:.2f} GB")
                    
                    # Simple GPU test
                    print("Testing GPU...")
                    test_tensor = torch.rand(10, 10, device=f'cuda:{gpu_device_id}')
                    test_result = test_tensor @ test_tensor
                    print(f"GPU test successful!")
                    
                    device = f'cuda:{gpu_device_id}'
                    device_info = f"GPU: {gpu_name}"
                    
                except Exception as e:
                    print(f"Error configuring GPU: {e}")
                    print("Falling back to CPU")
            
            # Store device info
            cls._instance.device = device
            
            # Display device information
            print("\n" + "=" * 50)
            print(f"DEVICE INFORMATION: Using {device_info}")
            print("=" * 50 + "\n")
            
            # Configure model cache directory
            model_cache_dir = getattr(config, 'MODEL_CACHE_DIR', None)
            if model_cache_dir:
                os.makedirs(model_cache_dir, exist_ok=True)
                os.environ['SENTENCE_TRANSFORMERS_HOME'] = model_cache_dir
                print(f"Model cache directory: {model_cache_dir}")
            
            # Load model directly without threading
            print(f"Loading model: {config.EMBEDDING_MODEL}")
            print("This may take a few minutes on first run...")
            
            try:
                start_time = time.time()
                
                # Suppress warnings during loading
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Load the model directly
                    cls._instance.model = SentenceTransformer(
                        config.EMBEDDING_MODEL,
                        device=device,
                        trust_remote_code=True
                    )
                
                elapsed = time.time() - start_time
                print(f"\nModel loaded successfully in {elapsed:.1f} seconds!")
                
            except Exception as e:
                print(f"\nError loading model: {e}")
                print(f"Error type: {type(e).__name__}")
                print("\nTroubleshooting:")
                print("1. Check internet connection")
                print("2. Try using a VPN if HuggingFace is blocked")
                print("3. Manually download the model")
                raise
            
            # Cache configuration
            cache_size = config.EMBEDDING_CACHE_SIZE
            cls._instance.get_embedding_cached = lru_cache(maxsize=cache_size)(cls._instance._get_embedding_impl)
            
            # Initialize other components
            cls._instance.memory_monitor = SystemMonitor()
            cls._instance.memory_monitor.start_monitoring()
            
            # Batch size configuration
            base_batch_size = config.EMBEDDING_BATCH_SIZE
            ram_info = psutil.virtual_memory()
            ram_usage_percent = ram_info.percent
            
            print(f"System RAM usage: {ram_usage_percent:.1f}%")
            
            if 'cuda' in device:
                # GPU mode
                if ram_usage_percent < 70:
                    cls._instance.optimal_batch_size = base_batch_size
                else:
                    cls._instance.optimal_batch_size = max(base_batch_size // 2, 32)
            else:
                # CPU mode - smaller batches
                cls._instance.optimal_batch_size = min(32, base_batch_size)
            
            cls._instance.current_batch_size = cls._instance.optimal_batch_size
            cls._instance._lock = threading.Lock()
            
            print(f"Optimal batch size: {cls._instance.optimal_batch_size}")
            print("EmbeddingModel initialization completed!")
            
        return cls._instance
    
    def get_embedding(self, text):
        """Convert text to embedding vector (with caching)"""
        if not text or text.isspace():
            return [0] * config.VECTOR_DIM
        
        # Text length limit
        max_text_length = 5000
        if len(text) > max_text_length:
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
            if isinstance(texts, str):
                texts = [texts]
            else:
                return []
        
        if not texts:
            return []
        
        # Filter empty texts
        valid_texts = []
        results = [[0] * config.VECTOR_DIM for _ in range(len(texts))]
        
        for i, text in enumerate(texts):
            if isinstance(text, str) and text and not text.isspace():
                if len(text) > 5000:
                    text = text[:5000]
                valid_texts.append((i, text))
        
        if not valid_texts:
            return results
        
        # Process in batches
        batch_size = self.current_batch_size
        
        for batch_start in range(0, len(valid_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_texts))
            batch = valid_texts[batch_start:batch_end]
            indices, batch_texts = zip(*batch)
            
            try:
                with torch.no_grad():
                    vectors = self.model.encode(batch_texts, show_progress_bar=False)
                
                for i, vector in enumerate(vectors):
                    results[indices[i]] = vector.tolist() if isinstance(vector, np.ndarray) else vector
                    
            except Exception as e:
                logging.error(f"Error in batch: {e}")
                # Fallback to individual processing
                for idx, text in zip(indices, batch_texts):
                    try:
                        vector = self.get_embedding(text)
                        results[idx] = vector
                    except Exception as inner_e:
                        logging.error(f"Error processing text: {inner_e}")
        
        return results
    
    def _adjust_batch_size(self):
        """Dynamically adjust batch size"""
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent
        
        with self._lock:
            if memory_percent > 85:
                self.current_batch_size = 1
            elif memory_percent > 70:
                self.current_batch_size = max(1, self.optimal_batch_size // 2)
            else:
                self.current_batch_size = self.optimal_batch_size
        
        return self.current_batch_size
    
    def _get_embedding_impl(self, text, original_text=None):
        """Actual embedding calculation implementation"""
        compute_text = original_text if original_text is not None else text
        
        if len(compute_text) > 10000:
            compute_text = compute_text[:10000]
        
        try:
            with torch.no_grad():
                vector = self.model.encode(compute_text)
                result = vector.tolist() if isinstance(vector, np.ndarray) else vector
            
            if isinstance(result, list) and len(result) == config.VECTOR_DIM:
                return result
            else:
                return [0] * config.VECTOR_DIM
                
        except Exception as e:
            logging.error(f"Error during embedding: {e}")
            return [0] * config.VECTOR_DIM
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.get_embedding_cached.cache_clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    
    def __del__(self):
        """Destructor"""
        try:
            if hasattr(self, 'memory_monitor'):
                self.memory_monitor.stop_monitoring()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
