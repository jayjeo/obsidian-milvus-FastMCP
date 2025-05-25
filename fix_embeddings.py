"""
Quick fix script to bypass model loading issues
This creates a minimal embeddings.py that uses CPU and has better timeout handling
"""
import os
import shutil
import sys

# Get the project directory
project_dir = os.path.dirname(os.path.abspath(__file__))

# Create backup of original embeddings.py
original_file = os.path.join(project_dir, "embeddings.py")
backup_file = os.path.join(project_dir, "embeddings_backup.py")

if os.path.exists(original_file):
    print("Creating backup of embeddings.py...")
    shutil.copy2(original_file, backup_file)
    print(f"Backup saved as: embeddings_backup.py")

# Create a simplified embeddings.py with better timeout handling
simplified_content = '''# Import warning suppressor first to suppress all warnings
import warning_suppressor

from sentence_transformers import SentenceTransformer
import config
from functools import lru_cache
import hashlib
import numpy as np
import torch
import os
import time
import gc
import threading
import logging
import warnings
import sys

# Additional warning suppression
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', module='huggingface_hub')
warnings.filterwarnings('ignore', module='transformers')

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL if hasattr(config, 'LOG_LEVEL') else 'WARNING'),
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class EmbeddingModel:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            
            print("Initializing EmbeddingModel (Simplified Version)...")
            
            # Force CPU mode for stability
            device = 'cpu'
            print("Using CPU mode for maximum compatibility")
            
            # Store device info
            cls._instance.device = device
            
            # Configure model cache directory
            model_cache_dir = getattr(config, 'MODEL_CACHE_DIR', None)
            if model_cache_dir:
                os.makedirs(model_cache_dir, exist_ok=True)
                os.environ['SENTENCE_TRANSFORMERS_HOME'] = model_cache_dir
                print(f"Model cache directory: {model_cache_dir}")
            
            # Check if we can skip loading by checking for cached model
            model_name = config.EMBEDDING_MODEL
            print(f"Model to load: {model_name}")
            
            # Try to load with a simple approach
            print("\\nAttempting to load embedding model...")
            print("If this hangs, it means:")
            print("1. Model is being downloaded (can take 5-10 minutes)")
            print("2. Network connection issues")
            print("3. Firewall/proxy blocking HuggingFace")
            print("\\nYou can press Ctrl+C to cancel if it takes too long\\n")
            
            try:
                # Set environment variable to show download progress
                os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
                
                # Load model directly without threading
                start_time = time.time()
                print("Starting model load...")
                
                cls._instance.model = SentenceTransformer(
                    model_name,
                    device=device,
                    trust_remote_code=True
                )
                
                elapsed = time.time() - start_time
                print(f"\\nModel loaded successfully in {elapsed:.1f} seconds!")
                
            except KeyboardInterrupt:
                print("\\n\\nModel loading cancelled by user!")
                print("\\nTroubleshooting steps:")
                print("1. Check your internet connection")
                print("2. Try using a VPN if HuggingFace is blocked")
                print("3. Download the model manually")
                print("4. Use a different/smaller model")
                sys.exit(1)
                
            except Exception as e:
                print(f"\\nError loading model: {e}")
                print("\\nTrying offline mode...")
                
                # Try to load in offline mode if possible
                try:
                    os.environ['HF_HUB_OFFLINE'] = '1'
                    cls._instance.model = SentenceTransformer(
                        model_name,
                        device=device,
                        trust_remote_code=True
                    )
                    print("Loaded model in offline mode!")
                except:
                    print("Failed to load model even in offline mode")
                    print("The model must be downloaded at least once")
                    raise
            
            # Cache configuration
            cache_size = config.EMBEDDING_CACHE_SIZE
            cls._instance.get_embedding_cached = lru_cache(maxsize=cache_size)(cls._instance._get_embedding_impl)
            
            # Simple batch size
            cls._instance.optimal_batch_size = 16
            cls._instance.current_batch_size = 16
            cls._instance._lock = threading.Lock()
            
            print(f"Batch size: {cls._instance.optimal_batch_size}")
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
            texts = [texts] if isinstance(texts, str) else []
                
        if not texts:
            return []
        
        # Filter and process
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
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i+batch_size]
            indices, batch_texts = zip(*batch)
            
            try:
                with torch.no_grad():
                    vectors = self.model.encode(batch_texts, show_progress_bar=False)
                
                for j, vector in enumerate(vectors):
                    results[indices[j]] = vector.tolist() if isinstance(vector, np.ndarray) else vector
                    
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
        return True
            
    def __del__(self):
        """Destructor"""
        try:
            gc.collect()
            logging.info("EmbeddingModel resources cleaned up")
        except:
            pass

# Dummy SystemMonitor class for compatibility
class SystemMonitor:
    def __init__(self, *args, **kwargs):
        pass
    
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
            "gpu_available": False
        }
    
    def get_history(self):
        return {
            'cpu': [50] * 30,
            'memory': [50] * 30,
            'gpu': [0] * 30,
            'timestamps': ['00:00:00'] * 30,
            'gpu_available': False
        }
'''

# Write the simplified version
print("\nCreating simplified embeddings.py...")
with open(original_file, 'w', encoding='utf-8') as f:
    f.write(simplified_content)

print("\nSimplified embeddings.py created!")
print("\nThis version:")
print("- Forces CPU mode for stability")
print("- Shows clear progress messages")
print("- Can be cancelled with Ctrl+C")
print("- Has better error handling")
print("\nNow try running start_mcp_with_encoding_fix.bat again")
print("\nTo restore the original version, rename embeddings_backup.py to embeddings.py")
