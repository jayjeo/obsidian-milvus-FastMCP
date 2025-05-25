"""
Simple test script to diagnose embedding model loading issues
"""
import os
import sys
import time
import warnings

# Add project directory to path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

print("=" * 60)
print("EMBEDDING MODEL LOADING TEST")
print("=" * 60)

# Import config first
try:
    import config
    print(f"Config loaded successfully")
    print(f"Model: {config.EMBEDDING_MODEL}")
    print(f"Model cache dir: {config.MODEL_CACHE_DIR}")
    print(f"Timeout: {config.MODEL_LOAD_TIMEOUT} seconds")
except Exception as e:
    print(f"Error loading config: {e}")
    sys.exit(1)

print("\n" + "-" * 60)

# Test direct model loading without our wrapper
print("\nTesting direct SentenceTransformer loading...")
print("(This bypasses our timeout mechanism to see raw behavior)")

try:
    # Suppress warnings
    warnings.filterwarnings('ignore')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    from sentence_transformers import SentenceTransformer
    import torch
    
    # Check PyTorch and CUDA
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Device selection
    device = 'cuda' if torch.cuda.is_available() and config.USE_GPU else 'cpu'
    print(f"Using device: {device}")
    
    # Set cache directory
    if config.MODEL_CACHE_DIR:
        os.makedirs(config.MODEL_CACHE_DIR, exist_ok=True)
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = config.MODEL_CACHE_DIR
        print(f"Model cache directory set to: {config.MODEL_CACHE_DIR}")
    
    # Try to load model
    print(f"\nAttempting to load model: {config.EMBEDDING_MODEL}")
    print("This may take a while if downloading for the first time...")
    
    start_time = time.time()
    
    # Create model
    model = SentenceTransformer(
        config.EMBEDDING_MODEL,
        device=device,
        trust_remote_code=True
    )
    
    elapsed = time.time() - start_time
    print(f"\nModel loaded successfully in {elapsed:.1f} seconds!")
    
    # Test embedding
    print("\nTesting embedding generation...")
    test_text = "This is a test sentence."
    embedding = model.encode(test_text)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding dimension: {len(embedding)}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
except KeyboardInterrupt:
    print("\n\nTest interrupted by user!")
    print("The model loading seems to be hanging.")
    
except Exception as e:
    print(f"\n\nError during test: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()

print("\nPress Enter to exit...")
input()
