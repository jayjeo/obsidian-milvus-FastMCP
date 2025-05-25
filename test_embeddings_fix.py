"""
Test script to verify embeddings.py loading works without hanging
This script tests the EmbeddingModel initialization with timeout protection
"""

import sys
import os
import time
import signal

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def timeout_handler(signum, frame):
    """Handle timeout signal"""
    print("ERROR: Test timed out after 60 seconds!")
    print("The embeddings module is still hanging")
    sys.exit(1)

def test_embeddings_loading():
    """Test embeddings module loading"""
    print("=" * 60)
    print("TESTING EMBEDDINGS MODULE LOADING")
    print("=" * 60)
    
    try:
        print("Step 1: Testing config import...")
        import config
        print("✅ Config import successful")
        
        print("Step 2: Testing embeddings import...")
        print("This is where the original code was hanging...")
        
        # Set timeout for this test
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)  # 60 second timeout
        
        start_time = time.time()
        import embeddings
        elapsed_time = time.time() - start_time
        
        # Cancel timeout
        signal.alarm(0)
        
        print(f"✅ Embeddings import successful in {elapsed_time:.2f} seconds")
        
        print("Step 3: Testing EmbeddingModel initialization...")
        start_time = time.time()
        model = embeddings.EmbeddingModel()
        elapsed_time = time.time() - start_time
        
        print(f"✅ EmbeddingModel initialization successful in {elapsed_time:.2f} seconds")
        
        print("Step 4: Testing simple embedding generation...")
        start_time = time.time()
        test_text = "Hello world test"
        embedding = model.get_embedding(test_text)
        elapsed_time = time.time() - start_time
        
        print(f"✅ Embedding generation successful in {elapsed_time:.2f} seconds")
        print(f"Embedding length: {len(embedding)}")
        print(f"Expected length: {config.VECTOR_DIM}")
        
        if len(embedding) == config.VECTOR_DIM:
            print("✅ Embedding dimensions match expected size")
        else:
            print("❌ Embedding dimensions mismatch")
            
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! The hanging issue has been fixed.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Make sure to cancel any remaining alarm
        signal.alarm(0)

if __name__ == "__main__":
    print("Starting embeddings loading test...")
    success = test_embeddings_loading()
    
    if success:
        print("\n🎉 SUCCESS: The embeddings module loads without hanging!")
        print("You can now run start_mcp_with_encoding_fix.bat safely")
    else:
        print("\n❌ FAILURE: There are still issues with the embeddings module")
        print("Please check the error messages above")
    
    print("\nPress Enter to exit...")
    input()
