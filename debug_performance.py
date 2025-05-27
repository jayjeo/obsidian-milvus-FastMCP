# Performance debugging script
import torch
import psutil
from embeddings import EmbeddingModel
import time

def debug_embedding_performance():
    print("🔍 PERFORMANCE DEBUGGING")
    print("=" * 50)
    
    # 1. Hardware Detection
    print("📊 Hardware Information:")
    print(f"   CPU Cores: {psutil.cpu_count()}")
    memory_info = psutil.virtual_memory()
    print(f"   RAM: {memory_info.total / (1024**3):.1f}GB ({memory_info.percent:.1f}% used)")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   GPU: {gpu_name}")
        print(f"   GPU Memory: {gpu_memory:.1f}GB")
    
    print()
    
    # 2. Initialize EmbeddingModel
    print("🚀 Initializing EmbeddingModel...")
    try:
        embedding_model = EmbeddingModel()
        print("✅ EmbeddingModel initialized successfully")
        
        # Check hardware profiler
        if hasattr(embedding_model, 'hardware_profiler'):
            profile = embedding_model.hardware_profiler.performance_profile
            print(f"   🎯 Detected Profile: {profile}")
        else:
            print("   ❌ No hardware profiler found")
        
        # Check batch optimizer
        if hasattr(embedding_model, 'batch_optimizer'):
            current_batch = embedding_model.batch_optimizer.current_batch_size
            min_batch = embedding_model.batch_optimizer.min_batch_size
            max_batch = embedding_model.batch_optimizer.max_batch_size
            print(f"   📦 Batch Size: {current_batch} (range: {min_batch}-{max_batch})")
        else:
            print("   ❌ No batch optimizer found")
            
    except Exception as e:
        print(f"❌ Error initializing EmbeddingModel: {e}")
        return
    
    print()
    
    # 3. Test Individual vs Batch Processing
    print("⚡ Testing Processing Methods...")
    
    test_texts = [
        "This is a test sentence for embedding performance testing.",
        "Another test sentence to evaluate batch processing speed.",
        "Third test sentence for comprehensive performance analysis.",
        "Fourth test sentence to check GPU utilization patterns.",
        "Fifth test sentence for memory usage optimization testing."
    ] * 20  # 100 total texts
    
    print(f"   📝 Test Data: {len(test_texts)} texts")
    
    # Individual processing test
    try:
        print("   🔄 Testing Individual Processing...")
        start_time = time.time()
        individual_vectors = []
        for text in test_texts:
            vector = embedding_model.get_embedding(text)
            individual_vectors.append(vector)
        individual_time = time.time() - start_time
        print(f"   ⏱️ Individual Time: {individual_time:.2f}s ({len(test_texts)/individual_time:.1f} texts/sec)")
    except Exception as e:
        print(f"   ❌ Individual processing failed: {e}")
        individual_time = float('inf')
    
    # Batch processing test
    try:
        print("   🚀 Testing Batch Processing...")
        start_time = time.time()
        
        # Clear GPU cache first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        batch_vectors = embedding_model.get_embeddings_batch_adaptive(test_texts)
        batch_time = time.time() - start_time
        print(f"   ⏱️ Batch Time: {batch_time:.2f}s ({len(test_texts)/batch_time:.1f} texts/sec)")
        
        # Check if batch processing actually worked
        if len(batch_vectors) == len(test_texts):
            print("   ✅ Batch processing successful")
            if batch_time < individual_time:
                speedup = individual_time / batch_time
                print(f"   🚀 Speedup: {speedup:.2f}x faster")
            else:
                print("   ⚠️ Batch processing slower than individual")
        else:
            print(f"   ❌ Batch processing failed: got {len(batch_vectors)} vectors for {len(test_texts)} texts")
            
    except Exception as e:
        print(f"   ❌ Batch processing failed: {e}")
        import traceback
        print(f"   Stack trace: {traceback.format_exc()}")
    
    print()
    
    # 4. System Resource Check
    print("💻 System Resource Analysis:")
    if hasattr(embedding_model, 'system_monitor'):
        try:
            status = embedding_model.system_monitor.get_system_status()
            print(f"   CPU: {status.get('cpu_percent', 0):.1f}%")
            print(f"   Memory: {status.get('memory_percent', 0):.1f}%")
            print(f"   GPU: {status.get('gpu_percent', 0):.1f}%")
            print(f"   GPU Memory: {status.get('gpu_memory_percent', 0):.1f}%")
        except Exception as e:
            print(f"   ❌ System monitoring error: {e}")
    else:
        print("   ❌ No system monitor found")
    
    print()
    print("🔧 RECOMMENDATIONS:")
    
    # Provide specific recommendations
    if hasattr(embedding_model, 'hardware_profiler'):
        profile = embedding_model.hardware_profiler.performance_profile
        if 'cpu' in profile.lower():
            print("   ⚠️ System detected as CPU-only mode")
            print("   💡 Check GPU detection and CUDA installation")
        elif 'low_end_gpu' in profile or 'very_low_end_gpu' in profile:
            print("   ⚠️ Low-end GPU detected")
            print("   💡 Consider increasing batch size manually")
    
    if hasattr(embedding_model, 'batch_optimizer'):
        current_batch = embedding_model.batch_optimizer.current_batch_size
        if current_batch < 32:
            print(f"   ⚠️ Very small batch size ({current_batch})")
            print("   💡 Try manually increasing batch size")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    debug_embedding_performance()
