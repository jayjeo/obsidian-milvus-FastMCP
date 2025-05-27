#!/usr/bin/env python3
"""
실제 파일로 배치 처리 성능 테스트
"""

import os
import time
import torch
from embeddings import EmbeddingModel
from obsidian_processor import ObsidianProcessor
from milvus_manager import MilvusManager
import config

def test_real_file_processing():
    """실제 파일로 배치 처리 테스트"""
    print("🧪 Starting real file processing test...")
    
    # 1. 시스템 상태 확인
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 2. EmbeddingModel 초기화
    embedding_model = EmbeddingModel()
    
    # 3. 작은 테스트 파일 생성
    test_dir = os.path.join(config.OBSIDIAN_VAULT_PATH, "batch_test")
    os.makedirs(test_dir, exist_ok=True)
    
    test_file = os.path.join(test_dir, "batch_test.md")
    
    # 테스트 파일 생성 (여러 청크가 생성되도록)
    test_content = """# Test Document for Batch Processing

## Section 1: Introduction
This is a test document designed to create multiple chunks for batch processing testing. The content needs to be long enough to be split into several chunks to properly test the batch embedding functionality.

## Section 2: Technical Details
Batch processing should significantly improve performance by processing multiple text chunks simultaneously instead of one by one. This reduces the overhead of individual GPU calls and maximizes throughput.

## Section 3: Performance Expectations
We expect to see:
- 5-10x speed improvement with batch processing
- Higher GPU utilization during processing
- More efficient memory usage
- Reduced processing time per chunk

## Section 4: Implementation Details
The batch processing implementation should use the get_embeddings_batch_adaptive method which automatically adjusts batch sizes based on system capabilities and current resource usage.

## Section 5: Monitoring
We will monitor:
- GPU utilization percentage
- Memory usage patterns
- Processing time per batch
- Overall throughput improvement

## Section 6: Conclusion
This test will help us verify that the batch processing optimization is working correctly and providing the expected performance improvements.

This document contains enough content to be split into multiple chunks, allowing us to test the batch processing functionality properly.
"""
    
    # 테스트 파일 작성
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print(f"✅ Test file created: {test_file}")
    
    # 4. ObsidianProcessor 초기화
    try:
        milvus_manager = MilvusManager()
        processor = ObsidianProcessor(milvus_manager)
        print("✅ ObsidianProcessor initialized")
    except Exception as e:
        print(f"❌ Error initializing processor: {e}")
        return
    
    # 5. 파일 처리 테스트
    print("\n🚀 Testing file processing with batch embedding...")
    
    try:
        # GPU 모니터링 시작
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated()
        
        start_time = time.time()
        
        # 파일 처리
        success = processor.process_file(test_file)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if torch.cuda.is_available():
            end_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = (peak_memory - start_memory) / 1024**2  # MB
        
        print(f"\n📊 Processing Results:")
        print(f"✅ Success: {success}")
        print(f"⏱️ Processing time: {processing_time:.2f} seconds")
        
        if torch.cuda.is_available():
            print(f"🔥 GPU memory used: {memory_used:.1f} MB")
            print(f"📈 Peak GPU memory: {peak_memory / 1024**2:.1f} MB")
        
        # 현재 배치 크기 확인
        if hasattr(processor.embedding_model, 'batch_optimizer'):
            batch_size = processor.embedding_model.batch_optimizer.current_batch_size
            print(f"📦 Current batch size: {batch_size}")
        
        # 6. 청크별 개별 처리와 비교
        print("\n🔄 Testing individual chunk processing for comparison...")
        
        # 파일에서 청크 추출
        chunks, metadata = processor._extract_chunks_from_file(test_file)
        if chunks:
            print(f"📄 File contains {len(chunks)} chunks")
            
            # 개별 처리 시간 측정
            start_time = time.time()
            individual_vectors = []
            for chunk in chunks:
                vector = embedding_model.get_embedding(chunk)
                individual_vectors.append(vector)
            individual_time = time.time() - start_time
            
            # 배치 처리 시간 측정
            start_time = time.time()
            batch_vectors = embedding_model.get_embeddings_batch_adaptive(chunks)
            batch_time = time.time() - start_time
            
            print(f"\n⚡ Performance Comparison:")
            print(f"Individual processing: {individual_time:.2f}s")
            print(f"Batch processing: {batch_time:.2f}s")
            if batch_time > 0:
                speedup = individual_time / batch_time
                print(f"🚀 Speedup: {speedup:.2f}x")
            
            # 벡터 품질 확인
            if len(batch_vectors) == len(individual_vectors):
                print(f"✅ Vector count matches: {len(batch_vectors)}")
            else:
                print(f"❌ Vector count mismatch: batch={len(batch_vectors)}, individual={len(individual_vectors)}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    # 7. 정리
    try:
        os.remove(test_file)
        os.rmdir(test_dir)
        print(f"🧹 Cleanup completed")
    except:
        pass

if __name__ == "__main__":
    test_real_file_processing()
