# Enhanced incremental embedding logic with dynamic optimization - FIXED VERSION
import os
import math
import time
import gc
from tqdm import tqdm

# Import centralized logger
from logger import get_logger

# Initialize module logger
logger = get_logger(__name__)

def estimate_chunk_count_optimized(file_size, chunk_size=1000, chunk_overlap=100, min_chunk_size=100):
    """Enhanced chunk count estimation with dynamic parameters"""
    if file_size < min_chunk_size:
        return 0
    
    # Try to get optimal chunk size from config or system
    try:
        import config
        if hasattr(config, 'CHUNK_SIZE'):
            chunk_size = getattr(config, 'CHUNK_SIZE', chunk_size)
        if hasattr(config, 'CHUNK_OVERLAP'):
            chunk_overlap = getattr(config, 'CHUNK_OVERLAP', chunk_overlap)
        if hasattr(config, 'CHUNK_MIN_SIZE'):
            min_chunk_size = getattr(config, 'CHUNK_MIN_SIZE', min_chunk_size)
    except:
        pass  # Use default values if config unavailable
    
    stride = chunk_size - chunk_overlap
    estimated_chunks = max(1, math.ceil((file_size - chunk_overlap) / stride))
    
    # Apply reasonable limits
    # Use config value for max chunks per file
    max_chunks_per_file = getattr(config, 'MAX_CHUNKS_PER_FILE', 1000)  # 1000 from config
    return min(estimated_chunks, max_chunks_per_file)

def get_system_performance_profile():
    """Get current system performance profile for optimization"""
    try:
        import psutil
        import torch
        
        # Memory information
        memory_info = psutil.virtual_memory()
        memory_gb = memory_info.total / (1024**3)
        memory_percent = memory_info.percent
        
        # CPU information
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # GPU information
        gpu_available = torch.cuda.is_available()
        gpu_memory_gb = 0
        if gpu_available:
            try:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except:
                gpu_memory_gb = 0
        
        # Determine performance profile
        if gpu_available and gpu_memory_gb >= 8:
            if memory_percent < 70:
                return "high_performance"
            else:
                return "gpu_memory_constrained"
        elif memory_gb >= 16 and cpu_count >= 8:
            if memory_percent < 70:
                return "high_cpu_performance"
            else:
                return "cpu_memory_constrained"
        else:
            return "standard_performance"
            
    except Exception as e:
        print(f"Warning: Could not determine system performance profile: {e}")
        return "standard_performance"

def optimize_batch_size_for_system():
    """Determine optimal batch size based on system capabilities"""
    profile = get_system_performance_profile()
    
    batch_sizes = {
        "high_performance": 128,           # Reduced from 256 for stability
        "gpu_memory_constrained": 64,     # Reduced from 128
        "high_cpu_performance": 32,       # Reduced from 64
        "cpu_memory_constrained": 16,     # Reduced from 32
        "standard_performance": 8         # Reduced from 16
    }
    
    return batch_sizes.get(profile, 16)

def process_incremental_embedding(processor):
    """ENHANCED: Incremental embedding with dynamic optimization and performance monitoring - FIXED"""
    logger.info("Starting ENHANCED incremental embedding with dynamic optimization")
    print(f"\n🚀 Starting ENHANCED incremental embedding with dynamic optimization")
    
    vault_path = processor.vault_path
    milvus = processor.milvus_manager
    embedding_model = processor.embedding_model
    
    # Performance monitoring setup
    start_time = time.time()  # Track overall process time
    profile_start = start_time  # Alias for compatibility with later code
    performance_stats = {
        "files_processed": 0,
        "files_skipped": 0,
        "files_deleted": 0,
        "processing_errors": 0,
        "total_chunks_processed": 0
    }
    logger.debug(f"Performance monitoring initialized: {performance_stats}")
    
    # Get system performance profile for optimization
    system_profile = get_system_performance_profile()
    optimal_batch_size = optimize_batch_size_for_system()
    logger.info(f"System profile: {system_profile}, Optimal batch size: {optimal_batch_size}")
    print(f"📊 System profile: {system_profile}")
    print(f"🎥 Optimal batch size: {optimal_batch_size}")
    
    # ENHANCED: Get existing file information from Milvus with better performance
    existing_files_info = {}   # path → updated_at (float)
    chunk_counts = {}          # path → chunk count

    try:
        logger.info("Querying existing files from Milvus database")
        print("🔍 Querying existing files from Milvus database...")
        
        # Use intelligent batch sizing from processor's DynamicBatchOptimizer if available
        if hasattr(processor, 'milvus_manager') and hasattr(processor.milvus_manager, '_get_optimal_query_limit'):
            max_limit = processor.milvus_manager._get_optimal_query_limit()
        else:
            max_limit = 16000  # Safe fallback - never exceed Milvus limit
        
        offset = 0
        total_queried = 0
        
        while True:
            try:
                # Use intelligent limit - already capped at 16000
                current_limit = max_limit
                
                results = milvus.collection.query(
                    expr="id >= 0",
                    output_fields=["path", "updated_at"], 
                    limit=current_limit,
                    offset=offset
                )
                
                if not results:
                    break
                
                for r in results:
                    path = r["path"]
                    ts = processor._normalize_timestamp(r.get("updated_at"))
                    existing_files_info[path] = ts
                    chunk_counts[path] = chunk_counts.get(path, 0) + 1
                    total_queried += 1
                
                offset += current_limit
                if len(results) < current_limit:
                    break
                
                # Progress feedback and memory management
                if total_queried % 10000 == 0:
                    logger.debug(f"Queried {total_queried} records from database")
                    print(f"📊 Queried {total_queried} records from database...")
                    gc.collect()  # Memory management
                    
            except Exception as e:
                logger.error(f"Error in batch query (offset {offset}): {e}", exc_info=True)
                print(f"⚠️ Error in batch query (offset {offset}): {e}")
                # Break on any error since we're using safe limits
                break
        
        logger.info(f"Found {len(existing_files_info)} unique files in database")
        print(f"✅ Found {len(existing_files_info)} unique files in database")
        
    except Exception as e:
        logger.error(f"Error querying existing file information: {e}", exc_info=True)
        print(f"\n⚠️ Error querying existing file information: {e}")
        print("💡 This error might occur if Milvus collection is empty or connection issues exist.")
        print("💡 Processing will continue...")
        logger.warning("Processing will continue despite error querying file information")

    # ENHANCED: File system scanning and decision making with optimization
    fs_paths = set()
    files_to_process = []
    skipped = []
    to_delete = []
    
    logger.info("Starting enhanced file system scanning")
    print("🔍 Enhanced file system scanning...")
    
    # Performance optimization: pre-compile supported extensions
    supported_extensions = ('.md', '.pdf')
    logger.debug(f"Using supported extensions: {supported_extensions}")
    
    scan_start_time = time.time()
    scanned_files = 0
    
    for root, _, files in os.walk(vault_path):
        # Skip hidden directories early
        if os.path.basename(root).startswith(('.', '_')):
            continue
            
        for fname in files:
            # Quick extension check
            if not fname.endswith(supported_extensions) or fname.startswith("."):
                continue
            
            scanned_files += 1
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, vault_path)
            fs_paths.add(rel_path)

            try:
                file_size = os.path.getsize(full_path)
                file_mtime = os.path.getmtime(full_path)
                est_chunks = estimate_chunk_count_optimized(file_size)
                
                # Check for potential problematic paths with special characters (e.g., Excalidraw files)
                has_special_chars = any(c in rel_path for c in "'\"()[]{},;")
                if has_special_chars:
                    logger.debug(f"Found path with special characters: {rel_path}")

                db_ts = existing_files_info.get(rel_path, None)
                db_chunks = chunk_counts.get(rel_path, 0)

                if db_ts is None:
                    # New file
                    logger.debug(f"New file detected: {rel_path}")
                    files_to_process.append((full_path, rel_path))
                    performance_stats["files_processed"] += 1
                    continue

                time_diff = abs(file_mtime - db_ts)

                # ENHANCED: Smart decision making with performance optimization
                if time_diff > 2.0:
                    # Definitely modified
                    logger.debug(f"Modified file detected: {rel_path} (time_diff: {time_diff:.2f}s)")
                    to_delete.append(rel_path)
                    files_to_process.append((full_path, rel_path))
                    performance_stats["files_processed"] += 1
                elif time_diff < 0.1:
                    # Very likely unchanged - check chunk consistency
                    logger.debug(f"Unchanged file detected: {rel_path} (time_diff: {time_diff:.2f}s)")

                    if db_chunks >= est_chunks:
                        skipped.append(rel_path)
                        performance_stats["files_skipped"] += 1
                    else:
                        # Incomplete processing - reprocess
                        to_delete.append(rel_path)
                        files_to_process.append((full_path, rel_path))
                        performance_stats["files_processed"] += 1
                else:
                    # VERIFY stage: ambiguous timestamp
                    if db_chunks < est_chunks:
                        to_delete.append(rel_path)
                        files_to_process.append((full_path, rel_path))
                        performance_stats["files_processed"] += 1
                    else:
                        skipped.append(rel_path)
                        performance_stats["files_skipped"] += 1
                        
            except Exception as e:
                # Check if path contains special characters that might cause issues
                has_special_chars = any(c in rel_path for c in "'\"()[]{},;")
                if has_special_chars:
                    logger.warning(f"Error processing file with special characters: {rel_path}: {e}", exc_info=True)
                    print(f"⚠️ Error processing file with special characters {rel_path}: {e}")
                else:
                    logger.error(f"Error processing file {rel_path}: {e}", exc_info=True)
                    print(f"⚠️ Error processing file {rel_path}: {e}")
                    
                performance_stats["processing_errors"] += 1
            
            # Progress feedback for large vaults
            if scanned_files % 1000 == 0:
                elapsed = time.time() - scan_start_time
                logger.info(f"Scanned {scanned_files} files in {elapsed:.1f}s")
                print(f"📊 Scanned {scanned_files} files in {elapsed:.1f}s...")

    # ENHANCED: Batch deletion processing
    all_deletions = to_delete + list(set(existing_files_info.keys()) - fs_paths)
    performance_stats["files_deleted"] = len(all_deletions)
    
    # Check for paths with special characters that might cause issues in batch deletion
    paths_with_special_chars = [path for path in all_deletions if any(c in path for c in "'\"()[]{},;")]
    if paths_with_special_chars:
        logger.info(f"Found {len(paths_with_special_chars)} paths with special characters that need special handling")
    
    logger.info(f"Processing summary - Files to process: {len(files_to_process)}, Skip: {len(skipped)}, Delete: {len(all_deletions)}")
    print(f"\n📊 Processing summary:")
    print(f"📝 Files to process: {len(files_to_process)}")
    print(f"⏭️ Files to skip: {len(skipped)}")
    print(f"🕹️ Files to delete: {len(all_deletions)}")
    
    # Batch deletion with performance optimization
    if all_deletions:
        logger.info(f"Starting batch deletion for {len(all_deletions)} files")
        print(f"\n🕹️ Processing {len(all_deletions)} file deletions...")
        try:
            # Process deletions in batches for better performance
            deletion_batch_size = 100  # Reduced batch size to prevent overloading
            logger.debug(f"Using deletion batch size of {deletion_batch_size}")
            
            for i in range(0, len(all_deletions), deletion_batch_size):
                batch = all_deletions[i:i + deletion_batch_size]
                
                # Log any paths with special characters in this batch
                special_chars_in_batch = [p for p in batch if any(c in p for c in "'\"()[]{},;")]
                if special_chars_in_batch:
                    logger.debug(f"Batch contains {len(special_chars_in_batch)} paths with special characters")
                
                for rel_path in batch:
                    milvus.mark_for_deletion(rel_path)
                    logger.debug(f"Marked for deletion: {rel_path}")
                
                # Execute batch deletion
                if (i + deletion_batch_size) % 500 == 0 or i + deletion_batch_size >= len(all_deletions):
                    logger.info(f"Executing pending deletions batch ({min(i + deletion_batch_size, len(all_deletions))}/{len(all_deletions)})")
                    milvus.execute_pending_deletions()
                    print(f"✅ Processed {min(i + deletion_batch_size, len(all_deletions))}/{len(all_deletions)} deletions")
                    
        except Exception as e:
            logger.error(f"Error during batch deletion: {e}", exc_info=True)
            print(f"\n⚠️ Error during batch deletion: {e}")

    # ENHANCED: File processing with adaptive batching
    if not files_to_process:
        logger.info("No files need processing")
        print("\n✅ No files need processing!")
    else:
        logger.info(f"Processing {len(files_to_process)} files with enhanced batching (batch size: {optimal_batch_size})")
        print(f"\n🚀 Processing {len(files_to_process)} files with enhanced batching...")
        
        processed_count = 0
        failed_count = 0
        failed_files = []
        
        # Check for files with special characters that might need careful handling
        special_char_files = [(full, rel) for full, rel in files_to_process if any(c in rel for c in "'\"()[]{},;")]
        if special_char_files:
            logger.info(f"Found {len(special_char_files)} files with special characters that need careful processing")
        
        # Use optimal batch size from system analysis
        processing_batch_size = min(optimal_batch_size, len(files_to_process))
        logger.debug(f"Using processing batch size of {processing_batch_size}")
        
        try:
            # Process files with progress bar and adaptive batching
            with tqdm(total=len(files_to_process), desc="Enhanced Processing", unit="files") as pbar:
                
                for i in range(0, len(files_to_process), processing_batch_size):
                    batch = files_to_process[i:i + processing_batch_size]
                    batch_start_time = time.time()
                    logger.debug(f"Starting batch {i//processing_batch_size + 1} with {len(batch)} files")
                    
                    # Process files in current batch
                    batch_processed = 0
                    for full_path, rel_path in batch:
                        try:
                            # Check for special characters before processing
                            has_special_chars = any(c in rel_path for c in "'\"()[]{},;")
                            if has_special_chars:
                                logger.debug(f"Processing file with special characters: {rel_path}")
                                
                            # Use processor's optimized file processing
                            logger.debug(f"Processing file: {rel_path}")
                            success = processor.process_file(full_path)
                            if success:
                                logger.debug(f"Successfully processed: {rel_path}")
                                processed_count += 1
                                batch_processed += 1
                                performance_stats["total_chunks_processed"] += 1
                            else:
                                logger.warning(f"Failed to process file: {rel_path}")
                                failed_count += 1
                                failed_files.append(rel_path)
                        except Exception as e:
                            failed_count += 1
                            failed_files.append(rel_path)
                            logger.error(f"Error processing {rel_path}: {e}", exc_info=True)
                            print(f"\n⚠️ Error processing {rel_path}: {e}")
                            
                        pbar.update(1)
                    
                    # Batch performance analysis
                    batch_time = time.time() - batch_start_time
                    if batch_time > 0:
                        throughput = len(batch) / batch_time
                        logger.debug(f"Batch processing complete: {batch_processed}/{len(batch)} files, {throughput:.1f} files/s")
                        pbar.set_postfix({
                            'Batch': f"{batch_processed}/{len(batch)}",
                            'Speed': f"{throughput:.1f} files/s"
                        })
                    
                    # Memory management between batches
                    if i > 0 and i % (processing_batch_size * 5) == 0:
                        logger.debug("Running garbage collection and clearing model cache")
                        gc.collect()
                        if hasattr(embedding_model, 'clear_cache'):
                            embedding_model.clear_cache()
                
        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user")
            print(f"\n⚠️ Processing interrupted by user")
        except Exception as e:
            logger.error(f"Error during enhanced processing: {e}", exc_info=True)
            print(f"\n⚠️ Error during enhanced processing: {e}")

    # Final deletion execution
    try:
        if hasattr(milvus, 'pending_deletions') and milvus.pending_deletions:
            # Check for paths with special characters in final deletions
            special_chars_in_pending = [p for p in milvus.pending_deletions if any(c in p for c in "'\"()[]{},;")]
            if special_chars_in_pending:
                logger.info(f"Final deletions include {len(special_chars_in_pending)} paths with special characters")
                
            logger.info(f"Executing final batch deletions ({len(milvus.pending_deletions)} files)")
            print(f"\n🕹️ Executing final batch deletions ({len(milvus.pending_deletions)} files)...")
            milvus.execute_pending_deletions()
    except Exception as e:
        logger.error(f"Error during final deletion: {e}", exc_info=True)
        print(f"\n⚠️ Error during final deletion: {e}")
    
    elapsed_time = time.time() - profile_start
    logger.info(f"Enhanced incremental embedding completed in {elapsed_time:.1f} seconds")
    
    # Display enhanced performance summary
    logger.info(f"Performance summary: Processed {processed_count}/{len(files_to_process)} files, Failed: {failed_count}")
    print(f"\n📈 Enhanced Performance Summary:")
    print(f"⏱ Total Time: {elapsed_time:.1f} seconds")
    print(f"📂 Files Processed: {processed_count}/{len(files_to_process)}")
    print(f"❌ Files Failed: {failed_count}")
    
    if failed_count > 0:
        # Only show a sample of failed files to avoid overwhelming output
        sample_size = min(5, len(failed_files))
        logger.warning(f"Failed files (sample): {', '.join(failed_files[:sample_size])}")
        print(f"⚠️ Sample of failed files: {', '.join(failed_files[:sample_size])}")
        
        # Log all failed files at debug level for troubleshooting
        if len(failed_files) > sample_size:
            logger.debug(f"All failed files: {', '.join(failed_files)}")
            
    # Log if any files with special characters failed
    special_char_failures = [f for f in failed_files if any(c in f for c in "'\"()[]{},;")]
    if special_char_failures:
        logger.warning(f"Failed files with special characters: {len(special_char_failures)}")
        logger.debug(f"Special character failures: {', '.join(special_char_failures[:10])}")

    print(f"🎯 Optimal batch size: {optimal_batch_size}")
    print("\n📈 Processing Statistics:")
    print(f"  🔄 Files unchanged (skipped): {performance_stats['files_skipped']}")
    print(f"  ✅ Files successfully processed: {processed_count}")
    print(f"  🗑️ Files deleted: {performance_stats['files_deleted']}")
    
    if failed_count > 0:
        print(f"\n⚠️ Processing errors: {failed_count}")
        print("Failed files:")
        for f in failed_files[:5]:  # Show first 5 failed files
            print(f"  - {f}")
        if len(failed_files) > 5:
            print(f"  - ... and {len(failed_files) - 5} more")
    
    # Performance metrics
    total_time = time.time() - start_time  # Calculate total time using the consistent start_time variable
    if total_time > 0:
        files_per_second = (processed_count + performance_stats['files_skipped']) / total_time
        logger.info(f"Performance metrics: {files_per_second:.1f} files/second")
        print(f"\n⚡ Performance: {files_per_second:.1f} files/second")
        
        if processed_count > 0:
            avg_processing_time = total_time / processed_count
            logger.info(f"Average processing time: {avg_processing_time:.2f} seconds/file")
            print(f"⚡ Average processing time: {avg_processing_time:.2f} seconds/file")

    return processed_count

# Utility functions for external use
def estimate_chunk_count(file_size, chunk_size=1000, chunk_overlap=100, min_chunk_size=100):
    """Legacy function - redirects to optimized version"""
    return estimate_chunk_count_optimized(file_size, chunk_size, chunk_overlap, min_chunk_size)
