import os
import re
import PyPDF2
import markdown
import json
import yaml  # Added PyYAML for better frontmatter parsing
import psutil
import colorama
from colorama import Fore, Style
import time
import threading
import gc
import sys
from datetime import datetime
from bs4 import BeautifulSoup
import config
from embeddings import EmbeddingModel
from tqdm import tqdm
from functools import lru_cache
from progress_monitor_cmd import ProgressMonitor

# Windows color support initialization
colorama.init()

class ObsidianProcessor:
    def __init__(self, milvus_manager):
        self.vault_path = config.OBSIDIAN_VAULT_PATH
        self.milvus_manager = milvus_manager
        self.embedding_model = EmbeddingModel()
        self.next_id = self._get_next_id()
        
        # GPU configuration
        self.use_gpu = config.USE_GPU
        self.device_idx = config.GPU_DEVICE_ID if hasattr(config, 'GPU_DEVICE_ID') else 0
        
        # Embedding progress tracking variables
        self.embedding_in_progress = False
        self.embedding_progress = {
            "total_files": 0,
            "processed_files": 0,
            "total_size": 0,
            "processed_size": 0,
            "start_time": None,
            "current_file": "",
            "estimated_time_remaining": "",
            "percentage": 0,
            "is_full_reindex": False,
            "cpu_percent": 0,
            "memory_percent": 0,
            "gpu_percent": 0,
            "gpu_memory_used": 0,
            "gpu_memory_total": 0,
            "current_batch_size": 0
        }
        
        # System resource usage limits
        self.max_cpu_percent = 85
        self.max_memory_percent = 80
        self.resource_check_interval = 2
        self.last_resource_check = 0
        self.dynamic_batch_size = config.BATCH_SIZE * 2
        self.min_batch_size = max(1, config.BATCH_SIZE // 2)
        self.max_batch_size = config.BATCH_SIZE * 4
        
        # Progress and resource monitoring manager
        self.monitor = ProgressMonitor(self)
        
        # Global processing timeout
        self.processing_timeout = 600  # 10 minutes (seconds)
        
        # OPTIMIZATION: Session cache for verification results
        self.verification_cache = {}
        
        # OPTIMIZATION: Performance thresholds for smart decision making
        self.FAST_SKIP_THRESHOLD = 0.1  # Files with time diff < 0.1s are very likely unchanged
        self.FAST_PROCESS_THRESHOLD = 2.0  # Files with time diff > 2.0s are definitely changed
        
    def _get_next_id(self):
        """Get next ID value"""
        results = self.milvus_manager.query("id >= 0", output_fields=["id"], limit=1)
        if not results:
            return 1
        return max([r['id'] for r in results]) + 1
        
    def _create_ascii_bar(self, percent, width=20):
        """Create ASCII graph bar from percentage value"""
        # Validate value range
        if not isinstance(percent, (int, float)) or percent < 0:
            percent = 0
        elif percent > 100:
            percent = 100
            
        # Calculate filled length (round to show at least 1 character if > 0)
        filled_length = max(1, int(width * percent / 100)) if percent > 0 else 0
        
        # Use different characters based on usage level
        if percent > 90:
            bar_char = '#'  # Very high
        elif percent > 70:
            bar_char = '='  # High
        elif percent > 50:
            bar_char = '-'  # Medium
        elif percent > 30:
            bar_char = '.'  # Low
        elif percent > 0:
            bar_char = '·'  # Very low
        else:
            bar_char = ' '  # 0%
        
        # Generate graph bar (limit max length)
        filled_length = min(filled_length, width)
        bar = bar_char * filled_length + ' ' * (width - filled_length)
        return bar
        
    def _check_system_resources(self):
        """Check system resource usage and adjust batch size"""
        current_time = time.time()
        
        if current_time - self.last_resource_check < self.resource_check_interval:
            return self.dynamic_batch_size
            
        self.last_resource_check = current_time
        
        # Call ProgressMonitor's _update_system_resources method
        self.monitor._update_system_resources()
        
        return self.dynamic_batch_size
        
    def _update_progress_stats(self):
        """Method to calculate embedding progress and estimated remaining time (file size based)"""
        if not self.embedding_in_progress:
            return
            
        # Calculate progress (file size based)
        total_size = self.embedding_progress["total_size"]
        processed_size = self.embedding_progress["processed_size"]
        
        # Also maintain file count for display purposes
        total_files = self.embedding_progress["total_files"]
        processed_files = self.embedding_progress["processed_files"]
        
        if total_size <= 0:
            self.embedding_progress["percentage"] = 0
            self.embedding_progress["estimated_time_remaining"] = "Calculating..."
            return
            
        # Calculate progress based on file size
        if total_size > 0:
            percentage = min(99, int((processed_size / total_size) * 100))  # Only show 100% when completely finished
            self.embedding_progress["percentage"] = percentage
        
        # Calculate estimated remaining time (file size based)
        if processed_size > 0 and self.embedding_progress["start_time"] is not None:
            elapsed_time = time.time() - self.embedding_progress["start_time"]
            bytes_per_second = processed_size / elapsed_time if elapsed_time > 0 else 0
            
            if bytes_per_second > 0:
                remaining_size = total_size - processed_size
                remaining_seconds = remaining_size / bytes_per_second
                
                # Format estimated time
                if remaining_seconds < 60:
                    time_str = f"{int(remaining_seconds)} seconds"
                elif remaining_seconds < 3600:
                    minutes = int(remaining_seconds / 60)
                    seconds = int(remaining_seconds % 60)
                    time_str = f"{minutes} minutes {seconds} seconds"
                else:
                    hours = int(remaining_seconds / 3600)
                    minutes = int((remaining_seconds % 3600) / 60)
                    time_str = f"{hours} hours {minutes} minutes"
                    
                self.embedding_progress["estimated_time_remaining"] = time_str
            else:
                self.embedding_progress["estimated_time_remaining"] = "Calculating..."
        else:
            self.embedding_progress["estimated_time_remaining"] = "Calculating..."
        
        # Update system resource usage
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            
            self.embedding_progress["cpu_percent"] = cpu_percent
            self.embedding_progress["memory_percent"] = memory_percent
        except Exception as e:
            # Ignore exceptions
            pass
        
    def start_monitoring(self):
        """Start all monitoring"""
        self.monitor.start()
        
    def stop_monitoring(self):
        """Stop all monitoring"""
        self.monitor.stop()
    
    def _normalize_timestamp(self, timestamp_value):
        """Normalize timestamp to float for reliable comparison"""
        try:
            if timestamp_value is None:
                return 0.0
            
            if isinstance(timestamp_value, (int, float)):
                return float(timestamp_value)
            
            if isinstance(timestamp_value, str):
                # Handle empty strings
                if not timestamp_value.strip():
                    return 0.0
                
                # Try direct float conversion
                try:
                    return float(timestamp_value)
                except ValueError:
                    # Try parsing as datetime string if needed
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                        return dt.timestamp()
                    except:
                        print(f"{Fore.YELLOW}[WARNING] Could not parse timestamp: {timestamp_value}{Style.RESET_ALL}")
                        return 0.0
            
            print(f"{Fore.YELLOW}[WARNING] Unknown timestamp type: {type(timestamp_value)}{Style.RESET_ALL}")
            return 0.0
            
        except Exception as e:
            print(f"{Fore.YELLOW}[WARNING] Error normalizing timestamp {timestamp_value}: {e}{Style.RESET_ALL}")
            return 0.0

    def _check_database_health(self):
        """OPTIMIZATION: Quick database health check to decide verification strategy"""
        try:
            # Sample 100 random records to check health
            sample_results = self.milvus_manager.query(
                expr="id >= 0",
                output_fields=["id", "vector", "chunk_text"],
                limit=100
            )
            
            if not sample_results:
                return "empty"
            
            # Check corruption rate in sample
            corrupted_count = 0
            for record in sample_results:
                vector = record.get("vector")
                chunk_text = record.get("chunk_text", "")
                
                if not vector or len(vector) == 0 or not chunk_text or len(chunk_text.strip()) < 5:
                    corrupted_count += 1
            
            corruption_rate = corrupted_count / len(sample_results)
            
            if corruption_rate > 0.1:  # More than 10% corrupted
                return "unhealthy"
            elif corruption_rate > 0.05:  # 5-10% corrupted
                return "warning"
            else:
                return "healthy"
                
        except Exception as e:
            print(f"{Fore.YELLOW}[WARNING] Health check failed: {e}{Style.RESET_ALL}")
            return "unknown"

    def _fast_decision_engine(self, file_path, file_mtime, existing_mtime, file_size):
        """OPTIMIZATION: 3-Tier fast decision making for file processing"""
        rel_path = os.path.relpath(file_path, self.vault_path)
        
        # Cache key for this file
        cache_key = f"{rel_path}:{file_mtime}:{file_size}"
        
        # Check cache first
        if cache_key in self.verification_cache:
            cached_result = self.verification_cache[cache_key]
            print(f"{Fore.BLUE}[CACHED] {rel_path}: {cached_result['decision']} ({cached_result['reason']}){Style.RESET_ALL}")
            return cached_result['decision'], cached_result['reason']
        
        # Calculate time difference
        time_diff = abs(file_mtime - existing_mtime) if existing_mtime > 0 else float('inf')
        
        decision = None
        reason = ""
        
        # TIER 1: Lightning Fast Decisions (90%+ of files)
        if time_diff > self.FAST_PROCESS_THRESHOLD:
            # File definitely modified - immediate process
            decision = "PROCESS"
            reason = f"definitely modified (time_diff: {time_diff:.2f}s)"
            print(f"{Fore.GREEN}[FAST-PROCESS] {rel_path}: {reason}{Style.RESET_ALL}")
            
        elif time_diff < self.FAST_SKIP_THRESHOLD:
            # File very likely unchanged - immediate skip (low risk)
            decision = "SKIP"
            reason = f"very likely unchanged (time_diff: {time_diff:.2f}s)"
            print(f"{Fore.CYAN}[FAST-SKIP] {rel_path}: {reason}{Style.RESET_ALL}")
            
        else:
            # TIER 2: Smart Batch Check for ambiguous cases
            decision = "VERIFY"
            reason = f"ambiguous timestamp (time_diff: {time_diff:.2f}s) - needs verification"
            print(f"{Fore.YELLOW}[NEED-VERIFY] {rel_path}: {reason}{Style.RESET_ALL}")
        
        # Cache the result
        result = {"decision": decision, "reason": reason}
        self.verification_cache[cache_key] = result
        
        return decision, reason

    def _batch_existence_check(self, suspect_files):
        """OPTIMIZATION: Batch check multiple files at once"""
        if not suspect_files:
            return {}
            
        try:
            # Build batch query for multiple files
            paths = [os.path.relpath(fp, self.vault_path) for fp, _ in suspect_files]
            path_conditions = " or ".join([f"path == '{path}'" for path in paths])
            
            # Single query to check all suspect files
            results = self.milvus_manager.query(
                expr=f"({path_conditions})",
                output_fields=["path", "id"],
                limit=len(paths) * 10  # Assume max 10 chunks per file
            )
            
            # Count chunks per file
            file_chunk_counts = {}
            for result in results:
                path = result.get("path")
                if path:
                    file_chunk_counts[path] = file_chunk_counts.get(path, 0) + 1
            
            print(f"{Fore.CYAN}[BATCH-CHECK] Verified {len(suspect_files)} files in single query{Style.RESET_ALL}")
            
            return file_chunk_counts
            
        except Exception as e:
            print(f"{Fore.YELLOW}[WARNING] Batch check failed: {e}{Style.RESET_ALL}")
            return {}

    def _smart_incremental_logic(self, files_info, existing_files_info):
        """OPTIMIZATION: Smart incremental processing with 3-tier verification"""
        files_to_process = []
        skipped_count = 0
        fast_decisions = 0
        batch_verifications = 0
        deep_verifications = 0
        
        # Step 1: Database health check
        db_health = self._check_database_health()
        print(f"{Fore.CYAN}[DB-HEALTH] Database status: {db_health}{Style.RESET_ALL}")
        
        if db_health == "unhealthy":
            print(f"{Fore.RED}[WARNING] Database appears corrupted (>10% corruption). Consider full reindex.{Style.RESET_ALL}")
            # Still proceed but with more thorough verification
        
        # Step 2: Fast decision phase
        suspect_files = []  # Files that need batch verification
        deep_verify_files = []  # Files that need deep verification
        
        for file_path, file_size in files_info:
            rel_path = os.path.relpath(file_path, self.vault_path)
            file_mtime = os.path.getmtime(file_path)
            existing_mtime = existing_files_info.get(rel_path, 0)
            
            # Normalize timestamp
            existing_mtime = self._normalize_timestamp(existing_mtime)
            
            # Fast decision engine
            decision, reason = self._fast_decision_engine(file_path, file_mtime, existing_mtime, file_size)
            
            if decision == "PROCESS":
                files_to_process.append((file_path, file_size))
                fast_decisions += 1
            elif decision == "SKIP":
                skipped_count += 1
                fast_decisions += 1
            elif decision == "VERIFY":
                suspect_files.append((file_path, file_size))
        
        # Step 3: Batch verification for suspect files
        if suspect_files:
            print(f"{Fore.CYAN}[BATCH-VERIFY] Checking {len(suspect_files)} suspect files in batch...{Style.RESET_ALL}")
            file_chunk_counts = self._batch_existence_check(suspect_files)
            
            for file_path, file_size in suspect_files:
                rel_path = os.path.relpath(file_path, self.vault_path)
                chunk_count = file_chunk_counts.get(rel_path, 0)
                
                if chunk_count == 0:
                    # No chunks found - definitely process
                    files_to_process.append((file_path, file_size))
                    print(f"{Fore.YELLOW}[BATCH-RESULT] {rel_path}: No chunks found - PROCESS{Style.RESET_ALL}")
                elif chunk_count < 3 and db_health in ["unhealthy", "warning"]:
                    # Few chunks and poor DB health - need deep verification
                    deep_verify_files.append((file_path, file_size))
                    print(f"{Fore.ORANGE}[BATCH-RESULT] {rel_path}: Few chunks + poor DB health - DEEP-VERIFY{Style.RESET_ALL}")
                else:
                    # Looks good - skip
                    skipped_count += 1
                    print(f"{Fore.GREEN}[BATCH-RESULT] {rel_path}: {chunk_count} chunks found - SKIP{Style.RESET_ALL}")
                
                batch_verifications += 1
        
        # Step 4: Deep verification only for truly suspect files
        if deep_verify_files and len(deep_verify_files) < 50:  # Limit deep verification
            print(f"{Fore.ORANGE}[DEEP-VERIFY] Performing detailed verification on {len(deep_verify_files)} files...{Style.RESET_ALL}")
            
            for file_path, file_size in deep_verify_files:
                rel_path = os.path.relpath(file_path, self.vault_path)
                
                # Perform detailed integrity check (original logic)
                has_valid_embedding = self._verify_embedding_data_integrity(file_path)
                
                if has_valid_embedding:
                    skipped_count += 1
                    print(f"{Fore.GREEN}[DEEP-RESULT] {rel_path}: Valid embedding - SKIP{Style.RESET_ALL}")
                else:
                    files_to_process.append((file_path, file_size))
                    print(f"{Fore.RED}[DEEP-RESULT] {rel_path}: Invalid embedding - PROCESS{Style.RESET_ALL}")
                
                deep_verifications += 1
        elif len(deep_verify_files) >= 50:
            # Too many files need deep verification - just process all of them
            print(f"{Fore.YELLOW}[OPTIMIZATION] Too many files ({len(deep_verify_files)}) need deep verification - processing all{Style.RESET_ALL}")
            files_to_process.extend(deep_verify_files)
        
        # Step 5: Performance summary
        total_files = len(files_info)
        print(f"\\n{Fore.CYAN}[PERFORMANCE SUMMARY]{Style.RESET_ALL}")
        print(f"Fast decisions: {fast_decisions}/{total_files} ({fast_decisions/total_files*100:.1f}%)")
        print(f"Batch verifications: {batch_verifications}/{total_files} ({batch_verifications/total_files*100:.1f}%)")
        print(f"Deep verifications: {deep_verifications}/{total_files} ({deep_verifications/total_files*100:.1f}%)")
        print(f"Files to process: {len(files_to_process)}")
        print(f"Files skipped: {skipped_count}")
        
        return files_to_process, skipped_count

    def _verify_embedding_data_integrity(self, file_path):
        """Deep verification - only used for truly suspect files"""
        try:
            rel_path = os.path.relpath(file_path, self.vault_path)
            
            # Check actual vector data and verify integrity
            results = self.milvus_manager.query(
                expr=f"path == '{rel_path}'",
                output_fields=["id", "vector", "chunk_text", "chunk_index"],
                limit=3  # Check only first 3 chunks for speed
            )
            
            if not results:
                return False
            
            # Quick validation: check first record only for speed
            record = results[0]
            vector = record.get("vector")
            chunk_text = record.get("chunk_text", "")
            
            # Simple checks
            if not vector or len(vector) == 0:
                return False
            if not chunk_text or len(chunk_text.strip()) < 10:
                return False
            if len(vector) != config.VECTOR_DIM:
                return False
            
            return True
            
        except Exception as e:
            print(f"{Fore.RED}[VERIFICATION ERROR] {file_path}: {e}{Style.RESET_ALL}")
            return False

    # [Rest of the methods remain the same...]
    # process_file, _extract_chunks_from_file, etc.
