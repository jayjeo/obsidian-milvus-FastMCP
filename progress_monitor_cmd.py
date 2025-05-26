"""
CMD-optimized progress monitor for embedding process
Simple, stable display for terminal output only
Enhanced with safe method calling for SystemMonitor
"""
import threading
import time
from datetime import datetime
import psutil
import os
import re
import sys
from colorama import Fore, Style, init
import io
import torch

# Enable ANSI escape sequences in Windows
init(autoreset=True)

# Custom stdout filter to suppress specific error messages
class OutputFilter(io.TextIOBase):
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        # 필터링할 패턴들을 정규표현식으로 정의
        self.filter_patterns = [
            # 시스템 관련 메시지
            re.compile(r'액세스 권한에 의해 숨겨진 소켓에 액세스를 시도했습니다'),  # 소켓 액세스 오류
            re.compile(r'WARNING: This is a development server'),  # Flask 개발 서버 경고
            re.compile(r'\* Serving Flask app'),  # Flask 앱 서빙 메시지
            re.compile(r'\* Debug mode'),  # 디버그 모드 메시지
            re.compile(r'\* Running on'),  # 실행 중인 주소 메시지
            re.compile(r'Press CTRL\+C to quit'),  # 종료 안내 메시지
            
            # 모델 처리 관련 메시지
            re.compile(r'Direct (transformer|tokenization|model) (processing|failed|forward)'),  # 모델 처리 관련 메시지
            re.compile(r'Found transformer module'),  # 트랜스포머 모듈 찾기 메시지
            re.compile(r'Direct transformer output (shape|device)'),  # 트랜스포머 출력 정보
            re.compile(r'Input tensor (device|shape)'),  # 입력 텐서 정보
            re.compile(r'Using (direct tokenization|standard encode method)'),  # 토큰화 방법 정보
            re.compile(r'(falling|fallback) (back|to) (encode|method)'),  # 폴백 관련 메시지
            re.compile(r'Fallback output tensor device'),  # 폴백 출력 텐서 디바이스
            
            # GPU 관련 메시지
            re.compile(r'GPU memory usage (during|after)'),  # GPU 메모리 사용량 정보
            re.compile(r'Performing additional GPU operations'),  # GPU 연산 정보
            re.compile(r'MODEL DEVICE CHECK'),  # 모델 디바이스 검사 메시지
            re.compile(r'Target device'),  # 디바이스 정보
            re.compile(r'Parameter devices'),  # 파라미터 디바이스 정보
            re.compile(r'Module devices'),  # 모듈 디바이스 정보
            re.compile(r'Current GPU memory usage'),  # 현재 GPU 메모리 사용량
            re.compile(r'Free GPU memory'),  # 남은 GPU 메모리
            re.compile(r'Input tensor device'),  # 입력 텐서 디바이스
        ]
    
    def write(self, text):
        # 필터링할 패턴이 있는지 확인
        for pattern in self.filter_patterns:
            if pattern.search(text):
                return len(text)  # 필터링된 메시지는 출력하지 않음
        
        # 필터링되지 않은 메시지는 원래 stdout으로 출력
        return self.original_stdout.write(text)
    
    def flush(self):
        self.original_stdout.flush()
    
    def isatty(self):
        return self.original_stdout.isatty()
    
    def fileno(self):
        return self.original_stdout.fileno()

# 로그 필터링 활성화
sys.stdout = OutputFilter(sys.stdout)

class ProgressMonitor:
    def __init__(self, processor):
        """
        Constructor
        
        Args:
            processor (ObsidianProcessor): Embedding processor object
        """
        self.processor = processor
        self.progress_update_interval = 0.5  # Longer interval to reduce flickering
        self.resource_check_interval = 1.0  # System resource check interval (seconds)
        
        # Events for thread control
        self.stop_progress_update_event = threading.Event()
        self.stop_resource_check_event = threading.Event()
        
        # Thread objects
        self.progress_update_thread = None
        self.resource_check_thread = None
        
        # Previous processed file information
        self.last_processed_file = ""
        self.last_processed_status = ""
        
        # Screen size and layout settings
        self.frame_width = 80  # Fixed frame width
        
        # Error logging settings
        self.error_logs = []
        self.max_error_logs = 10  # Maximum number of error logs to display
        self.error_display_time = 20  # Seconds to display each error
        self.error_timestamps = {}  # Timestamp for each error message
        
        # Force creation of embedding_progress dict if needed
        if not hasattr(processor, 'embedding_progress'):
            processor.embedding_progress = {
                "total_files": 0,
                "processed_files": 0,
                "current_file": "",
                "cpu_percent": 0,
                "memory_percent": 0
            }
    
    def _clear_screen(self):
        """Clear the screen completely for Windows and Unix systems"""
        if os.name == 'nt':  # Windows
            os.system('cls')
        else:  # Unix/Linux/MacOS
            os.system('clear')
    
    def _create_bar(self, percent, width=20):
        """Create a simple progress bar"""
        filled_length = int(width * percent / 100)
        bar = '█' * filled_length + '░' * (width - filled_length)
        return bar
    
    def add_error_log(self, error_message):
        """Add an error message to the log with timestamp"""
        # 중복 메시지 확인
        if error_message in self.error_logs:
            return
            
        # 타임스탬프 추가
        current_time = datetime.now().strftime("%H:%M:%S")
        timestamped_message = f"[{current_time}] {error_message}"
        
        # 에러 로그 추가
        self.error_logs.append(timestamped_message)
        self.error_timestamps[timestamped_message] = time.time()
        
        # 최대 개수 유지
        if len(self.error_logs) > self.max_error_logs:
            oldest = self.error_logs.pop(0)
            if oldest in self.error_timestamps:
                del self.error_timestamps[oldest]
    
    # ==============================================
    # SAFE SYSTEM MONITOR METHOD CALLS - NEW
    # ==============================================
    
    def _get_system_monitor_safe(self):
        """Safely get the system monitor object"""
        try:
            # Try to get milvus_manager from processor
            milvus_manager = getattr(self.processor, 'milvus_manager', None)
            if not milvus_manager:
                return None
            
            # Try to get memory_monitor from milvus_manager
            memory_monitor = getattr(milvus_manager, 'memory_monitor', None)
            return memory_monitor
        except Exception as e:
            print(f"Warning: Could not access system monitor: {e}")
            return None
    
    def _get_memory_status_safe(self):
        """Safely get memory status information"""
        system_monitor = self._get_system_monitor_safe()
        if not system_monitor:
            # Fallback: use psutil directly
            try:
                memory_info = psutil.virtual_memory()
                return {
                    "memory_status": "normal" if memory_info.percent < 80 else "high",
                    "memory_percent": memory_info.percent,
                    "available_memory_gb": memory_info.available / (1024**3),
                    "used_memory_gb": memory_info.used / (1024**3),
                    "total_memory_gb": memory_info.total / (1024**3)
                }
            except Exception:
                return {
                    "memory_status": "normal",
                    "memory_percent": 50,
                    "available_memory_gb": 8.0,
                    "used_memory_gb": 4.0,
                    "total_memory_gb": 16.0
                }
        
        # Try get_memory_status first
        try:
            return system_monitor.get_memory_status()
        except AttributeError:
            # Fallback to get_system_status
            try:
                system_status = system_monitor.get_system_status()
                return {
                    "memory_status": system_status.get("memory_status", "normal"),
                    "memory_percent": system_status.get("memory_percent", 50),
                    "available_memory_gb": 8.0,  # Default values
                    "used_memory_gb": 4.0,
                    "total_memory_gb": 16.0
                }
            except AttributeError:
                # Ultimate fallback: use psutil
                try:
                    memory_info = psutil.virtual_memory()
                    return {
                        "memory_status": "normal" if memory_info.percent < 80 else "high",
                        "memory_percent": memory_info.percent,
                        "available_memory_gb": memory_info.available / (1024**3),
                        "used_memory_gb": memory_info.used / (1024**3),
                        "total_memory_gb": memory_info.total / (1024**3)
                    }
                except Exception:
                    return {
                        "memory_status": "normal",
                        "memory_percent": 50,
                        "available_memory_gb": 8.0,
                        "used_memory_gb": 4.0,
                        "total_memory_gb": 16.0
                    }
    
    def _get_cpu_status_safe(self):
        """Safely get CPU status information"""
        system_monitor = self._get_system_monitor_safe()
        if not system_monitor:
            # Fallback: use psutil directly
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_count = psutil.cpu_count()
                return {
                    "cpu_percent": cpu_percent,
                    "cpu_cores": cpu_count,
                    "cpu_temperature": 65
                }
            except Exception:
                return {"cpu_percent": 50, "cpu_cores": 8, "cpu_temperature": 65}
        
        # Try get_cpu_status first
        try:
            return system_monitor.get_cpu_status()
        except AttributeError:
            # Fallback to get_system_status
            try:
                system_status = system_monitor.get_system_status()
                return {
                    "cpu_percent": system_status.get("cpu_percent", 50),
                    "cpu_cores": 8,  # Default value
                    "cpu_temperature": 65  # Default value
                }
            except AttributeError:
                # Ultimate fallback: use psutil
                try:
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    cpu_count = psutil.cpu_count()
                    return {
                        "cpu_percent": cpu_percent,
                        "cpu_cores": cpu_count,
                        "cpu_temperature": 65
                    }
                except Exception:
                    return {"cpu_percent": 50, "cpu_cores": 8, "cpu_temperature": 65}
    
    def _get_gpu_status_safe(self):
        """Safely get GPU status information"""
        system_monitor = self._get_system_monitor_safe()
        if not system_monitor:
            # Fallback: check GPU directly
            try:
                gpu_available = torch.cuda.is_available()
                if gpu_available:
                    used_memory = torch.cuda.memory_allocated(0)
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    gpu_percent = (used_memory / total_memory) * 100 if total_memory > 0 else 0
                    return {
                        "gpu_available": gpu_available,
                        "gpu_percent": gpu_percent,
                        "gpu_memory_used": used_memory,
                        "gpu_memory_total": total_memory
                    }
                else:
                    return {
                        "gpu_available": False,
                        "gpu_percent": 0,
                        "gpu_memory_used": 0,
                        "gpu_memory_total": 1  # Prevent division by zero
                    }
            except Exception:
                return {
                    "gpu_available": False,
                    "gpu_percent": 0,
                    "gpu_memory_used": 0,
                    "gpu_memory_total": 1
                }
        
        # Try get_gpu_status first
        try:
            return system_monitor.get_gpu_status()
        except AttributeError:
            # Fallback to get_system_status
            try:
                system_status = system_monitor.get_system_status()
                return {
                    "gpu_available": system_status.get("gpu_available", False),
                    "gpu_percent": system_status.get("gpu_percent", 0),
                    "gpu_memory_used": 0,  # Default values
                    "gpu_memory_total": 1   # Prevent division by zero
                }
            except AttributeError:
                # Ultimate fallback: check GPU directly
                try:
                    gpu_available = torch.cuda.is_available()
                    if gpu_available:
                        used_memory = torch.cuda.memory_allocated(0)
                        total_memory = torch.cuda.get_device_properties(0).total_memory
                        gpu_percent = (used_memory / total_memory) * 100 if total_memory > 0 else 0
                        return {
                            "gpu_available": gpu_available,
                            "gpu_percent": gpu_percent,
                            "gpu_memory_used": used_memory,
                            "gpu_memory_total": total_memory
                        }
                    else:
                        return {
                            "gpu_available": False,
                            "gpu_percent": 0,
                            "gpu_memory_used": 0,
                            "gpu_memory_total": 1
                        }
                except Exception:
                    return {
                        "gpu_available": False,
                        "gpu_percent": 0,
                        "gpu_memory_used": 0,
                        "gpu_memory_total": 1
                    }
    
    # ==============================================
    # END OF SAFE SYSTEM MONITOR METHOD CALLS
    # ==============================================
    
    def _display_progress(self):
        """Display progress information in a simple, stable format"""
        # Get progress data from processor
        progress_data = self.processor.embedding_progress
        
        # Extract progress information
        total_files = progress_data.get("total_files", 0)
        processed_files = progress_data.get("processed_files", 0)
        current_file = progress_data.get("current_file", "")
        cpu_percent = progress_data.get("cpu_percent", 0)
        memory_percent = progress_data.get("memory_percent", 0)
        gpu_percent = progress_data.get("gpu_percent", 0)
        start_time = progress_data.get("start_time", 0)
        total_size = progress_data.get("total_size", 0)
        processed_size = progress_data.get("processed_size", 0)
        
        # Calculate progress percentage
        percentage = 0
        # None 값 체크 추가
        if total_files is None:
            total_files = 0
        if processed_files is None:
            processed_files = 0
            
        if total_files > 0:
            percentage = min(99, int((processed_files / total_files) * 100))
        
        # Calculate estimated time remaining
        estimated_time_remaining = None
        # None 값 체크 추가
        if start_time is None:
            start_time = 0
        if processed_files is None:
            processed_files = 0
        if total_files is None:
            total_files = 0
            
        if start_time > 0 and processed_files > 0 and total_files > processed_files:
            elapsed_time = time.time() - start_time
            files_per_second = processed_files / elapsed_time if elapsed_time > 0 else 0
            if files_per_second > 0:
                remaining_files = total_files - processed_files
                remaining_seconds = remaining_files / files_per_second
                
                # Format time remaining
                if remaining_seconds < 60:
                    estimated_time_remaining = f"{int(remaining_seconds)} seconds"
                elif remaining_seconds < 3600:
                    estimated_time_remaining = f"{int(remaining_seconds / 60)} minutes"
                else:
                    hours = int(remaining_seconds / 3600)
                    minutes = int((remaining_seconds % 3600) / 60)
                    estimated_time_remaining = f"{hours} hours {minutes} minutes"
        
        # Prepare output lines
        output_lines = []
        
        # Add header
        output_lines.append(f"{Style.BRIGHT}{Fore.CYAN}Embedding Progress{Style.RESET_ALL}")
        output_lines.append("-" * 80)
        
        # Display progress bar
        output_lines.append(f"{Style.BRIGHT}Progress:{Style.RESET_ALL}     [{self._create_bar(percentage)}] {percentage}% ({processed_files}/{total_files} files)")
        
        # Display size progress if available
        if total_size > 0:
            size_percentage = min(99, int((processed_size / total_size) * 100))
            size_bar = self._create_bar(size_percentage)
            processed_mb = processed_size / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            output_lines.append(f"{Style.BRIGHT}Size:{Style.RESET_ALL}         [{size_bar}] {size_percentage}% ({processed_mb:.1f} MB/{total_mb:.1f} MB)")
        
        # Display estimated time remaining (using the value calculated earlier)
        # Always show the line, even if the calculation isn't available yet
        if not estimated_time_remaining:
            estimated_time_remaining = "Calculating..."
        output_lines.append(f"{Style.BRIGHT}Est. Time Remaining:{Style.RESET_ALL}    {Fore.CYAN}{estimated_time_remaining}{Fore.RESET}")
        
        # Display system resource usage
        cpu_color = Fore.GREEN
        if cpu_percent > 70:
            cpu_color = Fore.YELLOW
        if cpu_percent > 90:
            cpu_color = Fore.RED
        output_lines.append(f"{Style.BRIGHT}CPU Usage:{Style.RESET_ALL}    [{cpu_color}{self._create_bar(cpu_percent)}{Fore.RESET}] {cpu_color}{cpu_percent:.1f}%{Fore.RESET}")
        
        memory_color = Fore.GREEN
        if memory_percent > 70:
            memory_color = Fore.YELLOW
        if memory_percent > 90:
            memory_color = Fore.RED
        output_lines.append(f"{Style.BRIGHT}Memory Usage:{Style.RESET_ALL} [{memory_color}{self._create_bar(memory_percent)}{Fore.RESET}] {memory_color}{memory_percent:.1f}%{Fore.RESET}")
        
        # GPU usage if available
        try:
            # GPU 메모리 사용량 (MB)
            gpu_percent = progress_data.get("gpu_percent", 0)
            used_memory = progress_data.get("gpu_memory_used", 0)
            total_memory = progress_data.get("gpu_memory_total", 1)  # 0으로 나누기 방지
            
            # GPU 사용 가능 여부 확인
            if torch.cuda.is_available():
                # 색상 설정
                gpu_color = Fore.GREEN
                if gpu_percent > 70:
                    gpu_color = Fore.YELLOW
                if gpu_percent > 90:
                    gpu_color = Fore.RED
                
                # GPU 사용률 표시
                output_lines.append(f"{Style.BRIGHT}GPU Usage:{Style.RESET_ALL}    [{gpu_color}{self._create_bar(gpu_percent)}{Fore.RESET}] {gpu_color}{gpu_percent:.1f}%{Fore.RESET}")
                
                # GPU 메모리 사용량을 그래픽 바로 표시 (백분율)
                output_lines.append(f"{Style.BRIGHT}GPU Memory:{Style.RESET_ALL}   [{gpu_color}{self._create_bar(gpu_percent)}{Fore.RESET}] {gpu_color}{gpu_percent:.1f}%{Fore.RESET}")
            else:
                output_lines.append(f"{Style.BRIGHT}GPU Status:{Style.RESET_ALL}   [No CUDA compatible GPU detected]")
        except Exception as e:
            output_lines.append(f"{Style.BRIGHT}GPU Status:{Style.RESET_ALL}   [Error getting GPU info: {str(e)[:50]}]")
            
        # GPU 사용 설정 확인
        use_gpu = hasattr(self.processor, 'use_gpu') and self.processor.use_gpu
        if not use_gpu and torch.cuda.is_available():
            output_lines.append(f"{Style.BRIGHT}GPU Config:{Style.RESET_ALL}    [GPU available but disabled in settings]")
        elif not torch.cuda.is_available():
            output_lines.append(f"{Style.BRIGHT}GPU Config:{Style.RESET_ALL}    [No CUDA compatible GPU available]")
        else:
            output_lines.append(f"{Style.BRIGHT}GPU Config:{Style.RESET_ALL}    [GPU enabled and active]")

        
        # 현재 파일과 마지맵 처리 파일이 같고 스킵 상태인 경우 현재 파일 표시하지 않음
        display_current_file = current_file
        if self.last_processed_file and current_file == self.last_processed_file:
            if self.last_processed_status and ("Already exists, Skipping" in self.last_processed_status):
                display_current_file = ""  # 스킵된 파일은 표시하지 않음
        
        output_lines.append(f"\n{Style.BRIGHT}Current File:{Style.RESET_ALL} {display_current_file}")
        
        # Previous file - 항상 표시
        if self.last_processed_file:
            output_lines.append(f"{Style.BRIGHT}Last File:{Style.RESET_ALL}   {self.last_processed_file} - {self.last_processed_status}")
        
        output_lines.append("-" * 80)
        
        # Display error logs if any
        current_time = time.time()
        active_errors = []
        
        for error in self.error_logs:
            error_time = self.error_timestamps.get(error, 0)
            if current_time - error_time < self.error_display_time:
                active_errors.append(error)
        
        if active_errors:
            output_lines.append(f"{Style.BRIGHT}{Fore.RED}Error Logs:{Style.RESET_ALL}")
            for error in active_errors[-5:]:  # 최근 5개만 표시
                output_lines.append(f"{Fore.RED}{error}{Fore.RESET}")
            output_lines.append("-" * 80)
        
        # Clear screen and display all lines
        self._clear_screen()
        print("\n".join(output_lines))
    
    def _progress_update_thread_func(self):
        """Thread function for updating the progress display"""
        while not self.stop_progress_update_event.is_set():
            try:
                # 현재 파일 정보 가져오기
                last_current_file = self.processor.embedding_progress.get("current_file", "")
                
                # 파일 처리 상태 업데이트
                if last_current_file:
                    # 이전에 처리된 파일과 다른 경우 (새 파일 처리 시작)
                    if last_current_file != self.last_processed_file:
                        # 전체 재색인 모드인지 확인
                        is_full_reindex = False
                        if hasattr(self.processor, 'embedding_progress') and isinstance(self.processor.embedding_progress, dict):
                            is_full_reindex = self.processor.embedding_progress.get('is_full_reindex', False)
                        
                        # 전체 재색인 모드가 아닐 때만 "Already exists, Skipping" 메시지 표시
                        if not is_full_reindex:
                            self.last_processed_status = f"{Fore.GREEN}Already exists, Skipping{Fore.RESET}"
                        else:
                            self.last_processed_status = f"{Fore.GREEN}Processing...{Fore.RESET}"
                        
                        # 현재 파일을 마지막 처리 파일로 업데이트
                        self.last_processed_file = last_current_file
                    
                    # 파일 처리가 끝났을 때도 마지막 파일 정보 유지
                    processed_files = self.processor.embedding_progress.get("processed_files", 0)
                    total_files = self.processor.embedding_progress.get("total_files", 0)
                    
                    # None 값 체크 추가
                    if processed_files is None:
                        processed_files = 0
                    if total_files is None:
                        total_files = 0
                    
                    # 모든 파일이 처리되었고 현재 파일이 없으면 마지막 파일을 완료로 표시
                    if processed_files == total_files and not last_current_file and self.last_processed_file:
                        # 전체 재색인 모드인지 확인
                        is_full_reindex = False
                        if hasattr(self.processor, 'embedding_progress') and isinstance(self.processor.embedding_progress, dict):
                            is_full_reindex = self.processor.embedding_progress.get('is_full_reindex', False)
                        
                        # 전체 재색인 모드가 아닐 때만 "Already exists, Skipping" 메시지 표시
                        if not is_full_reindex:
                            self.last_processed_status = f"{Fore.GREEN}Already exists, Skipping{Fore.RESET}"
                        else:
                            self.last_processed_status = f"{Fore.GREEN}Processed{Fore.RESET}"
                    
                    # 이제 화면 업데이트 - 현재 파일 정보가 업데이트된 후에 화면 표시
                    self._display_progress()
            except Exception as e:
                # On error, just wait and try again
                print(f"Error in progress update thread: {e}")
            
            # Wait for next update
            time.sleep(self.progress_update_interval)
    
    def _resource_check_thread_func(self):
        """Thread function for resource monitoring"""
        while not self.stop_resource_check_event.is_set():
            try:
                self._update_system_resources()
            except Exception as e:
                print(f"Error in resource check thread: {e}")
            
            # Wait for next check
            time.sleep(self.resource_check_interval)
    
    def _update_system_resources(self):
        """Update system resource usage information - ENHANCED WITH SAFE CALLS"""
        try:
            # ENHANCED: Use safe methods to get system information
            # This prevents AttributeError if SystemMonitor methods are missing
            
            # Get CPU status safely
            cpu_status = self._get_cpu_status_safe()
            cpu_percent = cpu_status.get("cpu_percent", 50)
            
            # Get memory status safely
            memory_status = self._get_memory_status_safe()
            memory_percent = memory_status.get("memory_percent", 50)
            
            # Get GPU status safely
            gpu_status = self._get_gpu_status_safe()
            gpu_percent = gpu_status.get("gpu_percent", 0)
            gpu_memory_used = gpu_status.get("gpu_memory_used", 0)
            gpu_memory_total = gpu_status.get("gpu_memory_total", 1)
            
            # Update processor's embedding_progress with safe values
            self.processor.embedding_progress["cpu_percent"] = cpu_percent
            self.processor.embedding_progress["memory_percent"] = memory_percent
            self.processor.embedding_progress["gpu_percent"] = gpu_percent
            self.processor.embedding_progress["gpu_memory_used"] = gpu_memory_used
            self.processor.embedding_progress["gpu_memory_total"] = gpu_memory_total
            
        except Exception as e:
            # Ultimate fallback: set default values
            print(f"Warning: Error updating system resources: {e}")
            self.processor.embedding_progress["cpu_percent"] = 50
            self.processor.embedding_progress["memory_percent"] = 50
            self.processor.embedding_progress["gpu_percent"] = 0
            self.processor.embedding_progress["gpu_memory_used"] = 0
            self.processor.embedding_progress["gpu_memory_total"] = 1
    
    def start_monitoring(self):
        """Start resource monitoring and progress display"""
        # Stop any existing threads
        self.stop_monitoring()
        
        # Reset stop events
        self.stop_progress_update_event.clear()
        self.stop_resource_check_event.clear()
        
        # Start progress update thread
        self.progress_update_thread = threading.Thread(target=self._progress_update_thread_func)
        self.progress_update_thread.daemon = True
        self.progress_update_thread.start()
        
        # Start resource check thread
        self.resource_check_thread = threading.Thread(target=self._resource_check_thread_func)
        self.resource_check_thread.daemon = True
        self.resource_check_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring and progress display"""
        # Set stop events
        self.stop_progress_update_event.set()
        self.stop_resource_check_event.set()
        
        # Wait for threads to terminate
        if self.progress_update_thread and self.progress_update_thread.is_alive():
            self.progress_update_thread.join(timeout=1.0)
        
        if self.resource_check_thread and self.resource_check_thread.is_alive():
            self.resource_check_thread.join(timeout=1.0)
        
        # Reset thread objects
        self.progress_update_thread = None
        self.resource_check_thread = None
    
    # Start monitoring (maintain compatibility with previous version)
    def start(self):
        self.start_monitoring()
    
    # Stop monitoring (maintain compatibility with previous version)
    def stop(self):
        self.stop_monitoring()