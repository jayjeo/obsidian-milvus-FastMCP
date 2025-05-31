"""
CMD-optimized progress monitor for embedding process
Simple, stable display for terminal output only
Enhanced with safe method calling for SystemMonitor
FIXED: GPU usage and memory tracking with maximum values over time periods
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
# Force color output regardless of environment
os.environ['FORCE_COLOR'] = '1'
os.environ.pop('NO_COLOR', None)  # Remove NO_COLOR if exists

# Windows specific color enabling
if os.name == 'nt':
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

init(autoreset=True, convert=True, strip=False, wrap=True)

# Test if colors are working (you can remove this after testing)
if __name__ != "__main__":  # Only when imported as module
    test_colors = False  # Set to True to test colors on import
    if test_colors:
        print(f"{Fore.RED}RED{Fore.RESET} {Fore.GREEN}GREEN{Fore.RESET} {Fore.BLUE}BLUE{Fore.RESET} {Fore.YELLOW}YELLOW{Fore.RESET} {Fore.MAGENTA}MAGENTA{Fore.RESET} {Fore.CYAN}CYAN{Fore.RESET}")
        print(f"If you see colors above, colorama is working!")

# Try to import pynvml for accurate GPU monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available, using torch for GPU monitoring")

# Custom stdout filter to suppress specific error messages
class OutputFilter(io.TextIOBase):
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        # Never filter ANSI color codes
        self.ansi_pattern = re.compile(r'\x1b\[[0-9;]*m')
        # Filter patterns with regex
        self.filter_patterns = [
            # System related messages
            re.compile(r'액세스 권한에 의해 숨겨진 소켓에 액세스를 시도했습니다'),  # Socket access error
            re.compile(r'WARNING: This is a development server'),  # Flask dev server warning
            re.compile(r'\* Serving Flask app'),  # Flask app serving message
            re.compile(r'\* Debug mode'),  # Debug mode message
            re.compile(r'\* Running on'),  # Running address message
            re.compile(r'Press CTRL\+C to quit'),  # Quit instruction message
            
            # Model processing related messages
            re.compile(r'Direct (transformer|tokenization|model) (processing|failed|forward)'),  # Model processing messages
            re.compile(r'Found transformer module'),  # Transformer module found message
            re.compile(r'Direct transformer output (shape|device)'),  # Transformer output info
            re.compile(r'Input tensor (device|shape)'),  # Input tensor info
            re.compile(r'Using (direct tokenization|standard encode method)'),  # Tokenization method info
            re.compile(r'(falling|fallback) (back|to) (encode|method)'),  # Fallback related messages
            re.compile(r'Fallback output tensor device'),  # Fallback output tensor device
            
            # GPU related messages
            re.compile(r'GPU memory usage (during|after)'),  # GPU memory usage info
            re.compile(r'Performing additional GPU operations'),  # GPU operations info
            re.compile(r'MODEL DEVICE CHECK'),  # Model device check message
            re.compile(r'Target device'),  # Device info
            re.compile(r'Parameter devices'),  # Parameter device info
            re.compile(r'Module devices'),  # Module device info
            re.compile(r'Current GPU memory usage'),  # Current GPU memory usage
            re.compile(r'Free GPU memory'),  # Free GPU memory
            re.compile(r'Input tensor device'),  # Input tensor device
        ]
    
    def write(self, text):
        # Never filter text containing ANSI color codes
        if self.ansi_pattern.search(text):
            return self.original_stdout.write(text)
            
        # Check for filtering patterns
        for pattern in self.filter_patterns:
            if pattern.search(text):
                return len(text)  # Don't output filtered messages
        
        # Output unfiltered messages to original stdout
        return self.original_stdout.write(text)
    
    def flush(self):
        self.original_stdout.flush()
    
    def isatty(self):
        return self.original_stdout.isatty()
    
    def fileno(self):
        return self.original_stdout.fileno()

# Enable log filtering - but preserve color support
original_stdout = sys.stdout
filtered_stdout = OutputFilter(original_stdout)
# Only apply filter if not testing colors
if not os.environ.get('DISABLE_OUTPUT_FILTER', False):
    sys.stdout = filtered_stdout

class ProgressMonitor:
    def __init__(self, processor):
        """
        Constructor
        
        Args:
            processor (ObsidianProcessor): Embedding processor object
        """
        self.processor = processor
        
        # Force enable colors for this instance
        self.force_color = True  # Always use colors
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
        
        # GPU and Memory maximum value tracking - NEW
        self.max_tracking_duration = 5.0  # Track maximum for 5 seconds
        self.max_values_reset_time = time.time()
        self.max_gpu_percent = 0.0
        self.max_gpu_memory_percent = 0.0
        self.gpu_usage_history = []  # Store recent GPU usage values
        self.gpu_memory_history = []  # Store recent GPU memory values
        
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
    
    def _print_colored(self, text):
        """Print with colors forced - bypass any filtering"""
        # Direct write to terminal, bypassing filters
        if hasattr(sys, '__stdout__'):
            sys.__stdout__.write(text + '\n')
            sys.__stdout__.flush()
        else:
            # Fallback to original stdout if available
            if 'original_stdout' in globals():
                original_stdout.write(text + '\n')
                original_stdout.flush()
            else:
                print(text)
    
    def _create_bar(self, percent, width=20):
        """Create a simple progress bar"""
        filled_length = int(width * percent / 100)
        bar = '█' * filled_length + '░' * (width - filled_length)
        return bar
    
    def add_error_log(self, error_message):
        """Add an error message to the log with timestamp"""
        # Check for duplicate messages
        if error_message in self.error_logs:
            return
            
        # Add timestamp
        current_time = datetime.now().strftime("%H:%M:%S")
        timestamped_message = f"[{current_time}] {error_message}"
        
        # Add error log
        self.error_logs.append(timestamped_message)
        self.error_timestamps[timestamped_message] = time.time()
        
        # Maintain maximum count
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
        """Safely get GPU status information with enhanced accuracy using pynvml"""
        # Try pynvml first for more accurate GPU monitoring
        if PYNVML_AVAILABLE:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    
                    # Get GPU utilization (actual usage percentage)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_percent = utilization.gpu
                    
                    # Get memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_used = mem_info.used
                    memory_total = mem_info.total
                    memory_percent = (memory_used / memory_total) * 100 if memory_total > 0 else 0
                    
                    return {
                        "gpu_available": True,
                        "gpu_percent": float(gpu_percent),
                        "gpu_memory_used": memory_used,
                        "gpu_memory_total": memory_total,
                        "gpu_memory_percent": float(memory_percent)
                    }
            except Exception as e:
                print(f"Warning: pynvml error: {e}")
        
        # Fallback to system monitor
        system_monitor = self._get_system_monitor_safe()
        if system_monitor:
            try:
                return system_monitor.get_gpu_status()
            except AttributeError:
                pass
        
        # Final fallback: use torch
        try:
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                used_memory = torch.cuda.memory_allocated(0)
                total_memory = torch.cuda.get_device_properties(0).total_memory
                memory_percent = (used_memory / total_memory) * 100 if total_memory > 0 else 0
                
                # Since torch doesn't provide GPU utilization, estimate from memory usage
                gpu_percent = memory_percent
                
                return {
                    "gpu_available": gpu_available,
                    "gpu_percent": float(gpu_percent),
                    "gpu_memory_used": used_memory,
                    "gpu_memory_total": total_memory,
                    "gpu_memory_percent": float(memory_percent)
                }
            else:
                return {
                    "gpu_available": False,
                    "gpu_percent": 0.0,
                    "gpu_memory_used": 0,
                    "gpu_memory_total": 1,  # Prevent division by zero
                    "gpu_memory_percent": 0.0
                }
        except Exception:
            return {
                "gpu_available": False,
                "gpu_percent": 0.0,
                "gpu_memory_used": 0,
                "gpu_memory_total": 1,
                "gpu_memory_percent": 0.0
            }
    
    # ==============================================
    # NEW: MAXIMUM VALUE TRACKING METHODS
    # ==============================================
    
    def _update_max_values(self, current_gpu_percent, current_gpu_memory_percent):
        """Update maximum GPU values over time period"""
        current_time = time.time()
        
        # Add current values to history
        self.gpu_usage_history.append((current_time, current_gpu_percent))
        self.gpu_memory_history.append((current_time, current_gpu_memory_percent))
        
        # Remove old values (older than max_tracking_duration)
        cutoff_time = current_time - self.max_tracking_duration
        self.gpu_usage_history = [(t, v) for t, v in self.gpu_usage_history if t > cutoff_time]
        self.gpu_memory_history = [(t, v) for t, v in self.gpu_memory_history if t > cutoff_time]
        
        # Calculate maximum values from recent history
        if self.gpu_usage_history:
            self.max_gpu_percent = max(v for t, v in self.gpu_usage_history)
        else:
            self.max_gpu_percent = current_gpu_percent
            
        if self.gpu_memory_history:
            self.max_gpu_memory_percent = max(v for t, v in self.gpu_memory_history)
        else:
            self.max_gpu_memory_percent = current_gpu_memory_percent
        
        # Reset max values every tracking duration period
        if current_time - self.max_values_reset_time > self.max_tracking_duration:
            self.max_values_reset_time = current_time
    
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
        start_time = progress_data.get("start_time", 0)
        total_size = progress_data.get("total_size", 0)
        processed_size = progress_data.get("processed_size", 0)
        
        # Use maximum GPU values instead of current values
        display_gpu_percent = self.max_gpu_percent
        display_gpu_memory_percent = self.max_gpu_memory_percent
        
        # Calculate progress percentage
        percentage = 0
        # Check for None values
        if total_files is None:
            total_files = 0
        if processed_files is None:
            processed_files = 0
            
        if total_files > 0:
            percentage = min(99, int((processed_files / total_files) * 100))
        
        # Calculate estimated time remaining
        estimated_time_remaining = None
        # Check for None values
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
        output_lines.append(f"{Style.BRIGHT}{Fore.MAGENTA}Embedding Progress{Style.RESET_ALL}")
        output_lines.append(f"{Fore.BLUE}{"-" * 80}{Fore.RESET}")
        
        # Display progress bar
        output_lines.append(f"{Style.BRIGHT}{Fore.CYAN}Progress:{Style.RESET_ALL}     [{Fore.GREEN}{self._create_bar(percentage)}{Fore.RESET}] {Fore.YELLOW}{percentage}%{Fore.RESET} ({Fore.CYAN}{processed_files}{Fore.RESET}/{Fore.BLUE}{total_files}{Fore.RESET} files)")
        
        # Display size progress if available
        if total_size > 0:
            size_percentage = min(99, int((processed_size / total_size) * 100))
            size_bar = self._create_bar(size_percentage)
            processed_mb = processed_size / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            output_lines.append(f"{Style.BRIGHT}{Fore.MAGENTA}Size:{Style.RESET_ALL}         [{Fore.BLUE}{size_bar}{Fore.RESET}] {Fore.YELLOW}{size_percentage}%{Fore.RESET} ({Fore.CYAN}{processed_mb:.1f} MB{Fore.RESET}/{Fore.BLUE}{total_mb:.1f} MB{Fore.RESET})")
        
        # Display estimated time remaining (using the value calculated earlier)
        # Always show the line, even if the calculation isn't available yet
        if not estimated_time_remaining:
            estimated_time_remaining = "Calculating..."
        output_lines.append(f"{Style.BRIGHT}{Fore.LIGHTYELLOW_EX}Est. Time Remaining:{Style.RESET_ALL}    {Fore.LIGHTCYAN_EX}{estimated_time_remaining}{Fore.RESET}")
        
        # Display system resource usage
        cpu_color = Fore.GREEN
        if cpu_percent > 70:
            cpu_color = Fore.YELLOW
        if cpu_percent > 90:
            cpu_color = Fore.RED
        output_lines.append(f"{Style.BRIGHT}{Fore.LIGHTBLUE_EX}CPU Usage:{Style.RESET_ALL}    [{cpu_color}{self._create_bar(cpu_percent)}{Fore.RESET}] {cpu_color}{cpu_percent:.1f}%{Fore.RESET}")
        
        memory_color = Fore.GREEN
        if memory_percent > 70:
            memory_color = Fore.YELLOW
        if memory_percent > 90:
            memory_color = Fore.RED
        output_lines.append(f"{Style.BRIGHT}{Fore.LIGHTMAGENTA_EX}Memory Usage:{Style.RESET_ALL} [{memory_color}{self._create_bar(memory_percent)}{Fore.RESET}] {memory_color}{memory_percent:.1f}%{Fore.RESET}")
        
        # Enhanced GPU usage display with maximum tracking
        try:
            # GPU availability check
            if torch.cuda.is_available():
                # Color settings for GPU usage
                gpu_color = Fore.GREEN
                if display_gpu_percent > 70:
                    gpu_color = Fore.YELLOW
                if display_gpu_percent > 90:
                    gpu_color = Fore.RED
                
                # GPU memory color
                gpu_mem_color = Fore.GREEN
                if display_gpu_memory_percent > 70:
                    gpu_mem_color = Fore.YELLOW
                if display_gpu_memory_percent > 90:
                    gpu_mem_color = Fore.RED
                
                # Display GPU usage (5-second maximum)
                output_lines.append(f"{Style.BRIGHT}{Fore.LIGHTGREEN_EX}GPU Usage:{Style.RESET_ALL}    [{gpu_color}{self._create_bar(display_gpu_percent)}{Fore.RESET}] {gpu_color}{display_gpu_percent:.1f}% {Fore.LIGHTCYAN_EX}(5s max){Fore.RESET}")
                
                # Display GPU memory usage (5-second maximum)
                output_lines.append(f"{Style.BRIGHT}{Fore.LIGHTYELLOW_EX}GPU Memory:{Style.RESET_ALL}   [{gpu_mem_color}{self._create_bar(display_gpu_memory_percent)}{Fore.RESET}] {gpu_mem_color}{display_gpu_memory_percent:.1f}% {Fore.LIGHTCYAN_EX}(5s max){Fore.RESET}")
            else:
                output_lines.append(f"{Style.BRIGHT}{Fore.LIGHTRED_EX}GPU Status:{Style.RESET_ALL}   {Fore.RED}[No CUDA compatible GPU detected]{Fore.RESET}")
        except Exception as e:
            output_lines.append(f"{Style.BRIGHT}{Fore.LIGHTRED_EX}GPU Status:{Style.RESET_ALL}   {Fore.RED}[Error getting GPU info: {str(e)[:50]}]{Fore.RESET}")
            
        # GPU configuration check
        use_gpu = hasattr(self.processor, 'use_gpu') and self.processor.use_gpu
        if not use_gpu and torch.cuda.is_available():
            output_lines.append(f"{Style.BRIGHT}{Fore.LIGHTRED_EX}GPU Config:{Style.RESET_ALL}    {Fore.YELLOW}[GPU available but disabled in settings]{Fore.RESET}")
        elif not torch.cuda.is_available():
            output_lines.append(f"{Style.BRIGHT}{Fore.LIGHTRED_EX}GPU Config:{Style.RESET_ALL}    {Fore.RED}[No CUDA compatible GPU available]{Fore.RESET}")
        else:
            output_lines.append(f"{Style.BRIGHT}{Fore.LIGHTGREEN_EX}GPU Config:{Style.RESET_ALL}    {Fore.GREEN}[GPU enabled and active]{Fore.RESET}")

        # Display current file - avoid showing same file if it was skipped
        display_current_file = current_file
        if self.last_processed_file and current_file == self.last_processed_file:
            if self.last_processed_status and ("Already exists, Skipping" in self.last_processed_status):
                display_current_file = ""  # Don't show skipped files
        
        output_lines.append(f"\n{Style.BRIGHT}{Fore.LIGHTCYAN_EX}Current File:{Style.RESET_ALL} {Fore.WHITE}{display_current_file}{Fore.RESET}")
        
        # Previous file - always show
        if self.last_processed_file:
            output_lines.append(f"{Style.BRIGHT}{Fore.LIGHTBLUE_EX}Last File:{Style.RESET_ALL}   {Fore.WHITE}{self.last_processed_file}{Fore.RESET} - {self.last_processed_status}")
        
        output_lines.append(f"{Fore.BLUE}{"-" * 80}{Fore.RESET}")
        
        # Display error logs if any
        current_time = time.time()
        active_errors = []
        
        for error in self.error_logs:
            error_time = self.error_timestamps.get(error, 0)
            if current_time - error_time < self.error_display_time:
                active_errors.append(error)
        
        if active_errors:
            output_lines.append(f"{Style.BRIGHT}{Fore.RED}Error Logs:{Style.RESET_ALL}")
            for error in active_errors[-5:]:  # Show only recent 5
                output_lines.append(f"{Fore.RED}{error}{Fore.RESET}")
            output_lines.append(f"{Fore.BLUE}{"-" * 80}{Fore.RESET}")
        
        # Clear screen and display all lines
        self._clear_screen()
        
        # Try to output directly to terminal for color support
        output_text = "\n".join(output_lines)
        
        # Method 1: Direct to __stdout__ (original stdout before any wrapping)
        if hasattr(sys, '__stdout__'):
            sys.__stdout__.write(output_text + '\n')
            sys.__stdout__.flush()
        # Method 2: Use original_stdout if saved
        elif 'original_stdout' in globals():
            original_stdout.write(output_text + '\n')
            original_stdout.flush()
        # Method 3: Regular print as fallback
        else:
            print(output_text)
    
    def _progress_update_thread_func(self):
        """Thread function for updating the progress display"""
        while not self.stop_progress_update_event.is_set():
            try:
                # Get current file information
                last_current_file = self.processor.embedding_progress.get("current_file", "")
                
                # Update file processing status
                if last_current_file:
                    # If different from previous file (new file processing started)
                    if last_current_file != self.last_processed_file:
                        # Check if full reindex mode
                        is_full_reindex = False
                        if hasattr(self.processor, 'embedding_progress') and isinstance(self.processor.embedding_progress, dict):
                            is_full_reindex = self.processor.embedding_progress.get('is_full_reindex', False)
                        
                        # Show "Already exists, Skipping" message only when not in full reindex mode
                        if not is_full_reindex:
                            self.last_processed_status = f"{Fore.LIGHTGREEN_EX}Already exists, Skipping{Fore.RESET}"
                        else:
                            self.last_processed_status = f"{Fore.LIGHTYELLOW_EX}Processing...{Fore.RESET}"
                        
                        # Update current file to last processed file
                        self.last_processed_file = last_current_file
                    
                    # Also maintain last file info when file processing is complete
                    processed_files = self.processor.embedding_progress.get("processed_files", 0)
                    total_files = self.processor.embedding_progress.get("total_files", 0)
                    
                    # Check for None values
                    if processed_files is None:
                        processed_files = 0
                    if total_files is None:
                        total_files = 0
                    
                    # If all files processed and no current file, show last file as complete
                    if processed_files == total_files and not last_current_file and self.last_processed_file:
                        # Check if full reindex mode
                        is_full_reindex = False
                        if hasattr(self.processor, 'embedding_progress') and isinstance(self.processor.embedding_progress, dict):
                            is_full_reindex = self.processor.embedding_progress.get('is_full_reindex', False)
                        
                        # Show "Already exists, Skipping" message only when not in full reindex mode
                        if not is_full_reindex:
                            self.last_processed_status = f"{Fore.LIGHTGREEN_EX}Already exists, Skipping{Fore.RESET}"
                        else:
                            self.last_processed_status = f"{Fore.LIGHTGREEN_EX}Processed{Fore.RESET}"
                    
                    # Now update screen - display screen after current file info is updated
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
        """Update system resource usage information - ENHANCED WITH MAXIMUM VALUE TRACKING"""
        try:
            # ENHANCED: Use safe methods to get system information
            # This prevents AttributeError if SystemMonitor methods are missing
            
            # Get CPU status safely
            cpu_status = self._get_cpu_status_safe()
            cpu_percent = cpu_status.get("cpu_percent", 50)
            
            # Get memory status safely
            memory_status = self._get_memory_status_safe()
            memory_percent = memory_status.get("memory_percent", 50)
            
            # Get GPU status safely with enhanced accuracy
            gpu_status = self._get_gpu_status_safe()
            current_gpu_percent = gpu_status.get("gpu_percent", 0.0)
            current_gpu_memory_percent = gpu_status.get("gpu_memory_percent", 0.0)
            gpu_memory_used = gpu_status.get("gpu_memory_used", 0)
            gpu_memory_total = gpu_status.get("gpu_memory_total", 1)
            
            # Update maximum GPU values over time period - NEW
            self._update_max_values(current_gpu_percent, current_gpu_memory_percent)
            
            # Update processor's embedding_progress with safe values
            self.processor.embedding_progress["cpu_percent"] = cpu_percent
            self.processor.embedding_progress["memory_percent"] = memory_percent
            # Store both current and maximum values
            self.processor.embedding_progress["gpu_percent"] = current_gpu_percent
            self.processor.embedding_progress["gpu_memory_percent"] = current_gpu_memory_percent
            self.processor.embedding_progress["gpu_memory_used"] = gpu_memory_used
            self.processor.embedding_progress["gpu_memory_total"] = gpu_memory_total
            # Store maximum values for display
            self.processor.embedding_progress["max_gpu_percent"] = self.max_gpu_percent
            self.processor.embedding_progress["max_gpu_memory_percent"] = self.max_gpu_memory_percent
            
        except Exception as e:
            # Ultimate fallback: set default values
            print(f"Warning: Error updating system resources: {e}")
            self.processor.embedding_progress["cpu_percent"] = 50
            self.processor.embedding_progress["memory_percent"] = 50
            self.processor.embedding_progress["gpu_percent"] = 0
            self.processor.embedding_progress["gpu_memory_percent"] = 0
            self.processor.embedding_progress["gpu_memory_used"] = 0
            self.processor.embedding_progress["gpu_memory_total"] = 1
            self.processor.embedding_progress["max_gpu_percent"] = 0
            self.processor.embedding_progress["max_gpu_memory_percent"] = 0
    
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

# Test function to check if colors are working
def test_colors():
    """Test if terminal colors are working properly"""
    print("\nTesting colorama colors:")
    print(f"{Fore.RED}█ RED{Fore.RESET}")
    print(f"{Fore.GREEN}█ GREEN{Fore.RESET}")
    print(f"{Fore.BLUE}█ BLUE{Fore.RESET}")
    print(f"{Fore.YELLOW}█ YELLOW{Fore.RESET}")
    print(f"{Fore.MAGENTA}█ MAGENTA{Fore.RESET}")
    print(f"{Fore.CYAN}█ CYAN{Fore.RESET}")
    print(f"{Fore.LIGHTRED_EX}█ LIGHT RED{Fore.RESET}")
    print(f"{Fore.LIGHTGREEN_EX}█ LIGHT GREEN{Fore.RESET}")
    print(f"{Fore.LIGHTBLUE_EX}█ LIGHT BLUE{Fore.RESET}")
    print(f"{Fore.LIGHTYELLOW_EX}█ LIGHT YELLOW{Fore.RESET}")
    print(f"{Fore.LIGHTMAGENTA_EX}█ LIGHT MAGENTA{Fore.RESET}")
    print(f"{Fore.LIGHTCYAN_EX}█ LIGHT CYAN{Fore.RESET}")
    print("\nIf you see colors above, colorama is working!\n")

# Run color test if executed directly
if __name__ == "__main__":
    test_colors()
