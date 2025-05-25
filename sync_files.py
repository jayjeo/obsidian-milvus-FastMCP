"""
File sync helper - Copy missing files between PC and notebook
This script helps sync necessary files between different project locations
"""

import os
import shutil
import sys

def sync_files():
    """Sync required files to ensure both locations have all necessary files"""
    
    # List of critical files that must exist
    required_files = [
        "warning_suppressor.py",
        "mcp_server_helpers.py",
        "enhanced_search_engine.py",
        "hnsw_optimizer.py", 
        "advanced_rag.py",
        "config.py",
        "milvus_manager.py",
        "search_engine.py",
        "embeddings.py"
    ]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 60)
    print("OBSIDIAN-MILVUS FILE SYNC HELPER")
    print("=" * 60)
    print(f"\nCurrent directory: {current_dir}")
    print("\nChecking required files...")
    
    missing_files = []
    existing_files = []
    
    for file in required_files:
        file_path = os.path.join(current_dir, file)
        if os.path.exists(file_path):
            existing_files.append(file)
        else:
            missing_files.append(file)
    
    print(f"\n✅ Found {len(existing_files)} files:")
    for file in existing_files:
        print(f"   - {file}")
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} files:")
        for file in missing_files:
            print(f"   - {file}")
        
        print("\n⚠️  IMPORTANT: Copy these missing files from your main project folder!")
        print("\nIf you're syncing between PC and notebook, make sure to copy:")
        print("1. From PC: G:\\JJ Dropbox\\J J\\PythonWorks\\milvus\\obsidian-milvus-FastMCP")
        print("2. To Notebook: C:\\Users\\acube\\OneDrive\\Documents\\GitHub\\obsidian-milvus-FastMCP")
        
        # Create warning_suppressor.py if it's missing
        if "warning_suppressor.py" in missing_files:
            print("\n🔧 Creating warning_suppressor.py...")
            create_warning_suppressor(current_dir)
            print("✅ warning_suppressor.py created!")
    else:
        print("\n✅ All required files are present!")
    
    return missing_files

def create_warning_suppressor(directory):
    """Create the warning_suppressor.py file"""
    content = '''"""
Warning suppressor module to silence non-critical warnings across the application.
This module must be imported at the very beginning of any script to ensure
warnings are suppressed before any other imports.
"""

import warnings
import os
import sys
import logging

# Suppress all warnings by default
warnings.filterwarnings('ignore')

# Specific warning suppressions
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Suppress warnings from specific modules
warnings.filterwarnings('ignore', module='huggingface_hub')
warnings.filterwarnings('ignore', module='transformers')
warnings.filterwarnings('ignore', module='torch')
warnings.filterwarnings('ignore', module='sentence_transformers')
warnings.filterwarnings('ignore', module='urllib3')

# Suppress specific message patterns
warning_patterns = [
    ".*resume_download.*",
    ".*Parallelism.*",
    ".*torch.utils._pytree.*",
    ".*TypedStorage is deprecated.*",
    ".*weights_only.*",
    ".*huggingface_hub.*",
    ".*transformers.*",
    ".*tokenizers.*",
    ".*clean_up_tokenization_spaces.*",
    ".*flash_attn.*",
    ".*TqdmWarning.*",
]

for pattern in warning_patterns:
    warnings.filterwarnings('ignore', message=pattern)

# Environment variable to disable parallelism warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Suppress HTTP connection warnings
try:
    from urllib3.exceptions import InsecureRequestWarning
    warnings.filterwarnings('ignore', category=InsecureRequestWarning)
except ImportError:
    pass

# Configure logging to reduce noise
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('filelock').setLevel(logging.ERROR)

# Suppress stdout/stderr for specific operations if needed
class SuppressOutput:
    """Context manager to suppress stdout/stderr"""
    
    def __init__(self, suppress_stdout=True, suppress_stderr=True):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self._stdout = None
        self._stderr = None
        
    def __enter__(self):
        if self.suppress_stdout:
            self._stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        if self.suppress_stderr:
            self._stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress_stdout and self._stdout:
            sys.stdout.close()
            sys.stdout = self._stdout
        if self.suppress_stderr and self._stderr:
            sys.stderr.close()
            sys.stderr = self._stderr

# Export the context manager for use in other modules
__all__ = ['SuppressOutput']

# Apply warning suppression immediately
def apply_warning_suppression():
    """Reapply all warning suppressions (useful after other modules import)"""
    warnings.filterwarnings('ignore')
    for pattern in warning_patterns:
        warnings.filterwarnings('ignore', message=pattern)

# Auto-apply on import
apply_warning_suppression()
'''
    
    filepath = os.path.join(directory, "warning_suppressor.py")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    missing = sync_files()
    
    if missing:
        print("\n" + "=" * 60)
        print("ACTION REQUIRED:")
        print("1. Copy the missing files from your main project")
        print("2. Run this script again to verify")
        print("3. Then run start_mcp_with_encoding_fix.bat")
        print("=" * 60)
    else:
        print("\n✅ Ready to run start_mcp_with_encoding_fix.bat")
    
    input("\nPress Enter to exit...")
