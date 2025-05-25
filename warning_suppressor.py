"""
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
