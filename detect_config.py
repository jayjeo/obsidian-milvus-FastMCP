import sys
import os
from pathlib import Path

sys.path.insert(0, '.')
try:
    import config
    print('Config loaded successfully')
    
    def expand_path(path_str):
        if path_str.startswith('~/'):
            return str(Path.home() / path_str[2:])
        return path_str
    
    # Find conda
    conda_found = False
    conda_path = ''
    
    if getattr(config, 'AUTO_DETECT_CONDA', True):
        # Try common locations
        common_conda_paths = [
            str(Path.home() / 'Anaconda3' / 'Scripts' / 'conda.exe'),
            str(Path.home() / 'Miniconda3' / 'Scripts' / 'conda.exe'), 
            str(Path.home() / 'anaconda3' / 'Scripts' / 'conda.exe'),
            str(Path.home() / 'miniconda3' / 'Scripts' / 'conda.exe'),
            'C:/ProgramData/Anaconda3/Scripts/conda.exe',
            'C:/ProgramData/Miniconda3/Scripts/conda.exe'
        ]
        
        for path in common_conda_paths:
            if os.path.exists(path):
                conda_path = path
                conda_found = True
                print(f'Auto-detected conda: {path}')
                break
    
    if not conda_found and hasattr(config, 'CONDA_PATHS'):
        for path in config.CONDA_PATHS:
            expanded_path = expand_path(path)
            if os.path.exists(expanded_path):
                conda_path = expanded_path
                conda_found = True
                print(f'Found conda from config: {expanded_path}')
                break
    
    # Find mamba
    mamba_found = False
    mamba_path = ''
    
    if conda_found and getattr(config, 'AUTO_DETECT_MAMBA', True):
        common_mamba_paths = [
            str(Path.home() / '.local' / 'share' / 'mamba' / 'condabin' / 'mamba.bat'),
            str(Path.home() / 'Anaconda3' / 'Scripts' / 'mamba.exe'),
            str(Path.home() / 'Miniconda3' / 'Scripts' / 'mamba.exe'),
            'C:/ProgramData/Anaconda3/Scripts/mamba.exe'
        ]
        
        for path in common_mamba_paths:
            if os.path.exists(path):
                mamba_path = path
                mamba_found = True
                print(f'Auto-detected mamba: {path}')
                break
    
    if not mamba_found and conda_found and hasattr(config, 'MAMBA_PATHS'):
        for path in config.MAMBA_PATHS:
            expanded_path = expand_path(path)
            if os.path.exists(expanded_path):
                mamba_path = expanded_path
                mamba_found = True
                print(f'Found mamba from config: {expanded_path}')
                break
    
    # Save results
    with open('temp_config.bat', 'w') as f:
        f.write(f'set CONDA_FOUND={1 if conda_found else 0}\n')
        if conda_found:
            f.write(f'set "CONDA_PATH={conda_path}"\n')
        f.write(f'set MAMBA_FOUND={1 if mamba_found else 0}\n')
        if mamba_found:
            f.write(f'set "MAMBA_PATH={mamba_path}"\n')
    
    if conda_found:
        print(f'Conda ready: {conda_path}')
    else:
        print('Conda not found')
        
    if mamba_found:
        print(f'Mamba ready: {mamba_path}')
    else:
        print('Mamba not found')
        
except Exception as e:
    print(f'Error reading config: {e}')
    with open('temp_config.bat', 'w') as f:
        f.write('set CONDA_FOUND=0\n')
        f.write('set MAMBA_FOUND=0\n')
