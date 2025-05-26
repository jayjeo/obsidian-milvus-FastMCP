# NumPy Compatibility Fix Guide

## Problem Description

If you encounter an error like:
```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
```

This happens because `sentence-transformers` was compiled with NumPy 1.x but your system has NumPy 2.x installed, which is incompatible.

## Quick Solution

1. **Run the automatic fix (Recommended)**:
   ```batch
   fix_numpy_compatibility.bat
   ```

2. **Or run the diagnostic script**:
   ```batch
   diagnose_and_fix.bat
   ```

## Manual Solution

If the automatic fix doesn't work, follow these steps:

1. **Uninstall incompatible packages**:
   ```batch
   python -m pip uninstall -y numpy sentence-transformers
   ```

2. **Install compatible NumPy version**:
   ```batch
   python -m pip install "numpy<2,>=1.21.0"
   ```

3. **Reinstall sentence-transformers**:
   ```batch
   python -m pip install --no-cache-dir sentence-transformers>=2.2.2
   ```

4. **Verify the fix**:
   ```batch
   python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
   python -c "import sentence_transformers; print('Success!')"
   ```

## Why This Happens

- **sentence-transformers** depends on NumPy for tensor operations
- The current version was compiled against NumPy 1.x API
- NumPy 2.x has breaking changes that make it incompatible
- Until sentence-transformers is recompiled for NumPy 2.x, we need to use NumPy 1.x

## Prevention

The `requirements.txt` file has been updated to prevent this issue:
```
# NumPy compatibility fix (must be before other packages)
numpy<2,>=1.21.0
```

When installing packages, always use:
```batch
python -m pip install -r requirements.txt
```

## Verification

After fixing, run option 3 (incremental embedding) to test:
```batch
run-main.bat
# Choose option 3
```

## Additional Help

- Run `diagnose_and_fix.bat` for comprehensive environment checking
- Check Python version compatibility
- Ensure all packages are installed in the same environment

## Technical Details

- **Compatible NumPy versions**: 1.21.0 to 1.26.4
- **Incompatible NumPy versions**: 2.0.0 and above
- **sentence-transformers version**: 2.2.2 or higher
- **Python requirement**: 3.8 or higher

---

*This issue is common when NumPy 2.x is installed by default in newer Python environments. The fix is simple and permanent once applied correctly.*
