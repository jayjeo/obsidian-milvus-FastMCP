# Milvus Query Window Error Fix

## Problem
The error you encountered was:
```
MilvusException: (code=65535, message=invalid max query result window, (offset+limit) should be in range [1, 16384], but got 25000)
```

## Root Cause
The issue was in the `robust_incremental_embedding.py` and `milvus_manager.py` files where query limits were set too high, exceeding Milvus's maximum query window constraint of 16384.

## Fixed Files
1. **robust_incremental_embedding.py** - Reduced `max_limit` from 25000 to 16000 and added proper limit checking
2. **milvus_manager.py** - Reduced `max_limit` from 10000 to 8000 and added pagination safety checks

## Changes Made

### robust_incremental_embedding.py
- Changed limit from 25000 to 16000
- Added dynamic limit adjustment to prevent exceeding 16384
- Added error handling for query window limit errors

### milvus_manager.py
- Changed limit from 10000 to 8000
- Added proper pagination limit checking
- Improved error handling for query window constraints

## How to Test
1. Run `python main.py`
2. Select option 3 (Incremental & Deleted Cleanup)
3. The process should now complete without the query window error

## Technical Details
- Milvus has a hard limit where `(offset + limit) <= 16384`
- The fix ensures pagination never exceeds this constraint
- Fallback mechanisms handle edge cases where the limit is reached

## Backup Files
- Original files are backed up as `robust_incremental_embedding_backup.py`
- You can restore the original by renaming the backup file if needed

## English-Only Requirement
All code comments and output messages are now in English to prevent encoding issues as requested.
