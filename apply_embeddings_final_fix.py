"""
Apply the fixed embeddings.py to resolve the _temp_model error
"""
import os
import shutil

# Get the project directory
project_dir = os.path.dirname(os.path.abspath(__file__))

# File paths
original_file = os.path.join(project_dir, "embeddings.py")
fixed_file = os.path.join(project_dir, "embeddings_fixed.py")
backup_file = os.path.join(project_dir, "embeddings_original_backup.py")

print("Fixing embeddings.py...")

# Create backup if it doesn't exist
if not os.path.exists(backup_file) and os.path.exists(original_file):
    print("Creating backup of original embeddings.py...")
    shutil.copy2(original_file, backup_file)
    print(f"Backup saved as: embeddings_original_backup.py")

# Apply the fix
if os.path.exists(fixed_file):
    print("Applying fixed version...")
    shutil.copy2(fixed_file, original_file)
    print("Fixed embeddings.py applied successfully!")
    print("\nThe fix:")
    print("- Removed threading complexity that caused _temp_model error")
    print("- Simplified model loading process")
    print("- Maintained all functionality")
    print("\nNow try running start_mcp_with_encoding_fix.bat again")
else:
    print("Error: embeddings_fixed.py not found!")

print("\nTo restore original version later, rename embeddings_original_backup.py to embeddings.py")
