import os
import pytest
from pathlib import Path

def test_outdated_files_naming():
    """Test that all files in the outdated directory end with _outdated.md"""
    # Get the path to the outdated directory
    outdated_dir = Path(__file__).parent.parent / "realtime_with_restart" / "tasks_and_plans" / "implementation_plans" / "outdated"
    
    # Check if directory exists
    assert outdated_dir.exists(), f"Outdated directory not found at {outdated_dir}"
    
    # Get all files in the directory
    files = list(outdated_dir.glob("*.md"))
    
    # If there are no files, that's fine - the test passes
    if not files:
        return
        
    # Check each file
    for file in files:
        assert file.name.endswith("_outdated.md"), \
            f"File {file.name} in outdated directory does not end with _outdated.md" 