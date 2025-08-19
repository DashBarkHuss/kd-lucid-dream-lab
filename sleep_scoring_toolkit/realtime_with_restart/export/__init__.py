"""
Export module for the realtime_with_restart package.

This module provides CSV export functionality specifically for brainflow data, including:
- Memory-efficient buffer management for long recordings
- Incremental saving to prevent memory overflow
- Separate handling of main EEG data and sleep stage data
- Exact format preservation for compatibility
- Comprehensive validation and error handling

For detailed usage examples and validation rules, see the README.md and individual module docstrings.

Note: CSVManager should be imported directly from the csv submodule:
    from sleep_scoring_toolkit.realtime_with_restart.export.csv.manager import CSVManager
"""

__all__ = [] 