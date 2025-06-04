"""
Export module for the realtime_with_restart package.

This module provides CSV export functionality specifically for brainflow data, including:
- Memory-efficient buffer management for long recordings
- Incremental saving to prevent memory overflow
- Separate handling of main EEG data and sleep stage data
- Exact format preservation for compatibility
- Comprehensive validation and error handling

For detailed usage examples and validation rules, see the README.md and individual module docstrings.
"""

from .csv.manager import CSVManager

__all__ = ['CSVManager'] 