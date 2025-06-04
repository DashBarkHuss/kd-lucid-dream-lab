"""
Custom exceptions for CSV data management and validation.

This module contains all custom exceptions used by the CSV manager and related components.
Exceptions are organized in a hierarchy for better error handling and categorization.
"""

class CSVExportError(Exception):
    """Base exception for CSV export errors."""
    pass

class CSVValidationError(CSVExportError):
    """Raised when CSV validation fails."""
    pass

class CSVDataError(CSVExportError):
    """Raised when there are issues with the data being saved."""
    pass

class CSVFormatError(CSVExportError):
    """Raised when the CSV format is incorrect."""
    pass

class MissingOutputPathError(CSVExportError):
    """Raised when an output path is required but not set."""
    pass

class BufferError(Exception):
    """Base exception for buffer-related errors."""
    pass

class BufferOverflowError(BufferError):
    """Raised when buffer size limit is exceeded."""
    pass

class BufferStateError(BufferError):
    """Raised when buffer is in an invalid state for an operation."""
    pass

class BufferValidationError(BufferError):
    """Raised when buffer validation fails."""
    pass 