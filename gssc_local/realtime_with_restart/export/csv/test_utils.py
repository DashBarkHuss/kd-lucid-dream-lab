"""
Test utilities for CSV validation and comparison.

This module provides test-specific utilities for validating CSV files,
particularly for comparing exported data with reference data.
"""

import logging
from pathlib import Path
from typing import Union, Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class CSVComparisonResult:
    """Results of a CSV comparison."""
    matches: bool
    line_count_matches: bool
    actual_line_count: int
    expected_line_count: int
    mismatched_lines: List[Tuple[int, str, str]]  # [(line_num, expected, actual)]
    error_message: Optional[str] = None

def compare_csv_files(saved_path: Union[str, Path], 
                     reference_path: Union[str, Path], 
                     logger: Optional[logging.Logger] = None) -> CSVComparisonResult:
    """Compare a saved CSV file against a reference file for exact matches.
    
    This is a test utility function for verifying that exported CSV data
    matches the expected format and content exactly.
    
    Args:
        saved_path (Union[str, Path]): Path to the saved CSV to validate
        reference_path (Union[str, Path]): Path to the reference CSV to compare against
        logger (Optional[logging.Logger]): Logger for detailed comparison output
        
    Returns:
        CSVComparisonResult: Detailed results of the comparison
    """
    try:
        # Convert paths to Path objects
        saved_path = Path(saved_path)
        reference_path = Path(reference_path)
        
        # Read both CSVs as strings
        with open(saved_path, 'r') as f:
            saved_lines = f.readlines()
        with open(reference_path, 'r') as f:
            reference_lines = f.readlines()
        
        if logger:
            logger.info("\nCSV Comparison Results:")
            logger.info(f"Reference CSV path: {reference_path}")
            logger.info(f"Saved CSV path: {saved_path}")
            logger.info(f"Reference lines: {len(reference_lines)}")
            logger.info(f"Saved lines: {len(saved_lines)}")
        
        # Initialize result
        result = CSVComparisonResult(
            matches=True,
            line_count_matches=len(saved_lines) == len(reference_lines),
            actual_line_count=len(saved_lines),
            expected_line_count=len(reference_lines),
            mismatched_lines=[]
        )
        
        # Check for empty files
        if len(saved_lines) == 0 and len(reference_lines) == 0:
            result.matches = False
            result.error_message = "Both saved and reference CSV files are empty"
            if logger:
                logger.error("❌ Both saved and reference CSV files are empty.")
            return result
            
        # Check line count match
        if not result.line_count_matches:
            result.matches = False
            result.error_message = f"Line count mismatch: Reference={len(reference_lines)}, Saved={len(saved_lines)}"
            if logger:
                logger.error(f"❌ {result.error_message}")
                # Debug first few lines of both files
                logger.error("\nFirst 5 lines of reference file:")
                for i, line in enumerate(reference_lines[:5]):
                    logger.error(f"Reference line {i+1}: {line.strip()}")
                logger.error("\nFirst 5 lines of saved file:")
                for i, line in enumerate(saved_lines[:5]):
                    logger.error(f"Saved line {i+1}: {line.strip()}")
            return result
            
        if logger:
            logger.info(f"✅ Line count matches: {len(reference_lines)} lines")
        
        # Compare each line exactly
        for i, (saved_line, reference_line) in enumerate(zip(saved_lines, reference_lines)):
            if saved_line.strip() != reference_line.strip():
                result.matches = False
                result.mismatched_lines.append((i+1, reference_line.strip(), saved_line.strip()))
                
                if logger:
                    logger.error(f"❌ Line {i+1} does not match exactly:")
                    logger.error(f"Reference: {reference_line.strip()}")
                    logger.error(f"Saved:     {saved_line.strip()}")
                    # Debug surrounding lines for context
                    start = max(0, i-2)
                    end = min(len(saved_lines), i+3)
                    logger.error("\nContext from reference file:")
                    for j in range(start, end):
                        logger.error(f"Reference line {j+1}: {reference_lines[j].strip()}")
                    logger.error("\nContext from saved file:")
                    for j in range(start, end):
                        logger.error(f"Saved line {j+1}: {saved_lines[j].strip()}")
        
        if result.matches and logger:
            logger.info("✅ All lines match exactly")
            
        return result
            
    except Exception as e:
        if logger:
            logger.error(f"Failed to compare CSV files: {e}")
        return CSVComparisonResult(
            matches=False,
            line_count_matches=False,
            actual_line_count=0,
            expected_line_count=0,
            mismatched_lines=[],
            error_message=f"Failed to compare CSV files: {str(e)}"
        ) 