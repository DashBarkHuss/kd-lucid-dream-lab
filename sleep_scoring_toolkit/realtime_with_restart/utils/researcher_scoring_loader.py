#!/usr/bin/env python3

"""
Researcher Scoring File Loader for Real-Time Sleep Stage Comparison.

This module provides functionality to load and query researcher sleep stage
annotations from MATLAB .mat files, enabling real-time comparison between
automated model predictions and expert human scoring.

Classes:
    ResearcherScoringLoader: Main class for loading and querying .mat files
"""

import h5py
import numpy as np
from typing import Optional, Union


class ResearcherScoringLoader:
    """
    Loads researcher sleep stage scoring from MATLAB .mat files.
    
    Handles timestamp conversion between EEG absolute timestamps and 
    researcher relative timing, supporting real-time lookup of sleep stages.
    """
    
    def __init__(self, file_path: str, eeg_recording_start_timestamp: float):
        """
        Initialize the researcher scoring loader.
        
        Args:
            file_path: Path to the MATLAB .mat scoring file
            eeg_recording_start_timestamp: Unix timestamp when EEG recording started
                                         (used as reference for relative time conversion)
        
        Raises:
            FileNotFoundError: If the .mat file doesn't exist
            ValueError: If the file format is invalid or corrupted
        """
        self.file_path = file_path
        self.eeg_recording_start = eeg_recording_start_timestamp
        self.epoch_stages = None
        self.epoch_start_times = None
        self.num_epochs = 0
        
        # Load and validate the file
        self._load_mat_file()
        self._validate_data()
    
    def _load_mat_file(self):
        """Load the MATLAB .mat file using h5py for v7.3 format files."""
        try:
            with h5py.File(self.file_path, 'r') as f:
                if 'stageData' not in f:
                    raise ValueError("MATLAB file missing 'stageData' structure")
                
                stage_data = f['stageData']
                
                # Extract stages array (sleep stage values 0-7 for each epoch)
                stages_raw = stage_data['stages'][:]
                self.epoch_stages = stages_raw.flatten()
                
                # Extract stageTime array (epoch start times in minutes from recording start)
                stage_time_raw = stage_data['stageTime'][:]
                self.epoch_start_times = stage_time_raw.flatten()
                
                self.num_epochs = len(self.epoch_stages)
                
        except OSError as e:
            raise FileNotFoundError(f"Could not open MATLAB file: {self.file_path}") from e
        except Exception as e:
            raise ValueError(f"Error loading MATLAB file structure: {e}") from e
    
    def _validate_data(self):
        """Validate the loaded data for consistency and expected format."""
        if self.epoch_stages is None or self.epoch_start_times is None:
            raise ValueError("Failed to load required data from MATLAB file")
        
        if len(self.epoch_stages) != len(self.epoch_start_times):
            raise ValueError(f"Epoch stages ({len(self.epoch_stages)}) and epoch start times ({len(self.epoch_start_times)}) arrays have different lengths")
        
        if self.num_epochs == 0:
            raise ValueError("No epochs found in scoring file")
        
        # Validate stage values are in expected range (0-7)
        unique_stages = np.unique(self.epoch_stages)
        if np.any(unique_stages < 0) or np.any(unique_stages > 7):
            raise ValueError(f"Invalid stage values found: {unique_stages}")
        
        # Validate epoch start times are sequential and start at 0
        if not np.allclose(np.diff(self.epoch_start_times), 0.5, atol=0.01):
            raise ValueError("Epoch start times are not in expected 0.5-minute intervals")
        
        if not np.isclose(self.epoch_start_times[0], 0.0, atol=0.01):
            raise ValueError(f"Epoch start times should start at 0.0, found: {self.epoch_start_times[0]}")
    
    def _convert_eeg_timestamp_to_relative_minutes(self, eeg_absolute_timestamp: float) -> float:
        """
        Convert EEG absolute timestamp to relative minutes from recording start.
        
        Args:
            eeg_absolute_timestamp: Unix timestamp from EEG data
            
        Returns:
            Time in minutes from recording start
        """
        relative_seconds = eeg_absolute_timestamp - self.eeg_recording_start
        return relative_seconds / 60.0
    
    def _get_epoch_index_from_relative_minutes(self, relative_minutes: float) -> Optional[int]:
        """
        Calculate which epoch index a relative timestamp falls into.
        
        Args:
            relative_minutes: Time in minutes from recording start
            
        Returns:
            Epoch index (0-based) or None if outside valid range
        """
        # Direct calculation since epochs are evenly spaced at 0.5-minute intervals
        epoch_index = int(relative_minutes / 0.5)
        
        # Check bounds - must be within valid epoch range
        if epoch_index < 0 or epoch_index >= len(self.epoch_stages):
            return None
            
        # Verify timestamp falls within the calculated epoch (handle edge cases)
        epoch_start = epoch_index * 0.5
        epoch_end = epoch_start + 0.5
        
        if epoch_start <= relative_minutes < epoch_end:
            return epoch_index
            
        return None
    
    def get_researcher_score_for_timestamp(self, eeg_absolute_timestamp: float) -> Optional[int]:
        """
        Get researcher sleep stage score for a given EEG timestamp.
        
        Args:
            eeg_absolute_timestamp: Unix timestamp from EEG data
            
        Returns:
            Sleep stage (0-7) if timestamp falls within scored period, None otherwise
            
        Stage meanings (typical):
            0: Wake
            1: N1 (light sleep)
            2: N2 (deeper sleep)
            3: N3 (deep sleep)
            4: N4 (very deep sleep, older classification)
            5: REM (rapid eye movement)
            6-7: Movement artifacts, uncertain periods, or custom annotations
        """
        # Convert EEG absolute timestamp to relative minutes from recording start
        relative_minutes = self._convert_eeg_timestamp_to_relative_minutes(eeg_absolute_timestamp)
        
        # Get the epoch index for this timestamp
        epoch_index = self._get_epoch_index_from_relative_minutes(relative_minutes)
        
        if epoch_index is None:
            return None
            
        stage = int(self.epoch_stages[epoch_index])
        return stage
    
    def get_scoring_info(self) -> dict:
        """
        Get summary information about the loaded scoring data.
        
        Returns:
            Dictionary with scoring file statistics and metadata
        """
        stage_counts = np.bincount(self.epoch_stages.astype(int), minlength=8)
        
        return {
            'file_path': self.file_path,
            'num_epochs': self.num_epochs,
            'duration_minutes': self.epoch_start_times[-1] + 0.5 if len(self.epoch_start_times) > 0 else 0,
            'duration_seconds': (self.epoch_start_times[-1] + 0.5) * 60 if len(self.epoch_start_times) > 0 else 0,
            'stage_counts': {
                'Wake (0)': int(stage_counts[0]),
                'N1 (1)': int(stage_counts[1]),
                'N2 (2)': int(stage_counts[2]),
                'N3 (3)': int(stage_counts[3]),
                'N4 (4)': int(stage_counts[4]),
                'REM (5)': int(stage_counts[5]),
                'Stage 6': int(stage_counts[6]),
                'Stage 7': int(stage_counts[7])
            },
            'epoch_interval_minutes': 0.5,
            'epoch_interval_seconds': 30
        }
    
    def validate_file_format(self) -> bool:
        """
        Validate that the file follows expected format conventions.
        
        Returns:
            True if file format is valid, False otherwise
        """
        try:
            self._validate_data()
            return True
        except ValueError:
            return False