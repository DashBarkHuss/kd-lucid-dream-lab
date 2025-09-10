"""
Sleep stage classification constants for the sleep scoring toolkit.

This module defines standardized constants for sleep stage classifications
and channel configurations to prevent hardcoded assumptions and 
inconsistencies across the codebase.

The main issue this solves: GSSC model outputs REM as 4, but researcher 
scoring uses REM as 5, leading to confusion in tests and comparisons.
"""

# =============================================================================
# Standard Channel Configurations
# =============================================================================

# Standard EEG channels for sleep scoring
STANDARD_EEG_CHANNELS = ["C4", "C3", "F3", "F4"]

# Standard EOG channels for sleep scoring  
STANDARD_EOG_CHANNELS = ["L-HEOG", "R-HEOG"]

# =============================================================================
# Processing Configuration Constants
# =============================================================================

# Real-time processing uses 6 buffers for round-robin approach (0s, 5s, 10s, 15s, 20s, 25s offsets)
# Batch processing uses 1 buffer for sequential processing
REALTIME_BUFFER_COUNT = 6

# =============================================================================
# Sleep Stage Classifications
# =============================================================================

# GSSC Model Output Sleep Stages (0-4)
# These are the values returned by the GSSC model's predict_sleep_stage() method
class GSSCStages:
    """Sleep stage constants for GSSC model output (0-4 mapping)"""
    WAKE = 0
    N1 = 1
    N2 = 2
    N3 = 3
    REM = 4
    
    # Mapping for display/logging
    NAMES = {
        WAKE: "Wake",
        N1: "N1", 
        N2: "N2",
        N3: "N3",
        REM: "REM"
    }
    
    # Valid range for validation
    MIN_STAGE = WAKE
    MAX_STAGE = REM
    NUM_STAGES = 5
    
    @classmethod
    def is_valid(cls, stage: int) -> bool:
        """Check if stage value is valid for GSSC model"""
        return cls.MIN_STAGE <= stage <= cls.MAX_STAGE
    
    @classmethod
    def to_name(cls, stage: int) -> str:
        """Convert stage number to human-readable name"""
        return cls.NAMES.get(stage, f"Unknown({stage})")


# Researcher Scoring Sleep Stages (follows standard polysomnography scoring)  
# These are the values found in .mat files from researcher scoring
class ResearcherStages:
    """Sleep stage constants for researcher scoring (standard PSG mapping)"""
    WAKE = 0
    N1 = 1
    N2 = 2
    N3 = 3
    N4 = 4  # Older classification, rarely used
    REM = 5  # Note: Different from GSSC!
    MOVEMENT = 6  # Movement artifacts
    UNCERTAIN = 7  # Uncertain periods/custom annotations
    
    # Mapping for display/logging
    NAMES = {
        WAKE: "Wake",
        N1: "N1",
        N2: "N2", 
        N3: "N3",
        N4: "N4",
        REM: "REM",
        MOVEMENT: "Movement",
        UNCERTAIN: "Uncertain"
    }
    
    # Valid range for validation
    MIN_STAGE = WAKE
    MAX_STAGE = UNCERTAIN  
    NUM_CORE_STAGES = 6  # 0-5 (excluding movement/uncertain)
    
    @classmethod
    def is_valid(cls, stage: int) -> bool:
        """Check if stage value is valid for researcher scoring"""
        return cls.MIN_STAGE <= stage <= cls.MAX_STAGE
    
    @classmethod
    def is_core_sleep_stage(cls, stage: int) -> bool:
        """Check if stage is a core sleep stage (not movement/uncertain)"""
        return stage in [cls.WAKE, cls.N1, cls.N2, cls.N3, cls.N4, cls.REM]
    
    @classmethod
    def to_name(cls, stage: int) -> str:
        """Convert stage number to human-readable name"""
        return cls.NAMES.get(stage, f"Unknown({stage})")




