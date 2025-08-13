"""
Simple handler functions for sleep stage events.

These functions demonstrate how to create custom responses to sleep stage detection.
Each function takes a SleepStageEvent and performs some action.
"""

import logging
from typing import Optional

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

logger = logging.getLogger(__name__)


def play_sound(sound_file: str) -> bool:
    """Play a sound file asynchronously using multiple fallback methods.
    
    Args:
        sound_file: Path to the sound file to play
        
    Returns:
        bool: True if sound playback was initiated successfully, False otherwise
    """
    import subprocess
    import os
    
    # Method 1: Try macOS afplay command first (works in debugger) - NON-BLOCKING
    try:
        if os.system(f"which afplay > /dev/null 2>&1") == 0:
            # Use Popen for non-blocking execution
            process = subprocess.Popen(['afplay', sound_file], 
                                     stdout=subprocess.DEVNULL, 
                                     stderr=subprocess.DEVNULL)
            logger.info(f"Playing sound with afplay (async): {sound_file}")
            return True
    except Exception as e:
        logger.warning(f"afplay method failed: {e}")
    
    # Method 2: Try pygame if available - NON-BLOCKING
    if PYGAME_AVAILABLE:
        try:
            # Initialize pygame mixer if not already initialized
            if not pygame.mixer.get_init():
                pygame.mixer.init()
                
            # Load and play the sound (pygame is naturally non-blocking)
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
            logger.info(f"Playing sound with pygame (async): {sound_file}")
            return True
        except Exception as e:
            logger.warning(f"pygame method failed: {e}")
    
    # Method 3: Try system open command (macOS) - NON-BLOCKING
    try:
        process = subprocess.Popen(['open', sound_file],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        logger.info(f"Playing sound with open command (async): {sound_file}")
        return True
    except Exception as e:
        logger.warning(f"open command failed: {e}")
    
    logger.error(f"All audio playback methods failed for: {sound_file}")
    return False


def handler(sleep_stage_data):
    """Single unified handler for all sleep stage events.
    
    Args:
        sleep_stage_data: SleepStageEvent containing sleep stage information
    """
    # Play sound on high confidence N1 detection
    if sleep_stage_data.stage_text == 'N1' and sleep_stage_data.confidence > 0.6:
        sound_file = "sounds/n1_alert.wav"
        success = play_sound(sound_file)
        if success:
            print(f"🎯 HIGH CONFIDENCE N1 DETECTED! Sound played at {sleep_stage_data.timestamp:.2f}")
        else:
            print(f"🎯 HIGH CONFIDENCE N1 DETECTED at {sleep_stage_data.timestamp:.2f} (could not play sound)")