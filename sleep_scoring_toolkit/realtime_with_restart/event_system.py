"""
Core event system for sleep stage experiments.

This module provides the core event-driven architecture for triggering experiments
based on real-time sleep stage detection.
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SleepStageEvent:
    """Data class representing a sleep stage detection event."""
    
    sleep_stage: int  # 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
    timestamp: float  # Epoch start timestamp
    confidence: float # Prediction confidence (0.0-1.0)
    class_probabilities: np.ndarray  # Array of probabilities for all classes
    epoch_data: Optional[Any] = None  # Raw epoch data if needed
    buffer_id: Optional[int] = None   # Buffer ID that processed this epoch
    
    @property
    def stage_text(self) -> str:
        """Get human-readable sleep stage text."""
        stage_map = {
            0: 'Wake',
            1: 'N1', 
            2: 'N2',
            3: 'N3',
            4: 'REM'
        }
        return stage_map.get(self.sleep_stage, 'Unknown')


# No need for handler classes - we'll use simple functions
# Functions should take a SleepStageEvent and return bool (optional)


class EventDispatcher:
    """Central event dispatcher that manages event delivery to handler functions."""
    
    def __init__(self):
        """Initialize the event dispatcher."""
        self.handlers: List[Callable] = []  # Simple list of handler functions
        self.enabled = True
        self.event_count = 0
        
    def register_handler(self, handler_function: Callable[[SleepStageEvent], Optional[bool]]):
        """Register a handler function for new epoch events.
        
        Args:
            handler_function: Function that takes SleepStageEvent and optionally returns bool
        """
        self.handlers.append(handler_function)
        logger.info(f"Registered handler function {handler_function.__name__}")
            
    def unregister_handler(self, handler_function: Callable):
        """Unregister a handler function.
        
        Args:
            handler_function: Function to unregister
        """
        if handler_function in self.handlers:
            self.handlers.remove(handler_function)
            logger.info(f"Unregistered handler function {handler_function.__name__}")
        else:
            logger.warning(f"Handler function {handler_function.__name__} not found")
        
    def emit_sleep_stage_event(self, sleep_stage: int, timestamp: float, class_probabilities: np.ndarray,
                              epoch_data: Optional[Any] = None, buffer_id: Optional[int] = None):
        """Emit a new epoch event to all registered handlers.
        
        Args:
            sleep_stage: Detected sleep stage (0-4)
            timestamp: Timestamp of the epoch
            class_probabilities: Array of class probabilities
            epoch_data: Optional raw epoch data
            buffer_id: Optional buffer ID that processed this epoch
        """
        if not self.enabled:
            return
            
        # Calculate confidence as max probability
        confidence = float(np.max(class_probabilities)) if len(class_probabilities) > 0 else 0.0
        
        # Create event object
        event = SleepStageEvent(
            sleep_stage=sleep_stage,
            timestamp=timestamp,
            confidence=confidence,
            class_probabilities=class_probabilities,
            epoch_data=epoch_data,
            buffer_id=buffer_id
        )
        
        self.event_count += 1
        logger.debug(f"Emitting sleep stage event: {event.stage_text} (confidence: {confidence:.2f})")
        
        # Send to all handlers
        for handler_func in self.handlers:
            try:
                handler_func(event)
            except Exception as e:
                logger.error(f"Error in handler {handler_func.__name__}: {e}")
                        
    def _get_stage_name(self, sleep_stage: int) -> str:
        """Convert sleep stage integer to name.
        
        Args:
            sleep_stage: Sleep stage integer (0-4)
            
        Returns:
            str: Sleep stage name
        """
        stage_map = {
            0: 'Wake',
            1: 'N1',
            2: 'N2', 
            3: 'N3',
            4: 'REM'
        }
        return stage_map.get(sleep_stage, 'Unknown')
        
    def enable(self):
        """Enable the event dispatcher."""
        self.enabled = True
        logger.info("Event dispatcher enabled")
        
    def disable(self):
        """Disable the event dispatcher."""
        self.enabled = False
        logger.info("Event dispatcher disabled")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get dispatcher statistics.
        
        Returns:
            dict: Statistics about events and handlers
        """
        handler_info = [handler_func.__name__ for handler_func in self.handlers]
                
        return {
            'enabled': self.enabled,
            'total_events': self.event_count,
            'registered_handlers': handler_info
        }
        
    def reset_stats(self):
        """Reset event statistics."""
        self.event_count = 0
        logger.info("Event dispatcher statistics reset")