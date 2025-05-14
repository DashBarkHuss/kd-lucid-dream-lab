"""Core functionality for real-time EEG data processing."""

from .stream_manager import StreamManager
from .gap_handler import GapHandler

__all__ = ['StreamManager', 'GapHandler'] 