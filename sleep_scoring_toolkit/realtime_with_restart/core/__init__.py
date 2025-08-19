"""Core functionality for real-time EEG data processing."""

from .brainflow_child_process_manager import BrainFlowChildProcessManager
from .gap_handler import GapHandler

__all__ = ['BrainFlowChildProcessManager', 'GapHandler'] 