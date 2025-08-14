"""
Session Management Utilities

This module provides utilities for managing EEG recording sessions,
including timestamped file naming, session data saving, and signal handling.
"""

import logging
import signal
import atexit
import sys
from datetime import datetime

logger = logging.getLogger(__name__)

def generate_session_timestamp():
    """Generate a timestamp string for session identification.
    
    Returns:
        str: Timestamp in YYYY-MM-DD_HH-MM-SS format
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_session_csv_files(data_handler, session_timestamp):
    """Save and merge CSV files for the current session.
    
    Args:
        data_handler: The ReceivedStreamedDataHandler instance
        session_timestamp (str): Session timestamp for unique file naming
        
    Raises:
        Exception: If CSV saving/merging fails
    """
    if data_handler is None:
        raise ValueError("No data handler provided")
    
    # Generate timestamped filenames
    main_csv_filename = f'data_{session_timestamp}.csv'
    merged_csv_filename = f'merged_data_{session_timestamp}.csv'
    
    # Save and merge using the CSV manager
    # The CSV manager itself handles duplicate call protection via _session_finalized flag
    data_handler.data_manager.csv_manager.save_all_and_cleanup(
        output_path=main_csv_filename,
        merge_files=True, 
        merge_output_path=merged_csv_filename
    )
    
    logger.info(f"✅ CSV files saved: {main_csv_filename}, merged: {merged_csv_filename}")


def get_session_filenames(session_timestamp, merge_prefix="merged_data"):
    """Get the expected filenames for a session.
    
    Args:
        session_timestamp (str): Session timestamp
        merge_prefix (str): Prefix for merged file
        
    Returns:
        dict: Dictionary with 'main', 'sleep_stages', and 'merged' filenames
    """
    return {
        'main': f'data_{session_timestamp}.csv',
        'sleep_stages': f'sleep_stages_{session_timestamp}.csv',
        'merged': f'{merge_prefix}_{session_timestamp}.csv'
    }


def cleanup_and_merge_handler(data_handler, session_timestamp, 
                             streaming_board_manager=None, signal_num=None, frame=None):
    """Handle cleanup and CSV merging when script is terminated.
    
    Args:
        data_handler: The ReceivedStreamedDataHandler instance
        session_timestamp (str): Session timestamp for file naming
        merge_prefix (str): Prefix for merged output file
        streaming_board_manager: Optional streaming board manager to stop
        signal_num: Signal number if called from signal handler
        frame: Signal frame if called from signal handler
    """
    if signal_num is not None:
        logger.info(f"Received termination signal: {signal_num}. Saving and merging CSV files...")
    else:
        logger.info("Script terminating. Saving and merging CSV files...")
    
    try:
        if data_handler is not None:
            # Use the shared utility function to save and merge CSV files
            save_session_csv_files(data_handler, session_timestamp)
            data_handler.data_manager.cleanup()
            logger.info("✅ CSV files saved and merged successfully")
        else:
            logger.warning("No data handler found - cleanup skipped")
            
        if streaming_board_manager is not None:
            streaming_board_manager.stop_stream()
            logger.info("✅ Streaming board stopped")
            
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")
        # Still try basic cleanup
        try:
            if data_handler is not None:
                data_handler.data_manager.cleanup()
            if streaming_board_manager is not None:
                streaming_board_manager.stop_stream()
        except:
            pass
    finally:
        if signal_num is not None:
            sys.exit(0)


def setup_signal_handlers(data_handler, session_timestamp, 
                         streaming_board_manager=None):
    """Register signal handlers for graceful termination.
    
    Args:
        data_handler: The ReceivedStreamedDataHandler instance
        session_timestamp (str): Session timestamp for file naming
        merge_prefix (str): Prefix for merged output file
        streaming_board_manager: Optional streaming board manager to stop
    """
    def signal_handler(signal_num=None, frame=None):
        cleanup_and_merge_handler(
            data_handler,
            session_timestamp,
            streaming_board_manager,
            signal_num,
            frame
        )
    
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Kill command
    atexit.register(signal_handler)                # Normal Python exit
    logger.info("✅ Signal handlers registered for graceful CSV merging")