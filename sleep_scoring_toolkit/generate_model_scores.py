#!/usr/bin/env python3
"""
Generate complete sleep stage predictions from EEG CSV files.

This script provides fast, headless batch processing that generates sleep stage
predictions from BrainFlow CSV files without GUI and real-time overhead.

Usage:
    python generate_model_scores.py --input data.csv --output results.csv
    python generate_model_scores.py --input data.csv --output results.csv --montage minimal_sleep_montage --quiet
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from brainflow.board_shim import BoardShim, BoardIds
from sleep_scoring_toolkit.batch_processor import BatchProcessor
from sleep_scoring_toolkit.montage import Montage


def setup_logging(quiet: bool = False):
    """Setup logging configuration."""
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Suppress verbose MNE output during batch processing
    logging.getLogger('mne').setLevel(logging.WARNING)


def get_montage_by_name(montage_name: str):
    """Get montage instance by name."""
    montage_factory_map = {
        'default_sleep_montage': Montage.default_sleep_montage,
        'minimal_sleep_montage': Montage.minimal_sleep_montage,
        'eog_only_montage': Montage.eog_only_montage,
        'all_channels_montage': Montage.all_channels_montage
    }
    
    if montage_name not in montage_factory_map:
        available = list(montage_factory_map.keys())
        raise ValueError(f"Unknown montage '{montage_name}'. Available: {available}")
    
    return montage_factory_map[montage_name]()


def get_board_id_from_string(board_id_str: str) -> int:
    """Convert board ID string to BoardIds constant."""
    # Only include boards that exist in this BrainFlow version
    board_name_to_id_map = {}
    
    # Add boards that definitely exist
    if hasattr(BoardIds, 'CYTON_DAISY_BOARD'):
        board_name_to_id_map['CYTON_DAISY_BOARD'] = BoardIds.CYTON_DAISY_BOARD
    if hasattr(BoardIds, 'CYTON_BOARD'):
        board_name_to_id_map['CYTON_BOARD'] = BoardIds.CYTON_BOARD
    if hasattr(BoardIds, 'GANGLION_BOARD'):
        board_name_to_id_map['GANGLION_BOARD'] = BoardIds.GANGLION_BOARD
    if hasattr(BoardIds, 'SYNTHETIC_BOARD'):
        board_name_to_id_map['SYNTHETIC_BOARD'] = BoardIds.SYNTHETIC_BOARD
    if hasattr(BoardIds, 'UNICORN_BOARD'):
        board_name_to_id_map['UNICORN_BOARD'] = BoardIds.UNICORN_BOARD
    
    if board_id_str not in board_name_to_id_map:
        available = list(board_name_to_id_map.keys())
        raise ValueError(f"Unknown board ID '{board_id_str}'. Available: {available}")
    
    return board_name_to_id_map[board_id_str]


def get_board_configuration(board_id_str: str) -> dict:
    """Get board configuration from BrainFlow BoardShim."""
    try:
        board_id = get_board_id_from_string(board_id_str)
        
        return {
            'sampling_rate': BoardShim.get_sampling_rate(board_id),
            'timestamp_column': BoardShim.get_timestamp_channel(board_id),
            'exg_columns': BoardShim.get_exg_channels(board_id)
        }
    except Exception as e:
        raise ValueError(f"Failed to get configuration for board '{board_id_str}': {e}")


def parse_channel_names(channels_str: str, param_name: str) -> list:
    """Parse comma-separated channel names to list of strings."""
    if not channels_str.strip():
        return []  # Allow empty for EOG-only or EEG-only analysis
    
    try:
        channel_names = [ch.strip() for ch in channels_str.split(',') if ch.strip()]
        # Basic validation - just check they're non-empty strings
        for name in channel_names:
            if not name:
                raise ValueError(f"{param_name} channel name cannot be empty")
        return channel_names
    except ValueError as e:
        raise ValueError(f"Invalid {param_name}: {e}")


def parse_column_indices(indices_str: str) -> List[int]:
    """Parse comma-separated column indices to list of integers."""
    try:
        return [int(idx.strip()) for idx in indices_str.split(',') if idx.strip()]
    except ValueError as e:
        raise ValueError(f"Invalid column indices '{indices_str}': {e}")


def validate_args(args):
    """Validate command line arguments."""
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    output_dir = Path(args.output).parent
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    # Parse channel names (validation happens later in montage)
    args.eeg_channels_for_scoring = parse_channel_names(args.eeg_channels_for_scoring, "EEG channels")
    args.eog_channels_for_scoring = parse_channel_names(args.eog_channels_for_scoring, "EOG channels")
    
    # Validate board configuration vs manual configuration
    if args.board_id:
        # Automatic configuration from BrainFlow board
        try:
            board_config = get_board_configuration(args.board_id)
            args.sampling_rate = args.sampling_rate or board_config['sampling_rate']
            args.timestamp_column = args.timestamp_column or board_config['timestamp_column']
            args.exg_columns = args.exg_columns or ','.join(map(str, board_config['exg_columns']))
        except Exception as e:
            raise ValueError(f"Failed to configure board '{args.board_id}': {e}")
    else:
        # Manual configuration - require all parameters
        if not args.sampling_rate:
            raise ValueError("--sampling-rate is required when not using --board-id")
        if args.timestamp_column is None:
            raise ValueError("--timestamp-column is required when not using --board-id")
        if not args.exg_columns:
            raise ValueError("--exg-columns is required when not using --board-id")
    
    # Parse EXG column indices
    args.exg_columns = parse_column_indices(args.exg_columns)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='Generate complete sleep stage predictions from EEG CSV file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Automatic configuration with BrainFlow board (recommended)
    python generate_model_scores.py --input data.csv --output results.csv \\
        --board-id CYTON_DAISY_BOARD --montage minimal_sleep_montage \\
        --eeg-channels "C4,C3,F3,F4" --eog-channels "L-HEOG,R-HEOG"
    
    # Manual configuration for custom/non-BrainFlow data
    python generate_model_scores.py --input data.csv --output results.csv \\
        --sampling-rate 125 --timestamp-column 30 --eeg-column-indices "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" \\
        --montage minimal_sleep_montage --eeg-channels "C4,C3,F3,F4" --eog-channels "L-HEOG,R-HEOG"
    
    # Override board settings (board config + manual overrides)
    python generate_model_scores.py --input data.csv --output results.csv \\
        --board-id CYTON_DAISY_BOARD --sampling-rate 250 \\
        --montage minimal_sleep_montage --eeg-channels "C4,C3,F3,F4" --eog-channels "L-HEOG,R-HEOG"
        """
    )
    
    parser.add_argument(
        '--input', 
        required=True, 
        help='Input EEG CSV file (BrainFlow format)'
    )
    parser.add_argument(
        '--output', 
        required=True, 
        help='Output sleep stages CSV file'
    )
    parser.add_argument(
        '--montage', 
        required=True,
        help='Montage configuration - specify which channels to use (e.g. minimal_sleep_montage, eog_only_montage)'
    )
    # Board configuration (automatic setup)
    parser.add_argument(
        '--board-id',
        type=str,
        help='BrainFlow board ID for automatic configuration (e.g. CYTON_DAISY_BOARD, GANGLION_BOARD)'
    )
    
    # Manual configuration (required if no board-id)
    parser.add_argument(
        '--sampling-rate', 
        type=int,
        help='Sampling rate (Hz) - required if not using --board-id'
    )
    parser.add_argument(
        '--timestamp-column',
        type=int,
        help='CSV column index containing Unix timestamps - required if not using --board-id'
    )
    parser.add_argument(
        '--exg-columns',
        type=str,
        help='CSV column indices containing all electrode data (EEG+EOG, comma-separated, e.g. "0,1,2,3,10,11") - required if not using --board-id'
    )
    
    # Channel names (always required)
    parser.add_argument(
        '--eeg-channels-for-scoring',
        required=True,
        help='EEG channel labels selected for sleep scoring (comma-separated channel names, e.g. "C4,C3,F3,F4")'
    )
    parser.add_argument(
        '--eog-channels-for-scoring',
        required=True,
        help='EOG channel labels selected for sleep scoring (comma-separated channel names, e.g. "L-HEOG,R-HEOG" for default_sleep_montage)'
    )
    parser.add_argument(
        '--quiet', 
        action='store_true', 
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.quiet)
    logger = logging.getLogger(__name__)

    try:
        # Validate arguments
        validate_args(args)
        
        # Get montage configuration
        try:
            montage = get_montage_by_name(args.montage)
        except Exception as e:
            logger.error(f"Failed to load montage '{args.montage}': {e}")
            sys.exit(1)

        # Initialize batch processor (validation happens in StatefulInferenceManager)
        logger.info(f"Initializing processor with montage: {args.montage}")
        logger.info(f"EEG channels for scoring: {args.eeg_channels_for_scoring}, EOG channels for scoring: {args.eog_channels_for_scoring}")
        processor = BatchProcessor(
            montage=montage, 
            sampling_rate=args.sampling_rate,
            eeg_channels_for_scoring=args.eeg_channels_for_scoring,
            eog_channels_for_scoring=args.eog_channels_for_scoring,
            timestamp_column=args.timestamp_column,
            exg_columns=args.exg_columns,
            show_progress=not args.quiet
        )

        # Process all epochs sequentially
        logger.info("Starting batch processing...")
        results = processor.process_csv_file(args.input, args.output)
        
        # Report final results
        if not args.quiet:
            print(f"\n✅ Processing completed successfully!")
            print(f"📊 Epochs processed: {results['epochs_processed']}")
            print(f"⏱️  Processing time: {results['processing_time']:.2f} seconds")
            print(f"🚀 Speed ratio: {results['speed_ratio']:.1f}x real-time")
            print(f"💾 Output saved to: {results['output_path']}")
            print(f"📏 Data duration: {results['data_duration']:.1f} seconds")

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()