from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class ChannelConfig:
    label: str
    location: str
    channel_type: str  # 'EEG', 'EOG', 'EMG', 'OTHER'
    board: str  # 'CYTON_DAISY' or 'CYTON' or 'GANGLION'
    channel_number: int  # 1-16

class Montage:
    def __init__(self):
        self.channels: Dict[int, ChannelConfig] = {}
        self.reference = "Forehead"  # SRB
        self.ground = "Mastoid"
        self.bias = "Ground"
        
    def add_channel(self, channel_number: int, config: ChannelConfig):
        """Add a channel configuration"""
        self.channels[channel_number] = config
        
    @classmethod
    def default_sleep_montage(cls) -> 'Montage':
        """Create the default sleep montage configuration"""
        montage = cls()
        
        # EEG Channels (Cyton Board)
        eeg_channels = [
            ("F3", "Frontal Left"),
            ("F4", "Frontal Right"),
            ("C3", "Central Left"),
            ("C4", "Central Right"),
            ("O1", "Occipital Left"),
            ("O2", "Occipital Right"),
            ("T5", "Temporal Left"),
            ("T6", "Temporal Right")
        ]
        
        for i, (label, location) in enumerate(eeg_channels, 1):
            montage.add_channel(i, ChannelConfig(
                label=label,
                location=location,
                channel_type="EEG",
                board="CYTON_DAISY",
                channel_number=i
            ))
        
        # EOG Channels (Daisy Board)
        eog_channels = [
            ("T-EOG", "Top EOG"),
            ("B-EOG", "Bottom EOG"),
            ("R-HEOG", "Right Horizontal EOG"),
            ("L-HEOG", "Left Horizontal EOG")
        ]
        
        for i, (label, description) in enumerate(eog_channels, 9):
            montage.add_channel(i, ChannelConfig(
                label=label,
                location=description,
                channel_type="EOG",
                board="DAISY",
                channel_number=i
            ))
        
        # EMG and Other Channels (Daisy Board)
        other_channels = [
            ("EMG1", "Chin EMG", "EMG", (10, 100)),
            ("EMG2", "Leg EMG", "EMG", (10, 100)),
            ("Airflow", "Nasal Airflow", "OTHER", (0.1, 50)),
            ("Snoring", "Snoring", "OTHER", (10, 100))
        ]
        
        for i, (label, description, ch_type) in enumerate([("EMG1", "Chin EMG", "EMG"), ("EMG2", "Leg EMG", "EMG"), ("Airflow", "Nasal Airflow", "OTHER"), ("Snoring", "Snoring", "OTHER")], 13):
            montage.add_channel(i, ChannelConfig(
                label=label,
                location=description,
                channel_type=ch_type,
                board="DAISY",
                channel_number=i
            ))
        
        return montage
    
    @staticmethod
    def minimal_sleep_montage() -> 'Montage':
        """Create a minimal sleep montage without temporal channels and only lateral EOG.
        
        This montage includes:
        - EEG: F3, F4, C3, C4, O1, O2
        - EOG: R-LEOG, L-LEOG
        - EMG: EMG1, EMG2
        
        Returns:
            Montage: A montage with the specified channels
        """
        montage = Montage()
        
        # Add EEG channels with fixed indices
        eeg_channels = [
            (1, "F3", "Frontal Left"),
            (2, "F4", "Frontal Right"),
            (3, "C3", "Central Left"),
            (4, "C4", "Central Right"),
            (5, "O1", "Occipital Left"),
            (6, "O2", "Occipital Right")
        ]
        
        for channel_number, label, location in eeg_channels:
            montage.add_channel(channel_number, ChannelConfig(
                label=label,
                location=location,
                channel_type="EEG",
                board="CYTON_DAISY",
                channel_number=channel_number
            ))
        
        # Add EOG channels with fixed indices
        eog_channels = [
            (11, "R-LEOG", "Right Lateral EOG"),
            (12, "L-LEOG", "Left Lateral EOG")
        ]
        
        for channel_number, label, location in eog_channels:
            montage.add_channel(channel_number, ChannelConfig(
                label=label,
                location=location,
                channel_type="EOG",
                board="DAISY",
                channel_number=channel_number
            ))
            
        # Add EMG channels with fixed indices
        emg_channels = [
            (13, "EMG1", "Chin EMG 1"),
            (14, "EMG2", "Chin EMG 2")
        ]
        
        for channel_number, label, location in emg_channels:
            montage.add_channel(channel_number, ChannelConfig(
                label=label,
                location=location,
                channel_type="EMG",
                board="DAISY",
                channel_number=channel_number
            ))
        
        return montage
    
    @staticmethod
    def eog_only_montage() -> 'Montage':
        """Create an EOG-only montage with just lateral EOG channels.
        
        This montage includes only:
        - EOG: R-LEOG (channel 11), L-LEOG (channel 12)
        
        Returns:
            Montage: A montage with only EOG channels 11 and 12
        """
        montage = Montage()
        
        # Add EOG channels with fixed indices (channels 11 and 12 from full montage)
        eog_channels = [
            (11, "R-LEOG", "Right Lateral EOG"),
            (12, "L-LEOG", "Left Lateral EOG")
        ]
        
        for channel_number, label, location in eog_channels:
            montage.add_channel(channel_number, ChannelConfig(
                label=label,
                location=location,
                channel_type="EOG",
                board="DAISY",
                channel_number=channel_number
            ))
        
        return montage
    
    @staticmethod
    def all_channels_montage() -> 'Montage':
        """Create a montage that displays all 16 channels from the Cyton+Daisy board.
        
        This montage includes all channels 1-16 for debugging and data exploration.
        Channel types are set based on typical OpenBCI Cyton+Daisy configuration.
        
        Returns:
            Montage: A montage with all 16 channels
        """
        montage = Montage()
        
        # Add all 16 channels with appropriate labels and types
        all_channels = [
            (1, "CH1", "EEG"),
            (2, "CH2", "EEG"), 
            (3, "CH3", "EEG"),
            (4, "CH4", "EEG"),
            (5, "CH5", "EEG"),
            (6, "CH6", "EEG"),
            (7, "CH7", "EEG"),
            (8, "CH8", "EEG"),
            (9, "CH9", "EOG"),
            (10, "CH10", "EOG"),
            (11, "CH11", "EOG"),
            (12, "CH12", "EOG"),
            (13, "CH13", "EMG"),
            (14, "CH14", "EMG"),
            (15, "CH15", "OTHER"),
            (16, "CH16", "OTHER")
        ]
        
        for channel_number, label, channel_type in all_channels:
            # Use appropriate board designation
            board = "CYTON_DAISY" if channel_number <= 8 else "DAISY"
            
            montage.add_channel(channel_number, ChannelConfig(
                label=label,
                location=f"Channel {channel_number}",
                channel_type=channel_type,
                board=board,
                channel_number=channel_number
            ))
        
        return montage
    
    def get_channel_labels(self) -> List[str]:
        """Get list of channel labels in order"""
        return [self.channels[i].label for i in sorted(self.channels.keys())]
    
    def get_channel_types(self) -> List[str]:
        """Get list of channel types in order"""
        return [self.channels[i].channel_type for i in sorted(self.channels.keys())]
    
    def get_channel_boards(self) -> List[str]:
        """Get list of boards for each channel in order"""
        return [self.channels[i].board for i in sorted(self.channels.keys())]
    
    def get_electrode_channel_indices(self) -> List[int]:
        """Get 0-based electrode indices for extracting montage channels from epoch_data.
        
        This method returns the 0-based indices needed to extract the correct channels
        from the full epoch_data array (which contains all 16 board channels).
        
        For example:
        - EOG-only montage (channels 11, 12) returns [10, 11]
        - Minimal montage (channels 1,2,3,4,5,6,11,13,14) returns [0,1,2,3,4,5,10,12,13]
        
        Returns:
            List[int]: 0-based electrode indices to extract from epoch_data array
        """
        sorted_channel_numbers = sorted(self.channels.keys())
        return [ch_num - 1 for ch_num in sorted_channel_numbers]
    
    def convert_electrode_indices_to_montage_indices(self, electrode_indices: List[int]) -> List[int]:
        """Convert electrode channel mapping indices to montage indices.
        
        Args:
            electrode_indices: List of electrode channel mapping indices (0-based, where 0 = channel 1)
            
        Returns:
            List[int]: List of montage indices (positions in the montage's ordered channel list)
        """
        # First convert electrode indices to channel numbers
        channel_numbers = [idx + 1 for idx in electrode_indices]
        
        # Then convert channel numbers to montage indices
        sorted_channel_numbers = sorted(self.channels.keys())
        montage_indices = []
        
        for channel_num in channel_numbers:
            if channel_num in sorted_channel_numbers:
                montage_idx = sorted_channel_numbers.index(channel_num)
                montage_indices.append(montage_idx)
            else:
                raise ValueError(f"Channel {channel_num} does not exist in current montage")
                
        return montage_indices
    
    def validate_channel_indices_combination_types(self, eeg_combination_indices: List[int], eog_combination_indices: List[int]) -> None:
        """Validate that electrode channel mapping indices correspond to expected channel types in the montage.
        
        This ensures that the electrode channel mapping indices (where index 0 = channel 1, index 10 = channel 11, etc.)
        actually point to the right channel types (EEG/EOG) in the current montage configuration.
        
        Args:
            eeg_combination_indices: Electrode channel mapping indices that should point to EEG channels (e.g., [0, 1, 2] for channels 1, 2, 3)
            eog_combination_indices: Electrode channel mapping indices that should point to EOG channels (e.g., [10] for channel 11)
            
        Raises:
            ValueError: If any combination index doesn't match the expected channel type
        """
        # Convert electrode channel mapping indices to montage indices
        eeg_montage_indices = self.convert_electrode_indices_to_montage_indices(eeg_combination_indices)
        eog_montage_indices = self.convert_electrode_indices_to_montage_indices(eog_combination_indices)
        
        # Get ordered channel info for validation
        channel_types = self.get_channel_types()
        channel_labels = self.get_channel_labels()
        sorted_channel_numbers = sorted(self.channels.keys())
        
        # Check EEG combination indices
        for i, montage_idx in enumerate(eeg_montage_indices):
            electrode_idx = eeg_combination_indices[i]
            channel_num = sorted_channel_numbers[montage_idx]
            
            if channel_types[montage_idx] != 'EEG':
                raise ValueError(f"Channel {channel_num} (from electrode index {electrode_idx}, label '{channel_labels[montage_idx]}') is not an EEG channel in current montage. "
                               f"Channel type: {channel_types[montage_idx]}")
        
        # Check EOG combination indices  
        for i, montage_idx in enumerate(eog_montage_indices):
            electrode_idx = eog_combination_indices[i]
            channel_num = sorted_channel_numbers[montage_idx]
            
            if channel_types[montage_idx] != 'EOG':
                raise ValueError(f"Channel {channel_num} (from electrode index {electrode_idx}, label '{channel_labels[montage_idx]}') is not an EOG channel in current montage. "
                               f"Channel type: {channel_types[montage_idx]}") 