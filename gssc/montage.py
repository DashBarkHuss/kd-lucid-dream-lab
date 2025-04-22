from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class ChannelConfig:
    label: str
    location: str
    filter_range: tuple[float, float]
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
                filter_range=(0.1, 100),
                channel_type="EEG",
                board="CYTON_DAISY",
                channel_number=i
            ))
        
        # EOG Channels (Daisy Board)
        eog_channels = [
            ("T-EOG", "Top EOG"),
            ("B-EOG", "Bottom EOG"),
            ("R-LEOG", "Right Lateral EOG"),
            ("L-LEOG", "Left Lateral EOG")
        ]
        
        for i, (label, description) in enumerate(eog_channels, 9):
            montage.add_channel(i, ChannelConfig(
                label=label,
                location=description,
                filter_range=(0.1, 100),
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
        
        for i, (label, description, ch_type, filter_range) in enumerate(other_channels, 13):
            montage.add_channel(i, ChannelConfig(
                label=label,
                location=description,
                filter_range=filter_range,
                channel_type=ch_type,
                board="DAISY",
                channel_number=i
            ))
        
        return montage
    
    def get_channel_labels(self) -> List[str]:
        """Get list of channel labels in order"""
        return [self.channels[i].label for i in sorted(self.channels.keys())]
    
    def get_channel_types(self) -> List[str]:
        """Get list of channel types in order"""
        return [self.channels[i].channel_type for i in sorted(self.channels.keys())]
    
    def get_filter_ranges(self) -> List[tuple[float, float]]:
        """Get list of filter ranges in order"""
        return [self.channels[i].filter_range for i in sorted(self.channels.keys())]
    
    def get_channel_boards(self) -> List[str]:
        """Get list of boards for each channel in order"""
        return [self.channels[i].board for i in sorted(self.channels.keys())] 