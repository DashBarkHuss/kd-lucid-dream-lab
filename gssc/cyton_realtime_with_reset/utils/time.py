import pytz
from datetime import datetime

def format_timestamp_to_hawaii(ts):
    """Convert Unix timestamp to Hawaii time string"""
    hawaii_tz = pytz.timezone('Pacific/Honolulu')
    dt = datetime.fromtimestamp(ts, hawaii_tz)
    # Format with microseconds, then truncate to milliseconds, then add HST
    formatted = dt.strftime('%Y-%m-%d %I:%M:%S.%f')[:-3]
    return f"{formatted} {dt.strftime('%p')} HST"

def format_elapsed_time(seconds):
    """Format elapsed time in HH:MM:SS.mmm format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}" 