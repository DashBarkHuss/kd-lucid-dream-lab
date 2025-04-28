import datetime
import pytz

def format_timestamp_to_hawaii(timestamp):
    """Convert a Unix timestamp to Hawaii time string"""
    hawaii_tz = pytz.timezone('Pacific/Honolulu')
    utc_dt = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
    hawaii_dt = utc_dt.astimezone(hawaii_tz)
    return hawaii_dt.strftime('%Y-%m-%d %I:%M:%S.%f %p HST')

def format_elapsed_time(seconds):
    """Format elapsed time as HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}" 