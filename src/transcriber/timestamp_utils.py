"""Timestamp utilities for consistent time handling."""

import re
from typing import Optional


def format_timestamp(seconds: float) -> str:
    """Format seconds to [HH:MM:SS] format.
    
    Args:
        seconds: Time in seconds from start of audio
        
    Returns:
        Formatted timestamp string in [HH:MM:SS] format
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"


def parse_timestamp(timestamp: str) -> float:
    """Parse [HH:MM:SS] format to seconds.
    
    Args:
        timestamp: Timestamp string in [HH:MM:SS] format
        
    Returns:
        Time in seconds
        
    Raises:
        ValueError: If timestamp format is invalid
    """
    # Remove brackets and whitespace
    time_str = timestamp.strip().strip('[]')
    
    # Split on colons
    parts = time_str.split(':')
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp format: {timestamp}")
    
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        
        # Validate ranges
        if minutes >= 60 or seconds >= 60:
            raise ValueError(f"Invalid time values in timestamp: {timestamp}")
        
        return hours * 3600 + minutes * 60 + seconds
    except ValueError as e:
        raise ValueError(f"Invalid timestamp format: {timestamp}") from e


def parse_timestamp_from_text(text: str) -> Optional[float]:
    """Extract and parse the first timestamp found in text.
    
    Args:
        text: Text that may contain a timestamp in [HH:MM:SS] format
        
    Returns:
        Time in seconds if found, None otherwise
    """
    # Pattern to match [HH:MM:SS] format
    pattern = r'\[(\d{1,2}):(\d{2}):(\d{2})\]'
    match = re.search(pattern, text)
    
    if match:
        hours, minutes, seconds = map(int, match.groups())
        if minutes < 60 and seconds < 60:
            return hours * 3600 + minutes * 60 + seconds
    
    return None


def seconds_to_duration_str(seconds: float) -> str:
    """Convert seconds to a human-readable duration string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Duration string like "1h 23m 45s" or "23m 45s" or "45s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:  # Always show seconds if no other parts
        parts.append(f"{secs}s")
    
    return " ".join(parts)