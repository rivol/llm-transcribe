"""Tests for timestamp utilities."""

import pytest
from src.llm_transcribe.timestamp_utils import (
    format_timestamp,
    parse_timestamp,
    parse_timestamp_from_text,
    seconds_to_duration_str
)


class TestFormatTimestamp:
    """Test timestamp formatting."""
    
    def test_format_zero_seconds(self):
        """Test formatting zero seconds."""
        assert format_timestamp(0.0) == "[00:00:00]"
    
    def test_format_seconds_only(self):
        """Test formatting seconds only."""
        assert format_timestamp(45.0) == "[00:00:45]"
    
    def test_format_minutes_and_seconds(self):
        """Test formatting minutes and seconds."""
        assert format_timestamp(125.0) == "[00:02:05]"
    
    def test_format_hours_minutes_seconds(self):
        """Test formatting hours, minutes, and seconds."""
        assert format_timestamp(3665.0) == "[01:01:05]"
    
    def test_format_large_time(self):
        """Test formatting large time values."""
        assert format_timestamp(36125.0) == "[10:02:05]"
    
    def test_format_fractional_seconds(self):
        """Test that fractional seconds are truncated."""
        assert format_timestamp(65.7) == "[00:01:05]"


class TestParseTimestamp:
    """Test timestamp parsing."""
    
    def test_parse_zero_timestamp(self):
        """Test parsing zero timestamp."""
        assert parse_timestamp("[00:00:00]") == 0.0
    
    def test_parse_seconds_only(self):
        """Test parsing seconds only."""
        assert parse_timestamp("[00:00:45]") == 45.0
    
    def test_parse_minutes_and_seconds(self):
        """Test parsing minutes and seconds."""
        assert parse_timestamp("[00:02:05]") == 125.0
    
    def test_parse_hours_minutes_seconds(self):
        """Test parsing hours, minutes, and seconds."""
        assert parse_timestamp("[01:01:05]") == 3665.0
    
    def test_parse_large_time(self):
        """Test parsing large time values."""
        assert parse_timestamp("[10:02:05]") == 36125.0
    
    def test_parse_no_brackets(self):
        """Test parsing without brackets."""
        assert parse_timestamp("01:01:05") == 3665.0
    
    def test_parse_with_whitespace(self):
        """Test parsing with whitespace."""
        assert parse_timestamp("  [01:01:05]  ") == 3665.0
    
    def test_parse_invalid_format(self):
        """Test parsing invalid format raises ValueError."""
        with pytest.raises(ValueError):
            parse_timestamp("invalid")
    
    def test_parse_invalid_time_values(self):
        """Test parsing invalid time values raises ValueError."""
        with pytest.raises(ValueError):
            parse_timestamp("[00:60:00]")  # 60 minutes
        
        with pytest.raises(ValueError):
            parse_timestamp("[00:00:60]")  # 60 seconds
    
    def test_parse_non_numeric(self):
        """Test parsing non-numeric values raises ValueError."""
        with pytest.raises(ValueError):
            parse_timestamp("[aa:bb:cc]")


class TestParseTimestampFromText:
    """Test timestamp extraction from text."""
    
    def test_extract_timestamp_from_transcription_line(self):
        """Test extracting timestamp from transcription line."""
        text = "[00:01:30] Alice: Hello everyone"
        assert parse_timestamp_from_text(text) == 90.0
    
    def test_extract_timestamp_from_middle_of_text(self):
        """Test extracting timestamp from middle of text."""
        text = "Some text [00:02:15] more text"
        assert parse_timestamp_from_text(text) == 135.0
    
    def test_extract_first_timestamp_when_multiple(self):
        """Test extracting first timestamp when multiple exist."""
        text = "[00:01:00] Alice: Hello [00:02:00] Bob: Hi"
        assert parse_timestamp_from_text(text) == 60.0
    
    def test_no_timestamp_found(self):
        """Test when no timestamp is found."""
        text = "No timestamp here"
        assert parse_timestamp_from_text(text) is None
    
    def test_invalid_timestamp_format(self):
        """Test invalid timestamp format is ignored."""
        text = "[1:2:3] Alice: Hello"  # Single digits
        assert parse_timestamp_from_text(text) is None
    
    def test_invalid_time_values_ignored(self):
        """Test invalid time values are ignored."""
        text = "[00:60:00] Alice: Hello"  # 60 minutes
        assert parse_timestamp_from_text(text) is None


class TestSecondsToIntelligibleDuration:
    """Test human-readable duration formatting."""
    
    def test_zero_seconds(self):
        """Test zero seconds."""
        assert seconds_to_duration_str(0.0) == "0s"
    
    def test_seconds_only(self):
        """Test seconds only."""
        assert seconds_to_duration_str(45.0) == "45s"
    
    def test_minutes_and_seconds(self):
        """Test minutes and seconds."""
        assert seconds_to_duration_str(125.0) == "2m 5s"
    
    def test_minutes_only(self):
        """Test minutes only (no seconds)."""
        assert seconds_to_duration_str(120.0) == "2m"
    
    def test_hours_minutes_seconds(self):
        """Test hours, minutes, and seconds."""
        assert seconds_to_duration_str(3665.0) == "1h 1m 5s"
    
    def test_hours_only(self):
        """Test hours only."""
        assert seconds_to_duration_str(3600.0) == "1h"
    
    def test_hours_and_seconds(self):
        """Test hours and seconds (no minutes)."""
        assert seconds_to_duration_str(3605.0) == "1h 5s"
    
    def test_large_time(self):
        """Test large time values."""
        assert seconds_to_duration_str(36125.0) == "10h 2m 5s"


class TestTimestampUtilsRoundTrip:
    """Test round-trip conversion between formats."""
    
    def test_format_parse_roundtrip(self):
        """Test that format->parse gives original value."""
        original = 3665.0
        formatted = format_timestamp(original)
        parsed = parse_timestamp(formatted)
        assert parsed == original
    
    def test_parse_format_roundtrip(self):
        """Test that parse->format gives original string."""
        original = "[01:01:05]"
        parsed = parse_timestamp(original)
        formatted = format_timestamp(parsed)
        assert formatted == original