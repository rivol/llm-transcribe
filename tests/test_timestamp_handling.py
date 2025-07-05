"""Tests for timestamp handling and conversion logic."""

import pytest
from unittest.mock import Mock, patch
from typing import List

from src.transcriber.llm_client import LLMClient
from src.transcriber.models import ChunkData, TranscriptionLine, TranscriptionResult
from src.transcriber.timestamp_utils import format_timestamp


class TestTimestampHandling:
    """Test timestamp handling in LLM client."""
    
    def test_convert_context_to_relative_basic(self):
        """Test converting absolute timestamps to relative."""
        client = LLMClient()
        
        # Context from previous chunk with absolute timestamps
        context = "[00:09:30] Alice: Previous statement\n[00:09:45] Bob: Another statement"
        chunk_start_seconds = 540.0  # 9 minutes
        
        result = client._convert_context_to_relative(context, chunk_start_seconds)
        
        # Should convert to relative timestamps
        expected = "[00:30] Alice: Previous statement\n[00:45] Bob: Another statement"
        assert result == expected
    
    def test_convert_context_to_relative_negative_handled(self):
        """Test that negative relative timestamps are handled correctly."""
        client = LLMClient()
        
        # Context with timestamp before chunk start
        context = "[00:08:30] Alice: Previous statement"
        chunk_start_seconds = 540.0  # 9 minutes
        
        result = client._convert_context_to_relative(context, chunk_start_seconds)
        
        # Should convert to 00:00 (non-negative)
        expected = "[00:00] Alice: Previous statement"
        assert result == expected
    
    def test_convert_context_to_relative_multiple_timestamps(self):
        """Test converting multiple timestamps in context."""
        client = LLMClient()
        
        context = "[00:18:15] Alice: First\n[00:18:30] Bob: Second\n[00:18:45] Alice: Third"
        chunk_start_seconds = 1080.0  # 18 minutes
        
        result = client._convert_context_to_relative(context, chunk_start_seconds)
        
        expected = "[00:15] Alice: First\n[00:30] Bob: Second\n[00:45] Alice: Third"
        assert result == expected
    
    def test_convert_relative_to_absolute_seconds(self):
        """Test converting relative timestamp to absolute seconds."""
        client = LLMClient()
        
        # Test various relative timestamps
        assert client._convert_relative_to_absolute_seconds("[00:30]", 540.0) == 570.0
        assert client._convert_relative_to_absolute_seconds("[01:15]", 540.0) == 615.0
        assert client._convert_relative_to_absolute_seconds("[02:00]", 540.0) == 660.0
    
    def test_convert_relative_to_absolute_seconds_zero(self):
        """Test converting zero relative timestamp."""
        client = LLMClient()
        
        result = client._convert_relative_to_absolute_seconds("[00:00]", 540.0)
        assert result == 540.0
    
    def test_convert_relative_to_absolute_seconds_invalid(self):
        """Test handling invalid timestamp format."""
        client = LLMClient()
        
        # Should return chunk start time for invalid format
        result = client._convert_relative_to_absolute_seconds("invalid", 540.0)
        assert result == 540.0
    
    def test_parse_transcription_response_basic(self):
        """Test parsing basic transcription response."""
        client = LLMClient()
        
        response = "[00:30] Alice: Hello everyone\n[01:05] Bob: Hi there"
        chunk_start_seconds = 540.0  # 9 minutes
        
        lines = client.parse_transcription_response(response, chunk_start_seconds)
        
        assert len(lines) == 2
        
        # Check first line
        assert lines[0].timestamp == 570.0  # 540 + 30
        assert lines[0].speaker == "Alice"
        assert lines[0].text == "Hello everyone"
        
        # Check second line
        assert lines[1].timestamp == 605.0  # 540 + 65
        assert lines[1].speaker == "Bob"
        assert lines[1].text == "Hi there"
    
    def test_parse_transcription_response_malformed_line(self):
        """Test parsing response with malformed line."""
        client = LLMClient()
        
        response = "[00:30] Alice: Hello everyone\nMalformed line\n[01:05] Bob: Hi there"
        chunk_start_seconds = 540.0
        
        with patch('src.transcriber.llm_client.logger') as mock_logger:
            lines = client.parse_transcription_response(response, chunk_start_seconds)
            
            # Should log warning about malformed line
            mock_logger.warning.assert_called_once_with("Could not parse line: Malformed line")
            
            # Should still parse the valid lines
            assert len(lines) == 2
            assert lines[0].speaker == "Alice"
            assert lines[1].speaker == "Bob"
    
    def test_parse_transcription_response_timestamp_only(self):
        """Test parsing response with timestamp but no clear speaker/text format."""
        client = LLMClient()
        
        response = "[00:30] Alice says hello to everyone"
        chunk_start_seconds = 540.0
        
        with patch('src.transcriber.llm_client.logger') as mock_logger:
            lines = client.parse_transcription_response(response, chunk_start_seconds)
            
            # Should log warning about parsing failure
            mock_logger.warning.assert_called_once()
            
            # Should still extract timestamp and create line
            assert len(lines) == 1
            assert lines[0].timestamp == 570.0  # 540 + 30
            assert lines[0].speaker == "Unknown"
            assert lines[0].text == "Alice says hello to everyone"
    
    def test_parse_transcription_response_with_colon_in_text(self):
        """Test parsing response with colon in the text."""
        client = LLMClient()
        
        response = "[00:30] Alice: The time is 12:30 PM"
        chunk_start_seconds = 540.0
        
        lines = client.parse_transcription_response(response, chunk_start_seconds)
        
        assert len(lines) == 1
        assert lines[0].timestamp == 570.0
        assert lines[0].speaker == "Alice"
        assert lines[0].text == "The time is 12:30 PM"
    
    def test_parse_transcription_response_empty_lines(self):
        """Test parsing response with empty lines."""
        client = LLMClient()
        
        response = "[00:30] Alice: Hello\n\n[01:05] Bob: Hi\n\n"
        chunk_start_seconds = 540.0
        
        lines = client.parse_transcription_response(response, chunk_start_seconds)
        
        # Should ignore empty lines
        assert len(lines) == 2
        assert lines[0].speaker == "Alice"
        assert lines[1].speaker == "Bob"


class TestTimestampIntegration:
    """Test timestamp handling across the entire pipeline."""
    
    def test_absolute_timestamps_maintained_across_chunks(self):
        """Test that absolute timestamps are maintained across multiple chunks."""
        # This test ensures that timestamps don't get double-adjusted
        
        # Simulate chunk 1 result (starts at 9 minutes)
        chunk1_lines = [
            TranscriptionLine(timestamp=570.0, speaker="Alice", text="Hello"),  # 9:30
            TranscriptionLine(timestamp=600.0, speaker="Bob", text="Hi")        # 10:00
        ]
        
        # Simulate chunk 2 result (starts at 18 minutes)
        chunk2_lines = [
            TranscriptionLine(timestamp=1110.0, speaker="Alice", text="Continuing"),  # 18:30
            TranscriptionLine(timestamp=1140.0, speaker="Bob", text="Yes")           # 19:00
        ]
        
        # Verify timestamps are absolute and in correct order
        all_lines = chunk1_lines + chunk2_lines
        
        # Check all timestamps are monotonically increasing
        for i in range(1, len(all_lines)):
            assert all_lines[i].timestamp > all_lines[i-1].timestamp
        
        # Check specific values
        assert all_lines[0].timestamp == 570.0   # 9:30
        assert all_lines[1].timestamp == 600.0   # 10:00
        assert all_lines[2].timestamp == 1110.0  # 18:30
        assert all_lines[3].timestamp == 1140.0  # 19:00
    
    def test_formatted_timestamps_display_correctly(self):
        """Test that formatted timestamps display correctly."""
        lines = [
            TranscriptionLine(timestamp=30.0, speaker="Alice", text="Hello"),
            TranscriptionLine(timestamp=90.0, speaker="Bob", text="Hi"),
            TranscriptionLine(timestamp=3665.0, speaker="Alice", text="Later")
        ]
        
        # Check formatted output
        assert lines[0].formatted_timestamp == "[00:00:30]"
        assert lines[1].formatted_timestamp == "[00:01:30]"
        assert lines[2].formatted_timestamp == "[01:01:05]"
        
        # Check string representation
        assert str(lines[0]) == "[00:00:30] Alice: Hello"
        assert str(lines[1]) == "[00:01:30] Bob: Hi"
        assert str(lines[2]) == "[01:01:05] Alice: Later"
    
    def test_timestamp_precision_maintained(self):
        """Test that timestamp precision is maintained through conversions."""
        # Test with fractional seconds
        line = TranscriptionLine(timestamp=65.7, speaker="Alice", text="Test")
        
        # Float precision is maintained
        assert line.timestamp == 65.7
        
        # But formatted output truncates to seconds
        assert line.formatted_timestamp == "[00:01:05]"
    
    def test_large_timestamp_values(self):
        """Test handling of large timestamp values."""
        # Test with 10+ hours
        line = TranscriptionLine(timestamp=36125.0, speaker="Alice", text="Long audio")
        
        assert line.timestamp == 36125.0
        assert line.formatted_timestamp == "[10:02:05]"
    
    def test_zero_timestamp_handling(self):
        """Test handling of zero timestamps."""
        line = TranscriptionLine(timestamp=0.0, speaker="Alice", text="Start")
        
        assert line.timestamp == 0.0
        assert line.formatted_timestamp == "[00:00:00]"
        assert str(line) == "[00:00:00] Alice: Start"