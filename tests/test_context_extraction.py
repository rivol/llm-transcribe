"""Tests for context extraction logic."""

import pytest
from src.llm_transcribe.models import TranscriptionLine, TranscriptionResult


class TestContextExtraction:
    """Test context extraction from transcription results."""
    
    def test_get_last_minute_context_basic(self):
        """Test basic context extraction from last minute."""
        # Create lines spanning 10 minutes (chunk 0: 0-10 min)
        lines = [
            TranscriptionLine(timestamp=30.0, speaker="Alice", text="Early statement"),    # 0:30
            TranscriptionLine(timestamp=300.0, speaker="Bob", text="Middle statement"),    # 5:00
            TranscriptionLine(timestamp=540.0, speaker="Alice", text="Late statement"),    # 9:00
            TranscriptionLine(timestamp=570.0, speaker="Bob", text="Very late statement"), # 9:30
        ]
        
        result = TranscriptionResult(
            chunk_index=0,
            lines=lines,
            raw_response="mock response",
            model_used="test-model",
            processing_time_seconds=1.0
        )
        
        # For chunk 0, last minute starts at 9:00 (540 seconds)
        context = result.get_last_minute_context(chunk_duration_minutes=10)
        
        # Should include lines from 9:00 onwards
        expected_lines = [
            "[00:09:00] Alice: Late statement",
            "[00:09:30] Bob: Very late statement"
        ]
        assert context == "\n".join(expected_lines)
    
    def test_get_last_minute_context_chunk_1(self):
        """Test context extraction for chunk 1 (with proper offset calculation)."""
        # Create lines for chunk 1 (starts at 9 minutes, covers 9-19 minutes)
        lines = [
            TranscriptionLine(timestamp=540.0, speaker="Alice", text="Start of chunk"),     # 9:00
            TranscriptionLine(timestamp=600.0, speaker="Bob", text="Middle of chunk"),     # 10:00
            TranscriptionLine(timestamp=1080.0, speaker="Alice", text="Late in chunk"),    # 18:00
            TranscriptionLine(timestamp=1110.0, speaker="Bob", text="Very late in chunk"), # 18:30
        ]
        
        result = TranscriptionResult(
            chunk_index=1,
            lines=lines,
            raw_response="mock response",
            model_used="test-model",
            processing_time_seconds=1.0
        )
        
        # For chunk 1, last minute starts at 18:00 (1080 seconds)
        # chunk_start_minutes = 1 * (10 - 1) = 9 minutes
        # last_minute_start = 9 + 10 - 1 = 18 minutes = 1080 seconds
        context = result.get_last_minute_context(chunk_duration_minutes=10)
        
        # Should include lines from 18:00 onwards
        expected_lines = [
            "[00:18:00] Alice: Late in chunk",
            "[00:18:30] Bob: Very late in chunk"
        ]
        assert context == "\n".join(expected_lines)
    
    def test_get_last_minute_context_chunk_2(self):
        """Test context extraction for chunk 2."""
        # Create lines for chunk 2 (starts at 18 minutes, covers 18-28 minutes)
        lines = [
            TranscriptionLine(timestamp=1080.0, speaker="Alice", text="Start of chunk"),     # 18:00
            TranscriptionLine(timestamp=1200.0, speaker="Bob", text="Middle of chunk"),     # 20:00
            TranscriptionLine(timestamp=1620.0, speaker="Alice", text="Late in chunk"),     # 27:00
            TranscriptionLine(timestamp=1650.0, speaker="Bob", text="Very late in chunk"),  # 27:30
        ]
        
        result = TranscriptionResult(
            chunk_index=2,
            lines=lines,
            raw_response="mock response",
            model_used="test-model",
            processing_time_seconds=1.0
        )
        
        # For chunk 2, last minute starts at 27:00 (1620 seconds)
        # chunk_start_minutes = 2 * (10 - 1) = 18 minutes
        # last_minute_start = 18 + 10 - 1 = 27 minutes = 1620 seconds
        context = result.get_last_minute_context(chunk_duration_minutes=10)
        
        # Should include lines from 27:00 onwards
        expected_lines = [
            "[00:27:00] Alice: Late in chunk",
            "[00:27:30] Bob: Very late in chunk"
        ]
        assert context == "\n".join(expected_lines)
    
    def test_get_last_minute_context_no_lines_in_range(self):
        """Test context extraction when no lines are in the last minute."""
        # Create lines that are all before the last minute
        lines = [
            TranscriptionLine(timestamp=30.0, speaker="Alice", text="Early statement"),   # 0:30
            TranscriptionLine(timestamp=300.0, speaker="Bob", text="Middle statement"),  # 5:00
            TranscriptionLine(timestamp=450.0, speaker="Alice", text="Late statement"),  # 7:30
        ]
        
        result = TranscriptionResult(
            chunk_index=0,
            lines=lines,
            raw_response="mock response",
            model_used="test-model",
            processing_time_seconds=1.0
        )
        
        # For chunk 0, last minute starts at 9:00 (540 seconds)
        context = result.get_last_minute_context(chunk_duration_minutes=10)
        
        # Should return empty string since no lines are in the last minute
        assert context == ""
    
    def test_get_last_minute_context_empty_lines(self):
        """Test context extraction with empty lines list."""
        result = TranscriptionResult(
            chunk_index=0,
            lines=[],
            raw_response="mock response",
            model_used="test-model",
            processing_time_seconds=1.0
        )
        
        context = result.get_last_minute_context(chunk_duration_minutes=10)
        assert context == ""
    
    def test_get_last_minute_context_all_lines_in_range(self):
        """Test context extraction when all lines are in the last minute."""
        # Create lines that are all in the last minute
        lines = [
            TranscriptionLine(timestamp=540.0, speaker="Alice", text="First statement"),  # 9:00
            TranscriptionLine(timestamp=555.0, speaker="Bob", text="Second statement"),   # 9:15
            TranscriptionLine(timestamp=570.0, speaker="Alice", text="Third statement"),  # 9:30
            TranscriptionLine(timestamp=585.0, speaker="Bob", text="Fourth statement"),   # 9:45
        ]
        
        result = TranscriptionResult(
            chunk_index=0,
            lines=lines,
            raw_response="mock response",
            model_used="test-model",
            processing_time_seconds=1.0
        )
        
        context = result.get_last_minute_context(chunk_duration_minutes=10)
        
        # Should include all lines
        expected_lines = [
            "[00:09:00] Alice: First statement",
            "[00:09:15] Bob: Second statement",
            "[00:09:30] Alice: Third statement",
            "[00:09:45] Bob: Fourth statement"
        ]
        assert context == "\n".join(expected_lines)
    
    def test_get_last_minute_context_different_chunk_duration(self):
        """Test context extraction with different chunk duration."""
        # Create lines for a 5-minute chunk
        lines = [
            TranscriptionLine(timestamp=30.0, speaker="Alice", text="Early statement"),   # 0:30
            TranscriptionLine(timestamp=150.0, speaker="Bob", text="Middle statement"),  # 2:30
            TranscriptionLine(timestamp=240.0, speaker="Alice", text="Late statement"),  # 4:00
            TranscriptionLine(timestamp=270.0, speaker="Bob", text="Very late statement"), # 4:30
        ]
        
        result = TranscriptionResult(
            chunk_index=0,
            lines=lines,
            raw_response="mock response",
            model_used="test-model",
            processing_time_seconds=1.0
        )
        
        # For 5-minute chunk, last minute starts at 4:00 (240 seconds)
        context = result.get_last_minute_context(chunk_duration_minutes=5)
        
        # Should include lines from 4:00 onwards
        expected_lines = [
            "[00:04:00] Alice: Late statement",
            "[00:04:30] Bob: Very late statement"
        ]
        assert context == "\n".join(expected_lines)
    
    def test_get_last_minute_context_edge_case_exact_boundary(self):
        """Test context extraction with timestamp exactly at boundary."""
        # Create lines with one exactly at the boundary
        lines = [
            TranscriptionLine(timestamp=530.0, speaker="Alice", text="Just before"),   # 8:50
            TranscriptionLine(timestamp=540.0, speaker="Bob", text="Exactly at boundary"), # 9:00
            TranscriptionLine(timestamp=550.0, speaker="Alice", text="Just after"),   # 9:10
        ]
        
        result = TranscriptionResult(
            chunk_index=0,
            lines=lines,
            raw_response="mock response",
            model_used="test-model",
            processing_time_seconds=1.0
        )
        
        context = result.get_last_minute_context(chunk_duration_minutes=10)
        
        # Should include the line exactly at boundary and after
        expected_lines = [
            "[00:09:00] Bob: Exactly at boundary",
            "[00:09:10] Alice: Just after"
        ]
        assert context == "\n".join(expected_lines)
    
    def test_get_last_minute_context_overlap_scenario(self):
        """Test context extraction in overlap scenario between chunks."""
        # This tests the scenario where context from chunk N is used for chunk N+1
        
        # Chunk 0 result (0-10 minutes)
        chunk0_lines = [
            TranscriptionLine(timestamp=520.0, speaker="Alice", text="Before context"),  # 8:40
            TranscriptionLine(timestamp=540.0, speaker="Bob", text="Context line 1"),    # 9:00
            TranscriptionLine(timestamp=570.0, speaker="Alice", text="Context line 2"),  # 9:30
            TranscriptionLine(timestamp=590.0, speaker="Bob", text="Context line 3"),    # 9:50
        ]
        
        chunk0_result = TranscriptionResult(
            chunk_index=0,
            lines=chunk0_lines,
            raw_response="mock response",
            model_used="test-model",
            processing_time_seconds=1.0
        )
        
        # Get context from chunk 0 (should be lines from 9:00 onwards)
        context = chunk0_result.get_last_minute_context(chunk_duration_minutes=10)
        
        # This context would be used for chunk 1 (which starts at 9:00)
        # The context should help with speaker identification continuity
        expected_lines = [
            "[00:09:00] Bob: Context line 1",
            "[00:09:30] Alice: Context line 2",
            "[00:09:50] Bob: Context line 3"
        ]
        assert context == "\n".join(expected_lines)
        
        # This context would be passed to the LLM for chunk 1 processing
        # and then converted to relative timestamps (starting from [00:00:00])
        assert "Bob: Context line 1" in context
        assert "Alice: Context line 2" in context
        assert "Bob: Context line 3" in context