"""Tests for audio chunking functionality - basic validation tests only."""

import pytest
from src.llm_transcribe.models import ChunkData


class TestAudioChunking:
    """Test audio chunking logic - basic validation only."""
    
    def test_chunk_data_properties(self):
        """Test ChunkData model properties."""
        chunk = ChunkData(
            chunk_index=1,
            start_time_seconds=540.0,
            end_time_seconds=1140.0,
            audio_segment=None
        )
        
        assert chunk.chunk_index == 1
        assert chunk.start_time_seconds == 540.0
        assert chunk.end_time_seconds == 1140.0
        assert chunk.audio_segment is None
        
        # Test duration calculation
        assert chunk.end_time_seconds - chunk.start_time_seconds == 600.0  # 10 minutes
    
    def test_chunk_data_validation(self):
        """Test ChunkData validation."""
        # Test that end time must be greater than start time
        with pytest.raises(ValueError):
            ChunkData(
                chunk_index=0,
                start_time_seconds=600.0,
                end_time_seconds=300.0,  # End before start
                audio_segment=None
            )
        
        # Test that times must be non-negative
        with pytest.raises(ValueError):
            ChunkData(
                chunk_index=0,
                start_time_seconds=-10.0,  # Negative start time
                end_time_seconds=300.0,
                audio_segment=None
            )