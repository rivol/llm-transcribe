"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
import tempfile
import os

from src.llm_transcribe.models import Config, ChunkData, TranscriptionLine, TranscriptionResult


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return Config(
        chunk_duration_minutes=10,
        overlap_duration_minutes=1,
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=2000,
        verbose=False
    )


@pytest.fixture
def sample_chunks():
    """Create sample chunk data for testing."""
    return [
        ChunkData(
            chunk_index=0,
            start_time_seconds=0.0,
            end_time_seconds=600.0,  # 10 minutes
            audio_segment=None
        ),
        ChunkData(
            chunk_index=1,
            start_time_seconds=540.0,  # 9 minutes (1 minute overlap)
            end_time_seconds=1140.0,  # 19 minutes
            audio_segment=None
        ),
        ChunkData(
            chunk_index=2,
            start_time_seconds=1080.0,  # 18 minutes (1 minute overlap)
            end_time_seconds=1680.0,  # 28 minutes
            audio_segment=None
        )
    ]


@pytest.fixture
def sample_transcription_lines():
    """Create sample transcription lines for testing."""
    return [
        TranscriptionLine(timestamp=30.0, speaker="Alice", text="Hello everyone"),
        TranscriptionLine(timestamp=65.0, speaker="Bob", text="Hi there"),
        TranscriptionLine(timestamp=120.0, speaker="Alice", text="How are you doing?"),
        TranscriptionLine(timestamp=180.0, speaker="Charlie", text="I'm doing well, thanks")
    ]


@pytest.fixture
def sample_transcription_result():
    """Create a sample transcription result for testing."""
    lines = [
        TranscriptionLine(timestamp=30.0, speaker="Alice", text="Hello everyone"),
        TranscriptionLine(timestamp=65.0, speaker="Bob", text="Hi there"),
        TranscriptionLine(timestamp=120.0, speaker="Alice", text="How are you doing?")
    ]
    
    return TranscriptionResult(
        chunk_index=0,
        lines=lines,
        raw_response="[00:00:30] Alice: Hello everyone\n[00:01:05] Bob: Hi there\n[00:02:00] Alice: How are you doing?",
        model_used="gpt-4o-mini",
        processing_time_seconds=2.5
    )