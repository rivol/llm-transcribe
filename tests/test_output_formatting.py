"""Tests for output formatting and deduplication."""

import pytest
from pathlib import Path
import json
import tempfile

from src.llm_transcribe.output import OutputHandler
from src.llm_transcribe.models import (
    Config, ChunkData, TranscriptionLine, TranscriptionResult, TranscriptionJob
)


class TestOutputFormatting:
    """Test output formatting functionality."""
    
    def test_format_transcription_basic(self, temp_dir):
        """Test basic transcription formatting."""
        handler = OutputHandler()
        
        # Create sample transcription results
        lines1 = [
            TranscriptionLine(timestamp=30.0, speaker="Alice", text="Hello everyone"),
            TranscriptionLine(timestamp=65.0, speaker="Bob", text="Hi there")
        ]
        
        lines2 = [
            TranscriptionLine(timestamp=570.0, speaker="Alice", text="Continuing conversation"),
            TranscriptionLine(timestamp=605.0, speaker="Bob", text="Yes indeed")
        ]
        
        result1 = TranscriptionResult(
            chunk_index=0,
            lines=lines1,
            raw_response="mock",
            model_used="test-model",
            processing_time_seconds=1.0
        )
        
        result2 = TranscriptionResult(
            chunk_index=1,
            lines=lines2,
            raw_response="mock",
            model_used="test-model",
            processing_time_seconds=1.0
        )
        
        # Create job with results using temp files
        config = Config(chunk_duration_minutes=10, overlap_duration_minutes=1)
        
        input_file = temp_dir / "test.mp3"
        output_file = temp_dir / "test.txt"
        input_file.touch()  # Create the file so validation passes
        
        job = TranscriptionJob(
            input_file=input_file,
            output_file=output_file,
            model="test-model",
            config=config,
            results=[result1, result2]
        )
        
        formatted = handler.format_transcription(job)
        
        expected_lines = [
            "[00:00:30] Alice: Hello everyone",
            "[00:01:05] Bob: Hi there",
            "[00:09:30] Alice: Continuing conversation",
            "[00:10:05] Bob: Yes indeed"
        ]
        
        assert formatted == "\n".join(expected_lines)
    
    def test_format_transcription_empty_results(self, temp_dir):
        """Test formatting with empty results."""
        handler = OutputHandler()
        
        config = Config(chunk_duration_minutes=10, overlap_duration_minutes=1)
        
        input_file = temp_dir / "test.mp3"
        output_file = temp_dir / "test.txt"
        input_file.touch()
        
        job = TranscriptionJob(
            input_file=input_file,
            output_file=output_file,
            model="test-model",
            config=config,
            results=[]
        )
        
        formatted = handler.format_transcription(job)
        assert formatted == ""
    
    def test_format_transcription_out_of_order_chunks(self, temp_dir):
        """Test formatting with chunks provided out of order."""
        handler = OutputHandler()
        
        # Create results in reverse order
        lines1 = [TranscriptionLine(timestamp=30.0, speaker="Alice", text="First chunk")]
        lines2 = [TranscriptionLine(timestamp=570.0, speaker="Bob", text="Second chunk")]
        
        result1 = TranscriptionResult(chunk_index=0, lines=lines1, raw_response="", model_used="test", processing_time_seconds=1.0)
        result2 = TranscriptionResult(chunk_index=1, lines=lines2, raw_response="", model_used="test", processing_time_seconds=1.0)
        
        config = Config(chunk_duration_minutes=10, overlap_duration_minutes=1)
        
        input_file = temp_dir / "test.mp3"
        output_file = temp_dir / "test.txt"
        input_file.touch()
        
        # Provide results in wrong order
        job = TranscriptionJob(
            input_file=input_file,
            output_file=output_file,
            model="test-model",
            config=config,
            results=[result2, result1]  # Wrong order
        )
        
        formatted = handler.format_transcription(job)
        
        # Should be sorted by chunk index, so correct order in output
        expected_lines = [
            "[00:00:30] Alice: First chunk",
            "[00:09:30] Bob: Second chunk"
        ]
        
        assert formatted == "\n".join(expected_lines)


class TestDeduplication:
    """Test deduplication functionality for overlapping chunks."""
    
    def test_deduplicate_overlapping_content_basic(self, temp_dir):
        """Test basic deduplication of overlapping content."""
        handler = OutputHandler()
        
        # Create overlapping content
        # Chunk 0: 0-10 minutes
        lines1 = [
            TranscriptionLine(timestamp=30.0, speaker="Alice", text="Hello everyone"),
            TranscriptionLine(timestamp=540.0, speaker="Bob", text="Near end of chunk 0")
        ]
        
        # Chunk 1: 9-19 minutes (overlaps with chunk 0)
        lines2 = [
            TranscriptionLine(timestamp=540.0, speaker="Bob", text="Near end of chunk 0"),  # Duplicate
            TranscriptionLine(timestamp=600.0, speaker="Alice", text="Start of chunk 1"),
            TranscriptionLine(timestamp=1080.0, speaker="Bob", text="Near end of chunk 1")
        ]
        
        result1 = TranscriptionResult(chunk_index=0, lines=lines1, raw_response="", model_used="test", processing_time_seconds=1.0)
        result2 = TranscriptionResult(chunk_index=1, lines=lines2, raw_response="", model_used="test", processing_time_seconds=1.0)
        
        config = Config(chunk_duration_minutes=10, overlap_duration_minutes=1)
        
        input_file = temp_dir / "test.mp3"
        output_file = temp_dir / "test.txt"
        input_file.touch()
        
        job = TranscriptionJob(
            input_file=input_file,
            output_file=output_file,
            model="test-model",
            config=config,
            results=[result1, result2]
        )
        
        deduplicated = handler.deduplicate_overlapping_content(job)
        
        # Should include all unique lines in timestamp order
        expected_lines = [
            "[00:00:30] Alice: Hello everyone",
            "[00:09:00] Bob: Near end of chunk 0",
            "[00:10:00] Alice: Start of chunk 1",
            "[00:18:00] Bob: Near end of chunk 1"
        ]
        
        assert deduplicated == expected_lines
    
    def test_deduplicate_no_overlap(self, temp_dir):
        """Test deduplication when there's no actual overlap."""
        handler = OutputHandler()
        
        # Create non-overlapping content
        lines1 = [TranscriptionLine(timestamp=30.0, speaker="Alice", text="First chunk")]
        lines2 = [TranscriptionLine(timestamp=600.0, speaker="Bob", text="Second chunk")]
        
        result1 = TranscriptionResult(chunk_index=0, lines=lines1, raw_response="", model_used="test", processing_time_seconds=1.0)
        result2 = TranscriptionResult(chunk_index=1, lines=lines2, raw_response="", model_used="test", processing_time_seconds=1.0)
        
        config = Config(chunk_duration_minutes=10, overlap_duration_minutes=1)
        
        input_file = temp_dir / "test.mp3"
        output_file = temp_dir / "test.txt"
        input_file.touch()
        
        job = TranscriptionJob(
            input_file=input_file,
            output_file=output_file,
            model="test-model",
            config=config,
            results=[result1, result2]
        )
        
        deduplicated = handler.deduplicate_overlapping_content(job)
        
        expected_lines = [
            "[00:00:30] Alice: First chunk",
            "[00:10:00] Bob: Second chunk"
        ]
        
        assert deduplicated == expected_lines
    
    def test_deduplicate_empty_results(self, temp_dir):
        """Test deduplication with empty results."""
        handler = OutputHandler()
        
        config = Config(chunk_duration_minutes=10, overlap_duration_minutes=1)
        
        input_file = temp_dir / "test.mp3"
        output_file = temp_dir / "test.txt"
        input_file.touch()
        
        job = TranscriptionJob(
            input_file=input_file,
            output_file=output_file,
            model="test-model",
            config=config,
            results=[]
        )
        
        deduplicated = handler.deduplicate_overlapping_content(job)
        assert deduplicated == []
    
    def test_deduplicate_preserves_chronological_order(self, temp_dir):
        """Test that deduplication preserves chronological order."""
        handler = OutputHandler()
        
        # Create content with mixed timestamps
        lines1 = [
            TranscriptionLine(timestamp=30.0, speaker="Alice", text="Early"),
            TranscriptionLine(timestamp=580.0, speaker="Bob", text="Late in chunk 0")
        ]
        
        lines2 = [
            TranscriptionLine(timestamp=550.0, speaker="Alice", text="Middle overlap"),  # Earlier than last line of chunk 0
            TranscriptionLine(timestamp=620.0, speaker="Bob", text="After chunk 0")
        ]
        
        result1 = TranscriptionResult(chunk_index=0, lines=lines1, raw_response="", model_used="test", processing_time_seconds=1.0)
        result2 = TranscriptionResult(chunk_index=1, lines=lines2, raw_response="", model_used="test", processing_time_seconds=1.0)
        
        config = Config(chunk_duration_minutes=10, overlap_duration_minutes=1)
        
        input_file = temp_dir / "test.mp3"
        output_file = temp_dir / "test.txt"
        input_file.touch()
        
        job = TranscriptionJob(
            input_file=input_file,
            output_file=output_file,
            model="test-model",
            config=config,
            results=[result1, result2]
        )
        
        deduplicated = handler.deduplicate_overlapping_content(job)
        
        # Should skip the earlier timestamp from chunk 2
        expected_lines = [
            "[00:00:30] Alice: Early",
            "[00:09:40] Bob: Late in chunk 0",
            "[00:10:20] Bob: After chunk 0"
        ]
        
        assert deduplicated == expected_lines


class TestJSONExport:
    """Test JSON export functionality."""
    
    def test_format_for_json_basic(self, temp_dir):
        """Test basic JSON formatting."""
        handler = OutputHandler()
        
        lines = [TranscriptionLine(timestamp=30.0, speaker="Alice", text="Hello")]
        result = TranscriptionResult(
            chunk_index=0,
            lines=lines,
            raw_response="[00:00:30] Alice: Hello",
            model_used="test-model",
            processing_time_seconds=1.5
        )
        
        chunk = ChunkData(
            chunk_index=0,
            start_time_seconds=0.0,
            end_time_seconds=600.0,
            audio_segment=None
        )
        
        config = Config(chunk_duration_minutes=10, overlap_duration_minutes=1)
        
        input_file = temp_dir / "input.mp3"
        output_file = temp_dir / "output.txt"
        input_file.touch()
        
        job = TranscriptionJob(
            input_file=input_file,
            output_file=output_file,
            model="test-model",
            config=config,
            chunks=[chunk],
            results=[result]
        )
        
        json_data = handler.format_for_json(job)
        
        # Check structure
        assert json_data["input_file"] == str(input_file)
        assert json_data["output_file"] == str(output_file)
        assert json_data["model"] == "test-model"
        
        # Check chunks
        assert len(json_data["chunks"]) == 1
        assert json_data["chunks"][0]["chunk_index"] == 0
        assert json_data["chunks"][0]["start_time_seconds"] == 0.0
        assert json_data["chunks"][0]["end_time_seconds"] == 600.0
        
        # Check results
        assert len(json_data["results"]) == 1
        assert json_data["results"][0]["chunk_index"] == 0
        assert json_data["results"][0]["model_used"] == "test-model"
        assert json_data["results"][0]["processing_time_seconds"] == 1.5
        
        # Check lines (should use formatted timestamp)
        assert len(json_data["results"][0]["lines"]) == 1
        line_data = json_data["results"][0]["lines"][0]
        assert line_data["timestamp"] == "[00:00:30]"  # Formatted timestamp
        assert line_data["speaker"] == "Alice"
        assert line_data["text"] == "Hello"
    
    def test_format_for_json_multiple_lines(self, temp_dir):
        """Test JSON formatting with multiple lines."""
        handler = OutputHandler()
        
        lines = [
            TranscriptionLine(timestamp=30.0, speaker="Alice", text="Hello"),
            TranscriptionLine(timestamp=65.0, speaker="Bob", text="Hi there"),
            TranscriptionLine(timestamp=120.0, speaker="Alice", text="How are you?")
        ]
        
        result = TranscriptionResult(
            chunk_index=0,
            lines=lines,
            raw_response="mock response",
            model_used="test-model",
            processing_time_seconds=2.0
        )
        
        config = Config(chunk_duration_minutes=10, overlap_duration_minutes=1)
        
        input_file = temp_dir / "input.mp3"
        output_file = temp_dir / "output.txt"
        input_file.touch()
        
        job = TranscriptionJob(
            input_file=input_file,
            output_file=output_file,
            model="test-model",
            config=config,
            results=[result]
        )
        
        json_data = handler.format_for_json(job)
        
        # Check all lines are included with formatted timestamps
        lines_data = json_data["results"][0]["lines"]
        assert len(lines_data) == 3
        
        assert lines_data[0]["timestamp"] == "[00:00:30]"
        assert lines_data[1]["timestamp"] == "[00:01:05]"
        assert lines_data[2]["timestamp"] == "[00:02:00]"
        
        assert lines_data[0]["speaker"] == "Alice"
        assert lines_data[1]["speaker"] == "Bob"
        assert lines_data[2]["speaker"] == "Alice"


class TestFileWriting:
    """Test file writing functionality."""
    
    def test_write_transcription_file_basic(self, temp_dir):
        """Test basic transcription file writing."""
        handler = OutputHandler()
        
        lines = [
            TranscriptionLine(timestamp=30.0, speaker="Alice", text="Hello"),
            TranscriptionLine(timestamp=65.0, speaker="Bob", text="Hi there")
        ]
        
        result = TranscriptionResult(
            chunk_index=0,
            lines=lines,
            raw_response="mock",
            model_used="test-model",
            processing_time_seconds=1.0
        )
        
        config = Config(chunk_duration_minutes=10, overlap_duration_minutes=1)
        
        input_file = temp_dir / "input.mp3"
        output_file = temp_dir / "output.txt"
        input_file.touch()
        
        job = TranscriptionJob(
            input_file=input_file,
            output_file=output_file,
            model="test-model",
            config=config,
            results=[result]
        )
        
        handler.write_transcription_file(job, deduplicate=False)
        
        # Check file was created and contains expected content
        assert output_file.exists()
        content = output_file.read_text(encoding='utf-8')
        
        expected_lines = [
            "[00:00:30] Alice: Hello",
            "[00:01:05] Bob: Hi there"
        ]
        assert content == "\n".join(expected_lines)
    
    def test_write_transcription_file_with_deduplication(self, temp_dir):
        """Test transcription file writing with deduplication."""
        handler = OutputHandler()
        
        # Create overlapping content
        lines1 = [TranscriptionLine(timestamp=30.0, speaker="Alice", text="Hello")]
        lines2 = [
            TranscriptionLine(timestamp=30.0, speaker="Alice", text="Hello"),  # Duplicate
            TranscriptionLine(timestamp=65.0, speaker="Bob", text="Hi there")
        ]
        
        result1 = TranscriptionResult(chunk_index=0, lines=lines1, raw_response="", model_used="test", processing_time_seconds=1.0)
        result2 = TranscriptionResult(chunk_index=1, lines=lines2, raw_response="", model_used="test", processing_time_seconds=1.0)
        
        config = Config(chunk_duration_minutes=10, overlap_duration_minutes=1)
        
        input_file = temp_dir / "input.mp3"
        output_file = temp_dir / "output.txt"
        input_file.touch()
        
        job = TranscriptionJob(
            input_file=input_file,
            output_file=output_file,
            model="test-model",
            config=config,
            results=[result1, result2]
        )
        
        handler.write_transcription_file(job, deduplicate=True)
        
        # Check file was created and duplicates were removed
        assert output_file.exists()
        content = output_file.read_text(encoding='utf-8')
        
        expected_lines = [
            "[00:00:30] Alice: Hello",
            "[00:01:05] Bob: Hi there"
        ]
        assert content == "\n".join(expected_lines)
    
    def test_export_job_results_json(self, temp_dir):
        """Test JSON export to file."""
        handler = OutputHandler()
        
        lines = [TranscriptionLine(timestamp=30.0, speaker="Alice", text="Hello")]
        result = TranscriptionResult(chunk_index=0, lines=lines, raw_response="", model_used="test", processing_time_seconds=1.0)
        
        config = Config(chunk_duration_minutes=10, overlap_duration_minutes=1)
        
        input_file = temp_dir / "input.mp3"
        output_file = temp_dir / "output.txt"
        input_file.touch()
        
        job = TranscriptionJob(
            input_file=input_file,
            output_file=output_file,
            model="test-model",
            config=config,
            results=[result]
        )
        
        handler.export_job_results(job, export_formats=["json"])
        
        # Check JSON file was created
        json_file = temp_dir / "output.json"
        assert json_file.exists()
        
        # Check JSON content
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert data["input_file"] == str(input_file)
        assert len(data["results"]) == 1
        assert data["results"][0]["lines"][0]["timestamp"] == "[00:00:30]"