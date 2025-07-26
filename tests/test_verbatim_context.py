"""Tests for verbatim context repetition and deduplication."""

import pytest
from unittest.mock import Mock, patch
from typing import List

from src.llm_transcribe.llm_client import LLMClient
from src.llm_transcribe.models import ChunkData, TranscriptionLine, TranscriptionResult, TranscriptionJob, Config
from src.llm_transcribe.output import OutputHandler
from pathlib import Path


class TestVerbatimContextRepetition:
    """Test verbatim context repetition in LLM prompts."""
    
    def test_create_messages_with_context_includes_verbatim_instruction(self):
        """Test that context messages include verbatim repetition instructions."""
        client = LLMClient()
        
        context = "[00:30] Alice: Previous statement\n[00:45] Bob: Another statement"
        chunk_start_seconds = 540.0  # 9 minutes
        audio_bytes = b"fake_audio_data"
        
        messages = client.create_messages(audio_bytes, chunk_start_seconds, context)
        
        # Check that the user message contains verbatim instructions
        user_message = messages[1]["content"][0]["text"]
        
        assert "EXACTLY as shown (verbatim)" in user_message
        assert "repeating the context lines exactly" in user_message
        assert "Context from previous chunk:" in user_message
        assert "[00:30] Alice: Previous statement" in user_message
        assert "[00:45] Bob: Another statement" in user_message
    
    def test_create_messages_without_context_normal_prompt(self):
        """Test that messages without context use normal prompt."""
        client = LLMClient()
        
        audio_bytes = b"fake_audio_data"
        chunk_start_seconds = 0.0
        
        messages = client.create_messages(audio_bytes, chunk_start_seconds, None)
        
        # Check that the user message contains normal prompt
        user_message = messages[1]["content"][0]["text"]
        
        assert "Start timestamps from [00:00]" in user_message
        assert "verbatim" not in user_message
        assert "Context from previous chunk" not in user_message
    
    def test_system_message_includes_verbatim_example(self):
        """Test that system message includes verbatim context example."""
        client = LLMClient()
        
        system_message = client.system_message
        
        assert "When provided with context from a previous chunk:" in system_message
        assert "First output the context lines EXACTLY as shown" in system_message
        assert "Example with context:" in system_message
        assert "Your output should be:" in system_message


class TestVerbatimContextParsing:
    """Test parsing of responses that include verbatim context repetition."""
    
    def test_parse_response_with_verbatim_context_repetition(self):
        """Test parsing response where LLM repeated context verbatim."""
        client = LLMClient()
        
        # Simulate LLM response that includes verbatim context + new content
        response = """[00:30] Alice: Previous statement
[00:45] Bob: Another statement
[00:48] Alice: And now continuing with new content
[01:02] Bob: Yes, that makes sense"""
        
        chunk_start_seconds = 540.0  # 9 minutes
        
        lines = client.parse_transcription_response(response, chunk_start_seconds)
        
        assert len(lines) == 4
        
        # Check that all lines are parsed correctly with absolute timestamps
        assert lines[0].timestamp == 570.0  # 540 + 30
        assert lines[0].speaker == "Alice"
        assert lines[0].text == "Previous statement"
        
        assert lines[1].timestamp == 585.0  # 540 + 45
        assert lines[1].speaker == "Bob"
        assert lines[1].text == "Another statement"
        
        assert lines[2].timestamp == 588.0  # 540 + 48
        assert lines[2].speaker == "Alice"
        assert lines[2].text == "And now continuing with new content"
        
        assert lines[3].timestamp == 602.0  # 540 + 62
        assert lines[3].speaker == "Bob"
        assert lines[3].text == "Yes, that makes sense"
    
    def test_parse_response_handles_edge_case_missing_content(self):
        """Test that verbatim repetition catches content missing from previous chunk."""
        client = LLMClient()
        
        # Simulate case where previous chunk ended at [00:45] but there was actually
        # content at [00:47] that was missed, and current chunk's LLM caught it
        response = """[00:30] Alice: Previous statement
[00:45] Bob: Another statement
[00:47] Alice: Actually, one more thing
[00:50] Bob: What's that?
[01:05] Alice: We should review the budget"""
        
        chunk_start_seconds = 540.0
        
        lines = client.parse_transcription_response(response, chunk_start_seconds)
        
        assert len(lines) == 5
        
        # Verify the "missing" content at [00:47] is captured
        assert lines[2].timestamp == 587.0  # 540 + 47
        assert lines[2].speaker == "Alice"
        assert lines[2].text == "Actually, one more thing"


class TestDeduplicationWithVerbatimContext:
    """Test that deduplication works correctly with verbatim context repetition."""
    
    def test_deduplication_removes_verbatim_context_exactly(self, temp_dir):
        """Test that exact verbatim context repetition is properly deduplicated."""
        handler = OutputHandler()
        
        # Chunk 0 result
        chunk0_lines = [
            TranscriptionLine(timestamp=570.0, speaker="Alice", text="Previous statement"),   # 9:30
            TranscriptionLine(timestamp=585.0, speaker="Bob", text="Another statement"),     # 9:45
        ]
        
        # Chunk 1 result - LLM repeated context verbatim then continued
        chunk1_lines = [
            TranscriptionLine(timestamp=570.0, speaker="Alice", text="Previous statement"),   # 9:30 (verbatim)
            TranscriptionLine(timestamp=585.0, speaker="Bob", text="Another statement"),     # 9:45 (verbatim)
            TranscriptionLine(timestamp=588.0, speaker="Alice", text="New content here"),    # 9:48 (new)
            TranscriptionLine(timestamp=602.0, speaker="Bob", text="More new content"),      # 10:02 (new)
        ]
        
        result0 = TranscriptionResult(chunk_index=0, lines=chunk0_lines, raw_response="", model_used="test", processing_time_seconds=1.0)
        result1 = TranscriptionResult(chunk_index=1, lines=chunk1_lines, raw_response="", model_used="test", processing_time_seconds=1.0)
        
        config = Config(chunk_duration_minutes=10, overlap_duration_minutes=1)
        
        input_file = temp_dir / "test.mp3"
        output_file = temp_dir / "test.txt"
        input_file.touch()
        
        job = TranscriptionJob(
            input_file=input_file,
            output_file=output_file,
            model="test-model",
            config=config,
            results=[result0, result1]
        )
        
        deduplicated = handler.deduplicate_overlapping_content(job)
        
        # Should keep all unique lines in chronological order
        expected_lines = [
            "[00:09:30] Alice: Previous statement",
            "[00:09:45] Bob: Another statement",
            "[00:09:48] Alice: New content here",
            "[00:10:02] Bob: More new content"
        ]
        
        assert deduplicated == expected_lines
    
    def test_deduplication_handles_missing_content_edge_case(self, temp_dir):
        """Test deduplication when verbatim repetition includes content missing from previous chunk."""
        handler = OutputHandler()
        
        # Chunk 0 result - missed some content at end
        chunk0_lines = [
            TranscriptionLine(timestamp=570.0, speaker="Alice", text="Previous statement"),   # 9:30
            TranscriptionLine(timestamp=585.0, speaker="Bob", text="Another statement"),     # 9:45
            # Missing: content at 9:47 that should have been included
        ]
        
        # Chunk 1 result - LLM caught the missing content in verbatim repetition
        chunk1_lines = [
            TranscriptionLine(timestamp=570.0, speaker="Alice", text="Previous statement"),   # 9:30 (verbatim)
            TranscriptionLine(timestamp=585.0, speaker="Bob", text="Another statement"),     # 9:45 (verbatim)
            TranscriptionLine(timestamp=587.0, speaker="Alice", text="Actually, one more thing"), # 9:47 (was missing!)
            TranscriptionLine(timestamp=590.0, speaker="Bob", text="What's that?"),          # 9:50 (was missing!)
            TranscriptionLine(timestamp=605.0, speaker="Alice", text="We should review"),    # 10:05 (new)
        ]
        
        result0 = TranscriptionResult(chunk_index=0, lines=chunk0_lines, raw_response="", model_used="test", processing_time_seconds=1.0)
        result1 = TranscriptionResult(chunk_index=1, lines=chunk1_lines, raw_response="", model_used="test", processing_time_seconds=1.0)
        
        config = Config(chunk_duration_minutes=10, overlap_duration_minutes=1)
        
        input_file = temp_dir / "test.mp3"
        output_file = temp_dir / "test.txt"
        input_file.touch()
        
        job = TranscriptionJob(
            input_file=input_file,
            output_file=output_file,
            model="test-model",
            config=config,
            results=[result0, result1]
        )
        
        deduplicated = handler.deduplicate_overlapping_content(job)
        
        # Should keep all content, including the previously missing parts
        expected_lines = [
            "[00:09:30] Alice: Previous statement",
            "[00:09:45] Bob: Another statement",
            "[00:09:47] Alice: Actually, one more thing",  # This was missing from chunk 0!
            "[00:09:50] Bob: What's that?",                # This was missing from chunk 0!
            "[00:10:05] Alice: We should review"
        ]
        
        assert deduplicated == expected_lines
    
    def test_deduplication_with_single_digit_minutes_verbatim(self, temp_dir):
        """Test deduplication works with single digit minutes in verbatim context."""
        handler = OutputHandler()
        
        # Chunk 0 result
        chunk0_lines = [
            TranscriptionLine(timestamp=63.0, speaker="Alice", text="Early statement"),    # 1:03
            TranscriptionLine(timestamp=90.0, speaker="Bob", text="Another statement"),   # 1:30
        ]
        
        # Chunk 1 result - with single digit minute format in verbatim repetition
        chunk1_lines = [
            TranscriptionLine(timestamp=63.0, speaker="Alice", text="Early statement"),   # 1:03 (verbatim)
            TranscriptionLine(timestamp=90.0, speaker="Bob", text="Another statement"),  # 1:30 (verbatim)
            TranscriptionLine(timestamp=105.0, speaker="Alice", text="Continuing now"),  # 1:45 (new)
        ]
        
        result0 = TranscriptionResult(chunk_index=0, lines=chunk0_lines, raw_response="", model_used="test", processing_time_seconds=1.0)
        result1 = TranscriptionResult(chunk_index=1, lines=chunk1_lines, raw_response="", model_used="test", processing_time_seconds=1.0)
        
        config = Config(chunk_duration_minutes=2, overlap_duration_minutes=1)
        
        input_file = temp_dir / "test.mp3"
        output_file = temp_dir / "test.txt"
        input_file.touch()
        
        job = TranscriptionJob(
            input_file=input_file,
            output_file=output_file,
            model="test-model",
            config=config,
            results=[result0, result1]
        )
        
        deduplicated = handler.deduplicate_overlapping_content(job)
        
        expected_lines = [
            "[00:01:03] Alice: Early statement",
            "[00:01:30] Bob: Another statement",
            "[00:01:45] Alice: Continuing now"
        ]
        
        assert deduplicated == expected_lines


class TestIntegrationVerbatimContext:
    """Integration tests for verbatim context approach."""
    
    def test_full_transcription_with_verbatim_context_multiple_chunks(self, temp_dir):
        """Test complete transcription process with verbatim context across multiple chunks."""
        handler = OutputHandler()
        
        # Simulate 3 chunks with verbatim context repetition
        
        # Chunk 0: 0-10 minutes
        chunk0_lines = [
            TranscriptionLine(timestamp=30.0, speaker="Alice", text="Welcome to the meeting"),  # 0:30
            TranscriptionLine(timestamp=570.0, speaker="Bob", text="Thank you for joining"),   # 9:30
        ]
        
        # Chunk 1: 9-19 minutes (repeats 9:30 verbatim, adds content)
        chunk1_lines = [
            TranscriptionLine(timestamp=570.0, speaker="Bob", text="Thank you for joining"),   # 9:30 (verbatim)
            TranscriptionLine(timestamp=575.0, speaker="Alice", text="Let's begin"),           # 9:35 (new)
            TranscriptionLine(timestamp=1110.0, speaker="Bob", text="First topic is budget"), # 18:30 (new)
        ]
        
        # Chunk 2: 18-28 minutes (repeats 18:30 verbatim, adds content)
        chunk2_lines = [
            TranscriptionLine(timestamp=1110.0, speaker="Bob", text="First topic is budget"), # 18:30 (verbatim)
            TranscriptionLine(timestamp=1115.0, speaker="Alice", text="Great idea"),          # 18:35 (new)
            TranscriptionLine(timestamp=1650.0, speaker="Bob", text="Any questions?"),       # 27:30 (new)
        ]
        
        result0 = TranscriptionResult(chunk_index=0, lines=chunk0_lines, raw_response="", model_used="test", processing_time_seconds=1.0)
        result1 = TranscriptionResult(chunk_index=1, lines=chunk1_lines, raw_response="", model_used="test", processing_time_seconds=1.0)
        result2 = TranscriptionResult(chunk_index=2, lines=chunk2_lines, raw_response="", model_used="test", processing_time_seconds=1.0)
        
        config = Config(chunk_duration_minutes=10, overlap_duration_minutes=1)
        
        input_file = temp_dir / "test.mp3"
        output_file = temp_dir / "test.txt"
        input_file.touch()
        
        job = TranscriptionJob(
            input_file=input_file,
            output_file=output_file,
            model="test-model",
            config=config,
            results=[result0, result1, result2]
        )
        
        # Test deduplication
        deduplicated = handler.deduplicate_overlapping_content(job)
        
        expected_deduplicated_lines = [
            "[00:00:30] Alice: Welcome to the meeting",
            "[00:09:30] Bob: Thank you for joining",
            "[00:09:35] Alice: Let's begin",
            "[00:18:30] Bob: First topic is budget",
            "[00:18:35] Alice: Great idea",
            "[00:27:30] Bob: Any questions?"
        ]
        
        assert deduplicated == expected_deduplicated_lines
        
        # Test full formatting (includes duplicates)
        formatted = handler.format_transcription(job)
        
        expected_formatted_lines = [
            "[00:00:30] Alice: Welcome to the meeting",
            "[00:09:30] Bob: Thank you for joining",
            "[00:09:30] Bob: Thank you for joining",  # Verbatim repetition
            "[00:09:35] Alice: Let's begin",
            "[00:18:30] Bob: First topic is budget",
            "[00:18:30] Bob: First topic is budget",  # Verbatim repetition
            "[00:18:35] Alice: Great idea",
            "[00:27:30] Bob: Any questions?"
        ]
        
        assert formatted == "\n".join(expected_formatted_lines)