"""LLM client using LiteLLM."""

import base64
import logging
import re
import time
from typing import List, Optional

import litellm
from litellm import completion

from .models import ChunkData, TranscriptionLine, TranscriptionResult
from .timestamp_utils import format_timestamp, parse_timestamp_from_text

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with LLMs via LiteLLM."""
    
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model
        
        # System message for transcription
        self.system_message = """You are a professional transcriptionist. Your task is to transcribe audio content with high accuracy.

Instructions:
1. Listen to the audio carefully and transcribe all speech
2. Include timestamps for each speaker turn in the format [MM:SS]
3. IMPORTANT: Use timestamps relative to this audio chunk, starting from [00:00]
4. The first speaker should have a timestamp near [00:00], then increment naturally
5. Identify speakers by name when possible (e.g., "John", "Sarah")
6. If names are not clear, use roles when identifiable (e.g., "Manager", "Customer")
7. If neither names nor roles are clear, use "Speaker 1", "Speaker 2", etc.
8. Maintain speaker consistency throughout the transcription
9. Format each line as: [timestamp] Speaker: text
10. Be accurate with timestamps and speaker attribution
11. Include all speech, even brief responses like "yes", "okay", etc.
12. Do not add commentary or explanations, only transcribe what is said

Expected output format examples:

With named speakers:
[01:23] John: Welcome everyone to today's meeting.
[01:27] Sarah: Thank you, John. I'm excited to be here.
[01:30] John: Great! Let's start with the quarterly review.

With role-based speakers:
[02:15] Manager: How are we tracking against our goals?
[02:18] Analyst: We're about 15% ahead of schedule.
[02:22] Manager: Excellent news.

With numerical speakers:
[03:45] Speaker 1: Do we have the latest figures?
[03:47] Speaker 2: Yes, I can share those now.
[03:52] Speaker 1: Perfect, go ahead.

Include brief responses:
[04:10] Speaker 1: Are you ready to proceed?
[04:11] Speaker 2: Yes.
[04:12] Speaker 1: Okay, let's continue.

If you are provided with context from a previous chunk, use it to maintain speaker consistency and conversation flow."""
    
    def encode_audio_to_base64(self, audio_bytes: bytes) -> str:
        """Encode audio bytes to base64 string.
        
        Args:
            audio_bytes: Audio data as bytes
            
        Returns:
            Base64 encoded string
        """
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    def create_messages(self, audio_bytes: bytes, chunk_start_seconds: float, context: Optional[str] = None) -> List[dict]:
        """Create message array for LLM API.
        
        Args:
            audio_bytes: Audio data as bytes
            chunk_start_seconds: Start time of this chunk in seconds
            context: Optional context from previous chunk
            
        Returns:
            List of message dictionaries
        """
        messages = [
            {
                "role": "system",
                "content": self.system_message
            }
        ]
        
        # Create the main transcription prompt with relative timing instructions
        transcription_prompt = "Please transcribe this audio chunk. Start timestamps from [00:00], use [MM:SS] format, and increment naturally."
        if context:
            # Convert context timestamps to be relative to this chunk
            relative_context = self._convert_context_to_relative(context, chunk_start_seconds)
            transcription_prompt = f"Context from previous chunk (for speaker consistency):\n{relative_context}\n\nNow transcribe this new audio chunk. Start timestamps from [00:00] and increment naturally."
        
        # Add audio using Gemini's expected format
        audio_base64 = self.encode_audio_to_base64(audio_bytes)
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": transcription_prompt
                },
                {
                    "type": "file",
                    "file": {
                        "file_data": f"data:audio/wav;base64,{audio_base64}"
                    }
                }
            ]
        })
        
        return messages
    
    def _convert_context_to_relative(self, context: str, chunk_start_seconds: float) -> str:
        """Convert absolute timestamps in context to relative timestamps for current chunk.
        
        Args:
            context: Context string with absolute timestamps
            chunk_start_seconds: Start time of current chunk in seconds
            
        Returns:
            Context string with relative timestamps
        """
        def convert_timestamp(match):
            timestamp_str = match.group(1)  # HH:MM:SS
            try:
                time_parts = timestamp_str.split(':')
                absolute_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
                
                # Convert to relative seconds
                relative_seconds = absolute_seconds - chunk_start_seconds
                
                # Ensure non-negative (context should be from end of previous chunk)
                if relative_seconds < 0:
                    relative_seconds = 0
                
                # Convert back to MM:SS format for LLM
                minutes = int(relative_seconds // 60)
                seconds = int(relative_seconds % 60)
                return f"[{minutes:02d}:{seconds:02d}]"
            except (ValueError, IndexError):
                # Return original if parsing fails
                return match.group(0)
        
        # Replace all timestamps [HH:MM:SS] with relative versions
        return re.sub(r'\[(\d{2}:\d{2}:\d{2})\]', convert_timestamp, context)
    
    def _convert_relative_to_absolute_seconds(self, relative_timestamp: str, chunk_start_seconds: float) -> float:
        """Convert relative timestamp to absolute seconds.
        
        Args:
            relative_timestamp: Timestamp in format [MM:SS] relative to chunk start
            chunk_start_seconds: Start time of chunk in seconds
            
        Returns:
            Absolute timestamp in seconds
        """
        try:
            # Parse relative timestamp
            timestamp_str = relative_timestamp.strip('[]')
            time_parts = timestamp_str.split(':')
            # Handle MM:SS format
            if len(time_parts) == 2:
                relative_seconds = int(time_parts[0]) * 60 + int(time_parts[1])
            else:
                # Fallback for HH:MM:SS format (backwards compatibility)
                relative_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
            
            # Convert to absolute seconds
            return relative_seconds + chunk_start_seconds
        except (ValueError, IndexError):
            # Return 0 if parsing fails
            return chunk_start_seconds
    
    def _log_llm_messages(self, messages: List[dict]) -> None:
        """Log LLM messages in verbose mode, excluding audio data.
        
        Args:
            messages: List of message dictionaries to log
        """
        logger.debug("LLM Messages:")
        for i, message in enumerate(messages):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            
            # Handle different content types
            if isinstance(content, str):
                # Simple text content
                logger.debug(f"  Message {i+1} ({role}): {content}")
            elif isinstance(content, list):
                # Multimodal content (text + files)
                logger.debug(f"  Message {i+1} ({role}):")
                for j, item in enumerate(content):
                    if item.get("type") == "text":
                        text_content = item.get("text", "")
                        logger.debug(f"    Text part {j+1}: {text_content}")
                    elif item.get("type") == "file":
                        # Don't log file data, just metadata
                        file_info = item.get("file", {})
                        file_data = file_info.get("file_data", "")
                        if file_data.startswith("data:audio/"):
                            # Extract just the metadata, not the data
                            parts = file_data.split(",", 1)
                            mime_type = parts[0] if len(parts) > 0 else "unknown"
                            data_size = len(parts[1]) if len(parts) > 1 else 0
                            logger.debug(f"    File part {j+1}: {mime_type}, size: {data_size} chars")
                        else:
                            logger.debug(f"    File part {j+1}: {file_data[:100]}...")
                    else:
                        logger.debug(f"    Part {j+1}: {item}")
            else:
                logger.debug(f"  Message {i+1} ({role}): {content}")
    
    def parse_transcription_response(self, response_text: str, chunk_start_seconds: float) -> List[TranscriptionLine]:
        """Parse LLM response into TranscriptionLine objects.
        
        Args:
            response_text: Raw response from LLM with relative timestamps
            chunk_start_seconds: Start time of chunk to convert relative to absolute timestamps
            
        Returns:
            List of TranscriptionLine objects with absolute timestamps
        """
        lines = []
        
        # Split response into lines
        raw_lines = response_text.strip().split('\n')
        
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to parse timestamp and speaker
            # Expected format: [MM:SS] Speaker: text
            match = re.match(r'\[(\d{2}:\d{2})\]\s*([^:]+):\s*(.+)', line)
            
            if match:
                relative_timestamp = f"[{match.group(1)}]"
                speaker = match.group(2).strip()
                text = match.group(3).strip()
                
                # Convert relative timestamp to absolute seconds
                absolute_seconds = self._convert_relative_to_absolute_seconds(relative_timestamp, chunk_start_seconds)
                
                lines.append(TranscriptionLine(
                    timestamp=absolute_seconds,
                    speaker=speaker,
                    text=text
                ))
            else:
                # If parsing fails, log warning and try to extract what we can
                logger.warning(f"Could not parse line: {line}")
                
                # Try to find at least a timestamp
                timestamp_match = re.search(r'\[(\d{2}:\d{2})\]', line)
                if timestamp_match:
                    relative_timestamp = f"[{timestamp_match.group(1)}]"
                    # Convert relative timestamp to absolute seconds
                    absolute_seconds = self._convert_relative_to_absolute_seconds(relative_timestamp, chunk_start_seconds)
                    
                    # Use the rest as speaker + text
                    remaining = line.replace(timestamp_match.group(0), '').strip()
                    if ':' in remaining:
                        parts = remaining.split(':', 1)
                        speaker = parts[0].strip()
                        text = parts[1].strip()
                    else:
                        speaker = "Unknown"
                        text = remaining
                    
                    lines.append(TranscriptionLine(
                        timestamp=absolute_seconds,
                        speaker=speaker,
                        text=text
                    ))
        
        return lines
    
    def transcribe_chunk(self, chunk: ChunkData, audio_bytes: bytes, context: Optional[str] = None) -> TranscriptionResult:
        """Transcribe a single audio chunk.
        
        Args:
            chunk: ChunkData object
            audio_bytes: Audio data as bytes
            context: Optional context from previous chunk
            
        Returns:
            TranscriptionResult object
        """
        start_time = time.time()
        
        try:
            # Create messages
            messages = self.create_messages(audio_bytes, chunk.start_time_seconds, context)
            
            logger.info(f"Transcribing chunk {chunk.chunk_index} with model {self.model}")
            if context:
                logger.debug(f"Using context: {context[:100]}...")
            
            # Log message structure without the actual audio data
            logger.debug(f"Sending {len(messages)} messages, audio size: {len(audio_bytes)} bytes")
            
            # Log messages in verbose mode (excluding audio data)
            self._log_llm_messages(messages)
            
            # Make API call
            response = completion(
                model=self.model,
                messages=messages,
                max_tokens=4000,  # Generous limit for transcription
                temperature=0.1,  # Low temperature for consistent transcription
            )
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            # Log LLM response in verbose mode
            logger.debug(f"LLM Response:\n{response_text}")
            
            # Parse transcription lines (convert relative timestamps to absolute)
            lines = self.parse_transcription_response(response_text, chunk.start_time_seconds)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Transcribed chunk {chunk.chunk_index}: {len(lines)} lines in {processing_time:.2f}s")
            
            return TranscriptionResult(
                chunk_index=chunk.chunk_index,
                lines=lines,
                raw_response=response_text,
                model_used=self.model,
                processing_time_seconds=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error transcribing chunk {chunk.chunk_index}: {e}")
            
            # Return empty result with error info
            return TranscriptionResult(
                chunk_index=chunk.chunk_index,
                lines=[],
                raw_response=f"Error: {str(e)}",
                model_used=self.model,
                processing_time_seconds=processing_time
            )
    
    def test_connection(self) -> bool:
        """Test if the LLM client can connect and make a simple request.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Simple test message
            test_messages = [
                {"role": "user", "content": "Hello, please respond with 'OK' to confirm you're working."}
            ]
            
            # Log test messages in verbose mode
            self._log_llm_messages(test_messages)
            
            response = completion(
                model=self.model,
                messages=test_messages,
                max_tokens=10,
                temperature=0
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.debug(f"Test LLM Response: {response_text}")
            logger.info(f"Connection test successful: {response_text}")
            return True
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False