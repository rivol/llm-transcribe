"""LLM client using LiteLLM."""

import base64
import logging
import re
import time
from typing import List, Optional

import litellm
from litellm import completion

from .models import ChunkData, TranscriptionLine, TranscriptionResult

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with LLMs via LiteLLM."""
    
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model
        
        # System message for transcription
        self.system_message = """You are a professional transcriptionist. Your task is to transcribe audio content with high accuracy.

Instructions:
1. Listen to the audio carefully and transcribe all speech
2. Include timestamps for each speaker turn in the format [HH:MM:SS]
3. Identify speakers by name when possible (e.g., "John", "Sarah")
4. If names are not clear, use roles when identifiable (e.g., "Manager", "Customer")
5. If neither names nor roles are clear, use "Speaker 1", "Speaker 2", etc.
6. Maintain speaker consistency throughout the transcription
7. Format each line as: [timestamp] Speaker: text
8. Be accurate with timestamps and speaker attribution
9. Include all speech, even brief responses like "yes", "okay", etc.
10. Do not add commentary or explanations, only transcribe what is said

Expected output format examples:

With named speakers:
[00:01:23] John: Welcome everyone to today's meeting.
[00:01:27] Sarah: Thank you, John. I'm excited to be here.
[00:01:30] John: Great! Let's start with the quarterly review.

With role-based speakers:
[00:02:15] Manager: How are we tracking against our goals?
[00:02:18] Analyst: We're about 15% ahead of schedule.
[00:02:22] Manager: Excellent news.

With numerical speakers:
[00:03:45] Speaker 1: Do we have the latest figures?
[00:03:47] Speaker 2: Yes, I can share those now.
[00:03:52] Speaker 1: Perfect, go ahead.

Include brief responses:
[00:04:10] Speaker 1: Are you ready to proceed?
[00:04:11] Speaker 2: Yes.
[00:04:12] Speaker 1: Okay, let's continue.

If you are provided with context from a previous chunk, use it to maintain speaker consistency and conversation flow."""
    
    def encode_audio_to_base64(self, audio_bytes: bytes) -> str:
        """Encode audio bytes to base64 string.
        
        Args:
            audio_bytes: Audio data as bytes
            
        Returns:
            Base64 encoded string
        """
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    def create_messages(self, audio_bytes: bytes, context: Optional[str] = None) -> List[dict]:
        """Create message array for LLM API.
        
        Args:
            audio_bytes: Audio data as bytes
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
        
        # Create the main transcription prompt
        transcription_prompt = "Please transcribe this audio:"
        if context:
            transcription_prompt = f"Context from previous chunk (for speaker consistency):\n{context}\n\nNow transcribe this new audio chunk:"
        
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
    
    def parse_transcription_response(self, response_text: str) -> List[TranscriptionLine]:
        """Parse LLM response into TranscriptionLine objects.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            List of TranscriptionLine objects
        """
        lines = []
        
        # Split response into lines
        raw_lines = response_text.strip().split('\n')
        
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to parse timestamp and speaker
            # Expected format: [HH:MM:SS] Speaker: text
            match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s*([^:]+):\s*(.+)', line)
            
            if match:
                timestamp = f"[{match.group(1)}]"
                speaker = match.group(2).strip()
                text = match.group(3).strip()
                
                lines.append(TranscriptionLine(
                    timestamp=timestamp,
                    speaker=speaker,
                    text=text
                ))
            else:
                # If parsing fails, log warning and try to extract what we can
                logger.warning(f"Could not parse line: {line}")
                
                # Try to find at least a timestamp
                timestamp_match = re.search(r'\[(\d{2}:\d{2}:\d{2})\]', line)
                if timestamp_match:
                    timestamp = f"[{timestamp_match.group(1)}]"
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
                        timestamp=timestamp,
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
            messages = self.create_messages(audio_bytes, context)
            
            logger.info(f"Transcribing chunk {chunk.chunk_index} with model {self.model}")
            if context:
                logger.debug(f"Using context: {context[:100]}...")
            
            # Log message structure without the actual audio data
            logger.debug(f"Sending {len(messages)} messages, audio size: {len(audio_bytes)} bytes")
            
            # Make API call
            response = completion(
                model=self.model,
                messages=messages,
                max_tokens=4000,  # Generous limit for transcription
                temperature=0.1,  # Low temperature for consistent transcription
            )
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            # Parse transcription lines
            lines = self.parse_transcription_response(response_text)
            
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
            response = completion(
                model=self.model,
                messages=[
                    {"role": "user", "content": "Hello, please respond with 'OK' to confirm you're working."}
                ],
                max_tokens=10,
                temperature=0
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.info(f"Connection test successful: {response_text}")
            return True
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False