"""Pydantic models for transcriber."""

from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from pydantic import BaseModel, Field, validator
from pydub import AudioSegment


class Config(BaseModel):
    """Application configuration."""
    
    chunk_duration_minutes: int = Field(default=10, description="Duration of each audio chunk in minutes")
    overlap_duration_minutes: int = Field(default=1, description="Overlap duration between chunks in minutes")
    default_model: str = Field(default="gemini-2.5-flash", description="Default LLM model to use")
    supported_formats: List[str] = Field(
        default=["wav", "mp3", "m4a", "flac", "ogg", "aac"],
        description="Supported audio formats"
    )
    
    @validator('chunk_duration_minutes')
    def validate_chunk_duration(cls, v):
        if v <= 0:
            raise ValueError("Chunk duration must be positive")
        return v
    
    @validator('overlap_duration_minutes')
    def validate_overlap_duration(cls, v):
        if v < 0:
            raise ValueError("Overlap duration must be non-negative")
        return v


class ChunkData(BaseModel):
    """Audio chunk metadata."""
    
    start_time_seconds: float = Field(description="Start time of chunk in seconds")
    end_time_seconds: float = Field(description="End time of chunk in seconds")
    chunk_index: int = Field(description="Index of this chunk (0-based)")
    audio_segment: Optional[AudioSegment] = Field(default=None, description="Audio segment data")
    
    class Config:
        arbitrary_types_allowed = True  # Allow AudioSegment
    
    @validator('start_time_seconds')
    def validate_start_time(cls, v):
        if v < 0:
            raise ValueError("Start time must be non-negative")
        return v
    
    @validator('end_time_seconds')
    def validate_end_time(cls, v, values):
        if v < 0:
            raise ValueError("End time must be non-negative")
        if 'start_time_seconds' in values and v <= values['start_time_seconds']:
            raise ValueError("End time must be greater than start time")
        return v


class TranscriptionLine(BaseModel):
    """A single line of transcription."""
    
    timestamp: str = Field(description="Timestamp in format [HH:MM:SS]")
    speaker: str = Field(description="Speaker name or identifier")
    text: str = Field(description="Transcribed text")
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        # Basic validation for timestamp format [HH:MM:SS]
        if not v.startswith('[') or not v.endswith(']'):
            raise ValueError("Timestamp must be in format [HH:MM:SS]")
        return v
    
    def __str__(self) -> str:
        return f"{self.timestamp} {self.speaker}: {self.text}"


class TranscriptionResult(BaseModel):
    """Result of transcribing a single chunk."""
    
    chunk_index: int = Field(description="Index of the chunk that was transcribed")
    lines: List[TranscriptionLine] = Field(description="Transcription lines")
    raw_response: str = Field(description="Raw LLM response")
    model_used: str = Field(description="LLM model that was used")
    processing_time_seconds: float = Field(description="Time taken to process this chunk")
    
    @validator('processing_time_seconds')
    def validate_processing_time(cls, v):
        if v < 0:
            raise ValueError("Processing time must be non-negative")
        return v
    
    @property
    def text(self) -> str:
        """Get the full transcription text."""
        return "\n".join(str(line) for line in self.lines)
    
    def get_last_minute_context(self, chunk_duration_minutes: int = 10) -> str:
        """Extract the last minute of transcription for context."""
        if not self.lines:
            return ""
        
        # Calculate the last minute timestamp
        chunk_start_minutes = self.chunk_index * (chunk_duration_minutes - 1)  # Account for overlap
        last_minute_start = chunk_start_minutes + chunk_duration_minutes - 1
        
        # Convert to seconds for comparison
        last_minute_start_seconds = last_minute_start * 60
        
        context_lines = []
        for line in self.lines:
            # Parse timestamp to get seconds
            timestamp_str = line.timestamp.strip('[]')
            try:
                time_parts = timestamp_str.split(':')
                line_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
                
                if line_seconds >= last_minute_start_seconds:
                    context_lines.append(line)
            except (ValueError, IndexError):
                # Skip lines with invalid timestamps
                continue
        
        return "\n".join(str(line) for line in context_lines)


class TranscriptionJob(BaseModel):
    """Complete transcription job information."""
    
    input_file: Path = Field(description="Input audio file path")
    output_file: Path = Field(description="Output transcription file path")
    model: str = Field(description="LLM model to use")
    config: Config = Field(description="Configuration for this job")
    chunks: List[ChunkData] = Field(default_factory=list, description="Audio chunks")
    results: List[TranscriptionResult] = Field(default_factory=list, description="Transcription results")
    started_at: Optional[datetime] = Field(default=None, description="When the job started")
    completed_at: Optional[datetime] = Field(default=None, description="When the job completed")
    
    @validator('input_file')
    def validate_input_file(cls, v):
        if not v.exists():
            raise ValueError(f"Input file does not exist: {v}")
        return v
    
    @property
    def is_completed(self) -> bool:
        """Check if the job is completed."""
        return self.completed_at is not None
    
    @property
    def total_duration_seconds(self) -> float:
        """Get total processing time."""
        if not self.is_completed or not self.started_at:
            return 0.0
        return (self.completed_at - self.started_at).total_seconds()
    
    @property
    def final_transcription(self) -> str:
        """Get the final merged transcription."""
        if not self.results:
            return ""
        
        # Sort results by chunk index
        sorted_results = sorted(self.results, key=lambda r: r.chunk_index)
        
        # Combine all transcriptions
        all_lines = []
        for result in sorted_results:
            all_lines.extend(result.lines)
        
        return "\n".join(str(line) for line in all_lines)