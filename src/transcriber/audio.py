"""Audio processing using pydub."""

import logging
from pathlib import Path
from typing import List

from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from .models import ChunkData, Config

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio file processing and chunking."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def load_audio(self, file_path: Path) -> AudioSegment:
        """Load audio file and return AudioSegment.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            AudioSegment object
            
        Raises:
            FileNotFoundError: If file doesn't exist
            CouldntDecodeError: If file can't be decoded
            ValueError: If file format is not supported
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Check if file extension is supported
        file_extension = file_path.suffix.lower().lstrip('.')
        if file_extension not in self.config.supported_formats:
            raise ValueError(
                f"Unsupported audio format: {file_extension}. "
                f"Supported formats: {', '.join(self.config.supported_formats)}"
            )
        
        try:
            logger.info(f"Loading audio file: {file_path}")
            audio = AudioSegment.from_file(str(file_path))
            logger.info(f"Audio loaded: {len(audio) / 1000:.2f}s duration, {audio.frame_rate}Hz sample rate")
            return audio
        except CouldntDecodeError as e:
            raise CouldntDecodeError(f"Could not decode audio file {file_path}: {e}")
    
    def create_chunks(self, audio: AudioSegment) -> List[ChunkData]:
        """Split audio into overlapping chunks.
        
        Args:
            audio: AudioSegment to split
            
        Returns:
            List of ChunkData objects
        """
        chunks = []
        
        # Convert durations to milliseconds
        chunk_duration_ms = self.config.chunk_duration_minutes * 60 * 1000
        overlap_duration_ms = self.config.overlap_duration_minutes * 60 * 1000
        step_size_ms = chunk_duration_ms - overlap_duration_ms
        
        total_duration_ms = len(audio)
        
        logger.info(f"Creating chunks: {chunk_duration_ms / 1000:.0f}s chunks with {overlap_duration_ms / 1000:.0f}s overlap")
        
        chunk_index = 0
        start_time_ms = 0
        
        while start_time_ms < total_duration_ms:
            # Calculate end time for this chunk
            end_time_ms = min(start_time_ms + chunk_duration_ms, total_duration_ms)
            
            # Extract audio segment for this chunk
            chunk_audio = audio[start_time_ms:end_time_ms]
            
            # Create ChunkData object
            chunk_data = ChunkData(
                start_time_seconds=start_time_ms / 1000,
                end_time_seconds=end_time_ms / 1000,
                chunk_index=chunk_index,
                audio_segment=chunk_audio
            )
            
            chunks.append(chunk_data)
            
            logger.debug(f"Created chunk {chunk_index}: {chunk_data.start_time_seconds:.1f}s - {chunk_data.end_time_seconds:.1f}s")
            
            # Move to next chunk
            chunk_index += 1
            start_time_ms += step_size_ms
            
            # Break if we've reached the end
            if end_time_ms >= total_duration_ms:
                break
        
        logger.info(f"Created {len(chunks)} chunks from {total_duration_ms / 1000:.2f}s audio")
        return chunks
    
    def export_chunk_to_bytes(self, chunk: ChunkData, format: str = "wav") -> bytes:
        """Export audio chunk to bytes.
        
        Args:
            chunk: ChunkData with audio segment
            format: Audio format for export
            
        Returns:
            Audio data as bytes
        """
        if chunk.audio_segment is None:
            raise ValueError("ChunkData has no audio segment")
        
        # Export to bytes
        from io import BytesIO
        buffer = BytesIO()
        chunk.audio_segment.export(buffer, format=format)
        return buffer.getvalue()
    
    def get_audio_info(self, audio: AudioSegment) -> dict:
        """Get information about audio file.
        
        Args:
            audio: AudioSegment to analyze
            
        Returns:
            Dictionary with audio information
        """
        return {
            "duration_seconds": len(audio) / 1000,
            "sample_rate": audio.frame_rate,
            "channels": audio.channels,
            "sample_width": audio.sample_width,
            "frame_count": audio.frame_count(),
            "max_possible_amplitude": audio.max_possible_amplitude
        }
    
    def process_file(self, file_path: Path) -> tuple[AudioSegment, List[ChunkData]]:
        """Process audio file and return audio and chunks.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (AudioSegment, List[ChunkData])
        """
        # Load audio
        audio = self.load_audio(file_path)
        
        # Create chunks
        chunks = self.create_chunks(audio)
        
        # Log summary
        info = self.get_audio_info(audio)
        logger.info(f"Processed audio file: {info['duration_seconds']:.2f}s, {len(chunks)} chunks")
        
        return audio, chunks