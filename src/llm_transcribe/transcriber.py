"""Core transcription engine."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from .audio import AudioProcessor
from .llm_client import LLMClient
from .models import Config, TranscriptionJob, TranscriptionResult

logger = logging.getLogger(__name__)


class TranscriptionEngine:
    """Core engine for orchestrating transcription process."""
    
    def __init__(self, config: Config, model: str = "gemini-2.5-flash"):
        self.config = config
        self.model = model
        self.audio_processor = AudioProcessor(config)
        self.llm_client = LLMClient(model)
    
    def create_job(self, input_file: Path, output_file: Path, model: Optional[str] = None, context: Optional[str] = None) -> TranscriptionJob:
        """Create a new transcription job.
        
        Args:
            input_file: Path to input audio file
            output_file: Path to output transcription file
            model: Optional model override
            context: Optional context about the meeting for better transcription
            
        Returns:
            TranscriptionJob object
        """
        job_model = model or self.model
        
        job = TranscriptionJob(
            input_file=input_file,
            output_file=output_file,
            model=job_model,
            config=self.config,
            context=context
        )
        
        return job
    
    def prepare_job(self, job: TranscriptionJob) -> TranscriptionJob:
        """Prepare job by processing audio and creating chunks.
        
        Args:
            job: TranscriptionJob to prepare
            
        Returns:
            Updated TranscriptionJob with chunks
        """
        logger.info(f"Preparing transcription job for {job.input_file}")
        
        # Process audio file
        audio, chunks = self.audio_processor.process_file(job.input_file)
        
        # Update job with chunks
        job.chunks = chunks
        
        logger.info(f"Job prepared: {len(chunks)} chunks to process")
        return job
    
    def extract_context(self, previous_result: TranscriptionResult) -> str:
        """Extract context from previous transcription result.
        
        Args:
            previous_result: Previous chunk's transcription result
            
        Returns:
            Context string for next chunk
        """
        if not previous_result.lines:
            return ""
        
        # Get the last minute of transcription
        context = previous_result.get_last_minute_context(self.config.chunk_duration_minutes)
        
        if context:
            logger.debug(f"Extracted context: {len(context)} characters")
            return context
        else:
            # Fallback: use last few lines if timestamp parsing fails
            last_lines = previous_result.lines[-3:]  # Last 3 lines
            fallback_context = "\n".join(str(line) for line in last_lines)
            logger.debug(f"Using fallback context: {len(fallback_context)} characters")
            return fallback_context
    
    
    def process_chunk(self, job: TranscriptionJob, chunk_index: int, progress_callback: Optional[Callable] = None) -> TranscriptionResult:
        """Process a single chunk.
        
        Args:
            job: TranscriptionJob being processed
            chunk_index: Index of chunk to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            TranscriptionResult for the chunk
        """
        if chunk_index >= len(job.chunks):
            raise ValueError(f"Chunk index {chunk_index} out of range")
        
        chunk = job.chunks[chunk_index]
        
        # Extract context from previous chunk if available
        context = None
        if chunk_index > 0 and job.results:
            # Find the previous result
            previous_results = [r for r in job.results if r.chunk_index == chunk_index - 1]
            if previous_results:
                context = self.extract_context(previous_results[0])
        
        # Convert audio to bytes
        audio_bytes = self.audio_processor.export_chunk_to_bytes(chunk)
        
        # Call progress callback if provided
        if progress_callback:
            progress_callback(chunk_index, len(job.chunks), f"Processing chunk {chunk_index + 1}/{len(job.chunks)}")
        
        # Transcribe chunk (pass job context and chunk context)
        result = self.llm_client.transcribe_chunk(chunk, audio_bytes, context, job.context)
        
        # Timestamps are already absolute from llm_client.parse_transcription_response
        return result
    
    def process_job(self, job: TranscriptionJob, progress_callback: Optional[Callable] = None) -> TranscriptionJob:
        """Process entire transcription job.
        
        Args:
            job: TranscriptionJob to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            Completed TranscriptionJob
        """
        logger.info(f"Starting transcription job: {len(job.chunks)} chunks")
        
        job.started_at = datetime.now()
        
        try:
            # Process each chunk
            for chunk_index in range(len(job.chunks)):
                logger.info(f"Processing chunk {chunk_index + 1}/{len(job.chunks)}")
                
                result = self.process_chunk(job, chunk_index, progress_callback)
                job.results.append(result)
                
                # Log progress
                if result.lines:
                    logger.info(f"Chunk {chunk_index + 1} completed: {len(result.lines)} lines")
                else:
                    logger.warning(f"Chunk {chunk_index + 1} completed with no transcription")
            
            job.completed_at = datetime.now()
            logger.info(f"Transcription job completed in {job.total_duration_seconds:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing job: {e}")
            job.completed_at = datetime.now()
            raise
        
        return job
    
    def transcribe_file(self, 
                       input_file: Path, 
                       output_file: Path, 
                       model: Optional[str] = None,
                       context: Optional[str] = None,
                       progress_callback: Optional[Callable] = None) -> TranscriptionJob:
        """Transcribe an audio file from start to finish.
        
        Args:
            input_file: Path to input audio file
            output_file: Path to output transcription file
            model: Optional model override
            context: Optional context about the meeting for better transcription
            progress_callback: Optional callback for progress updates
            
        Returns:
            Completed TranscriptionJob
        """
        logger.info(f"Starting transcription: {input_file} -> {output_file}")
        
        # Create and prepare job
        job = self.create_job(input_file, output_file, model, context)
        job = self.prepare_job(job)
        
        # Process job
        job = self.process_job(job, progress_callback)
        
        return job
    
    def test_setup(self) -> bool:
        """Test that the transcription engine is properly configured.
        
        Returns:
            True if setup is valid, False otherwise
        """
        try:
            # Test LLM connection
            if not self.llm_client.test_connection():
                logger.error("LLM client connection test failed")
                return False
            
            logger.info("Transcription engine setup test passed")
            return True
            
        except Exception as e:
            logger.error(f"Setup test failed: {e}")
            return False