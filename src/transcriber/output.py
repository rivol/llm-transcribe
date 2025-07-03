"""Output handling and formatting."""

import logging
from pathlib import Path
from typing import List, Optional

from .models import TranscriptionJob, TranscriptionResult

logger = logging.getLogger(__name__)


class OutputHandler:
    """Handles formatting and writing transcription outputs."""
    
    def __init__(self):
        pass
    
    def format_transcription(self, job: TranscriptionJob) -> str:
        """Format transcription job results into final output.
        
        Args:
            job: Completed TranscriptionJob
            
        Returns:
            Formatted transcription text
        """
        if not job.results:
            return ""
        
        # Sort results by chunk index to ensure proper order
        sorted_results = sorted(job.results, key=lambda r: r.chunk_index)
        
        # Collect all lines from all chunks
        all_lines = []
        for result in sorted_results:
            all_lines.extend(result.lines)
        
        # Format each line
        formatted_lines = []
        for line in all_lines:
            formatted_lines.append(str(line))
        
        return "\n".join(formatted_lines)
    
    def deduplicate_overlapping_content(self, job: TranscriptionJob) -> List[str]:
        """Remove duplicate content from overlapping chunks.
        
        Args:
            job: TranscriptionJob with results
            
        Returns:
            List of formatted lines with duplicates removed
        """
        if not job.results:
            return []
        
        # Sort results by chunk index
        sorted_results = sorted(job.results, key=lambda r: r.chunk_index)
        
        all_lines = []
        last_timestamp_seconds = -1
        
        for result in sorted_results:
            for line in result.lines:
                # Parse timestamp to get seconds
                timestamp_str = line.timestamp.strip('[]')
                try:
                    time_parts = timestamp_str.split(':')
                    line_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
                    
                    # Only include lines that are after the last timestamp
                    # This helps remove duplicates from overlapping chunks
                    if line_seconds > last_timestamp_seconds:
                        all_lines.append(str(line))
                        last_timestamp_seconds = line_seconds
                        
                except (ValueError, IndexError):
                    # If timestamp parsing fails, include the line anyway
                    all_lines.append(str(line))
        
        return all_lines
    
    def write_transcription_file(self, job: TranscriptionJob, deduplicate: bool = True) -> None:
        """Write transcription to output file.
        
        Args:
            job: Completed TranscriptionJob
            deduplicate: Whether to remove duplicate content from overlaps
        """
        if not job.results:
            logger.warning("No results to write")
            return
        
        logger.info(f"Writing transcription to {job.output_file}")
        
        # Format content
        if deduplicate:
            lines = self.deduplicate_overlapping_content(job)
            content = "\n".join(lines)
        else:
            content = self.format_transcription(job)
        
        # Ensure output directory exists
        job.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        with open(job.output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Transcription written: {len(content)} characters")
    
    def create_summary_report(self, job: TranscriptionJob) -> str:
        """Create a summary report of the transcription job.
        
        Args:
            job: Completed TranscriptionJob
            
        Returns:
            Summary report as string
        """
        if not job.is_completed:
            return "Job not completed"
        
        # Calculate statistics
        total_lines = sum(len(result.lines) for result in job.results)
        total_processing_time = sum(result.processing_time_seconds for result in job.results)
        avg_processing_time = total_processing_time / len(job.results) if job.results else 0
        
        # Count successful chunks
        successful_chunks = sum(1 for result in job.results if result.lines)
        failed_chunks = len(job.results) - successful_chunks
        
        # Get audio duration if available
        audio_duration = 0
        if job.chunks:
            audio_duration = job.chunks[-1].end_time_seconds
        
        report = f"""Transcription Summary Report
{'=' * 50}

Input File: {job.input_file}
Output File: {job.output_file}
Model Used: {job.model}

Audio Information:
- Duration: {audio_duration:.2f} seconds ({audio_duration/60:.2f} minutes)
- Total Chunks: {len(job.chunks)}
- Chunk Duration: {job.config.chunk_duration_minutes} minutes
- Overlap Duration: {job.config.overlap_duration_minutes} minutes

Processing Results:
- Successful Chunks: {successful_chunks}
- Failed Chunks: {failed_chunks}
- Total Lines: {total_lines}
- Total Processing Time: {total_processing_time:.2f} seconds
- Average Time per Chunk: {avg_processing_time:.2f} seconds
- Job Duration: {job.total_duration_seconds:.2f} seconds

Performance:
- Audio/Processing Ratio: {audio_duration/total_processing_time:.2f}x
- Real-time Factor: {total_processing_time/audio_duration:.2f}
"""
        
        return report
    
    def write_summary_report(self, job: TranscriptionJob, report_file: Optional[Path] = None) -> None:
        """Write summary report to file.
        
        Args:
            job: Completed TranscriptionJob
            report_file: Optional path for report file (defaults to output_file.report.txt)
        """
        if report_file is None:
            report_file = job.output_file.with_suffix('.report.txt')
        
        report = self.create_summary_report(job)
        
        logger.info(f"Writing summary report to {report_file}")
        
        # Ensure output directory exists
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Summary report written: {report_file}")
    
    def format_for_json(self, job: TranscriptionJob) -> dict:
        """Format transcription job as JSON-serializable dictionary.
        
        Args:
            job: TranscriptionJob to format
            
        Returns:
            Dictionary representation
        """
        return {
            "input_file": str(job.input_file),
            "output_file": str(job.output_file),
            "model": job.model,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "total_duration_seconds": job.total_duration_seconds,
            "chunks": [
                {
                    "chunk_index": chunk.chunk_index,
                    "start_time_seconds": chunk.start_time_seconds,
                    "end_time_seconds": chunk.end_time_seconds,
                }
                for chunk in job.chunks
            ],
            "results": [
                {
                    "chunk_index": result.chunk_index,
                    "lines": [
                        {
                            "timestamp": line.timestamp,
                            "speaker": line.speaker,
                            "text": line.text
                        }
                        for line in result.lines
                    ],
                    "model_used": result.model_used,
                    "processing_time_seconds": result.processing_time_seconds
                }
                for result in job.results
            ],
            "final_transcription": job.final_transcription
        }
    
    def export_job_results(self, job: TranscriptionJob, export_formats: List[str] = None) -> None:
        """Export job results in multiple formats.
        
        Args:
            job: Completed TranscriptionJob
            export_formats: List of formats to export ('txt', 'json', 'report')
        """
        if export_formats is None:
            export_formats = ['txt']
        
        for format_type in export_formats:
            if format_type == 'txt':
                self.write_transcription_file(job)
            elif format_type == 'json':
                import json
                json_file = job.output_file.with_suffix('.json')
                data = self.format_for_json(job)
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                logger.info(f"JSON export written: {json_file}")
            elif format_type == 'report':
                self.write_summary_report(job)
            else:
                logger.warning(f"Unknown export format: {format_type}")