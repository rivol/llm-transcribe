"""Command line interface for transcriber."""

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.prompt import Confirm

from .models import Config
from .output import OutputHandler
from .transcriber import TranscriptionEngine

# Create console for rich output
console = Console()

# Create Typer app
app = typer.Typer(
    name="transcriber",
    help="Audio transcription using Large Language Models",
    add_completion=False,
)

# Global variables
config = Config()
output_handler = OutputHandler()


def setup_logging(verbose: bool = False):
    """Setup logging with rich handler."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=False)]
    )
    
    # Reduce noise from external libraries
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pydub").setLevel(logging.WARNING)


def progress_callback(current: int, total: int, message: str):
    """Callback for progress updates."""
    # This will be used by the progress bar context
    pass


@app.command()
def main(
    audio_file: Path = typer.Argument(
        ...,
        help="Path to the audio file to transcribe",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    model: str = typer.Option(
        "gemini-2.5-flash",
        "-m", "--model",
        help="LLM model to use for transcription"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "-o", "--output",
        help="Output file path (default: input_file.txt)"
    ),
    chunk_duration: int = typer.Option(
        10,
        "--chunk-duration",
        help="Duration of each audio chunk in minutes"
    ),
    overlap_duration: int = typer.Option(
        1,
        "--overlap-duration",
        help="Overlap duration between chunks in minutes"
    ),
    export_formats: Optional[str] = typer.Option(
        None,
        "--export",
        help="Export formats: txt,json,report (comma-separated)"
    ),
    verbose: bool = typer.Option(
        False,
        "-v", "--verbose",
        help="Enable verbose logging"
    ),
    test: bool = typer.Option(
        False,
        "--test",
        help="Test setup and exit"
    ),
    yes: bool = typer.Option(
        False,
        "-y", "--yes",
        help="Skip confirmation prompts"
    )
):
    """Transcribe audio files using Large Language Models."""
    
    # Setup logging
    setup_logging(verbose)
    
    # Update config with command line options
    config.chunk_duration_minutes = chunk_duration
    config.overlap_duration_minutes = overlap_duration
    
    # Determine output file
    if output is None:
        output = audio_file.with_suffix('.txt')
    
    # Validate audio file extension
    file_extension = audio_file.suffix.lower().lstrip('.')
    if file_extension not in config.supported_formats:
        console.print(f"[red]Error:[/red] Unsupported audio format: {file_extension}")
        console.print(f"Supported formats: {', '.join(config.supported_formats)}")
        raise typer.Exit(1)
    
    # Parse export formats
    export_format_list = ['txt']  # Default
    if export_formats:
        export_format_list = [fmt.strip() for fmt in export_formats.split(',')]
    
    # Show configuration
    console.print(f"[blue]Input:[/blue] {audio_file}")
    console.print(f"[blue]Output:[/blue] {output}")
    console.print(f"[blue]Model:[/blue] {model}")
    console.print(f"[blue]Chunk Duration:[/blue] {chunk_duration} minutes")
    console.print(f"[blue]Overlap Duration:[/blue] {overlap_duration} minutes")
    console.print(f"[blue]Export Formats:[/blue] {', '.join(export_format_list)}")
    
    # Create transcription engine
    try:
        engine = TranscriptionEngine(config, model)
    except Exception as e:
        console.print(f"[red]Error creating transcription engine:[/red] {e}")
        raise typer.Exit(1)
    
    # Test setup if requested
    if test:
        console.print("[yellow]Testing setup...[/yellow]")
        if engine.test_setup():
            console.print("[green]Setup test passed![/green]")
            raise typer.Exit(0)
        else:
            console.print("[red]Setup test failed![/red]")
            raise typer.Exit(1)
    
    # Check if output file exists
    if output.exists() and not yes:
        if not Confirm.ask(f"Output file {output} already exists. Overwrite?"):
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(0)
    
    # Create and prepare job
    try:
        console.print("[yellow]Preparing transcription job...[/yellow]")
        job = engine.create_job(audio_file, output, model)
        job = engine.prepare_job(job)
        
        console.print(f"[green]Job prepared:[/green] {len(job.chunks)} chunks to process")
        
        # Estimate processing time
        audio_duration = job.chunks[-1].end_time_seconds if job.chunks else 0
        estimated_time = audio_duration * 0.3  # Rough estimate: 30% of audio duration
        console.print(f"[blue]Estimated processing time:[/blue] {estimated_time:.0f} seconds")
        
    except Exception as e:
        console.print(f"[red]Error preparing job:[/red] {e}")
        raise typer.Exit(1)
    
    # Process job with progress bar
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            
            task = progress.add_task("Transcribing...", total=len(job.chunks))
            
            def update_progress(current: int, total: int, message: str):
                progress.update(task, completed=current, description=message)
            
            # Process the job
            job = engine.process_job(job, update_progress)
            
            # Complete the progress bar
            progress.update(task, completed=len(job.chunks), description="Transcription complete!")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Transcription interrupted by user.[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Error during transcription:[/red] {e}")
        raise typer.Exit(1)
    
    # Export results
    try:
        console.print("[yellow]Exporting results...[/yellow]")
        output_handler.export_job_results(job, export_format_list)
        
        # Show completion message
        console.print(f"[green]Transcription completed successfully![/green]")
        console.print(f"[blue]Output written to:[/blue] {output}")
        
        # Show summary
        total_lines = sum(len(result.lines) for result in job.results)
        console.print(f"[blue]Total lines:[/blue] {total_lines}")
        console.print(f"[blue]Processing time:[/blue] {job.total_duration_seconds:.2f} seconds")
        
        # Show report file if generated
        if 'report' in export_format_list:
            report_file = output.with_suffix('.report.txt')
            console.print(f"[blue]Summary report:[/blue] {report_file}")
        
    except Exception as e:
        console.print(f"[red]Error exporting results:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def test_setup(
    model: str = typer.Option(
        "gemini-2.5-flash",
        "-m", "--model",
        help="LLM model to test"
    ),
    verbose: bool = typer.Option(
        False,
        "-v", "--verbose",
        help="Enable verbose logging"
    )
):
    """Test the transcription setup."""
    setup_logging(verbose)
    
    console.print(f"[blue]Testing setup with model:[/blue] {model}")
    
    try:
        engine = TranscriptionEngine(config, model)
        if engine.test_setup():
            console.print("[green]Setup test passed![/green]")
            console.print("✓ LLM connection working")
            console.print("✓ All dependencies available")
        else:
            console.print("[red]Setup test failed![/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error during setup test:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    from . import __version__
    console.print(f"transcriber v{__version__}")


if __name__ == "__main__":
    app()