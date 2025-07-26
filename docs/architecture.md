# LLM Transcribe Architecture

## Overview

LLM Transcribe is a Python CLI tool that transcribes audio files using Large Language Models (LLMs). It handles long audio files by splitting them into overlapping chunks, transcribing each chunk with context from previous chunks, and producing timestamped transcriptions with speaker identification.

## Technology Stack

- **CLI**: Typer - Modern, type-safe CLI framework
- **LLM Integration**: LiteLLM - Unified interface for all LLM providers
- **Data Models**: Pydantic - Type-safe data validation and configuration
- **Audio Processing**: pydub - Simple Python audio manipulation

## Core Components

### 1. CLI Interface (`cli.py`)
- **Responsibility**: Command-line interface using Typer
- **Key Functions**:
  - Parse arguments (`-m model`, `-o output`, audio file path)
  - Validate input files and parameters with rich error messages
  - Display help and usage information with Typer's auto-generation
  - Progress bars and user-friendly output

### 2. Data Models (`models.py`)
- **Responsibility**: Pydantic models for all data structures
- **Key Models**:
  - `Config`: Application settings and defaults
  - `ChunkData`: Audio chunk metadata (start_time, end_time, audio_segment)
  - `TranscriptionResult`: LLM response with timestamps and speakers
  - `TranscriptionChunk`: Individual chunk transcription data
- **Benefits**:
  - Type safety and validation
  - Easy serialization/deserialization
  - Auto-generated documentation

### 3. Audio Processor (`audio.py`)
- **Responsibility**: Audio file handling and chunking using pydub
- **Key Functions**:
  - Load and validate audio files (all formats supported by pydub)
  - Split audio into 10-minute chunks with 1-minute overlap
  - Generate ChunkData objects for each segment
  - Handle format conversion if needed
- **Implementation Notes**:
  - Uses pydub for all audio operations
  - Chunk timing: [0:00-10:00], [9:00-20:00], [19:00-30:00], etc.
  - No temporary files needed - audio kept in memory

### 4. LLM Client (`llm_client.py`)
- **Responsibility**: LiteLLM wrapper for unified LLM access
- **Key Functions**:
  - Send audio + context to any LLM provider via LiteLLM
  - Handle API rate limiting and retries
  - Parse and validate LLM responses
  - Support all models through LiteLLM (gemini-2.5-flash, gpt-4, claude, etc.)
- **Benefits**:
  - No provider-specific code needed
  - Automatic API key management
  - Built-in retry logic and error handling

### 5. Transcription Engine (`transcriber.py`)
- **Responsibility**: Core orchestration with integrated context management
- **Key Functions**:
  - Coordinate audio processing and LLM calls
  - Manage context between chunks (integrated, not separate module)
  - Handle progress tracking and error recovery
  - Extract last 1 minute of previous transcription for context
  - Format context for next chunk's system message
  - Merge chunk transcriptions into final output
- **Context Management**:
  - Extract relevant context from previous TranscriptionResult
  - Maintain speaker continuity across chunks
  - Handle edge cases (silence, background noise)

### 6. Output Handler (`output.py`)
- **Responsibility**: Result formatting and file I/O
- **Key Functions**:
  - Format TranscriptionResult objects into final output
  - Handle timestamp adjustments and formatting
  - Write final transcription file
  - Support different output formats (if needed in future)
- **Output Format**:
  ```
  [01:23:45] John: We should start.
  [01:24:02] Speaker 2: I agree, let's begin.
  ```

## Data Flow

```
Audio File → Audio Processor → ChunkData[]
                ↓
ChunkData → Transcription Engine → LLM Client → TranscriptionResult
                ↓
Previous Context ← Integrated Context Management
                ↓
Output Handler → Final Transcript File
```

### Detailed Flow
1. **Input Processing**: Typer CLI validates input file and parameters
2. **Audio Chunking**: Audio processor loads file with pydub and creates ChunkData objects
3. **Iterative Transcription**: For each ChunkData:
   - Extract context from previous TranscriptionResult (if exists)
   - Send audio chunk + context to LLM via LiteLLM
   - Receive and validate TranscriptionResult
   - Store for next iteration's context
4. **Output Generation**: Output handler formats all TranscriptionResults into final file

## Key Design Decisions

### Chunking Strategy
- **10-minute chunks**: Balance between context preservation and LLM limits
- **1-minute overlap**: Ensures continuity and prevents context loss
- **Sliding window**: Each chunk starts 9 minutes after the previous

### Context Preservation
- **Last 1 minute rule**: Only the final minute of previous transcription is used as context
- **Speaker continuity**: Context includes speaker names to maintain consistency
- **Format consistency**: Context is formatted similarly to expected output

### LLM Integration
- **Unified interface**: LiteLLM provides single interface for all providers
- **Error handling**: Built-in retry logic and rate limiting via LiteLLM
- **Response validation**: Pydantic models ensure output matches expected format

### Speaker Identification
- **Priority**: Named speakers (John, Cate) > Role-based (Manager, Salesperson) > Numerical (Speaker 1, Speaker 2)
- **Consistency**: Speaker labels maintained across chunks using context
- **Fallback**: Unknown speakers get numerical labels starting from Speaker 1

## Extension Points

### New LLM Providers
- No code changes needed - LiteLLM supports 100+ providers out of the box
- Just update model configuration in Pydantic models

### Audio Format Support
- Already handled by pydub - supports all common formats
- No additional code needed for new formats

### Output Formats
- Add new format methods to `output.py`
- Add CLI options via Typer for format selection
- Use Pydantic models for format-specific configuration

## Technical Considerations

### Memory Management
- Audio kept in memory via pydub (acceptable for typical file sizes)
- No temporary files needed
- Context size limited to last 1 minute of transcription

### Error Handling
- Pydantic validation for all data structures
- LiteLLM handles API failures and retries automatically
- Typer provides rich error messages with helpful formatting

### Performance
- Sequential processing keeps memory usage predictable
- Typer progress bars for user feedback
- Future: async LLM calls if needed

### Security
- LiteLLM handles secure API key management
- Pydantic validates all inputs
- No temporary files to clean up

## Implementation Priority

1. **Data Models**: Pydantic models for all data structures
2. **Audio Processing**: pydub-based chunking and validation
3. **LLM Client**: LiteLLM wrapper for transcription
4. **Core Engine**: Orchestration with integrated context management
5. **CLI & Output**: Typer interface and result formatting

## Project Structure

```
llm_transcribe/
├── cli.py              # Typer-based CLI interface
├── models.py           # Pydantic data models
├── audio.py            # pydub audio processing  
├── llm_client.py       # LiteLLM wrapper
├── transcriber.py      # Core orchestration engine
└── output.py           # Result formatting and file I/O
```

## Benefits of This Architecture

- **Simplicity**: 6 focused modules vs 9+ in original design
- **Maintainability**: Leverage mature libraries instead of custom abstractions
- **Type Safety**: Pydantic models provide validation and documentation
- **Flexibility**: LiteLLM supports any provider without code changes
- **Developer Experience**: Typer provides excellent CLI ergonomics
- **Future-proof**: Easy to extend without architectural changes