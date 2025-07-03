# Transcriber

A Python CLI tool that generates accurate transcriptions for audio files using Large Language Models (LLMs). Handles long audio files by processing them in overlapping chunks while maintaining speaker continuity and context.

## Usage

```sh
# Simplest way, result will be in audio.txt
transcriber audio.wav

# Specify model and output filename
transcriber -m gemini-2.5-flash -o audio.out.txt audio.wav

# Get help
transcriber --help
```

## How it Works

1. **Audio Processing**: Audio file is split into 10-minute chunks with 1-minute overlap.
   - Chunks are technically up to 11 minutes long for context preservation
   - Timing: [0:00-10:00], [9:00-20:00], [19:00-30:00], etc.
   - Supports all common audio formats (WAV, MP3, M4A, etc.)

2. **LLM Transcription**: Each chunk is transcribed using your chosen LLM.
   - LLM receives: system message, audio chunk, context from previous chunk
   - Context includes the last 1 minute of previous transcription for continuity
   - Output format: timestamped lines with speaker identification

3. **Context Management**: Maintains speaker continuity across chunks.
   - Previous chunk's final minute provides context for the next chunk
   - Speaker names and roles are preserved across the entire transcription
   - Handles transitions and maintains conversation flow

4. **Result Merging**: Individual chunk transcriptions are merged into final output.
   - Timestamps are adjusted to reflect absolute time in the original audio
   - Speaker labels are consistent throughout the entire transcription
   - Duplicate content in overlapping regions is intelligently handled

## Output Format

The transcription output includes timestamps and speaker identification:

```
[01:23:45] John: We should start the meeting now.
[01:23:52] Sarah: I agree. Let me pull up the agenda.
[01:24:05] Speaker 1: Can everyone hear me clearly?
[01:24:08] John: Yes, you're coming through fine.
```

### Speaker Identification Priority

1. **Named speakers**: When identifiable by name (John, Sarah, etc.)
2. **Role-based**: When identifiable by role (Manager, Salesperson, etc.)
3. **Numerical fallback**: Unknown speakers labeled as "Speaker 1", "Speaker 2", etc.

## Supported Models

Works with any LLM provider supported by LiteLLM:
- **Gemini**: `gemini-2.5-flash`, `gemini-1.5-pro`
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`
- **Anthropic**: `claude-3-5-sonnet`, `claude-3-5-haiku`
- **And many more**: See [LiteLLM documentation](https://docs.litellm.ai/docs/providers)

## Requirements

- Python 3.13
- FFmpeg (for audio processing)
- API key for your chosen LLM provider

## Configuration

Set your API key as an environment variable:

```sh
# For Gemini
export GOOGLE_API_KEY=your_api_key_here

# For OpenAI
export OPENAI_API_KEY=your_api_key_here

# For Anthropic
export ANTHROPIC_API_KEY=your_api_key_here
```
