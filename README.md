# LLM Transcribe

LLM Transcribe is a Python CLI tool that leverages state-of-the-art Large Language Models
to convert audio files into richly annotated transcriptions with timestamps,
speaker identification, and emotional context.

Built for developers and technical users who need high-quality transcription with minimal setup,
it supports all major LLM providers (Google Gemini, OpenAI, Anthropic) through a single interface,
handles arbitrarily long audio files—from quick voice notes to multi-hour meetings—and
produces structured output that captures not just what was said but how it was said,
including non-verbal cues like laughter, hesitation, or frustration.

## Installation

We recommend using [uv](https://docs.astral.sh/uv/)
```sh
uv tool install llm-transcribe
```

After installation, the `llm-transcribe` command will be available in your PATH.

## Usage

```sh
# Simplest way, result will be in audio.txt
llm-transcribe audio.wav

# Specify model and output filename
llm-transcribe -m gemini-2.5-flash -o audio.out.txt audio.wav

# Test setup without processing audio
llm-transcribe audio.wav --test

# Add context for better transcription
llm-transcribe -c "Meeting between John and Kate about Q3 revenue projections" audio.wav

# Get help
llm-transcribe --help
```

## How it Works

1. **Audio Processing**: Audio file is split into 10-minute chunks with 1-minute overlap using a sliding-window approach.
   - Chunks are technically up to 11 minutes long for context preservation
   - Timing: [0:00-10:00], [9:00-20:00], [19:00-30:00], etc.
   - Supports all common audio formats (WAV, MP3, M4A, etc.)

2. **LLM Transcription**: Each chunk is transcribed using modern LLMs instead of traditional speech-to-text methods.
   - Audio converted to base64 and sent with carefully crafted prompts
   - Context from previous chunk's final minute maintains speaker consistency
   - Superior accuracy through understanding context and capturing emotional nuances
   - Output format: timestamped lines with speaker identification and non-verbal cues

3. **Context Management**: Maintains conversation flow across chunks.
   - Previous chunk's final minute provides context for the next chunk
   - Speaker names and roles preserved throughout entire transcription
   - Handles complex timestamp arithmetic (chunk-relative to absolute time)

4. **Production Ready**: Built on modern Python stack with robust error handling.
   - Retry logic for API failures and rate limiting
   - Cost tracking and provider abstraction via LiteLLM
   - Supports all major LLM providers through unified interface

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
