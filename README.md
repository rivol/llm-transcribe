# Transcriber

Generates transcriptions for audio files:

```sh
# Simplest way, result will be in audio.txt
transcriber audio.wav

# Specify model and output filename
transcriber -m gemini-2.5-flash -o audio.out.txt audio.wav
```


## How it Works

1. Audio file is split into 10 minute chunks.
   NB: Chunks will have 1-minute overlap, so they're technically up to 11 minutes long. First chunk will be 0:00 - 10:00, second will be 9:00 - 20:00, third one 19:00 - 30:00, etc.
2. Each chunk is transcribed with LLM.
   LLM will receive: system message, audio chunk, last 1 minute of previous chunk's transcription output (so that it can continue).
   LLM will output lines with timestamp, speaker, text. E.g "[01:23:45] John: We should start."
   If speaker's cannot be identified by name (e.g John, Cate) or role (e.g manager, salesperson), they should be labeled numerically starting with "Speaker 1".
