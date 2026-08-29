# Hebrew Transcriber

A tiny local CLI for transcribing Hebrew audio and video with [ivrit.ai](https://huggingface.co/ivrit-ai).

It uses:

- `faster-whisper` with `ivrit-ai/whisper-large-v3-turbo-ct2`
- `mlx-whisper` with `mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx`
  on Apple Silicon
- FFmpeg for video/audio conversion
- optional speaker diarization with `ivrit-ai/pyannote-speaker-diarization-3.1`

This is the same general workflow we use to transcribe episodes of **מפתחים מחוץ לקופסה**.

## What it creates

Give it an audio or video file:

```bash
python transcribe.py interview.mp4
```

It creates:

```text
output/
├── interview.txt
└── interview.srt
```

Each run also prints a timing summary for audio preparation, model loading,
transcription, file writing, optional diarization, and total wall-clock time.

For speaker diarization:

```bash
python transcribe.py interview.mp4 --speakers 2
```

It will also try to create:

```text
output/interview_with_speakers.txt
```

Example:

```text
SPEAKER_00:
ברוכים הבאים לפרק החדש.

SPEAKER_01:
כיף להיות פה.
```

Diarization distinguishes speakers, but it does not automatically know their real names.

## macOS setup

### 1. Install FFmpeg

With Homebrew:

```bash
brew install ffmpeg
```

### 2. Create a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

## Basic usage

Audio:

```bash
python transcribe.py recording.mp3
```

Video:

```bash
python transcribe.py interview.mp4
```

Known number of speakers:

```bash
python transcribe.py podcast.mp4 --speakers 2
```

Unknown number of speakers, but still run diarization:

```bash
python transcribe.py meeting.m4a --diarize
```

Choose a different output directory:

```bash
python transcribe.py interview.mp4 --output-dir transcripts
```

## Why `language="he"` is set explicitly

The ivrit.ai Whisper models are fine-tuned specifically for Hebrew. This script always passes `language="he"` instead of relying on automatic language detection.

## Video files

For video input, the script automatically runs FFmpeg and converts the audio to:

- WAV
- mono
- 16 kHz

Conceptually:

```text
MP4 / MOV
    |
    v
 FFmpeg
    |
    v
   WAV
    |
    +----------------+
    |                |
    v                v
 Whisper         Diarization
    |                |
    v                v
text + time      speaker + time
    |                |
    +-------+--------+
            |
            v
       TXT / SRT
```

## Hugging Face and diarization

Speaker diarization models may require accepting model terms or authenticating with Hugging Face, depending on the model and its upstream dependencies.

If transcription works but diarization fails, the script deliberately keeps the `.txt` and `.srt` files instead of failing the whole job.

## Performance

The default configuration is:

```text
backend=auto
device=auto
compute_type=int8
diarization_device=auto
cpu_threads=auto
```

On an Apple Silicon Mac with MLX installed, the first automatic run benchmarks
MLX and CTranslate2 on the same 60-second sample and caches the faster backend.
Later runs immediately reuse that choice. On other machines, automatic mode
uses CTranslate2. You can force or refresh the choice with:

```bash
python transcribe.py audio.mp3 --backend mlx
python transcribe.py audio.mp3 --backend ctranslate2
python transcribe.py audio.mp3 --retune-backend
```

The engines use different acceleration backends:

- MLX Whisper uses the Apple GPU on Apple Silicon.
- faster-whisper uses CTranslate2. It automatically uses CUDA on supported
  NVIDIA systems, but CTranslate2 does not support Apple Metal/MPS, so it uses
  the optimized ARM CPU path on Apple Silicon.
- speaker diarization uses PyTorch. It automatically prefers CUDA, then Apple
  Metal/MPS, then CPU. If an automatically selected GPU cannot run a model
  operation, the diarization pass retries on CPU.

You can override either decision:

```bash
python transcribe.py audio.mp3 --device cpu --diarization-device cpu
```

When CTranslate2 runs on CPU, the CLI also benchmarks a short sample with
several thread counts and caches the fastest result for that machine, model,
compute type, and CTranslate2 version. Override or refresh it with:

```bash
python transcribe.py audio.mp3 --cpu-threads 8
python transcribe.py audio.mp3 --retune-cpu-threads
```

For supported NVIDIA machines you can try:

```bash
python transcribe.py audio.mp3 --device cuda --compute-type float16 --diarization-device cuda
```

## Claude Code / Codex prompt

If you prefer to let a coding agent set everything up for you, paste this prompt into Claude Code or Codex:

```text
Set up this Hebrew transcription repository on my Mac.

1. Inspect the repository first.
2. Verify Homebrew and FFmpeg are installed. If FFmpeg is missing, install it with Homebrew.
3. Create a Python virtual environment in .venv.
4. Install requirements.txt.
5. Verify `python transcribe.py --help` works.
6. Do not change the transcription models unless there is a compatibility problem.
7. If I provide an audio or video file, run the transcription command for it.
8. Use --speakers when I tell you the known number of speakers.
9. Verify the generated TXT and SRT files exist in output/.
10. If diarization requires Hugging Face authentication or accepting model terms, explain exactly what I need to do and continue with normal transcription so I still get TXT and SRT.
11. Fix any local dependency or compatibility problem you encounter and explain the change briefly.
```

Then you can simply tell the agent:

```text
Transcribe ~/Downloads/interview.mp4. There are 2 speakers.
```

## Notes

- The first run downloads the selected model from Hugging Face.
- Large audio files can take a while on CPU.
- Speaker labels such as `SPEAKER_00` are identities within the recording, not real-world names.
- This project is intentionally small. It is meant to be easy to read, copy, modify, and run locally.
