# Hebrew Transcriber

Transcribe Hebrew audio and video locally. The tool creates a plain-text
transcript, subtitles, and optionally a version labeled by speaker.

It automatically finds the fastest transcription engine for your computer.
On Apple Silicon Macs, it can use the Apple GPU through MLX.

## Install on macOS

You only need to do this once.

### 1. Install Homebrew

Open **Terminal** and run:

```bash
brew --version
```

If that prints a version number, continue to step 2. If it says
`command not found`, install Homebrew from [brew.sh](https://brew.sh), then
close and reopen Terminal.

### 2. Install Python, FFmpeg, and Git

Copy and paste this command into Terminal:

```bash
brew install python@3.12 ffmpeg git
```

### 3. Download and install Hebrew Transcriber

Copy and paste this entire block:

```bash
git clone https://github.com/shahar84/hebrew-transcription.git
cd hebrew-transcription
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python transcribe.py --help
```

Installation is successful when the last command displays the available
options without an error.

## Transcribe your first file

While still in the same Terminal window, run:

```bash
python transcribe.py "/full/path/to/video.mp4"
```

The quotation marks are important when the path contains spaces. On macOS, you
can type `python transcribe.py ` and then drag the file from Finder into the
Terminal window to insert its complete path.

For example:

```bash
python transcribe.py "/Users/your-name/Downloads/interview.mp4"
```

The results appear in the repository's `output` folder:

```text
output/
├── interview.txt
└── interview.srt
```

- `.txt` is the readable transcript.
- `.srt` contains timed subtitles for video players and editors.

## Use it again later

Every time you open a new Terminal window, run these commands first:

```bash
cd ~/hebrew-transcription
source .venv/bin/activate
```

Then transcribe a file:

```bash
python transcribe.py "/full/path/to/file.mp4"
```

When you are finished, you can leave the virtual environment with:

```bash
deactivate
```

## Identify different speakers

If you know the number of speakers, add `--speakers`:

```bash
python transcribe.py "/full/path/to/interview.mp4" --speakers 2
```

This also creates:

```text
output/interview_with_speakers.txt
```

The labels look like `SPEAKER_00` and `SPEAKER_01`. The software separates the
voices, but it does not know their real names.

Speaker identification may require a free Hugging Face account and permission
to use the diarization model. After accepting any model terms, log in from
Terminal with:

```bash
hf auth login
```

Normal transcription still produces `.txt` and `.srt` files if speaker
identification cannot run.

## Automatic speed selection

The normal command uses `--backend auto`. On the first run, the software:

1. Tests CTranslate2 and, when available, Apple MLX on the same 60-second sample.
2. Selects the faster engine.
3. Saves that choice for future runs.

The first run is therefore slower because it downloads the models and performs
the one-time tests. Later runs start immediately with the saved engine.

To repeat the speed test:

```bash
python transcribe.py "audio.mp3" --retune-backend
```

To choose an engine yourself:

```bash
python transcribe.py "audio.mp3" --backend mlx
python transcribe.py "audio.mp3" --backend ctranslate2
```

MLX works only on Apple Silicon Macs. On other computers, automatic mode uses
CTranslate2. If MLX fails during an automatic run, the software retries with
CTranslate2.

When CTranslate2 uses the CPU, the software also tests different CPU thread
counts once and remembers the fastest setting. To repeat that test:

```bash
python transcribe.py "audio.mp3" --retune-cpu-threads
```

## More useful commands

Transcribe audio:

```bash
python transcribe.py "recording.mp3"
```

Transcribe video with two speakers:

```bash
python transcribe.py "interview.mp4" --speakers 2
```

Run speaker identification without specifying the number of speakers:

```bash
python transcribe.py "meeting.m4a" --diarize
```

Choose a different output folder:

```bash
python transcribe.py "interview.mp4" --output-dir transcripts
```

Show every available option:

```bash
python transcribe.py --help
```

## Update the project

From the repository folder:

```bash
git pull
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Common problems

### `brew: command not found`

Install Homebrew from [brew.sh](https://brew.sh), then reopen Terminal.

### Python cannot be found

Install Python and create the virtual environment using its full path:

```bash
brew install python@3.12
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
```

### `ffmpeg: command not found`

Run:

```bash
brew install ffmpeg
```

### The file cannot be found

Put quotation marks around the complete path, especially when it contains
spaces or Hebrew characters:

```bash
python transcribe.py "/Users/your-name/Downloads/My Video/video.mp4"
```

### Speaker identification fails

Confirm that you accepted the Hugging Face model terms and ran:

```bash
hf auth login
```

## What runs behind the scenes

- [ivrit.ai](https://huggingface.co/ivrit-ai) provides the Hebrew Whisper models.
- `mlx-whisper` uses the Apple GPU on supported Apple Silicon Macs.
- `faster-whisper` and CTranslate2 provide the CPU and NVIDIA CUDA engine.
- FFmpeg converts video and audio to the required format.
- `pyannote.audio` optionally separates speakers.

Everything runs locally after the models have been downloaded.
