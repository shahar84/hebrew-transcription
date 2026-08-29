# Hebrew Transcriber

Transcribe Hebrew audio and video locally. The tool creates a plain-text
transcript, subtitles, and optionally a version labeled by speaker.

It automatically finds the fastest transcription engine for your computer.
On Apple Silicon Macs, it can use the Apple GPU through MLX.

It supports macOS, Linux, and Windows. Linux and Windows can use an NVIDIA GPU
through CUDA when the required NVIDIA libraries are installed; otherwise they
run CTranslate2 on the CPU.

## Table of contents

- [What to expect](#what-to-expect)
- [Quick start](#quick-start)
- [Install on macOS](#install-on-macos)
- [Install on Linux](#install-on-linux)
- [Install on Windows](#install-on-windows)
- [Transcribe your first file](#transcribe-your-first-file)
- [Supported formats](#supported-formats)
- [Use it again later](#use-it-again-later)
- [Identify different speakers](#identify-different-speakers)
- [Automatic speed selection](#automatic-speed-selection)
- [More useful commands](#more-useful-commands)
- [Update the project](#update-the-project)
- [Limitations and FAQ](#limitations-and-faq)
- [Common problems](#common-problems)
- [Uninstalling](#uninstalling)
- [What runs behind the scenes](#what-runs-behind-the-scenes)
- [Contributing and issues](#contributing-and-issues)
- [License](#license)

## What to expect

- **Model downloads:** the current
  [CTranslate2 model](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ct2)
  is **1.622 GB** and the current
  [MLX model](https://huggingface.co/mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx)
  is **1.614 GB**. Apple Silicon automatic mode downloads both, for **3.24 GB
  total**. Linux, Windows, and Intel Macs download only the 1.622 GB
  CTranslate2 model.
- **Disk space:** on Apple Silicon, allow at least **5 GB free** for the two
  models and Python dependencies, plus space for your own media and output.
  Dependency sizes vary by operating system and CUDA setup, so there is no
  single exact total for every Linux or Windows computer.
- **First-run time:** the first run downloads the models and performs one-time
  speed tests, so its total time depends mostly on your internet connection and
  hardware. On the M5 Max used to test this project, expect roughly one minute
  of one-time tuning after the downloads; a 6-minute video then takes about
  8-10 seconds on later MLX runs. CPU and NVIDIA timings will differ.

## Quick start

macOS or Linux:

```bash
git clone https://github.com/shahar84/hebrew-transcription.git
cd hebrew-transcription
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python transcribe.py --help
```

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

## Install on Linux

These commands are for Ubuntu and Debian. Other Linux distributions need the
equivalent Python 3.10-or-newer, FFmpeg, Git, and virtual-environment packages.
Confirm that `python3 --version` reports Python 3.10 or newer before continuing.

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg git
git clone https://github.com/shahar84/hebrew-transcription.git
cd hebrew-transcription
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python transcribe.py --help
```

For NVIDIA acceleration, install **CUDA 12 with cuBLAS and cuDNN 9** as
described in the
[faster-whisper GPU requirements](https://github.com/SYSTRAN/faster-whisper#gpu).
The normal `--backend auto` mode will detect a working NVIDIA GPU; no MLX
installation is needed on Linux.

## Install on Windows

First install [Python 3.12](https://www.python.org/downloads/),
[Git](https://git-scm.com/download/win), and
[FFmpeg](https://ffmpeg.org/download.html). Make sure each command is available
in `PATH`, then open PowerShell and run:

```powershell
git clone https://github.com/shahar84/hebrew-transcription.git
cd hebrew-transcription
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python transcribe.py --help
```

For NVIDIA acceleration, install **CUDA 12 with cuBLAS and cuDNN 9** as
described in the
[faster-whisper GPU requirements](https://github.com/SYSTRAN/faster-whisper#gpu).
Automatic mode uses the GPU when CTranslate2 detects it and otherwise uses the
CPU.

## Transcribe your first file

While still in the same Terminal or PowerShell window, run:

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

For example, the text file contains one transcribed segment per line:

```text
שלום וברוכים הבאים.
היום נדבר על בינה מלאכותית.
תודה שהצטרפתם אלינו.
```

The subtitle file includes the same text with timestamps:

```srt
1
00:00:00,000 --> 00:00:03,200
שלום וברוכים הבאים.
```

## Supported formats

The tool accepts these file types:

- **Audio:** `.wav`, `.mp3`, `.m4a`, `.aac`, `.flac`, `.ogg`, `.opus`
- **Video:** `.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`, `.m4v`

Audio and video that need conversion are converted to mono, 16 kHz WAV with
FFmpeg before transcription. An unsupported extension stops the run with an
`Unsupported file format` error; no transcript or subtitle file is created.

## Use it again later

Every time you open a new Terminal or PowerShell window, first change into
wherever you cloned the repository. For example:

```bash
cd /path/to/hebrew-transcription
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
cd C:\path\to\hebrew-transcription
.venv\Scripts\Activate.ps1
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

Speaker identification uses the
[configured ivrit.ai diarization model](https://huggingface.co/ivrit-ai/pyannote-speaker-diarization-3.1).
It is currently public, but if Hugging Face asks for authentication or displays
terms for your account, accept them on that page and then log in from Terminal:

```bash
hf auth login
```

Normal transcription still produces `.txt` and `.srt` files if speaker
identification cannot run.

## Automatic speed selection

MLX and CTranslate2 are two different engines that do the same transcription
job at different speeds, depending on your hardware.

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

## Limitations and FAQ

### Is there a maximum file length or size?

The code does not set a maximum. The practical limit depends on available disk
space, memory, and processing time. Non-WAV inputs temporarily use additional
disk space while FFmpeg creates a 16 kHz WAV file.

### Is internet access required after the first run?

Not for normal transcription after the required models are cached locally.
Internet access is still needed to download a missing model, authenticate with
Hugging Face, or rerun a backend test when one of its models is not cached.

### Does accuracy vary with the recording?

Yes. Background noise, overlapping speakers, very quiet audio, strong accents,
and low-quality recordings can reduce transcription and speaker-label accuracy.

### Can it transcribe languages other than Hebrew?

No. The included models are fine-tuned for Hebrew, and the code explicitly sets
the language to Hebrew. There is no command-line option to select another
language.

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

## Uninstalling

Before deleting the virtual environment, run this command from the activated
repository environment:

```bash
hf cache delete
```

Select `ivrit-ai/whisper-large-v3-turbo-ct2` and
`mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx` to remove the downloaded
transcription models and reclaim the model space documented in
[What to expect](#what-to-expect). If you no longer need speaker identification,
you can also select `ivrit-ai/pyannote-speaker-diarization-3.1`,
`ivrit-ai/pyannote-segmentation-3.0`, and
`pyannote/wespeaker-voxceleb-resnet34-LM` when they appear in the cache list.

To keep the repository but remove its Python environment:

```bash
deactivate
rm -rf .venv
```

To remove the complete cloned repository on macOS or Linux, leave its folder
and delete it. This also deletes its `output` folder, so copy any transcripts
you want to keep first:

```bash
deactivate
cd ..
rm -rf hebrew-transcription
```

On Windows PowerShell:

```powershell
deactivate
cd ..
Remove-Item -Recurse -Force hebrew-transcription
```

## What runs behind the scenes

- [ivrit.ai](https://huggingface.co/ivrit-ai) provides the Hebrew Whisper models.
- `mlx-whisper` uses the Apple GPU on supported Apple Silicon Macs.
- `faster-whisper` and CTranslate2 provide the CPU and NVIDIA CUDA engine.
- FFmpeg converts video and audio to the required format.
- `pyannote.audio` optionally separates speakers.

Everything runs locally after the models have been downloaded.

## Contributing and issues

Report bugs, request improvements, or propose contributions on the repository's
[GitHub Issues page](https://github.com/shahar84/hebrew-transcription/issues).

## License

This project is available under the [MIT License](LICENSE).
