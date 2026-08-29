# Hebrew Transcriber

Transcribe Hebrew audio and video locally. The tool creates a plain-text
transcript, subtitles, and optionally a version labeled by speaker.

It automatically finds the fastest transcription engine for your computer.
On Apple Silicon Macs, it can use the Apple GPU through MLX.

It supports macOS, Linux, and Windows. Linux and Windows can use an NVIDIA GPU
through CUDA when the required NVIDIA libraries are installed; otherwise they
run CTranslate2 on the CPU.

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

## Table of contents

- [What to expect](#what-to-expect)
- [Quick start](#quick-start)
- [Install on macOS](#install-on-macos)
- [Transcribe your first file](#transcribe-your-first-file)
- [Supported formats](#supported-formats)
- [Use it again later](#use-it-again-later)
- [Identify different speakers](#identify-different-speakers)
- [Advanced usage](#advanced-usage)
- [Limitations and FAQ](#limitations-and-faq)
- [Common problems](#common-problems)
- [Uninstalling](#uninstalling)
- [What runs behind the scenes](#what-runs-behind-the-scenes)
- [Contributing and issues](#contributing-and-issues)
- [License](#license)

### More docs

- [Linux install](docs/linux-install.md)
- [Windows install](docs/windows-install.md)
- [Limitations and FAQ](docs/faq.md)
- [Uninstalling](docs/uninstalling.md)
- [Advanced usage](docs/advanced-usage.md)

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

Using Linux or Windows? See [Linux install](docs/linux-install.md) or
[Windows install](docs/windows-install.md).

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

## Advanced usage

See [Advanced usage](docs/advanced-usage.md) for speed tuning and additional commands.

## Limitations and FAQ

See [Limitations and FAQ](docs/faq.md).

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

See [Uninstalling](docs/uninstalling.md).

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
