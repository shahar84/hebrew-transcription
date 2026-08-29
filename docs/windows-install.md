# Install on Windows

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
