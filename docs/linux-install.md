# Install on Linux

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
