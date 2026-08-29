# Advanced usage

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
