# Uninstalling

Before deleting the virtual environment, run this command from the activated
repository environment:

```bash
hf cache delete
```

Select `ivrit-ai/whisper-large-v3-turbo-ct2` and
`mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx` to remove the downloaded
transcription models and reclaim the model space documented in
[What to expect](../README.md#what-to-expect). If you no longer need speaker identification,
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
