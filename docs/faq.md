# Limitations and FAQ

## Is there a maximum file length or size?

The code does not set a maximum. The practical limit depends on available disk
space, memory, and processing time. Non-WAV inputs temporarily use additional
disk space while FFmpeg creates a 16 kHz WAV file.

## Is internet access required after the first run?

Not for normal transcription after the required models are cached locally.
Internet access is still needed to download a missing model, authenticate with
Hugging Face, or rerun a backend test when one of its models is not cached.

## Does accuracy vary with the recording?

Yes. Background noise, overlapping speakers, very quiet audio, strong accents,
and low-quality recordings can reduce transcription and speaker-label accuracy.

## Can it transcribe languages other than Hebrew?

No. The included models are fine-tuned for Hebrew, and the code explicitly sets
the language to Hebrew. There is no command-line option to select another
language.
