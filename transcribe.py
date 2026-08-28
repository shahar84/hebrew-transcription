#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

WHISPER_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"
DIARIZATION_MODEL = "ivrit-ai/pyannote-speaker-diarization-3.1"

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


@dataclass
class Segment:
    start: float
    end: float
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Hebrew transcription using ivrit.ai and faster-whisper."
    )
    parser.add_argument("input", type=Path, help="Audio or video file to transcribe")
    parser.add_argument(
        "--speakers",
        type=int,
        default=None,
        help="Known number of speakers. Enables speaker diarization.",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Run speaker diarization even when speaker count is unknown.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for generated files (default: output)",
    )
    parser.add_argument(
        "--model",
        default=WHISPER_MODEL,
        help=f"Whisper model (default: {WHISPER_MODEL})",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="faster-whisper device, e.g. cpu or cuda (default: cpu)",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="faster-whisper compute type (default: int8)",
    )
    return parser.parse_args()


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg is not installed. On macOS run: brew install ffmpeg"
        )


def prepare_audio(source: Path, workdir: Path) -> Path:
    suffix = source.suffix.lower()

    if suffix in AUDIO_EXTENSIONS and suffix == ".wav":
        return source

    if suffix not in AUDIO_EXTENSIONS | VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {suffix or '(no extension)'}")

    require_ffmpeg()
    target = workdir / "audio.wav"

    print("Extracting/converting audio with FFmpeg...")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(target),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr[-2000:]}")
    return target


def transcribe_audio(
    audio_path: Path, model_name: str, device: str, compute_type: str
) -> list[Segment]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install -r requirements.txt"
        ) from exc

    print("Loading transcription model...")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    print("Transcribing Hebrew audio...")
    raw_segments, _ = model.transcribe(
        str(audio_path),
        language="he",
        vad_filter=True,
    )

    return [
        Segment(segment.start, segment.end, segment.text.strip())
        for segment in raw_segments
        if segment.text.strip()
    ]


def format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def write_txt(segments: Iterable[Segment], output_path: Path) -> None:
    output_path.write_text(
        "\n".join(segment.text for segment in segments) + "\n",
        encoding="utf-8",
    )


def write_srt(segments: Iterable[Segment], output_path: Path) -> None:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            f"{index}\n"
            f"{format_srt_timestamp(segment.start)} --> "
            f"{format_srt_timestamp(segment.end)}\n"
            f"{segment.text}\n"
        )
    output_path.write_text("\n".join(blocks), encoding="utf-8")


def run_diarization(audio_path: Path, speakers: int | None):
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio is not installed. Run: pip install -r requirements.txt"
        ) from exc

    print("Loading speaker diarization model...")
    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL)

    print("Running speaker diarization...")
    kwargs = {}
    if speakers is not None:
        kwargs["num_speakers"] = speakers

    return pipeline(str(audio_path), **kwargs)


def speaker_for_segment(diarization, segment: Segment) -> str:
    midpoint = (segment.start + segment.end) / 2
    best_speaker = "UNKNOWN"
    best_overlap = 0.0

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        overlap = max(
            0.0,
            min(segment.end, turn.end) - max(segment.start, turn.start),
        )
        if turn.start <= midpoint <= turn.end:
            overlap += 10_000
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker

    return best_speaker


def write_speaker_transcript(
    segments: list[Segment], diarization, output_path: Path
) -> None:
    lines: list[str] = []
    current_speaker = None

    for segment in segments:
        speaker = speaker_for_segment(diarization, segment)
        if speaker != current_speaker:
            if lines:
                lines.append("")
            lines.append(f"{speaker}:")
            current_speaker = speaker
        lines.append(segment.text)

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    source = args.input.expanduser().resolve()

    if not source.exists():
        print(f"Error: file not found: {source}", file=sys.stderr)
        return 2

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = source.stem
    txt_path = output_dir / f"{stem}.txt"
    srt_path = output_dir / f"{stem}.srt"
    speakers_path = output_dir / f"{stem}_with_speakers.txt"

    try:
        with tempfile.TemporaryDirectory(prefix="hebrew-transcriber-") as temp:
            workdir = Path(temp)
            audio_path = prepare_audio(source, workdir)

            segments = transcribe_audio(
                audio_path,
                model_name=args.model,
                device=args.device,
                compute_type=args.compute_type,
            )

            print("Saving transcript...")
            write_txt(segments, txt_path)
            write_srt(segments, srt_path)

            should_diarize = args.diarize or args.speakers is not None
            if should_diarize:
                try:
                    diarization = run_diarization(audio_path, args.speakers)
                    write_speaker_transcript(
                        segments, diarization, speakers_path
                    )
                except Exception as exc:
                    print(
                        f"Speaker diarization failed, but transcription was saved: {exc}",
                        file=sys.stderr,
                    )

        print("Done.")
        print(f"TXT: {txt_path}")
        print(f"SRT: {srt_path}")
        if speakers_path.exists():
            print(f"Speakers: {speakers_path}")
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
