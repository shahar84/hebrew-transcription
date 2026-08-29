#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import shutil
import subprocess
import sys
import tempfile
import time
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


def add_timing(
    timings: dict[str, float] | None, name: str, elapsed: float
) -> None:
    if timings is not None:
        timings[name] = timings.get(name, 0.0) + elapsed


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60

    if hours:
        return f"{hours}h {minutes:02d}m {remaining_seconds:04.1f}s"
    if minutes:
        return f"{minutes}m {remaining_seconds:04.1f}s"
    return f"{remaining_seconds:.1f}s"


def print_timing_summary(timings: dict[str, float], total: float) -> None:
    print("Timing:")
    for name, elapsed in timings.items():
        print(f"  {name}: {format_duration(elapsed)}")
    print(f"  Total: {format_duration(total)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="faster-whisper device (default: auto; Apple Silicon uses CPU)",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="faster-whisper compute type (default: int8)",
    )
    parser.add_argument(
        "--diarization-device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="pyannote device (default: auto; prefers CUDA, then Apple MPS)",
    )
    return parser.parse_args(argv)


def select_transcription_device(requested: str, ctranslate2_module=None) -> str:
    """Choose a CTranslate2 device. Apple MPS is not supported by CTranslate2."""
    if ctranslate2_module is None:
        try:
            import ctranslate2 as ctranslate2_module
        except ImportError as exc:
            raise RuntimeError(
                "CTranslate2 is not installed. Run: pip install -r requirements.txt"
            ) from exc

    cuda_available = ctranslate2_module.get_cuda_device_count() > 0
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested, but CTranslate2 found no CUDA GPU.")
    return requested


def select_diarization_device(requested: str, torch_module=None) -> str:
    """Choose the fastest available PyTorch device for speaker diarization."""
    # Let unsupported MPS operations fall back to CPU instead of aborting a job.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is not installed. Run: pip install -r requirements.txt"
            ) from exc

    cuda_available = torch_module.cuda.is_available()
    mps_backend = getattr(torch_module.backends, "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())

    if requested == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested, but PyTorch found no CUDA GPU.")
    if requested == "mps" and not mps_available:
        raise RuntimeError("MPS was requested, but PyTorch found no Apple GPU.")
    return requested


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
    audio_path: Path,
    model_name: str,
    device: str,
    compute_type: str,
    timings: dict[str, float] | None = None,
) -> list[Segment]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install -r requirements.txt"
        ) from exc

    print("Loading transcription model...")
    started = time.perf_counter()
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    add_timing(timings, "Whisper model loading", time.perf_counter() - started)

    print("Transcribing Hebrew audio...")
    started = time.perf_counter()
    raw_segments, _ = model.transcribe(
        str(audio_path),
        language="he",
        vad_filter=True,
    )

    segments = [
        Segment(segment.start, segment.end, segment.text.strip())
        for segment in raw_segments
        if segment.text.strip()
    ]
    add_timing(timings, "Transcription", time.perf_counter() - started)
    return segments


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


def run_diarization(
    audio_path: Path,
    speakers: int | None,
    device: str = "auto",
    timings: dict[str, float] | None = None,
):
    selected_device = select_diarization_device(device)

    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio is not installed. Run: pip install -r requirements.txt"
        ) from exc

    print("Loading speaker diarization model...")
    started = time.perf_counter()
    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL)
    add_timing(
        timings, "Diarization model loading", time.perf_counter() - started
    )

    kwargs = {}
    if speakers is not None:
        kwargs["num_speakers"] = speakers

    print(f"Running speaker diarization on {selected_device.upper()}...")
    started = time.perf_counter()
    active_device = selected_device
    try:
        pipeline.to(torch.device(selected_device))
        result = pipeline(str(audio_path), **kwargs)
    except (RuntimeError, NotImplementedError) as exc:
        if device != "auto" or selected_device == "cpu":
            raise
        print(
            f"{selected_device.upper()} diarization failed; retrying on CPU: {exc}",
            file=sys.stderr,
        )
        del pipeline
        gc.collect()
        if selected_device == "mps":
            torch.mps.empty_cache()
        elif selected_device == "cuda":
            torch.cuda.empty_cache()

        load_started = time.perf_counter()
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL)
        add_timing(
            timings,
            "Diarization model loading",
            time.perf_counter() - load_started,
        )
        pipeline.to(torch.device("cpu"))
        result = pipeline(str(audio_path), **kwargs)
        active_device = "cpu"

    if active_device == "mps":
        torch.mps.synchronize()
    elif active_device == "cuda":
        torch.cuda.synchronize()

    add_timing(timings, "Diarization", time.perf_counter() - started)
    del pipeline
    gc.collect()
    if active_device == "mps":
        torch.mps.empty_cache()
    elif active_device == "cuda":
        torch.cuda.empty_cache()
    return result


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
    total_started = time.perf_counter()
    timings: dict[str, float] = {}
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
            started = time.perf_counter()
            audio_path = prepare_audio(source, workdir)
            add_timing(
                timings, "Audio preparation", time.perf_counter() - started
            )

            transcription_device = select_transcription_device(args.device)
            print(f"Transcription device: {transcription_device.upper()}")
            segments = transcribe_audio(
                audio_path,
                model_name=args.model,
                device=transcription_device,
                compute_type=args.compute_type,
                timings=timings,
            )

            print("Saving transcript...")
            started = time.perf_counter()
            write_txt(segments, txt_path)
            write_srt(segments, srt_path)
            add_timing(
                timings, "Writing TXT/SRT", time.perf_counter() - started
            )

            should_diarize = args.diarize or args.speakers is not None
            if should_diarize:
                try:
                    diarization = run_diarization(
                        audio_path,
                        args.speakers,
                        device=args.diarization_device,
                        timings=timings,
                    )
                    started = time.perf_counter()
                    write_speaker_transcript(
                        segments, diarization, speakers_path
                    )
                    add_timing(
                        timings,
                        "Writing speaker transcript",
                        time.perf_counter() - started,
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
        print_timing_summary(timings, time.perf_counter() - total_started)
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
