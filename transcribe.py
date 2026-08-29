#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Iterable

WHISPER_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"
MLX_WHISPER_MODEL = "mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx"
DIARIZATION_MODEL = "ivrit-ai/pyannote-speaker-diarization-3.1"

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
CPU_TUNING_SAMPLE_SECONDS = 30
BACKEND_TUNING_SAMPLE_SECONDS = 60


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


def parse_cpu_threads(value: str) -> str | int:
    if value == "auto":
        return value
    try:
        threads = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be 'auto' or a positive integer") from exc
    if threads < 1:
        raise argparse.ArgumentTypeError("must be 'auto' or a positive integer")
    return threads


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Hebrew transcription using ivrit.ai with CTranslate2 or MLX."
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
        "--backend",
        choices=("auto", "ctranslate2", "mlx"),
        default="auto",
        help="transcription engine (default: benchmark once and cache the fastest)",
    )
    parser.add_argument(
        "--model",
        default=WHISPER_MODEL,
        help=f"CTranslate2 Whisper model (default: {WHISPER_MODEL})",
    )
    parser.add_argument(
        "--mlx-model",
        default=MLX_WHISPER_MODEL,
        help=f"MLX Whisper model (default: {MLX_WHISPER_MODEL})",
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
        "--cpu-threads",
        type=parse_cpu_threads,
        default="auto",
        help="CPU threads for faster-whisper (default: benchmark once and cache)",
    )
    parser.add_argument(
        "--retune-cpu-threads",
        action="store_true",
        help="Ignore the cached CPU thread result and benchmark again",
    )
    parser.add_argument(
        "--retune-backend",
        action="store_true",
        help="Ignore the cached backend result and benchmark both engines again",
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


def cpu_thread_candidates(cpu_count: int | None = None) -> list[int]:
    count = max(1, cpu_count or os.cpu_count() or 1)
    candidates = {min(count, value) for value in (4, 8, 12)}
    candidates.update({max(1, count - 4), max(1, count - 2), count})
    return sorted(candidates)


def cpu_tuning_cache_path() -> Path:
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "hebrew-transcription" / "cpu-thread-tuning.json"


def cpu_tuning_key(model_name: str, compute_type: str) -> str:
    try:
        import ctranslate2

        ctranslate2_version = ctranslate2.__version__
    except (ImportError, AttributeError):
        ctranslate2_version = "unknown"

    details = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "model": model_name,
        "compute_type": compute_type,
        "ctranslate2": ctranslate2_version,
    }
    encoded = json.dumps(details, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_cached_cpu_threads(cache_path: Path, key: str) -> int | None:
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        threads = data.get("entries", {}).get(key, {}).get("threads")
        return threads if isinstance(threads, int) and threads > 0 else None
    except (AttributeError, FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def save_cpu_thread_tuning(
    cache_path: Path, key: str, threads: int, scores: dict[int, float]
) -> None:
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        data = {"version": 1, "entries": {}}
    if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
        data = {"version": 1, "entries": {}}

    entries = data.setdefault("entries", {})
    entries[key] = {
        "threads": threads,
        "scores": {str(candidate): elapsed for candidate, elapsed in scores.items()},
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def benchmark_cpu_threads(
    audio_path: Path,
    model_name: str,
    compute_type: str,
    candidates: Iterable[int] | None = None,
) -> tuple[int, dict[int, float]]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install -r requirements.txt"
        ) from exc

    candidates = list(candidates or cpu_thread_candidates())
    scores: dict[int, float] = {}
    reference_text: str | None = None
    print(
        "Auto-tuning faster-whisper CPU threads "
        f"on a {CPU_TUNING_SAMPLE_SECONDS}s sample..."
    )

    for threads in candidates:
        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type=compute_type,
            cpu_threads=threads,
        )
        started = time.perf_counter()
        raw_segments, _ = model.transcribe(
            str(audio_path),
            language="he",
            vad_filter=True,
            clip_timestamps=f"0,{CPU_TUNING_SAMPLE_SECONDS}",
        )
        segments = list(raw_segments)
        elapsed = time.perf_counter() - started
        text = "".join(segment.text for segment in segments)
        scores[threads] = elapsed
        print(f"  {threads} threads: {format_duration(elapsed)}")
        if reference_text is None:
            reference_text = text
        elif text != reference_text:
            print(
                f"Warning: {threads}-thread benchmark output differed.",
                file=sys.stderr,
            )
        del segments, raw_segments, model
        gc.collect()

    best_threads = min(scores, key=scores.get)
    print(f"Selected {best_threads} CPU threads.")
    return best_threads, scores


def select_cpu_threads(
    requested: str | int,
    audio_path: Path,
    model_name: str,
    compute_type: str,
    retune: bool = False,
    cache_path: Path | None = None,
    benchmark_function=benchmark_cpu_threads,
) -> int:
    if isinstance(requested, int):
        return requested

    cache_path = cache_path or cpu_tuning_cache_path()
    key = cpu_tuning_key(model_name, compute_type)
    if not retune:
        cached = load_cached_cpu_threads(cache_path, key)
        if cached is not None:
            print(f"Using cached CPU thread tuning: {cached} threads")
            return cached

    threads, scores = benchmark_function(audio_path, model_name, compute_type)
    try:
        save_cpu_thread_tuning(cache_path, key, threads, scores)
    except OSError as exc:
        print(f"Warning: could not save CPU tuning cache: {exc}", file=sys.stderr)
    return threads


def mlx_is_available(
    system: str | None = None,
    machine: str | None = None,
    find_spec=importlib.util.find_spec,
) -> bool:
    system = system or platform.system()
    machine = machine or platform.machine()
    return (
        system == "Darwin"
        and machine == "arm64"
        and find_spec("mlx_whisper") is not None
    )


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def backend_tuning_cache_path() -> Path:
    return cpu_tuning_cache_path().with_name("backend-tuning.json")


def backend_tuning_key(
    ctranslate2_model: str,
    mlx_model: str,
    device: str,
    compute_type: str,
) -> str:
    details = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "ctranslate2_model": ctranslate2_model,
        "mlx_model": mlx_model,
        "device": device,
        "compute_type": compute_type,
        "faster_whisper": package_version("faster-whisper"),
        "mlx_whisper": package_version("mlx-whisper"),
    }
    encoded = json.dumps(details, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_cached_backend(cache_path: Path, key: str) -> str | None:
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        backend = data.get("entries", {}).get(key, {}).get("backend")
        return backend if backend in {"ctranslate2", "mlx"} else None
    except (AttributeError, FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def save_backend_tuning(
    cache_path: Path, key: str, backend: str, scores: dict[str, float]
) -> None:
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        data = {"version": 1, "entries": {}}
    if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
        data = {"version": 1, "entries": {}}

    entries = data.setdefault("entries", {})
    entries[key] = {"backend": backend, "scores": scores}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def choose_transcription_backend(
    requested: str,
    *,
    mlx_available: bool,
    cache_path: Path | None = None,
    cache_key: str | None = None,
    retune: bool = False,
) -> str | None:
    """Return a backend, or None when auto mode needs a benchmark."""
    if requested == "mlx":
        if not mlx_available:
            raise RuntimeError(
                "MLX was requested, but mlx-whisper is unavailable. "
                "MLX requires an Apple Silicon Mac and the mlx-whisper package."
            )
        return "mlx"
    if requested == "ctranslate2":
        return "ctranslate2"
    if not mlx_available:
        return "ctranslate2"
    if retune:
        return None
    if cache_path is not None and cache_key is not None:
        return load_cached_backend(cache_path, cache_key)
    return None


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


def prepare_backend_benchmark_sample(audio_path: Path, workdir: Path) -> Path:
    sample_path = workdir / "backend-benchmark.wav"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-t",
        str(BACKEND_TUNING_SAMPLE_SECONDS),
        "-acodec",
        "pcm_s16le",
        str(sample_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg failed to create a benchmark sample:\n"
            f"{result.stderr[-2000:]}"
        )
    return sample_path


def transcribe_audio(
    audio_path: Path,
    model_name: str,
    device: str,
    compute_type: str,
    cpu_threads: int = 0,
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
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
    )
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


def transcribe_audio_mlx(
    audio_path: Path,
    model_name: str,
    timings: dict[str, float] | None = None,
    mlx_whisper_module=None,
    mlx_core_module=None,
) -> list[Segment]:
    try:
        if mlx_whisper_module is None:
            import mlx_whisper as mlx_whisper_module
        if mlx_core_module is None:
            import mlx.core as mlx_core_module
    except ImportError as exc:
        raise RuntimeError(
            "mlx-whisper is not installed. Run: pip install -r requirements.txt"
        ) from exc

    print("Transcribing Hebrew audio with MLX...")
    started = time.perf_counter()
    result = mlx_whisper_module.transcribe(
        str(audio_path),
        path_or_hf_repo=model_name,
        language="he",
        verbose=False,
    )
    mlx_core_module.synchronize()
    add_timing(timings, "MLX transcription", time.perf_counter() - started)
    return [
        Segment(float(segment["start"]), float(segment["end"]), text)
        for segment in result["segments"]
        if (text := segment["text"].strip())
    ]


def prefetch_hugging_face_model(model_name: str) -> None:
    model_path = Path(model_name).expanduser()
    if model_path.exists() or "/" not in model_name:
        return
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=model_name)


def benchmark_transcription_backends(
    audio_path: Path,
    *,
    ctranslate2_model: str,
    mlx_model: str,
    device: str,
    compute_type: str,
    cpu_threads: int,
    ctranslate2_function=transcribe_audio,
    mlx_function=transcribe_audio_mlx,
) -> tuple[str, dict[str, float]]:
    print(
        "Auto-tuning the transcription backend "
        f"on a {BACKEND_TUNING_SAMPLE_SECONDS}s sample..."
    )
    print("Ensuring both benchmark models are downloaded...")
    for backend_name, model_name in (
        ("CTranslate2", ctranslate2_model),
        ("MLX", mlx_model),
    ):
        try:
            prefetch_hugging_face_model(model_name)
        except Exception as exc:
            print(
                f"  Could not prefetch the {backend_name} model: {exc}",
                file=sys.stderr,
            )
    scores: dict[str, float] = {}
    transcripts: dict[str, str] = {}

    try:
        started = time.perf_counter()
        segments = ctranslate2_function(
            audio_path,
            model_name=ctranslate2_model,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
        scores["ctranslate2"] = time.perf_counter() - started
        transcripts["ctranslate2"] = " ".join(
            segment.text for segment in segments
        )
        print(f"  CTranslate2: {format_duration(scores['ctranslate2'])}")
        del segments
    except Exception as exc:
        print(f"  CTranslate2 benchmark failed: {exc}", file=sys.stderr)
    finally:
        gc.collect()

    try:
        started = time.perf_counter()
        segments = mlx_function(audio_path, model_name=mlx_model)
        scores["mlx"] = time.perf_counter() - started
        transcripts["mlx"] = " ".join(segment.text for segment in segments)
        print(f"  MLX: {format_duration(scores['mlx'])}")
        del segments
    except Exception as exc:
        print(f"  MLX benchmark failed: {exc}", file=sys.stderr)
    finally:
        gc.collect()

    if not scores:
        raise RuntimeError("Both transcription backend benchmarks failed.")
    if (
        len(transcripts) == 2
        and transcripts["ctranslate2"].split() != transcripts["mlx"].split()
    ):
        print(
            "Note: backend benchmark transcripts differed; selection is based on speed.",
            file=sys.stderr,
        )

    backend = min(scores, key=scores.get)
    print(f"Selected {backend} transcription backend.")
    return backend, scores


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

            mlx_available = mlx_is_available()
            backend_cache_path = backend_tuning_cache_path()
            backend_key = backend_tuning_key(
                args.model,
                args.mlx_model,
                args.device,
                args.compute_type,
            )
            backend = choose_transcription_backend(
                args.backend,
                mlx_available=mlx_available,
                cache_path=backend_cache_path,
                cache_key=backend_key,
                retune=args.retune_backend,
            )

            transcription_device = "cpu"
            cpu_threads = 0

            if backend != "mlx":
                transcription_device = select_transcription_device(args.device)
                print(f"CTranslate2 device: {transcription_device.upper()}")
            if backend != "mlx" and transcription_device == "cpu":
                started = time.perf_counter()
                cpu_threads = select_cpu_threads(
                    args.cpu_threads,
                    audio_path,
                    args.model,
                    args.compute_type,
                    retune=args.retune_cpu_threads,
                )
                add_timing(
                    timings, "CPU thread tuning", time.perf_counter() - started
                )
                print(f"Transcription CPU threads: {cpu_threads}")

            if backend is None:
                started = time.perf_counter()
                benchmark_audio = prepare_backend_benchmark_sample(
                    audio_path, workdir
                )
                backend, scores = benchmark_transcription_backends(
                    benchmark_audio,
                    ctranslate2_model=args.model,
                    mlx_model=args.mlx_model,
                    device=transcription_device,
                    compute_type=args.compute_type,
                    cpu_threads=cpu_threads,
                )
                try:
                    save_backend_tuning(
                        backend_cache_path, backend_key, backend, scores
                    )
                except OSError as exc:
                    print(
                        f"Warning: could not save backend tuning cache: {exc}",
                        file=sys.stderr,
                    )
                add_timing(
                    timings, "Backend tuning", time.perf_counter() - started
                )
            else:
                print(f"Transcription backend: {backend}")

            if backend == "mlx":
                try:
                    segments = transcribe_audio_mlx(
                        audio_path,
                        model_name=args.mlx_model,
                        timings=timings,
                    )
                except Exception as exc:
                    if args.backend != "auto":
                        raise
                    print(
                        f"MLX transcription failed; retrying with CTranslate2: {exc}",
                        file=sys.stderr,
                    )
                    transcription_device = select_transcription_device(args.device)
                    cpu_threads = 0
                    if transcription_device == "cpu":
                        started = time.perf_counter()
                        cpu_threads = select_cpu_threads(
                            args.cpu_threads,
                            audio_path,
                            args.model,
                            args.compute_type,
                            retune=args.retune_cpu_threads,
                        )
                        add_timing(
                            timings,
                            "CPU thread tuning",
                            time.perf_counter() - started,
                        )
                    segments = transcribe_audio(
                        audio_path,
                        model_name=args.model,
                        device=transcription_device,
                        compute_type=args.compute_type,
                        cpu_threads=cpu_threads,
                        timings=timings,
                    )
            else:
                segments = transcribe_audio(
                    audio_path,
                    model_name=args.model,
                    device=transcription_device,
                    compute_type=args.compute_type,
                    cpu_threads=cpu_threads,
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
