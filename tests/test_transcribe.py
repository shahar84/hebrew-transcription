from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "transcribe.py"
SPEC = importlib.util.spec_from_file_location("transcribe", MODULE_PATH)
transcribe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = transcribe
SPEC.loader.exec_module(transcribe)


def fake_ctranslate2(cuda_devices):
    return SimpleNamespace(get_cuda_device_count=lambda: cuda_devices)


def fake_torch(cuda_available=False, mps_available=False):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps_available)
        ),
    )


def test_device_defaults_are_automatic():
    args = transcribe.parse_args(["recording.wav"])
    assert args.backend == "auto"
    assert args.device == "auto"
    assert args.diarization_device == "auto"
    assert args.cpu_threads == "auto"


@pytest.mark.parametrize(("value", "expected"), [("auto", "auto"), ("14", 14)])
def test_parse_cpu_threads(value, expected):
    assert transcribe.parse_cpu_threads(value) == expected


@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_parse_cpu_threads_rejects_invalid_values(value):
    with pytest.raises(Exception, match="positive integer"):
        transcribe.parse_cpu_threads(value)


def test_cpu_thread_candidates_are_bounded_and_sorted():
    candidates = transcribe.cpu_thread_candidates(18)
    assert candidates == [4, 8, 12, 14, 16, 18]


def test_cpu_thread_tuning_is_cached(tmp_path, monkeypatch):
    cache_path = tmp_path / "tuning.json"
    monkeypatch.setattr(transcribe, "cpu_tuning_key", lambda model, compute: "key")
    calls = []

    def benchmark(audio_path, model_name, compute_type):
        calls.append(audio_path)
        return 14, {4: 2.0, 14: 1.0}

    first = transcribe.select_cpu_threads(
        "auto",
        tmp_path / "audio.wav",
        "model",
        "int8",
        cache_path=cache_path,
        benchmark_function=benchmark,
    )
    second = transcribe.select_cpu_threads(
        "auto",
        tmp_path / "audio.wav",
        "model",
        "int8",
        cache_path=cache_path,
        benchmark_function=benchmark,
    )

    assert first == second == 14
    assert len(calls) == 1


def test_malformed_cpu_thread_cache_is_replaced(tmp_path, monkeypatch):
    cache_path = tmp_path / "tuning.json"
    cache_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(transcribe, "cpu_tuning_key", lambda model, compute: "key")

    selected = transcribe.select_cpu_threads(
        "auto",
        tmp_path / "audio.wav",
        "model",
        "int8",
        cache_path=cache_path,
        benchmark_function=lambda *args: (8, {8: 1.0}),
    )

    assert selected == 8
    assert transcribe.load_cached_cpu_threads(cache_path, "key") == 8


def test_explicit_cpu_threads_skip_tuning(tmp_path):
    assert (
        transcribe.select_cpu_threads(
            8,
            tmp_path / "audio.wav",
            "model",
            "int8",
            benchmark_function=lambda *args: pytest.fail("unexpected benchmark"),
        )
        == 8
    )


def test_mlx_availability_requires_apple_silicon_and_package():
    found = lambda name: object()
    missing = lambda name: None

    assert transcribe.mlx_is_available("Darwin", "arm64", found)
    assert not transcribe.mlx_is_available("Darwin", "x86_64", found)
    assert not transcribe.mlx_is_available("Linux", "arm64", found)
    assert not transcribe.mlx_is_available("Darwin", "arm64", missing)


def test_auto_backend_uses_cached_winner(tmp_path):
    cache_path = tmp_path / "backend.json"
    transcribe.save_backend_tuning(
        cache_path, "machine", "mlx", {"ctranslate2": 5.0, "mlx": 1.0}
    )

    assert (
        transcribe.choose_transcription_backend(
            "auto",
            mlx_available=True,
            cache_path=cache_path,
            cache_key="machine",
        )
        == "mlx"
    )


def test_auto_backend_requests_benchmark_without_cache(tmp_path):
    assert (
        transcribe.choose_transcription_backend(
            "auto",
            mlx_available=True,
            cache_path=tmp_path / "missing.json",
            cache_key="machine",
        )
        is None
    )


def test_auto_backend_uses_ctranslate2_when_mlx_is_unavailable():
    assert (
        transcribe.choose_transcription_backend("auto", mlx_available=False)
        == "ctranslate2"
    )


def test_explicit_mlx_rejects_unsupported_machine():
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        transcribe.choose_transcription_backend("mlx", mlx_available=False)


def test_backend_benchmark_selects_fastest(monkeypatch, tmp_path):
    times = iter([0.0, 5.0, 10.0, 11.0])
    monkeypatch.setattr(transcribe.time, "perf_counter", lambda: next(times))

    def run_ctranslate2(*args, **kwargs):
        return [transcribe.Segment(0, 1, "שלום")]

    def run_mlx(*args, **kwargs):
        return [transcribe.Segment(0, 1, "שלום")]

    backend, scores = transcribe.benchmark_transcription_backends(
        tmp_path / "sample.wav",
        ctranslate2_model="ct2",
        mlx_model="mlx",
        device="cpu",
        compute_type="int8",
        cpu_threads=8,
        ctranslate2_function=run_ctranslate2,
        mlx_function=run_mlx,
    )

    assert backend == "mlx"
    assert scores == {"ctranslate2": 5.0, "mlx": 1.0}


def test_backend_benchmark_uses_working_engine(monkeypatch, tmp_path):
    times = iter([0.0, 1.0, 2.0])
    monkeypatch.setattr(transcribe.time, "perf_counter", lambda: next(times))

    backend, scores = transcribe.benchmark_transcription_backends(
        tmp_path / "sample.wav",
        ctranslate2_model="ct2",
        mlx_model="mlx",
        device="cpu",
        compute_type="int8",
        cpu_threads=8,
        ctranslate2_function=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("broken")
        ),
        mlx_function=lambda *args, **kwargs: [
            transcribe.Segment(0, 1, "שלום")
        ],
    )

    assert backend == "mlx"
    assert scores == {"mlx": 1.0}


def test_mlx_transcription_converts_segments(tmp_path):
    fake_whisper = SimpleNamespace(
        transcribe=lambda *args, **kwargs: {
            "segments": [
                {"start": 0, "end": 1.25, "text": " שלום "},
                {"start": 1.25, "end": 2, "text": "  "},
            ]
        }
    )
    fake_core = SimpleNamespace(synchronize=lambda: None)

    segments = transcribe.transcribe_audio_mlx(
        tmp_path / "audio.wav",
        "model",
        mlx_whisper_module=fake_whisper,
        mlx_core_module=fake_core,
    )

    assert segments == [transcribe.Segment(0.0, 1.25, "שלום")]


def test_transcription_device_uses_cuda_when_ctranslate2_supports_it():
    assert (
        transcribe.select_transcription_device("auto", fake_ctranslate2(1))
        == "cuda"
    )


def test_transcription_device_uses_cpu_without_cuda():
    assert (
        transcribe.select_transcription_device("auto", fake_ctranslate2(0))
        == "cpu"
    )


def test_transcription_device_rejects_unavailable_cuda():
    with pytest.raises(RuntimeError, match="no CUDA GPU"):
        transcribe.select_transcription_device("cuda", fake_ctranslate2(0))


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [
        (True, True, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_diarization_device_chooses_fastest_available(
    cuda_available, mps_available, expected
):
    assert (
        transcribe.select_diarization_device(
            "auto", fake_torch(cuda_available, mps_available)
        )
        == expected
    )


def test_diarization_device_rejects_unavailable_mps():
    with pytest.raises(RuntimeError, match="no Apple GPU"):
        transcribe.select_diarization_device("mps", fake_torch())


def test_auto_diarization_retries_on_cpu_after_gpu_failure(
    monkeypatch, tmp_path, capsys
):
    class FakePipeline:
        devices = []

        @classmethod
        def from_pretrained(cls, model):
            return cls()

        def to(self, device):
            self.device = device.type
            self.devices.append(self.device)

        def __call__(self, audio_path, **kwargs):
            if self.device == "mps":
                raise RuntimeError("unsupported MPS operation")
            return "cpu-result"

    monkeypatch.setattr(
        transcribe, "select_diarization_device", lambda requested: "mps"
    )
    monkeypatch.setitem(
        sys.modules,
        "pyannote.audio",
        SimpleNamespace(Pipeline=FakePipeline),
    )

    result = transcribe.run_diarization(tmp_path / "audio.wav", 3)

    assert result == "cpu-result"
    assert FakePipeline.devices == ["mps", "cpu"]
    assert "retrying on CPU" in capsys.readouterr().err


def test_srt_timestamp():
    assert transcribe.format_srt_timestamp(0) == "00:00:00,000"
    assert transcribe.format_srt_timestamp(65.432) == "00:01:05,432"
    assert transcribe.format_srt_timestamp(3661.001) == "01:01:01,001"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0.0s"),
        (12.34, "12.3s"),
        (65.4, "1m 05.4s"),
        (3661.2, "1h 01m 01.2s"),
    ],
)
def test_format_duration(seconds, expected):
    assert transcribe.format_duration(seconds) == expected


def test_print_timing_summary(capsys):
    transcribe.print_timing_summary(
        {"Transcription": 65.4, "Diarization": 12.34},
        total=80,
    )

    assert capsys.readouterr().out.splitlines() == [
        "Timing:",
        "  Transcription: 1m 05.4s",
        "  Diarization: 12.3s",
        "  Total: 1m 20.0s",
    ]


def test_write_txt(tmp_path):
    segments = [
        transcribe.Segment(0, 1, "שלום"),
        transcribe.Segment(1, 2, "עולם"),
    ]
    target = tmp_path / "out.txt"
    transcribe.write_txt(segments, target)
    assert target.read_text(encoding="utf-8") == "שלום\nעולם\n"


def test_write_srt(tmp_path):
    segments = [transcribe.Segment(0, 1.5, "שלום")]
    target = tmp_path / "out.srt"
    transcribe.write_srt(segments, target)
    text = target.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:01,500" in text
    assert "שלום" in text
