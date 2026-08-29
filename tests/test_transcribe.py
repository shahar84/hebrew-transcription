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
    assert args.device == "auto"
    assert args.diarization_device == "auto"


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
