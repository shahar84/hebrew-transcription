from pathlib import Path
import importlib.util

MODULE_PATH = Path(__file__).resolve().parents[1] / "transcribe.py"
SPEC = importlib.util.spec_from_file_location("transcribe", MODULE_PATH)
transcribe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(transcribe)


def test_srt_timestamp():
    assert transcribe.format_srt_timestamp(0) == "00:00:00,000"
    assert transcribe.format_srt_timestamp(65.432) == "00:01:05,432"
    assert transcribe.format_srt_timestamp(3661.001) == "01:01:01,001"


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
