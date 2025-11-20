import os
import queue
import subprocess
import threading

import numpy as np
import pytest

from audfprint2.utils import audio


class DummyStream:
    """A minimal file-like object returning predefined chunks."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def read(self, blocksize):
        if self._chunks:
            return self._chunks.pop(0)
        return b""

    def readline(self):
        return self.read(-1)


class DummyLineStream(DummyStream):
    """Provides readline semantics for stderr parsing."""

    def readline(self):
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class DummyProcess:
    """Stand-in for subprocess.Popen."""

    def __init__(self, stderr_lines):
        self.stdout = DummyStream([b"\x00\x00", b""])
        self.stderr = DummyLineStream(list(stderr_lines))
        self.returncode = None

    def kill(self):  # pragma: no cover - exercised via close()
        self.returncode = 0

    def wait(self):
        self.returncode = 0
        return 0


def test_ffmpeg_audio_file_parses_metadata(monkeypatch, tmp_path):
    """FFmpegAudioFile should parse stderr metadata and stream frames."""

    # Create a real file so FFmpegAudioFile path checks pass.
    wav_path = tmp_path / "fake.wav"
    wav_path.write_bytes(b"00")

    stderr_lines = [
        b"ffmpeg version",  # ignored boilerplate
        b"Duration: 00:00:01.0, start: 0.000000, bitrate: 705 kb/s",  # duration line
        b"Audio: pcm_s16le, 44100 Hz, stereo, s16, 1411 kb/s",  # audio line to parse
    ]

    def fake_popen(*args, **kwargs):
        return DummyProcess(stderr_lines)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    reader_threads = []

    def tracked_thread_start(self):
        reader_threads.append(self)
        threading.Thread.start(self)

    # Track created QueueReaderThread instances for graceful shutdown.
    monkeypatch.setattr(audio.QueueReaderThread, "start", tracked_thread_start)

    fobj = audio.FFmpegAudioFile(str(wav_path), channels=1, sample_rate=8000, block_size=2)

    # Consume data via iterator protocol to exercise read_data loop.
    frames = list(fobj)
    assert frames  # At least one block should be yielded
    assert fobj.sample_rate == 8000
    assert fobj.channels == 1
    assert fobj.duration == pytest.approx(1.0, abs=0.1)

    # Closing should terminate the dummy process and threads cleanly.
    fobj.close()
    for t in reader_threads:
        t.join(timeout=1)


def test_audio_read_ffmpeg_converts_frames(monkeypatch):
    """audio_read_ffmpeg should convert byte frames to float arrays."""

    class FakeFFmpegFile:
        def __init__(self, *_, sample_rate=None, channels=None, **__):
            self.sample_rate = sample_rate or 22050
            self.channels = channels or 2

        def __iter__(self):
            # Two stereo frames, little-endian 16-bit ints
            yield (np.array([0, 32767, -32768, 16384], dtype=np.int16).tobytes())
            yield b""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(audio, "FFmpegAudioFile", FakeFFmpegFile)

    data, sr = audio.audio_read_ffmpeg("irrelevant", sr=16000, channels=2)

    assert sr == 16000
    assert data.dtype == np.float32
    # Stereo frames should reshape to (2, n)
    assert data.shape[0] == 2
    assert np.max(np.abs(data)) <= 1.0


def test_buf_to_float_scaling():
    """buf_to_float should scale integer buffers to [-1, 1) floats."""

    int_values = np.array([0, 32767, -32768], dtype=np.int16)
    scaled = audio.buf_to_float(int_values.tobytes(), n_bytes=2, dtype=np.float64)

    assert scaled.dtype == np.float64
    assert np.isclose(scaled[1], 0.999969482, atol=1e-6)
    assert np.isclose(scaled[2], -1.0, atol=1e-6)
