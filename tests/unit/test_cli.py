"""Tests for CLI helpers that don't require invoking subprocesses."""

import os
from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile

from audfprint2 import cli
from audfprint2.core import analyzer
from audfprint2.utils import audio, stft


class TestFilenameHelpers:
    def test_filename_list_iterator_handles_plain_names(self):
        items = list(cli.filename_list_iterator(["track"], wavdir="/tmp", wavext=".wav", listflag=False))
        assert items == ["/tmp/track.wav"]

    def test_filename_list_iterator_reads_lists(self, tmp_path):
        listing = tmp_path / "files.txt"
        listing.write_text("a\nb\n")
        items = list(cli.filename_list_iterator([str(listing)], wavdir="/base", wavext=".mp3", listflag=True))
        assert items == ["/base/a.mp3", "/base/b.mp3"]

    def test_ensure_dir_creates_nested_directories(self, tmp_path):
        target = tmp_path / "nested" / "dir"
        cli.ensure_dir(str(target))
        assert target.exists()


class TestPrecomputePaths:
    def test_file_precompute_uses_precomputed_hashes(self, tmp_path):
        anlz = analyzer.Analyzer()
        precomp_dir = tmp_path / "precomp"
        source = tmp_path / "source.afpt"
        hashes = np.array([[0, 1], [2, 3]], dtype=np.int32)
        analyzer.hashes_save(source, hashes)

        messages = cli.file_precompute_peaks_or_hashes(
            anlz,
            str(source),
            precompdir=str(precomp_dir),
            hashes_not_peaks=True,
        )

        stored_files = list(precomp_dir.rglob("*.afpt"))
        assert stored_files, "Expected a precomputed file to be written"
        assert any("wrote" in msg for msg in messages)
        loaded = analyzer.hashes_load(stored_files[0])
        assert loaded == [(0, 1), (2, 3)]


class TestSTFT:
    def test_frame_generates_expected_shape(self):
        data = np.arange(10)
        result = stft.frame(data, window_length=4, hop_length=2)
        assert result.shape == (4, 4)
        np.testing.assert_array_equal(result[0], data[:4])

    def test_periodic_hann_properties(self):
        window = stft.periodic_hann(4)
        assert len(window) == 4
        # Periodic Hann should start at zero and be symmetric
        assert window[0] == 0.0
        assert window[-1] == pytest.approx(0.5)
        assert window[-2] == pytest.approx(1.0)
        assert window.max() == pytest.approx(window[2])

    def test_stft_runs_on_simple_signal(self):
        signal = np.ones(16)
        result = stft.stft(signal, n_fft=8, hop_length=4)
        # Expect n_fft/2 + 1 frequency bins
        assert result.shape[0] == 5
        assert result.shape[1] > 0


class TestAudioRead:
    def test_audio_read_wav_via_scipy(self, tmp_path, monkeypatch):
        monkeypatch.setattr(audio, "HAVE_FFMPEG", False)
        sr = 8000
        samples = np.array([0, 32767, -32768], dtype=np.int16)
        stereo = np.stack([samples, samples], axis=-1)
        wav_path = tmp_path / "tone.wav"
        scipy.io.wavfile.write(wav_path, sr, stereo)

        data, returned_sr = audio.audio_read(str(wav_path), sr=sr, channels=1)

        assert returned_sr == sr
        # Mono conversion averages the two channels
        assert data.shape[0] == samples.shape[0]
        assert np.max(np.abs(data)) <= 1.0
