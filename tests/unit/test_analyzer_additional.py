import numpy as np
import pytest

from audfprint2.core import analyzer, hash_table


def test_find_peaks_handles_zero_signal():
    """A zero-valued signal should emit debug but still succeed."""

    a = analyzer.Analyzer()
    zeros = np.zeros(256)
    peaks = a.find_peaks(zeros, sr=a.target_sr)
    assert peaks == []


def test_wavfile2peaks_continue_on_error(monkeypatch):
    """When fail_on_error is False, wavfile2peaks should swallow I/O errors."""

    a = analyzer.Analyzer()
    a.fail_on_error = False
    monkeypatch.setattr(analyzer.audio, "audio_read", lambda *args, **kwargs: (_ for _ in ()).throw(IOError("boom")))

    peaks = a.wavfile2peaks("missing.wav")
    assert peaks == []
    assert a.soundfilecount == 1


def test_wavfile2hashes_multi_shift(monkeypatch):
    """wavfile2hashes should concatenate hashes from shifted peak lists."""

    a = analyzer.Analyzer()
    a.shifts = 2
    a.mindt = 1

    peak_a = [(0, 1), (1, 2)]
    peak_b = [(2, 3)]

    monkeypatch.setattr(a, "wavfile2peaks", lambda filename, shifts=None: [peak_a, peak_b])

    hashes = a.wavfile2hashes("dummy.wav")
    combined_landmarks = a.peaks2landmarks(peak_a + peak_b)
    expected = set(map(tuple, analyzer.landmarks2hashes(combined_landmarks)))
    observed = set(map(tuple, hashes))
    assert observed
    assert expected.issuperset(observed)


def test_extract_features_and_glob2hashtable(tmp_path, monkeypatch):
    """extract_features should reuse global analyzer and feed glob2hashtable."""

    # Create a small precomputed hash file
    hash_path = tmp_path / "track.afpt"
    sample_hashes = [(0, 1), (2, 3)]
    analyzer.hashes_save(hash_path, sample_hashes)

    class Track:
        fn_audio = str(hash_path)

    features = analyzer.extract_features(Track(), density=15.0, n_fft=256, n_hop=128, sr=8000)
    assert len(features) == len(sample_hashes)

    # glob2hashtable should ingest the precomputed hashes
    glob_pattern = str(hash_path)
    monkeypatch.setattr(analyzer.time, "clock", lambda: analyzer.time.perf_counter(), raising=False)
    ht = analyzer.glob2hashtable(glob_pattern, density=10.0)
    assert ht.names


def test_ingest_returns_duration(tmp_path):
    """ingest should compute duration from precomputed peaks file."""

    a = analyzer.Analyzer()
    ht = hash_table.HashTable(hashbits=6, depth=3, maxtime=64)

    peak_path = tmp_path / "sample.afpk"
    peaks = [(0, 1), (5, 2)]
    analyzer.peaks_save(peak_path, peaks)

    dur, nh = a.ingest(ht, str(peak_path))
    assert nh > 0
    assert dur == pytest.approx((max(p[0] for p in peaks) * a.n_hop) / a.target_sr)
