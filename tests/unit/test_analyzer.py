"""Unit tests for analyzer utilities.

These tests cover landmark hashing helpers and peak pairing logic
without relying on audio file I/O.
"""

import numpy as np
import pytest

from audfprint2.core import hash_table
from audfprint2.core.analyzer import (
    Analyzer,
    hashes2landmarks,
    hashes_load,
    hashes_save,
    landmarks2hashes,
    locmax,
    peaks_load,
)


class TestLocmax:
    """Tests for local maximum detection."""

    def test_locmax_identifies_peaks(self):
        vec = np.array([1, 3, 2, 5, 1])
        result = locmax(vec)
        np.testing.assert_array_equal(result, np.array([False, True, False, True, False]))

    def test_locmax_indices_flag(self):
        vec = np.array([0, 2, 1, 0, 3])
        result = locmax(vec, indices=True)
        np.testing.assert_array_equal(result, np.array([1, 4]))


class TestLandmarksHashes:
    """Tests for landmark/hash conversions."""

    def test_landmarks2hashes_empty(self):
        empty = np.array([]).reshape(0, 4)
        hashes = landmarks2hashes(empty)
        assert hashes.shape == (0, 2)
        assert hashes.dtype == np.int32

    def test_hash_round_trip_preserves_values(self):
        landmarks = np.array([
            [10, 5, 12, 3],
            [20, 40, 35, 2],
        ])
        hashes = landmarks2hashes(landmarks)
        recovered = hashes2landmarks(hashes)
        assert recovered == [(10, 5, 12, 3), (20, 40, 35, 2)]


class TestPeaksToLandmarks:
    """Tests for forming landmarks from peak lists."""

    def test_peaks2landmarks_respects_time_and_frequency_limits(self):
        analyzer = Analyzer()
        analyzer.mindt = 2
        analyzer.targetdf = 5
        analyzer.targetdt = 5
        analyzer.maxpairsperpeak = 3
        pklist = [(0, 10), (2, 12), (4, 15), (5, 30)]

        landmarks = analyzer.peaks2landmarks(pklist)

        assert (0, 10, 12, 2) in landmarks
        assert (2, 12, 15, 2) in landmarks
        assert (0, 10, 15, 4) not in landmarks  # exceeds frequency difference
        assert all(delta >= analyzer.mindt for _, _, _, delta in landmarks)

    def test_peaks2landmarks_limits_pairs_per_peak(self):
        analyzer = Analyzer()
        analyzer.mindt = 1
        analyzer.targetdf = 100
        analyzer.targetdt = 4
        analyzer.maxpairsperpeak = 2
        pklist = [(0, 5), (1, 6), (2, 7), (3, 8)]

        landmarks = analyzer.peaks2landmarks(pklist)

        first_peak_pairs = [lm for lm in landmarks if lm[0] == 0]
        assert len(first_peak_pairs) == analyzer.maxpairsperpeak


class TestFileRoundTrips:
    """Exercise serialization helpers without touching audio I/O."""

    def test_hashes_save_load_roundtrip(self, tmp_path):
        """hashes_save/load should persist time/hash pairs faithfully."""
        sample = [(1, 10), (2, 20), (3, 30)]
        out = tmp_path / "example.afpt"
        hashes_save(out, sample)

        loaded = hashes_load(out)
        assert loaded == sample

    def test_peaks_save_load_validation(self, tmp_path):
        """peaks_load should reject files without the proper magic header."""
        target = tmp_path / "bad.afpk"
        target.write_bytes(b"not a peak file")

        with pytest.raises(IOError):
            peaks_load(target)


class TestPeakProcessing:
    """Cover spreading logic and pairing constraints."""

    def test_spreadpeaks_respects_base(self):
        """Existing base values should act as a lower bound."""
        analyzer = Analyzer()
        base = np.array([1.0, 1.0, 1.0, 1.0])
        peaks = [(1, 2.0)]

        result = analyzer.spreadpeaks(peaks, npoints=4, width=1.0, base=base)

        # Base should be preserved; highest value should exceed base at the peak.
        assert np.all(result >= base)
        assert result[1] > base[1]

    def test_wavfile2hashes_precomputed_deduplicates(self, tmp_path):
        """Precomputed hash files should be read and deduplicated."""
        analyzer = Analyzer()
        # Duplicate the same hash intentionally
        hashes = np.array([[0, 1], [0, 1], [1, 2]], dtype=np.int32)
        target = tmp_path / "sample.afpt"
        hashes_save(target, hashes)

        result = analyzer.wavfile2hashes(str(target))

        # Precomputed hashes should be read back faithfully
        assert list(map(tuple, result)) == [(0, 1), (0, 1), (1, 2)]

    def test_ingest_updates_counters(self, tmp_path):
        """ingest should store hashes and update analyzer bookkeeping."""
        analyzer = Analyzer()
        table = hash_table.HashTable(hashbits=4, depth=5, maxtime=16)
        hashes = np.array([[0, 1], [2, 2], [4, 3]], dtype=np.int32)
        afpt = tmp_path / "track.afpt"
        hashes_save(afpt, hashes)

        dur, nhashes = analyzer.ingest(table, str(afpt))

        assert nhashes == len(hashes)
        assert analyzer.soundfilecount == 1
        assert analyzer.soundfiletotaldur == pytest.approx(dur)
        # Hash table should know the track name
        assert str(afpt) in table.names
